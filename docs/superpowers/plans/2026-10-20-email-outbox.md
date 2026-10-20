# Email Campaign Outbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send admin email campaigns through a per-recipient outbox at a capped
hourly rate, and let admins see and resend per recipient from the Email tab.

**Architecture:** Each campaign recipient gets a `BulkEmailDelivery` row. A
Celery beat task (`drain_email_outbox`, every 5 minutes) calls
`accounts.outbox.drain_outbox()`, which sends waiting rows up to the hourly
allowance and classifies SMTP failures into "fail this row", "skip this row",
or "pause everything for an hour". A pure module, `accounts/outbox_rules.py`,
holds the rate-limit test and the rebuild logic for old campaigns. Both the
engine and a data migration use it.

**Tech Stack:** Django (5.2 locally, **4.0 in the production Docker image**),
Celery beat with Redis, the Django SMTP email backend, Bootstrap 5 templates.

**Spec:** `docs/superpowers/specs/2026-10-20-email-outbox-design.md`

## Global Constraints

- Code and migrations must run on Django 4.0. Don't use features newer than
  4.0, such as `UniqueConstraint(violation_error_message=…)` or `db_comment`.
- Hourly cap: setting `EMAIL_CAMPAIGN_HOURLY_LIMIT`, read from the environment,
  default `200`, counted over the last 60 minutes of `sent_at`.
- Pause length: 60 minutes. Stuck threshold: 30 minutes. Recipient 4xx gives
  up after 3 attempts.
- Rate limit test: the reply text contains `limit` and `exceed`
  (case-insensitive), whatever the code.
- Exact copy:
  - `"Sending was interrupted; this person may or may not have received it."`
  - `"Not sent: hourly sending limit reached"`
  - `"Not sent (rebuilt from the campaign's totals)"`
- Never resend automatically. A row goes from `sent` or `failed` back to
  `waiting` only through the admin resend view.
- Password reset sending (`accounts.views.CustomPasswordResetView`) is not
  touched.
- Staff-only views use the existing `staff_required` decorator. Changes use
  `require_POST`.
- Run tests from `Trust-AI-Platform/` with:
  `../djangofaithvenv/Scripts/python.exe manage.py test <labels> --settings=faithDev.settings_test --noinput`
  This uses SQLite and has no `select_for_update`, so claiming must use a
  conditional `UPDATE`.

## Review Focus

- Two campaigns waiting at once: the older campaign's rows go first (test in
  Task 3: `test_older_campaign_goes_first`).
- More sent in the last hour than the limit, for example after the limit was
  lowered: the allowance is 0, not negative, and nothing is sent (Task 3:
  `test_allowance_never_negative`).
- A user deactivated, deleted, or with a changed email after queueing:
  deactivated or deleted users fail with a reason, and a changed email still
  goes to the address that was queued (Task 3:
  `test_deactivated_and_deleted_users_fail_with_reason`,
  `test_sends_to_queued_address`).
- Resend with delivery ids from another campaign, or a double submit: other
  campaigns' rows are ignored, and resending rows that are already waiting
  changes nothing (Task 5: `test_resend_ignores_other_campaigns_rows`,
  `test_resend_twice_is_harmless`).
- A campaign with 1,000+ recipients on the Details page: paged 100 per page,
  and the status filter survives paging (Task 5:
  `test_details_pages_and_keeps_filter`).

---

### Task 1: Pure rules module

**Files:**
- Create: `Trust-AI-Platform/accounts/outbox_rules.py`
- Test: `Trust-AI-Platform/accounts/test_email_outbox.py` (new; later tasks add
  classes to it)

**Interfaces:**
- Produces, in `accounts/outbox_rules.py`, with no Django model imports:
  - `INTERRUPTED_REASON`, `RATE_LIMIT_REASON`, `REBUILT_REASON` (exact copy
    above)
  - `is_rate_limit(text: str) -> bool`
  - `reconstruct_deliveries(recipient_ids: list[int], sent_count: int, error_summary: str, in_progress: bool) -> dict[int, tuple[str, str]]`
    maps each user id to `(status, error)`, where status is `'sent'`,
    `'failed'` or `'waiting'`.

- [ ] **Step 1: Write the failing tests** (`SimpleTestCase` class `OutboxRulesTests`)

```python
PAPAKI = "(554, b'5.7.0 Your message could not be sent. The limit on the number of allowed outgoing messages was exceeded. Try again later.')"

def test_papaki_reply_is_rate_limit(self):
    self.assertTrue(is_rate_limit(PAPAKI))
    self.assertTrue(is_rate_limit('452 Rate LIMIT Exceeded'))
    self.assertFalse(is_rate_limit("550 5.1.1 <x@y.eu>: user unknown"))

def test_reconstruct_matches_real_campaign(self):
    ids = list(range(1, 441))
    summary = '\n'.join(f'User {i}: {PAPAKI}' for i in range(301, 321))
    rows = reconstruct_deliveries(ids, 300, summary, in_progress=False)
    self.assertEqual([i for i in ids if rows[i][0] == 'sent'], list(range(1, 301)))
    self.assertEqual(rows[301], ('failed', RATE_LIMIT_REASON))
    self.assertEqual(rows[400], ('failed', REBUILT_REASON))

def test_reconstruct_named_non_limit_error_keeps_line(self):
    rows = reconstruct_deliveries([1, 2, 3, 4], 3, 'User 2: inactive account or missing email', False)
    self.assertEqual(rows[2], ('failed', 'User 2: inactive account or missing email'))
    self.assertEqual([rows[i][0] for i in (1, 3, 4)], ['sent', 'sent', 'sent'])

def test_reconstruct_in_progress_leaves_rest_waiting(self):
    rows = reconstruct_deliveries([1, 2, 3], 1, '', in_progress=True)
    self.assertEqual([rows[i][0] for i in (1, 2, 3)], ['sent', 'waiting', 'waiting'])
```

- [ ] **Step 2: Run** `... test accounts.test_email_outbox` and check it fails with `ImportError`.
- [ ] **Step 3: Implement** `is_rate_limit` and `reconstruct_deliveries`, following spec "Existing campaigns" steps 1–3. Parse named lines with `re.findall(r'^User (\d+): (.*)$', summary, re.M)`. A named line that `is_rate_limit` gets `RATE_LIMIT_REASON`; any other named line keeps the whole `User N: …` line as its reason.
- [ ] **Step 4: Run** the tests and check they pass.
- [ ] **Step 5: Commit** `Add outbox rules for rate limits and rebuilding old campaigns`

---

### Task 2: Models, schema migration, admin

**Files:**
- Modify: `Trust-AI-Platform/accounts/models.py` (after `BulkEmailCampaign`)
- Create: `Trust-AI-Platform/accounts/migrations/0006_bulkemaildelivery_emailoutboxstate.py` (generated)
- Modify: `Trust-AI-Platform/accounts/admin.py`
- Test: `accounts/test_email_outbox.py`, class `CampaignCountsTests`

**Interfaces:**
- Produces:
  - `BulkEmailDelivery(campaign FK related_name='deliveries' CASCADE, user FK User null SET_NULL related_name='bulk_email_deliveries', email EmailField, recipient_name CharField(150, blank), status CharField(20) default 'waiting', error TextField(blank), attempts PositiveIntegerField(0), claimed_at/sent_at DateTimeField(null), reconstructed BooleanField(False), created_at auto_now_add, updated_at auto_now)`
    - Constants `STATUS_WAITING='waiting'`, `STATUS_SENDING='sending'`, `STATUS_SENT='sent'`, `STATUS_FAILED='failed'`.
    - `Meta`: ordering `['id']`. `UniqueConstraint(fields=['campaign', 'user'], condition=Q(user__isnull=False), name='bulk_email_delivery_unique_user')`. Indexes on `(status, sent_at)` and `(campaign, status)`.
  - `EmailOutboxState(paused_until DateTimeField null, pause_reason TextField blank, updated_at auto_now)`
    - `@classmethod load() -> EmailOutboxState` returns the singleton (pk=1), creating it if needed.
    - `is_paused(now) -> bool`
  - `BulkEmailCampaign.refresh_counts() -> None`: recomputes `sent_count`, `failed_count` and `status` from its rows. Sets `started_at` when it is null and a row has been sent. Sets `completed_at` when no rows are waiting or sending, and clears it otherwise. Saves those fields.
  - `BulkEmailCampaign.waiting_count` (property): the number of waiting and sending rows.

- [ ] **Step 1: Write the failing tests**

```python
def test_status_follows_rows(self):
    # campaign with 3 deliveries; helper make_rows(campaign, ['waiting','waiting','waiting'])
    c.refresh_counts(); self.assertEqual(c.status, 'queued')
    set_statuses(['sent', 'waiting', 'waiting']); c.refresh_counts()
    self.assertEqual((c.status, c.sent_count, c.waiting_count), ('sending', 1, 2))
    self.assertIsNone(c.completed_at)
    set_statuses(['sent', 'sent', 'sent']); c.refresh_counts()
    self.assertEqual(c.status, 'completed'); self.assertIsNotNone(c.completed_at)
    set_statuses(['sent', 'failed', 'sent']); c.refresh_counts()
    self.assertEqual((c.status, c.failed_count), ('partial', 1))
    set_statuses(['failed', 'failed', 'failed']); c.refresh_counts()
    self.assertEqual(c.status, 'failed')

def test_one_row_per_user_per_campaign(self):
    with self.assertRaises(IntegrityError):
        BulkEmailDelivery.objects.create(campaign=c, user=u, email='a@b.eu')  # second time

def test_outbox_state_singleton(self):
    self.assertEqual(EmailOutboxState.load().pk, EmailOutboxState.load().pk)
    s = EmailOutboxState.load(); s.paused_until = now + timedelta(minutes=5); s.save()
    self.assertTrue(EmailOutboxState.load().is_paused(now))
    self.assertFalse(EmailOutboxState.load().is_paused(now + timedelta(minutes=6)))
```

- [ ] **Step 2: Run** them and check they fail.
- [ ] **Step 3: Implement** the models and `refresh_counts`, using the status rules in spec "The sending job". Then run `makemigrations accounts`. Open the generated file and check it uses only `CreateModel`, `AddConstraint`, `AddIndex` and `Q`, which all work on Django 4.0. Register `BulkEmailDelivery` in `admin.py` as read-only (`list_display` id, campaign, email, status, sent_at, attempts; `list_filter` status, reconstructed; `search_fields` email, recipient_name; every field in `readonly_fields`).
- [ ] **Step 4: Run** the tests and check they pass. Run `... test accounts` too: it should show no new failures.
- [ ] **Step 5: Commit** `Add per-recipient email delivery rows and outbox pause state`

---

### Task 3: Sending engine, beat task, settings

**Files:**
- Create: `Trust-AI-Platform/accounts/outbox.py`
- Modify: `Trust-AI-Platform/accounts/tasks.py`. Replace the file's task. Move `_plain_text_message` into `outbox.py` as `plain_text_message`.
- Modify: `Trust-AI-Platform/faithDev/settings.py`. Add next to the email settings: `EMAIL_CAMPAIGN_HOURLY_LIMIT = int(os.environ.get('EMAIL_CAMPAIGN_HOURLY_LIMIT', '200'))`. Add next to the Celery settings: `CELERY_BEAT_SCHEDULE = {'drain-email-outbox': {'task': 'accounts.tasks.drain_email_outbox', 'schedule': 300.0}}`.
- Test: `accounts/test_email_outbox.py`, class `DrainOutboxTests`, using `@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend', EMAIL_CAMPAIGN_HOURLY_LIMIT=200)`

**Interfaces:**
- Consumes: Task 1 rules, Task 2 models.
- Produces, in `accounts/outbox.py`:
  - `PAUSE_FOR = timedelta(minutes=60)`, `STUCK_AFTER = timedelta(minutes=30)`, `MAX_TEMPORARY_ATTEMPTS = 3`
  - `plain_text_message(campaign, user) -> str` (the old `_plain_text_message`, unchanged)
  - `build_message(delivery, connection) -> EmailMultiAlternatives`. It goes to `delivery.email`, uses the same subject, plain text and `email/admin_bulk_email.html` context as the old task, and `DEFAULT_FROM_EMAIL`.
  - `claim(delivery_id: int, now) -> bool`: a conditional update from `waiting` to `sending`, which sets `claimed_at=now` and `attempts=F('attempts')+1`. Returns True only if exactly one row changed.
  - `remaining_allowance(now) -> int`: `max(0, limit - rows with sent_at >= now - 1h)`
  - `release_stuck(now) -> int`: rows in `sending` with `claimed_at < now - STUCK_AFTER` become `failed` with `INTERRUPTED_REASON`.
  - `pause(reason: str, now) -> None`
  - `drain_outbox(now=None) -> dict` with keys `sent`, `failed`, `deferred` (ints), `paused` (bool) and `reason` (str).
- Produces, in `accounts/tasks.py`: `@shared_task drain_email_outbox() -> dict`, which calls `drain_outbox()`. `send_bulk_email_campaign` is deleted.

`drain_outbox` order of work:
1. `release_stuck`.
2. Return if paused.
3. Work out the allowance, and return if it's 0.
4. Open the connection. If that fails, pause and return.
5. Go through waiting rows ordered by `campaign__created_at, campaign_id, id`. `claim` each one, and stop once `sent` reaches the allowance.
6. Classify each failure with the spec's "Failure rules" table. Check `is_rate_limit(str(exc))` first. Then:
   - `SMTPRecipientsRefused`: take the code from `exc.recipients`.
   - Other `SMTPResponseException`: `exc.smtp_code`.
   - `SMTPSenderRefused` or `SMTPAuthenticationError`: pause.
   - `SMTPServerDisconnected`, `socket.timeout` or `OSError` during send: interrupted.
   - Any other exception: fail the row with `str(exc)`.
7. Close the connection, then call `refresh_counts()` on every campaign touched.

- [ ] **Step 1: Write the failing tests.** Use a helper `queue(campaign, users)` that creates waiting rows, and a helper `failing_backend(exc_at: dict[int, Exception])` that patches `locmem.EmailBackend.send_messages` to raise on the Nth call. Tests:
  - `test_sends_waiting_rows_and_updates_campaign`: 3 rows → 3 in `mail.outbox`, all `sent` with `sent_at`, campaign `completed`.
  - `test_respects_hourly_allowance`: limit 200 with 150 rows sent 30 minutes ago and 100 waiting → exactly 50 sent.
  - `test_allowance_never_negative`: 250 rows sent in the last hour, limit 200 → `remaining_allowance == 0`, nothing sent.
  - `test_older_campaign_goes_first`: allowance 2, campaigns A (older) and B each have 2 rows → A's two are sent, B's wait.
  - `test_papaki_limit_pauses_and_keeps_waiting`: raise `SMTPDataError(554, PAPAKI_BYTES)` on the 2nd send. Expect 1 sent, the 2nd and later rows `waiting` with the error noted, `EmailOutboxState.load().is_paused(now)`, and a second `drain_outbox(now + 10 min)` sending nothing. `drain_outbox(now + 61 min)` sends the rest.
  - `test_papaki_limit_at_recipient_stage_also_pauses`: the same with `SMTPRecipientsRefused({'x@y.eu': (554, PAPAKI_BYTES)})`.
  - `test_permanent_5xx_fails_only_that_row`: `SMTPRecipientsRefused({...: (550, b'5.1.1 user unknown')})` on row 1 → row 1 `failed` with the message, rows 2–3 sent, not paused.
  - `test_temporary_4xx_skips_then_fails_after_three`: a 451 on the same row in three runs (each 61 minutes apart) → `waiting` after runs 1 and 2, other rows sent, `failed` after run 3.
  - `test_disconnect_mid_send_is_uncertain_and_pauses`: `SMTPServerDisconnected` → that row is `failed` with `INTERRUPTED_REASON`, later rows `waiting`, paused.
  - `test_cannot_connect_pauses_without_touching_rows`: patch `accounts.outbox.get_connection` so `open()` raises `OSError` → every row `waiting` with attempts 0, paused.
  - `test_claim_is_exclusive`: `claim(id)` returns True, then False. A row already in `sending` is not sent by `drain_outbox`.
  - `test_stuck_sending_becomes_uncertain_failure`: a row `sending` with `claimed_at` 31 minutes ago → `failed` with `INTERRUPTED_REASON`, never sent.
  - `test_deactivated_and_deleted_users_fail_with_reason`: one user `is_active=False`, one user deleted (row `user=None`) → both `failed` with a reason, nothing sent to them.
  - `test_sends_to_queued_address`: the row's email is `old@x.eu` and `user.email` was changed → message `to == ['old@x.eu']`.
  - `test_beat_schedule_runs_drain_task`: `settings.CELERY_BEAT_SCHEDULE['drain-email-outbox']['task'] == 'accounts.tasks.drain_email_outbox'`, and the task is registered in `celery.current_app.tasks`.
- [ ] **Step 2: Run** them and check they fail.
- [ ] **Step 3: Implement** `outbox.py`, the new task and the settings as specified above.
- [ ] **Step 4: Run** `... test accounts.test_email_outbox` and check it passes.
- [ ] **Step 5: Commit** `Send campaign email from the outbox at a capped hourly rate`

---

### Task 4: Queue on send, data migration for old campaigns

**Files:**
- Modify: `Trust-AI-Platform/accounts/admin_views.py`, `admin_send_bulk_email` (around lines 152–220), and the import at line 24.
- Create: `Trust-AI-Platform/accounts/migrations/0007_backfill_bulk_email_deliveries.py`
- Modify: `Trust-AI-Platform/accounts/tests.py`, `AdminBulkEmailTests` (lines 651–820)
- Test: `accounts/test_email_outbox.py`, classes `QueueOnSendTests` and `BackfillMigrationTests`

**Interfaces:**
- Consumes: `reconstruct_deliveries` (Task 1), models (Task 2), `drain_outbox` (Task 3).
- Produces: `admin_send_bulk_email` creates the campaign, its recipients and one `waiting` delivery per recipient inside `transaction.atomic()`. Each delivery has the email and `recipient_name = get_full_name() or username`. The view returns the same JSON keys as now, with `status='queued'`, and calls no Celery task. The migration module exposes `backfill(apps, schema_editor)`.

- [ ] **Step 1: Write the failing tests**
  - `QueueOnSendTests.test_send_creates_waiting_rows_and_sends_nothing`: post as in `AdminBulkEmailTests._send` → `len(mail.outbox) == 0`, two `waiting` deliveries, campaign `queued`.
  - `QueueOnSendTests.test_password_reset_sends_immediately_while_campaign_waits`: with a queued campaign and the outbox paused, POST `reverse('password_reset')` with a teacher's email → `len(mail.outbox) == 1` straight away, and the campaign rows are still `waiting`.
  - `BackfillMigrationTests.test_rebuilds_partial_campaign`: a campaign with 440 recipients, `sent_count=300`, `failed_count=140`, `status='partial'`, `completed_at=T` and a Papaki `error_summary` naming users 301–320 by position. Call `backfill(django.apps.apps, None)`. Expect 440 rows, all `reconstructed`. The first 300 by id are `sent` with `sent_at == T`, and the rest are `failed`.
  - `test_rebuilds_in_progress_campaign_as_waiting`: `status='sending'`, `sent_count=1` of 3 → 1 `sent`, 2 `waiting`.
  - `test_backfill_skips_campaigns_that_already_have_rows`: running it twice creates no duplicates.
- [ ] **Step 2: Run** them and check they fail.
- [ ] **Step 3: Implement** the view change. Write the migration: depends on `0006`; `RunPython(backfill, migrations.RunPython.noop)`; `backfill` iterates over campaigns without deliveries and uses `reconstruct_deliveries(..., in_progress=status in ('queued', 'sending'))`, then `bulk_create`. Afterwards, update `AdminBulkEmailTests`: each test that asserts on `mail.outbox` or `sent_count` calls `drain_outbox()` after `_send(...)` before asserting. Drop the `CELERY_TASK_*` overrides.
- [ ] **Step 4: Run** `... test accounts` and check there are no failures in `AdminBulkEmailTests` or `test_email_outbox`.
- [ ] **Step 5: Commit** `Queue campaigns in the outbox and rebuild rows for past campaigns`

---

### Task 5: Details page, resend, Email tab

**Files:**
- Modify: `Trust-AI-Platform/accounts/admin_views.py`. Add two views, and add `outbox_state` to `admin_dashboard`'s context.
- Modify: `Trust-AI-Platform/accounts/urls.py`. Add after line 42:
  - `path('admin/email/campaigns/<int:campaign_id>/', admin_views.admin_bulk_email_campaign, name='admin_bulk_email_campaign')`
  - `path('admin/email/campaigns/<int:campaign_id>/resend/', admin_views.admin_bulk_email_resend, name='admin_bulk_email_resend')`
- Create: `Trust-AI-Platform/accounts/templates/accounts/admin_email_campaign.html`. It extends `main.html` with `{% block atcontent %}`, styled like the Email tab cards.
- Modify: `Trust-AI-Platform/accounts/templates/accounts/_admin_email_tab.html` (Recent campaigns table, lines 163–193)
- Test: `accounts/test_email_outbox.py`, class `CampaignDetailsTests`

**Interfaces:**
- Consumes: models (Task 2), `EmailOutboxState.load()`.
- Produces:
  - `admin_bulk_email_campaign(request, campaign_id)`: GET, staff only.
    - Query param `status` in `all|waiting|sent|failed`, default `all`. `waiting` also includes `sending`.
    - `Paginator` of 100 per page, with `page` as the query param.
    - Context: `campaign`, `page_obj`, `status_filter`, `counts` (dict with keys `all`, `waiting`, `sent`, `failed`), `outbox_state`, and `resent` (int from the `resent` query param, default 0).
  - `admin_bulk_email_resend(request, campaign_id)`: `require_POST`, staff only.
    - `mode=failed` sets every `failed` row with a user to `waiting`.
    - `mode=selected` sets rows whose ids are in `delivery_ids`, belong to this campaign, have a user, and aren't already `waiting` or `sending`, to `waiting`.
    - Clears `error`, keeps `attempts`, calls `campaign.refresh_counts()`, and redirects to the details URL with `?resent=<n>`.

What the pages show:
- **Details page:**
  - Header: the subject, the audience (`target_summary`), the counts and the status badge.
  - When `outbox_state.is_paused(now)`: an alert reading "Sending paused until HH:MM: <reason>".
  - Filter pills with counts.
  - A table with a checkbox, name, email, status badge (Sending shows as "Sending"), time sent and reason, plus a small "reconstructed" note on those rows. Rows with no user have a disabled checkbox.
  - Buttons: **Resend to all failed (N)** and **Resend to selected**. Each submits a POST form, with a JS `confirm()` that states the count. For selected rows, the confirm also says "X of these people already received it" when ticked rows have `data-status="sent"`.
  - When `resent > 0`: a success alert reading "Queued N emails again."
- **Email tab:** the Delivery cell shows `N sent`, `N waiting` (when > 0) and `N failed` (when > 0). The table gets a **Details** link per row. When the outbox is paused, the same pause alert appears above the table. Below the confirm modal's "This action queues individual emails…" line, add: "Up to {{ hourly_limit }} emails go out per hour; sending starts within 5 minutes." `hourly_limit` comes from settings, through the dashboard context.

- [ ] **Step 1: Write the failing tests**
  - `test_details_lists_recipients_with_status_and_reason`: rows sent, failed (reason "550 user unknown") and waiting → 200, and the page contains each email and the reason.
  - `test_details_filter_failed`: `?status=failed` → only the failed row's email appears.
  - `test_details_pages_and_keeps_filter`: 250 failed rows → page 3 has 50 rows, and the next/previous links keep `status=failed`.
  - `test_resend_failed_requeues_only_failed_rows_with_users`: one failed, one failed with `user=None`, one sent → only the first becomes `waiting`. The redirect has `resent=1`.
  - `test_resend_selected_can_requeue_sent_rows`: select a sent row → `waiting`.
  - `test_resend_ignores_other_campaigns_rows`: an id from another campaign → that row is unchanged.
  - `test_resend_twice_is_harmless`: the second identical POST → `resent=0`, and the rows are unchanged.
  - `test_details_and_resend_are_staff_only`: a regular user gets 403 on both. GET on resend gets 405.
  - `test_email_tab_shows_waiting_count_details_link_and_pause`: the dashboard contains the details URL and "waiting". With a pause set, it contains "Sending paused until".
- [ ] **Step 2: Run** them and check they fail.
- [ ] **Step 3: Implement** the views, URLs, details template and tab changes.
- [ ] **Step 4: Run** `... test accounts` and check it passes with no new failures.
- [ ] **Step 5: Commit** `Show campaign recipients on the Email tab and let admins resend`

---

### Task 6: Verify on the production stack (Django 4.0, Postgres)

**Files:** none changed unless a check fails.

- [ ] **Step 1: Run the new tests in the production image** from the repo root:
  `MSYS_NO_PATHCONV=1 docker compose run --rm --no-deps -v "$(pwd -W)/Trust-AI-Platform:/app" web python manage.py test accounts.test_email_outbox --noinput`.
  Expect `OK`. Postgres enforces the FK and the conditional unique constraint, and Django 4.0 runs the migrations.
- [ ] **Step 2: Run the whole `accounts` suite in the same image.** Compare failures by test name with the baseline in the test notes, so any failure in `accounts` that isn't already known is fixed.
- [ ] **Step 3: Write the deploy note into the final handoff message:**
  - stop `celery`
  - pull and build
  - `migrate`
  - start `celery`, then **recreate** `celerybeat`, because it now has a schedule
  - check `docker compose logs celerybeat` shows `drain-email-outbox`
