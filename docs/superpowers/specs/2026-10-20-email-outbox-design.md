# Email Campaign Outbox — Design

Date: 2026-10-20
Status: approved 2026-10-20

## Why

Admins send email campaigns to teachers from the **Email** tab of the admin
dashboard (`accounts/admin_views.py: admin_send_bulk_email`, task
`accounts/tasks.py: send_bulk_email_campaign`). The task sends to every
recipient in one go over one SMTP connection.

The platform mailbox (`info@trust-ai-lab.eu`, Papaki shared hosting) has an
hourly sending cap. The admin isn't sure of the exact number, but believes it's
about 300 an hour. A campaign to all teachers hit the cap after about 300
emails. Papaki refused every message after that, and the task counted those
recipients as "failed". Nobody can tell who got the email: the campaign stores
only totals (`sent_count`, `failed_count`) and the first 20 error lines.

The admin wants two things:

1. Large campaigns should reach everyone, spread over time, without hitting the
   cap.
2. An outbox they can check, showing who got each campaign and who didn't, so
   they can choose who to resend to.

## Goals

- Each campaign recipient gets a row in an outbox with a status: waiting,
  sending, sent or failed.
- A background job sends waiting rows at **at most 200 campaign emails per
  rolling hour**, configurable in settings. Under a ~300/hour cap, this leaves
  about 100 per hour for password resets.
- When Papaki refuses because of the limit, the user stays **waiting**, not
  failed. Sending pauses and resumes later on its own.
- A user is marked **failed** only when retrying can't help (see "Failure
  rules").
- The admin can open a campaign and see each recipient's name, email, status,
  time sent and failure reason, filtered by status.
- The admin can resend to the users they choose: all failed users, or the
  ones they tick. Nothing is resent automatically.
- Campaigns sent before this change get reconstructed outbox rows (see
  "Existing campaigns") so the admin can resend the one that hit the cap.

## Non-goals

- Password reset emails stay exactly as they are. They are sent straight away
  from the request, outside the queue. They are the only other email the app
  sends today. Any notification emails added later should also skip the queue.
- No unsubscribe links, open or click tracking, or bounce processing. Bounces
  that arrive later in the Papaki inbox are not detected.
- No change to how campaigns are written or targeted (all teachers, selected
  teachers, organizations), or to the email template.
- No switch to another email provider.
- No automatic resending.

## Anti-goals (failure even if it "works")

- **A user gets the same campaign email twice** without the admin choosing to
  resend it to them. Preventing duplicates matters more than speed.
- **A user who was only delayed shows as failed.**
- **Password resets are slowed or blocked** by a running campaign.
- **The system keeps retrying Papaki every few minutes** while it's refusing
  mail, which could get the account flagged.

## Constraints

- Production runs the Docker image, which uses **Django 4.0**. The local venv
  uses 5.2. Migrations and code must work on 4.0 (see commit a75353e8 for an
  earlier 4.0 migration fix).
- The database is Postgres in production. Tests run locally on SQLite with
  `faithDev.settings_test` (see the repo's test notes).
- Celery and Redis are already deployed. A `celerybeat` service runs
  `celery -A faithDev beat -l info`, but **no beat schedule is configured** in
  settings yet. This feature adds the first entry (`CELERY_BEAT_SCHEDULE`).
- The Celery worker runs tasks concurrently, and a slow run can overlap the
  next scheduled one. Two runs of the sending job must never send the same row.
- Email settings come from the environment (`faithDev/settings.py`). The new cap
  follows the same pattern: `EMAIL_CAMPAIGN_HOURLY_LIMIT`, default `200`.

## Decided by the admin

- Hourly cap for campaign email: **200**. Password resets go out immediately,
  outside the queue.
- Campaigns that miss people are **not** resent automatically. The admin
  reviews the list and resends.
- Campaigns go out **in the order they were sent**. A second campaign waits
  for the first.

## How it works

### Outbox rows

A new model in `accounts` (name left to the builder; `BulkEmailDelivery` fits
the existing naming) with one row per campaign recipient:

- campaign (FK, cascade)
- user (FK, `SET_NULL`, so deleting a user keeps the history)
- email address, copied when the campaign is queued (the address the admin
  confirmed)
- recipient display name, copied as well, so the list still makes sense if the
  user is deleted
- status: `waiting`, `sending`, `sent`, `failed`
- error message (blank unless failed or waiting after a refusal)
- attempts (int)
- `sent_at`, `created_at`, `updated_at`
- `reconstructed` (bool, true only for rows rebuilt from old campaigns)

Unique on (campaign, user) where the user is not null. A resend reuses the
existing row: back to `waiting`, error cleared, attempts kept.

`admin_send_bulk_email` creates the rows in the same transaction that creates
the campaign, then returns. It no longer calls `send_bulk_email_campaign.delay`.
The campaign shows **Queued** until its first email is sent.

### The sending job

A Celery beat task runs every 5 minutes. **My call:** 5 minutes is short
enough that a small campaign goes out quickly, and long enough to keep the load
light.

Each run:

1. Stops at once if sending is paused (see below).
2. Works out the allowance: `EMAIL_CAMPAIGN_HOURLY_LIMIT` minus the rows whose
   `sent_at` is within the last 60 minutes, across all campaigns.
3. Claims up to that many `waiting` rows, oldest campaign first, then by row
   id. Claiming must be safe if two runs overlap, for example with
   `select_for_update(skip_locked=True)` and setting the rows to `sending` in
   the same transaction, or with a single-run lock in Redis. The builder
   chooses.
4. Sends each claimed row over one SMTP connection, using the existing message
   building code (`_plain_text_message`, `email/admin_bulk_email.html`).
5. Updates each row as it goes, so a crash loses at most the row being sent.

Campaign counts and status follow from the rows. The existing `status`,
`sent_count`, `failed_count` and `completed_at` fields stay, and are updated
from the rows after each run so the current templates keep working. The
statuses:

- queued: all rows waiting
- sending: some sent, some waiting
- completed: all sent
- partial: none waiting, some failed
- failed: none waiting, none sent

### Failure rules

There are two kinds of problem. **Account-wide** problems stop all sending,
because every email would fail the same way. **Recipient** problems affect
only one row.

| What happened | Row becomes | Sending |
|---|---|---|
| Sent | `sent`, `sent_at` set | continues |
| Recipient refused with a 5xx code that is about the address (unknown user, no such mailbox, bad address) and **not** a rate limit message | `failed`, with the server's message | continues |
| Recipient refused with a 4xx code (temporary, e.g. "451 try again later" for one domain) | stays `waiting`, attempts + 1, skipped for this run | continues |
| The same row has hit a recipient 4xx 3 times | `failed`, with the last message | continues |
| User deleted or deactivated, or no email | `failed`, with the reason | continues |
| Rate limit message (any code; see below), sender refused, login failure, can't connect | back to `waiting`, error noted | **pause** |
| Connection dropped or timed out **while a message was being sent** | that row `failed` with "Sending was interrupted; this person may or may not have received it." Other claimed rows go back to `waiting` | **pause** |

**My call (recipient 4xx):** a temporary refusal for one recipient must not
hold up the whole queue every hour. Rows are skipped and given up after 3
tries, which is what `attempts` is for.

**My call (interrupted send):** when the connection drops mid-message, Papaki
may have accepted it. Marking it waiting could send a duplicate, so it's
treated like a stuck row: failed, with an uncertain note, and the admin
decides. Problems that happen while connecting or logging in, before any
message is sent, are safe to retry, so those rows go back to waiting. Open
the connection separately from sending each message, so the two cases can be
told apart.

**Rate limit messages.** Papaki rejects over-limit mail with a code that
normally means permanent failure. This is the real reply from the campaign that
hit the cap (sent 300, then 140 refused):

```
(554, b'5.7.0 Your message could not be sent. The limit on the number of
allowed outgoing messages was exceeded. Try again later.')
```

So the SMTP code alone can't be trusted. A reply counts as a rate limit, and
**pauses** sending, whatever its code, if its text contains "limit" together
with "exceed" (case-insensitive). That covers this message and the usual
variants ("exceeded the max emails per hour", "rate limit exceeded"). Keep the
matching in one small function with a test that uses the exact text above.

**My call (other 5xx):** a 5xx that isn't a rate limit fails only that row, at
any stage. That includes content or policy rejections. They can be seen and
resent from the list. The rate limit check runs first, so the cap can never
mark people as failed.

**The cap itself:** the admin believes Papaki allows about 300 an hour, and it
did accept 300 in a row before refusing. If the real window turns out to be
daily, 200 an hour will reach it. The pause still handles that: sending tries
again once an hour until Papaki accepts mail again. Nobody is marked failed.

**Pause:** stop this run and store "paused until now + 60 minutes", with the
reason. The builder chooses where to store it: Redis or a small model. Runs
before that time do nothing. The Email tab shows "Sending paused until 15:40:
<reason>" so the admin can see a wrong password or a blocked port. There's no
limit on the number of pauses: a broken setup retries once an hour and stays
visible.

**Rows stuck in `sending`:** a worker can die mid-send, and then nobody knows
whether the email went out. After 30 minutes in `sending`, the row becomes
`failed` with the message "Sending was interrupted; this person may or may not
have received it." It is never resent automatically, so a duplicate is
impossible. The admin decides.

### Checking the outbox

In the **Recent campaigns** table on the Email tab, each campaign shows waiting,
sent and failed counts, the status badge, and a **Details** link.

Details is a staff-only page that lists the campaign's recipients:

- name, email, status, time sent, failure reason
- a "reconstructed" note on rows rebuilt from old campaigns
- a status filter (all, waiting, sent, failed), defaulting to all
- paging or a scroll-friendly table (campaigns can have 1,000+ rows)

### Resending

On the Details page:

- **Resend to all failed** sets every `failed` row back to `waiting`.
- **Resend to selected** sets the ticked rows back to `waiting`, whatever their
  status. If any ticked rows were already sent, the confirmation says how many
  ("12 of these people already received it").
- Both ask for confirmation with the count, and only work on POST, like the
  existing send flow.
- Rows of deleted users can't be resent.

Resent rows go back into the normal queue under the hourly cap.

### Existing campaigns

A data migration creates outbox rows for every campaign that exists before
this change, all marked `reconstructed = True`. The old task went through
recipients in id order, so:

1. Read the user ids named in the campaign's `error_summary` (lines look like
   `User 123: …`). Those users become `failed`, with that line as the reason.
   If the line is a rate limit message, the reason reads "Not sent: hourly
   sending limit reached" instead.
2. Go through the remaining recipients in id order. The first `sent_count` of
   them become `sent` (`sent_at` = the campaign's `completed_at`).
3. Everyone after that becomes `failed`, with the reason "Not sent (rebuilt
   from the campaign's totals)".

For the campaign that hit the cap ("Explore what's new in Trust-AI!": 300
sent, 140 failed, all 20 named errors are the rate limit), this gives the
first 300 recipients by id as sent and the other 140 as not sent. That is the
list the admin will resend from.

This is a best guess, not a record. The Details page shows the "reconstructed"
note on those rows.

**My call (campaigns still in progress at deploy):** if an old campaign is still
`queued` or `sending`, step 2 still applies, so its first `sent_count`
recipients are `sent`. The rest become `waiting` instead of failed, and the
new job finishes them. Nobody who already got it gets it twice.

**Deploy order:** stop the `celery` worker, deploy and run the migration, then
start the worker again. Remove the `send_bulk_email_campaign` task, so a
leftover message for it in Redis fails as an unregistered task instead of
sending a whole campaign outside the queue.

## Not in this version

- **Cancel campaign** and **Try now** (end a pause early) buttons. The admin
  was asked and didn't choose, so they're left out for now. Both are easy to
  add later.

## Left to the builder

- Model, field, URL and template names.
- Whether Details is its own page or a modal.
- How claiming and the pause are stored (database or Redis), as long as the
  anti-goals hold.
- Exact SMTP error classification code, following the table above.
- Admin (`accounts/admin.py`) registration for the new model.

## Testing

Required tests:

- Sending a campaign creates one waiting row per recipient and sends nothing
  immediately.
- One run sends no more than the allowance: the cap minus the emails sent in
  the last hour.
- A 4xx or connection error leaves the row waiting and pauses sending. Runs
  during the pause send nothing.
- A 5xx recipient refusal fails only that row and sending continues.
- Papaki's exact `554 5.7.0 … limit … exceeded` reply pauses sending and
  leaves the row waiting, at the recipient stage and at the data stage.
- A recipient 4xx skips that row for the run, and the third one fails it.
- A connection dropped mid-message fails only that row, with the uncertain
  note, and pauses.
- Two overlapping runs never send the same row.
- A row stuck in `sending` for more than 30 minutes becomes failed and is not
  resent.
- Resend to failed and resend to selected put rows back to waiting. Staff only,
  POST only.
- The reconstruction migration on a campaign with `sent_count=300`, 1,000
  recipients and two named errors gives 300 sent, 2 failed with their reasons,
  and 698 failed as not sent.
- Password reset still sends straight away while a campaign is queued.

Use the locmem email backend, with a stub backend that raises chosen SMTP
exceptions for the failure tests.
