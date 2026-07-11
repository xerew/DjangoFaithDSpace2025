# Proposal Edit Tracking — Design Spec

## Goal

Capture teacher edits to AI-generated activity proposals as structured, timestamped, per-revision data — so acceptance/modification behavior can be analyzed later (e.g. for a paper on teacher acceptance and modification of LLM-generated content). Today the platform only keeps the *latest* edited state (`UserProposalReview.teacher_edited_json`); it has no record of when an edit happened, how many revisions occurred, or what specifically changed.

## Scope

- `Trust-AI-Platform/authoringtool/models.py` — new `ActivityProposalEditEvent` model; new fields on `UserProposalReview`
- `Trust-AI-Platform/authoringtool/views.py` — `edit_proposal_json` gains diff-and-log logic
- `Trust-AI-Platform/authoringtool/admin.py` — register the new model for spot-checking
- One new migration
- **Out of scope:** any UI change (no diff viewer, no "edited" badge), any export/CSV tooling, any change to `accept_proposal`, `reject_proposal`, or the `tasks.py` apply-to-scenario logic. `teacher_edited_json` keeps its current meaning and is still the field `tasks.py` reads when applying an accepted proposal.

---

## 1. Current State (confirmed by reading the code)

- `ActivityProposal.json_action` / `json_translated_action` (models.py:580-581) hold the original LLM output as JSON strings, produced by `parse_llm_proposal` (tasks.py:2774-2883). Confirmed keys relevant to teacher editing: `activity_name`, `content`, `explanation`, `answers` (list of `{text, is_correct, weight}`).
- `UserProposalReview.teacher_edited_json` (models.py:674) is a `JSONField`, populated by `edit_proposal_json` (views.py:2478-2514) with a dict shaped `{"activity_name": str, "content": str, "explanation": str, "answers": [{"text": str}, ...]}`. Note answers here only carry `text` — `is_correct`/`weight` are not present in the edited shape. This is existing behavior and is not being changed.
- `reviewed_at` (models.py:673) is `auto_now=True`, so it is overwritten on every save of a `UserProposalReview` — including accept, reject, and edit. It cannot be used to answer "when was this edited."
- `tasks.py` (~line 3116-3128, in `apply_user_proposals_to_new_scenario`) reads `review.teacher_edited_json` as the override, falling back to `proposal.json_translated_action or proposal.json_action` as the base. This read path is unaffected by this design.

## 2. New Model: `ActivityProposalEditEvent`

Add to `authoringtool/models.py`, after `UserProposalReview`:

```python
class ActivityProposalEditEvent(models.Model):
    review = models.ForeignKey(
        'UserProposalReview', on_delete=models.CASCADE, related_name='edit_events'
    )
    edit_number = models.PositiveIntegerField()
    edited_json = models.JSONField()
    changed_fields = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Activity Proposal Edit Event"
        verbose_name_plural = "Activity Proposal Edit Events"
        ordering = ['review', 'edit_number']
        constraints = [
            models.UniqueConstraint(
                fields=['review', 'edit_number'], name='unique_review_edit_number'
            )
        ]

    def __str__(self):
        return f"Edit #{self.edit_number} on review {self.review_id}"
```

- `review` — scopes the event to the same (proposal, teacher) pair as the existing review.
- `edit_number` — 1-indexed sequence within that review (1st edit, 2nd edit, ...). The unique constraint prevents duplicate numbering if a request is ever double-submitted.
- `edited_json` — full snapshot of the edited state at this revision (same shape as `teacher_edited_json` today), so the complete revision history is reconstructable even though `teacher_edited_json` itself only ever holds the latest.
- `changed_fields` — see §4 below.
- `created_at` — the real per-edit timestamp `reviewed_at` cannot provide.

## 3. New Fields on `UserProposalReview`

Add to `authoringtool/models.py`:

```python
    was_edited = models.BooleanField(default=False)
    edit_count = models.PositiveIntegerField(default=0)
```

Cheap denormalized counters so `UserProposalReview.objects.filter(was_edited=True)` or aggregate queries don't require a join to `ActivityProposalEditEvent` for the common case (e.g. "how many accepted proposals were edited before acceptance").

## 4. Diff Computation

Runs inside `edit_proposal_json`, once per save, comparing the newly submitted `data` dict against a baseline.

**Baseline selection:**
- If this is the first edit for this review (`review.edit_count == 0`), baseline is parsed from `proposal.json_translated_action or proposal.json_action` (same precedence as `tasks.py`).
- Otherwise, baseline is the most recent `ActivityProposalEditEvent.edited_json` for this review (i.e. each edit diffs against the *previous* edit, not always against the original — this gives per-revision granularity, e.g. "how many separate passes did the teacher make and what did each one touch").

**Per-field diff, comparing baseline vs. new `data`:**

For each of `activity_name`, `content`, `explanation` (plain strings; missing baseline key treated as `""`):

```python
def _string_field_diff(old_val, new_val):
    old_val = old_val or ""
    new_val = new_val or ""
    return {
        "changed": old_val != new_val,
        "char_delta": len(new_val) - len(old_val),
    }
```

For `answers` (compare the list of `text` values only, since that's the only key both the baseline and the edited shape reliably carry):

```python
def _answers_field_diff(old_answers, new_answers):
    old_texts = [a.get("text", "") for a in (old_answers or [])]
    new_texts = [a.get("text", "") for a in (new_answers or [])]
    return {
        "changed": old_texts != new_texts,
        "char_delta": sum(len(t) for t in new_texts) - sum(len(t) for t in old_texts),
        "count_delta": len(new_texts) - len(old_texts),
    }
```

`changed_fields` stored on the event:

```python
changed_fields = {
    "activity_name": _string_field_diff(baseline.get("activity_name"), data.get("activity_name")),
    "content": _string_field_diff(baseline.get("content"), data.get("content")),
    "explanation": _string_field_diff(baseline.get("explanation"), data.get("explanation")),
    "answers": _answers_field_diff(baseline.get("answers"), data.get("answers")),
}
```

## 5. Wiring into `edit_proposal_json`

Modify `authoringtool/views.py:2478-2514`. The existing body up through building `data` (lines 2489-2501) is unchanged. Insert the logging step before the existing `review.teacher_edited_json = data` / `review.save()`:

```python
@group_required('teachers')
def edit_proposal_json(request, scenario_id, pk):
    proposal = get_object_or_404(ActivityProposal, pk=pk)
    user = request.user

    review, created = UserProposalReview.objects.get_or_create(
        proposal=proposal,
        user=user,
        defaults={'status': 'new'}
    )

    data = {
        "activity_name": request.POST.get("activity_name", ""),
        "content": request.POST.get("content", ""),
        "explanation": request.POST.get("explanation", ""),
        "answers": []
    }
    for i in range(1, 20):
        key = f"answer_text_{i}"
        val = request.POST.get(key)
        if val:
            data["answers"].append({"text": val.strip()})

    # ── Log this revision as an edit event ──────────────────────────────
    if review.edit_count == 0:
        base_raw = proposal.json_translated_action or proposal.json_action
        try:
            baseline = json.loads(base_raw) if isinstance(base_raw, str) else (base_raw or {})
        except (json.JSONDecodeError, TypeError):
            baseline = {}
    else:
        last_event = review.edit_events.order_by('-edit_number').first()
        baseline = last_event.edited_json if last_event else {}

    changed_fields = {
        "activity_name": _string_field_diff(baseline.get("activity_name"), data.get("activity_name")),
        "content": _string_field_diff(baseline.get("content"), data.get("content")),
        "explanation": _string_field_diff(baseline.get("explanation"), data.get("explanation")),
        "answers": _answers_field_diff(baseline.get("answers"), data.get("answers")),
    }

    ActivityProposalEditEvent.objects.create(
        review=review,
        edit_number=review.edit_count + 1,
        edited_json=data,
        changed_fields=changed_fields,
    )
    review.edit_count += 1
    review.was_edited = True
    # ──────────────────────────────────────────────────────────────────────

    review.teacher_edited_json = data

    if review.status not in ['accepted', 'rejected']:
        review.status = 'new'

    review.save()
    return redirect("proposal_list", scenario_id=scenario_id)
```

`_string_field_diff` and `_answers_field_diff` (§4) are added as module-level helper functions in `views.py`, near `edit_proposal_json`.

The existing `print(...)` debug lines in the current implementation (views.py:2505-2506, 2513) are removed as part of this edit — they are leftover debug output on a path this task is already touching, not scope creep.

## 6. Admin Registration

Add to `authoringtool/admin.py`, after the existing `UserProposalReviewAdmin` (admin.py:633-638), following the same pattern as `ActivityProposalAdmin`:

```python
# ─── ActivityProposalEditEvent ─────────────────────────────────────────────────

@admin.register(ActivityProposalEditEvent)
class ActivityProposalEditEventAdmin(admin.ModelAdmin):
    list_display = ('id', 'review', 'edit_number', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('review__user__username', 'review__proposal__id')
    raw_id_fields = ('review',)
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
```

Add `ActivityProposalEditEvent` to the import block at admin.py:16.

Also update `UserProposalReviewAdmin.list_display` (admin.py:635) to include the new counters:

```python
    list_display = ('id', 'proposal', 'user', 'status', 'was_edited', 'edit_count', 'reviewed_at')
```

## 7. Migration

One migration, auto-generatable via `makemigrations`, containing:
- `CreateModel` for `ActivityProposalEditEvent`
- `AddField` for `UserProposalReview.was_edited` (default `False`)
- `AddField` for `UserProposalReview.edit_count` (default `0`)

No data migration needed — existing `UserProposalReview` rows correctly default to `was_edited=False`, `edit_count=0` (they have no edit history to backfill; `teacher_edited_json` may already be populated on some existing rows from before this change, but reconstructing retroactive `ActivityProposalEditEvent` rows for those is not possible since intermediate revisions were never stored — this is an accepted gap for historical data).

## 8. What Does NOT Change

- `accept_proposal`, `reject_proposal` (views.py:2443-2468) — untouched.
- `tasks.py` apply-to-scenario logic — still reads `review.teacher_edited_json` exactly as before.
- `teacher_edited_json` field itself — still holds only the latest edited state; still what downstream code consumes.
- No template/UI changes.
- No export/CSV tooling — deferred until analysis needs are known.
