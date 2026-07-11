# Proposal Edit Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log every teacher edit to an AI-generated activity proposal as a timestamped, per-revision event with a per-field diff, so acceptance/modification behavior can be analyzed later (research instrumentation for a paper on teacher acceptance and modification of LLM-generated content).

**Architecture:** Two sequential tasks — the data model first (one new model, two new fields, one migration, admin registration), then the diff logic and view wiring that populates it. No UI changes. No changes to `accept_proposal`, `reject_proposal`, or the `tasks.py` apply-to-scenario logic.

**Tech Stack:** Django 5.1 · SQLite (dev) — model + view change only, no JS/template work

## Global Constraints

- No UI changes: no diff viewer, no "edited" badge, no template edits.
- No export/CSV tooling — deferred to a future task.
- Do NOT modify `accept_proposal`, `reject_proposal` (`Trust-AI-Platform/authoringtool/views.py`), or `apply_user_proposals_to_new_scenario` (`Trust-AI-Platform/authoringtool/tasks.py`).
- `UserProposalReview.teacher_edited_json` keeps its exact current meaning (latest edited state, read by `tasks.py`) — do not repurpose it.
- Each edit's diff is computed against the immediately prior edit (or the original LLM proposal for the first edit) — not always against the original.
- Baseline precedence for the first edit is `proposal.json_translated_action or proposal.json_action`, matching the precedence already used in `tasks.py` (~line 3119).

---

### Task 1: Data model — `ActivityProposalEditEvent`, review counters, migration, admin

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/models.py`
- Modify: `Trust-AI-Platform/authoringtool/admin.py`
- Modify: `Trust-AI-Platform/authoringtool/tests.py`
- Create: one Django migration under `Trust-AI-Platform/authoringtool/migrations/`

**Interfaces:**
- Produces: `ActivityProposalEditEvent` model with fields `review` (FK to `UserProposalReview`, `related_name='edit_events'`), `edit_number` (int), `edited_json` (JSONField), `changed_fields` (JSONField), `created_at` (auto_now_add). Unique constraint on `(review, edit_number)`.
- Produces: `UserProposalReview.was_edited` (BooleanField, default `False`) and `UserProposalReview.edit_count` (PositiveIntegerField, default `0`).
- Consumed by: Task 2 (the view wiring reads `review.edit_count`, `review.edit_events`, and creates `ActivityProposalEditEvent` rows).

- [ ] **Step 1: Write the failing tests**

  Add these imports to the top of `Trust-AI-Platform/authoringtool/tests.py` (extend the existing `from authoringtool.models import (...)` block — do not create a second import line):

  ```python
  from django.db import IntegrityError, transaction
  ```

  Add `ActivityProposalEditEvent`, `ActivityProposal`, `ActivityType`, `UserProposalReview` to the existing `from authoringtool.models import (...)` block at the top of the file (some may already be present — only add the ones missing).

  Append this test class to the end of `Trust-AI-Platform/authoringtool/tests.py`:

  ```python
  class ActivityProposalEditEventModelTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('teacher_edit1', password='pass')
          self.scenario = Scenario.objects.create(
              name='Edit Event Scenario', created_by=self.user, updated_by=self.user
          )
          self.phase = Phase.objects.create(
              name='Phase 1', scenario=self.scenario, created_by=self.user, updated_by=self.user
          )
          self.activity_type = ActivityType.objects.create(
              name='Explanation', created_by=self.user, updated_by=self.user
          )
          self.activity = Activity.objects.create(
              name='Act 1', text='Hello', scenario=self.scenario, phase=self.phase,
              activity_type=self.activity_type, created_by=self.user, updated_by=self.user,
          )
          self.proposal = ActivityProposal.objects.create(
              scenario=self.scenario, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='raw', translated_action='raw',
              json_action=json.dumps({
                  "activity_name": "Act 1", "content": "Old content",
                  "explanation": "Old exp", "answers": [],
              }),
              json_translated_action=json.dumps({
                  "activity_name": "Act 1", "content": "Old content",
                  "explanation": "Old exp", "answers": [],
              }),
          )
          self.review = UserProposalReview.objects.create(proposal=self.proposal, user=self.user)

      def test_review_defaults(self):
          self.assertFalse(self.review.was_edited)
          self.assertEqual(self.review.edit_count, 0)

      def test_create_edit_event(self):
          event = ActivityProposalEditEvent.objects.create(
              review=self.review, edit_number=1,
              edited_json={
                  "activity_name": "Act 1", "content": "New content",
                  "explanation": "Old exp", "answers": [],
              },
              changed_fields={"content": {"changed": True, "char_delta": 3}},
          )
          self.assertEqual(self.review.edit_events.count(), 1)
          self.assertEqual(event.edit_number, 1)

      def test_unique_edit_number_per_review(self):
          ActivityProposalEditEvent.objects.create(
              review=self.review, edit_number=1, edited_json={}, changed_fields={},
          )
          with self.assertRaises(IntegrityError):
              with transaction.atomic():
                  ActivityProposalEditEvent.objects.create(
                      review=self.review, edit_number=1, edited_json={}, changed_fields={},
                  )
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  Run (from `Trust-AI-Platform/`):
  ```bash
  python manage.py test authoringtool.tests.ActivityProposalEditEventModelTests -v 2
  ```
  Expected: FAIL with `ImportError: cannot import name 'ActivityProposalEditEvent'` (or `AttributeError` on `was_edited`/`edit_count` if the import line is adjusted manually).

- [ ] **Step 3: Add the model and fields**

  In `Trust-AI-Platform/authoringtool/models.py`, replace:

  ```python
      reviewed_at = models.DateTimeField(auto_now=True)
      teacher_edited_json = models.JSONField(null=True, blank=True)
      rejection_reasons = models.JSONField(default=list, blank=True)

      class Meta:
          unique_together = ('proposal', 'user')  # one review per user per proposal
  ```

  with:

  ```python
      reviewed_at = models.DateTimeField(auto_now=True)
      teacher_edited_json = models.JSONField(null=True, blank=True)
      rejection_reasons = models.JSONField(default=list, blank=True)
      was_edited = models.BooleanField(default=False)
      edit_count = models.PositiveIntegerField(default=0)

      class Meta:
          unique_together = ('proposal', 'user')  # one review per user per proposal
  ```

  Then, in the same file, replace:

  ```python
      def reject(self, reasons=None):
          """Transition any status -> REJECTED, recording reasons.
          Q-update (−1 on chosen action, positive nudges per reason) handled by post_save signal."""
          if self.status == 'rejected':
              return
          self.rejection_reasons = reasons or []
          self.status = 'rejected'
          self.save(update_fields=['status', 'reviewed_at', 'rejection_reasons'])

  # ─────────────────────────────────────────────────────────────────────────────
  # Signals: ensure Q-value updates even if status set directly in admin/UI
  # ─────────────────────────────────────────────────────────────────────────────
  ```

  with:

  ```python
      def reject(self, reasons=None):
          """Transition any status -> REJECTED, recording reasons.
          Q-update (−1 on chosen action, positive nudges per reason) handled by post_save signal."""
          if self.status == 'rejected':
              return
          self.rejection_reasons = reasons or []
          self.status = 'rejected'
          self.save(update_fields=['status', 'reviewed_at', 'rejection_reasons'])


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

  # ─────────────────────────────────────────────────────────────────────────────
  # Signals: ensure Q-value updates even if status set directly in admin/UI
  # ─────────────────────────────────────────────────────────────────────────────
  ```

- [ ] **Step 4: Generate the migration**

  From `Trust-AI-Platform/`:
  ```bash
  python manage.py makemigrations authoringtool
  ```
  Expected: a new migration file listing `Add field was_edited to userproposalreview`, `Add field edit_count to userproposalreview`, and `Create model ActivityProposalEditEvent`.

- [ ] **Step 5: Apply the migration**

  ```bash
  python manage.py migrate authoringtool
  ```
  Expected: `Applying authoringtool.00XX_...  OK`.

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  python manage.py test authoringtool.tests.ActivityProposalEditEventModelTests -v 2
  ```
  Expected: `OK` (3 tests).

- [ ] **Step 7: Register the model in admin**

  In `Trust-AI-Platform/authoringtool/admin.py`, update the import at line 16 — replace:

  ```python
      ActivityFlag, ActivityProposal, QValue, UserProposalReview, Language,
  ```

  with:

  ```python
      ActivityFlag, ActivityProposal, ActivityProposalEditEvent, QValue, UserProposalReview, Language,
  ```

  Then replace the existing `UserProposalReviewAdmin` block:

  ```python
  @admin.register(UserProposalReview)
  class UserProposalReviewAdmin(admin.ModelAdmin):
      list_display = ('id', 'proposal', 'user', 'status', 'reviewed_at')
      list_filter = ('status', 'reviewed_at')
      search_fields = ('user__username', 'proposal__id')
      ordering = ('-reviewed_at',)
  ```

  with:

  ```python
  @admin.register(UserProposalReview)
  class UserProposalReviewAdmin(admin.ModelAdmin):
      list_display = ('id', 'proposal', 'user', 'status', 'was_edited', 'edit_count', 'reviewed_at')
      list_filter = ('status', 'reviewed_at')
      search_fields = ('user__username', 'proposal__id')
      ordering = ('-reviewed_at',)


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

- [ ] **Step 8: Manually verify in admin**

  ```bash
  python manage.py runserver
  ```
  Navigate to `/admin/authoringtool/userproposalreview/` — confirm the `was_edited` and `edit_count` columns appear (both `False`/`0` for existing rows). Navigate to `/admin/authoringtool/activityproposaleditevent/` — confirm the empty list page loads without error.

- [ ] **Step 9: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/models.py Trust-AI-Platform/authoringtool/admin.py Trust-AI-Platform/authoringtool/tests.py Trust-AI-Platform/authoringtool/migrations/
  git commit -m "Add ActivityProposalEditEvent model and review edit counters"
  ```

---

### Task 2: Diff computation and view wiring

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/views.py`
- Modify: `Trust-AI-Platform/authoringtool/tests.py`

**Interfaces:**
- Consumes: `ActivityProposalEditEvent` model, `UserProposalReview.was_edited`/`edit_count` (from Task 1)
- Produces: `_string_field_diff(old_val, new_val) -> dict` and `_answers_field_diff(old_answers, new_answers) -> dict` helper functions in `views.py`; `edit_proposal_json` now logs an `ActivityProposalEditEvent` on every save while leaving `teacher_edited_json`, `status`, and the redirect behavior unchanged.

- [ ] **Step 1: Write the failing tests**

  Ensure `reverse` and `Group` are imported in `Trust-AI-Platform/authoringtool/tests.py` (both are already imported for other test classes in this file — verify, don't duplicate).

  Append this test class to the end of `Trust-AI-Platform/authoringtool/tests.py`:

  ```python
  class EditProposalJsonViewTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.user = User.objects.create_user('teacher_edit2', password='pass')
          g, _ = Group.objects.get_or_create(name='teachers')
          self.user.groups.add(g)
          self.client.login(username='teacher_edit2', password='pass')

          self.scenario = Scenario.objects.create(
              name='Edit View Scenario', created_by=self.user, updated_by=self.user
          )
          self.phase = Phase.objects.create(
              name='Phase 1', scenario=self.scenario, created_by=self.user, updated_by=self.user
          )
          self.activity_type = ActivityType.objects.create(
              name='Explanation', created_by=self.user, updated_by=self.user
          )
          self.activity = Activity.objects.create(
              name='Act 1', text='Hello', scenario=self.scenario, phase=self.phase,
              activity_type=self.activity_type, created_by=self.user, updated_by=self.user,
          )
          self.proposal = ActivityProposal.objects.create(
              scenario=self.scenario, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='raw', translated_action='raw',
              json_action=json.dumps({
                  "activity_name": "Act 1",
                  "content": "Original content",
                  "explanation": "Original explanation",
                  "answers": [{"text": "A. Old answer", "is_correct": True, "weight": 3}],
              }),
              json_translated_action='',
          )

      def _post_edit(self, **overrides):
          data = {
              'activity_name': 'Act 1',
              'content': 'Original content',
              'explanation': 'Original explanation',
              'answer_text_1': 'A. Old answer',
          }
          data.update(overrides)
          url = reverse('edit_proposal_json', args=[self.scenario.id, self.proposal.id])
          return self.client.post(url, data)

      def test_first_edit_creates_event_diffed_against_original_proposal(self):
          self._post_edit(content='Revised content is longer now')

          review = UserProposalReview.objects.get(proposal=self.proposal, user=self.user)
          self.assertTrue(review.was_edited)
          self.assertEqual(review.edit_count, 1)

          events = list(review.edit_events.order_by('edit_number'))
          self.assertEqual(len(events), 1)
          event = events[0]
          self.assertEqual(event.edit_number, 1)
          self.assertTrue(event.changed_fields['content']['changed'])
          self.assertFalse(event.changed_fields['explanation']['changed'])
          self.assertFalse(event.changed_fields['answers']['changed'])

      def test_second_edit_diffs_against_previous_edit_not_original(self):
          self._post_edit(content='First revision')
          self._post_edit(content='First revision, refined further')

          review = UserProposalReview.objects.get(proposal=self.proposal, user=self.user)
          self.assertEqual(review.edit_count, 2)

          events = list(review.edit_events.order_by('edit_number'))
          self.assertEqual(len(events), 2)

          second_event = events[1]
          expected_delta = len('First revision, refined further') - len('First revision')
          self.assertEqual(second_event.changed_fields['content']['char_delta'], expected_delta)

      def test_answers_count_delta_tracks_added_answer(self):
          self._post_edit(answer_text_1='A. Old answer', answer_text_2='B. New second answer')

          review = UserProposalReview.objects.get(proposal=self.proposal, user=self.user)
          event = review.edit_events.get(edit_number=1)
          self.assertEqual(event.changed_fields['answers']['count_delta'], 1)
          self.assertTrue(event.changed_fields['answers']['changed'])

      def test_teacher_edited_json_still_holds_latest_state_only(self):
          self._post_edit(content='First revision')
          self._post_edit(content='Second revision')

          review = UserProposalReview.objects.get(proposal=self.proposal, user=self.user)
          self.assertEqual(review.teacher_edited_json['content'], 'Second revision')
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  python manage.py test authoringtool.tests.EditProposalJsonViewTests -v 2
  ```
  Expected: FAIL — `review.was_edited` stays `False`, `review.edit_count` stays `0`, `review.edit_events.count()` is `0` (the `AttributeError`/lookup failures from Task 1 are already resolved; these failures are `AssertionError`s showing the view doesn't log events yet).

- [ ] **Step 3: Add the diff helpers and wire them into the view**

  In `Trust-AI-Platform/authoringtool/views.py`, replace:

  ```python
  @group_required('teachers')
  def edit_proposal_json(request, scenario_id, pk):
      proposal = get_object_or_404(ActivityProposal, pk=pk)
      user = request.user

      # Get or create the review
      review, created = UserProposalReview.objects.get_or_create(
          proposal=proposal,
          user=user,
          defaults={'status': 'new'}
      )

      # Build JSON from POST data
      data = {
          "activity_name": request.POST.get("activity_name", ""),
          "content": request.POST.get("content", ""),
          "explanation": request.POST.get("explanation", ""),
          "answers": []
      }

      for i in range(1, 20):  # allow up to 20 answers
          key = f"answer_text_{i}"
          val = request.POST.get(key)
          if val:
              data["answers"].append({"text": val.strip()})

      review.teacher_edited_json = data

      print("✅ Type of final data:", type(data))  # should be <class 'dict'>
      print("✅ Dumped JSON string:", json.dumps(data, ensure_ascii=False))

      # ✅ Keep review visible after editing
      if review.status not in ['accepted', 'rejected']:
          review.status = 'new'

      review.save()
      print("RAW saved value:", review.teacher_edited_json)
      return redirect("proposal_list", scenario_id=scenario_id)
  ```

  with:

  ```python
  def _string_field_diff(old_val, new_val):
      old_val = old_val or ""
      new_val = new_val or ""
      return {
          "changed": old_val != new_val,
          "char_delta": len(new_val) - len(old_val),
      }


  def _answers_field_diff(old_answers, new_answers):
      old_texts = [a.get("text", "") for a in (old_answers or [])]
      new_texts = [a.get("text", "") for a in (new_answers or [])]
      return {
          "changed": old_texts != new_texts,
          "char_delta": sum(len(t) for t in new_texts) - sum(len(t) for t in old_texts),
          "count_delta": len(new_texts) - len(old_texts),
      }


  @group_required('teachers')
  def edit_proposal_json(request, scenario_id, pk):
      proposal = get_object_or_404(ActivityProposal, pk=pk)
      user = request.user

      # Get or create the review
      review, created = UserProposalReview.objects.get_or_create(
          proposal=proposal,
          user=user,
          defaults={'status': 'new'}
      )

      # Build JSON from POST data
      data = {
          "activity_name": request.POST.get("activity_name", ""),
          "content": request.POST.get("content", ""),
          "explanation": request.POST.get("explanation", ""),
          "answers": []
      }

      for i in range(1, 20):  # allow up to 20 answers
          key = f"answer_text_{i}"
          val = request.POST.get(key)
          if val:
              data["answers"].append({"text": val.strip()})

      # Log this revision as an edit event, diffed against the previous
      # revision (or the original LLM proposal for the first edit).
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

      review.teacher_edited_json = data

      # Keep review visible after editing
      if review.status not in ['accepted', 'rejected']:
          review.status = 'new'

      review.save()
      return redirect("proposal_list", scenario_id=scenario_id)
  ```

  Then update the model import at the top of the same file — replace:

  ```python
  from .models import Scenario, Phase, ActivityType, Activity, Answer, AnswerFeedback, NextQuestionLogic, QuestionBunch, EvQuestionBranching, Simulation, UserAnswer, UserScenarioScore, SchoolDepartment, ExperimentLL, RemoteLabSession, VRARExperiment, ActivityProposal, UserProposalReview, Language, Subject
  ```

  with:

  ```python
  from .models import Scenario, Phase, ActivityType, Activity, Answer, AnswerFeedback, NextQuestionLogic, QuestionBunch, EvQuestionBranching, Simulation, UserAnswer, UserScenarioScore, SchoolDepartment, ExperimentLL, RemoteLabSession, VRARExperiment, ActivityProposal, ActivityProposalEditEvent, UserProposalReview, Language, Subject
  ```

- [ ] **Step 4: Run the tests to verify they pass**

  ```bash
  python manage.py test authoringtool.tests.EditProposalJsonViewTests -v 2
  ```
  Expected: `OK` (4 tests).

- [ ] **Step 5: Run the full authoringtool test suite**

  ```bash
  python manage.py test authoringtool -v 2
  ```
  Expected: `OK` — confirms nothing in the existing suite (import/export, proposal accept/reject, etc.) regressed.

- [ ] **Step 6: Manually verify end-to-end**

  ```bash
  python manage.py runserver
  ```
  As a teacher user, open an existing scenario's proposal list (`/authoringtool/scenarios/<scenario_id>/proposals/`), edit a proposal's text through the existing edit form, and submit. Then check `/admin/authoringtool/activityproposaleditevent/` — confirm a new row appears with `edit_number=1` and a `changed_fields` value reflecting the edit. Edit the same proposal again and confirm a second row appears with `edit_number=2`.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/views.py Trust-AI-Platform/authoringtool/tests.py
  git commit -m "Log per-field diff on each proposal edit via ActivityProposalEditEvent"
  ```
