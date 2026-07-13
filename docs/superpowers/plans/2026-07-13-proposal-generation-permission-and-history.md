# Proposal Generation Permission and History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict proposal *generation* to a scenario's creator (or an admin), and stop destroying past generation batches on regenerate — instead archive them and add a read-only history UI so a scenario's creator can look back at what was proposed and decided in previous runs.

**Architecture:** Three sequential tasks. Task 1 lays the data foundation: a new `ProposalGenerationRun` model with a `start_new()` classmethod that atomically archives the previous run and opens a new one, plus a migration that backfills existing `ActivityProposal` rows into a synthetic first run so nothing already in the database is orphaned. Task 2 wires that model into the actual generation task (replacing its unconditional delete-everything behavior), adds the creator-only permission gate to the trigger endpoint, and fixes a real correctness bug this change would otherwise introduce (`apply_proposals_to_cloned_scenario`'s accepted-proposals query has no run scoping today — harmless only because old data is always deleted; once it isn't, that query needs scoping too). Task 3 builds the read-only history UI on top of the now-preserved data.

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Celery (`CELERY_TASK_ALWAYS_EAGER=True` in test settings) · Postgres (production) · SQLite (dev/test)

## Global Constraints

- **Generation permission:** `@group_required('teachers')` (already defined in `authoringtool/views.py:49`) plus the exact ownership check already used on ~10 sibling views in this file — `if scenario.created_by != request.user and not is_admin_user(request.user): return <403>` (`is_admin_user` already defined at `views.py:44`). `trigger_llm_context_task` is an AJAX/JSON endpoint, so its 403 response is `JsonResponse({"error": ...}, status=403)`, not `HttpResponseForbidden` — matching its own existing response shape (see `views.py:2350-2352`), not the HTML-page pattern the other gated views use.
- **Viewing/history access is NOT tightened.** `proposal_list_view` (the live review page) is `@login_required` only today, with no creator restriction — this plan does not change that, and the new history views match it exactly (`@login_required` only). Only *generation* becomes creator-gated. This is a deliberate, already-existing asymmetry (viewing is open, generation is about to become restricted), not something to "fix" as part of this plan.
- **Exactly one current run per scenario**, enforced at the database level with a Postgres partial unique constraint (`UniqueConstraint(fields=['scenario'], condition=Q(is_current=True), name='unique_current_run_per_scenario')`), not just application discipline. `ProposalGenerationRun.start_new()` flips the old run and creates the new one inside one `transaction.atomic()` block so the constraint is never violated mid-write.
- **No more deletion.** The two `.delete()` calls currently in `generate_llm_context_for_scenario` (`authoringtool/tasks.py:2599,2601`) are removed entirely and replaced with `ProposalGenerationRun.start_new(...)`. Old proposals, their `UserProposalReview`s, and their `ActivityProposalEditEvent` audit trails simply stay in the database forever. No pagination/pruning of history is in scope.
- **`ActivityProposal.generation_run` stays nullable** (`null=True, blank=True`) at the schema level, even though every code path that creates an `ActivityProposal` from this point forward always sets it. This sidesteps Django's mandatory interactive "provide a one-off default" prompt that `makemigrations` requires when converting an existing nullable-with-data field to non-nullable — a prompt a non-interactive subagent dispatch cannot answer. This matches the same pattern already used elsewhere in this codebase (e.g. `Announcement.created_by` is nullable via `SET_NULL` even though it's normally always set).
- **`proposal_list_view` must filter to the current run** (`generation_run__is_current=True` added to its existing `ActivityProposal.objects.filter(scenario=myScenario)` query) so live review behavior is completely unchanged from today — same counts, same auto-created reviews — just scoped to the live batch instead of all-time.
- **`apply_proposals_to_cloned_scenario`'s accepted-reviews query must also gain `generation_run__is_current=True` scoping** — required, not optional. Without it, "Create Personalised Scenario" would start silently materializing accepted decisions from old, superseded batches once history is no longer deleted. This query is extracted into a small standalone function, `get_accepted_reviews_for_personal_scenario(original_scenario, user)`, specifically so it can be unit-tested without needing to invoke the full scenario-cloning pipeline (which has its own heavy dependencies unrelated to this fix).
- **`generate_llm_context_for_scenario` itself cannot be unit-tested end-to-end** — it makes real HTTP calls to a local Ollama server and does RAG/PDF indexing, none of which are available in this sandbox (this is a pre-existing gap, not one this plan introduces — no existing test in this codebase invokes this task's full body either; `ActivityProposalEditEventModelTests`/`EditProposalJsonViewTests` already establish the precedent of creating `ActivityProposal` rows directly rather than running the generation task). This plan's tests instead cover the parts that ARE independently testable: the `ProposalGenerationRun.start_new()` lifecycle (Task 1), the permission gate and the two scoping fixes (Task 2, via direct fixture creation across multiple runs — no LLM calls needed), and the new history views (Task 3).
- Migration file names below use placeholder numbers — confirm the actual filename `makemigrations` generates at each step and use that exact name for the following step's `dependencies`, rather than assuming the number shown here is exact.

---

### Task 1: `ProposalGenerationRun` model, `start_new()`, migration with backfill, admin

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/models.py`
- Modify: `Trust-AI-Platform/authoringtool/admin.py`
- Modify: `Trust-AI-Platform/authoringtool/tests.py`
- Create (via `makemigrations`): `Trust-AI-Platform/authoringtool/migrations/0057_proposalgenerationrun_activityproposal_generation_run.py` (exact name may differ)
- Create (hand-filled data migration): `Trust-AI-Platform/authoringtool/migrations/0058_backfill_proposal_generation_runs.py` (exact name may differ)

**Interfaces:**
- Produces: `ProposalGenerationRun` model (`scenario`, `created_by`, `created_at`, `is_current`) with classmethod `ProposalGenerationRun.start_new(scenario, created_by) -> ProposalGenerationRun`. `ActivityProposal.generation_run` (nullable FK). Consumed by Task 2 (the generation task, the two scoping fixes) and Task 3 (the history views).

- [ ] **Step 1: Write the failing tests**

  In `Trust-AI-Platform/authoringtool/tests.py`, add `ProposalGenerationRun` to the existing model import block — replace:
  ```python
  from authoringtool.models import (
      Activity,
      ActivityProposal,
      ActivityProposalEditEvent,
      ActivityType,
      Answer,
      AnswerFeedback,
      NextQuestionLogic,
      Phase,
      Scenario,
      SchoolDepartment,
      UserAnswer,
      UserProposalReview,
      UserScenarioScore,
  )
  ```
  with:
  ```python
  from authoringtool.models import (
      Activity,
      ActivityProposal,
      ActivityProposalEditEvent,
      ActivityType,
      Answer,
      AnswerFeedback,
      NextQuestionLogic,
      Phase,
      ProposalGenerationRun,
      Scenario,
      SchoolDepartment,
      UserAnswer,
      UserProposalReview,
      UserScenarioScore,
  )
  ```

  Then append at the end of the file:
  ```python
  class ProposalGenerationRunModelTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('run_owner', password='pass')
          self.scenario = Scenario.objects.create(
              name='Run Scenario', created_by=self.user, updated_by=self.user
          )

      def test_start_new_creates_current_run(self):
          run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.assertTrue(run.is_current)
          self.assertEqual(run.scenario, self.scenario)
          self.assertEqual(run.created_by, self.user)

      def test_start_new_archives_previous_current_run(self):
          first_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          second_run = ProposalGenerationRun.start_new(self.scenario, self.user)

          first_run.refresh_from_db()
          self.assertFalse(first_run.is_current)
          self.assertTrue(second_run.is_current)

      def test_only_one_current_run_per_scenario(self):
          ProposalGenerationRun.start_new(self.scenario, self.user)
          with self.assertRaises(IntegrityError):
              with transaction.atomic():
                  ProposalGenerationRun.objects.create(
                      scenario=self.scenario, created_by=self.user, is_current=True,
                  )

      def test_different_scenarios_can_each_have_a_current_run(self):
          other_scenario = Scenario.objects.create(
              name='Other Run Scenario', created_by=self.user, updated_by=self.user
          )
          run1 = ProposalGenerationRun.start_new(self.scenario, self.user)
          run2 = ProposalGenerationRun.start_new(other_scenario, self.user)
          self.assertTrue(run1.is_current)
          self.assertTrue(run2.is_current)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.ProposalGenerationRunModelTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `ImportError: cannot import name 'ProposalGenerationRun'`.

- [ ] **Step 3: Add the model**

  In `Trust-AI-Platform/authoringtool/models.py`, insert immediately before `class ActivityProposal(models.Model):` (currently at line 560):
  ```python
  class ProposalGenerationRun(models.Model):
      scenario = models.ForeignKey('Scenario', on_delete=models.CASCADE, related_name='proposal_generation_runs')
      created_by = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, related_name='triggered_proposal_generation_runs')
      created_at = models.DateTimeField(auto_now_add=True)
      is_current = models.BooleanField(default=True)

      class Meta:
          verbose_name = "Proposal Generation Run"
          verbose_name_plural = "Proposal Generation Runs"
          ordering = ['-created_at']
          constraints = [
              models.UniqueConstraint(
                  fields=['scenario'],
                  condition=models.Q(is_current=True),
                  name='unique_current_run_per_scenario',
              ),
          ]

      def __str__(self):
          status = 'current' if self.is_current else 'archived'
          return f"Run for '{self.scenario.name}' @ {self.created_at:%Y-%m-%d %H:%M} ({status})"

      @classmethod
      def start_new(cls, scenario, created_by):
          """Archive the scenario's current run (if any) and start a new current one, atomically."""
          with transaction.atomic():
              cls.objects.filter(scenario=scenario, is_current=True).update(is_current=False)
              return cls.objects.create(scenario=scenario, created_by=created_by, is_current=True)


  ```

  Then, in the `ActivityProposal` class (now immediately below), add the `generation_run` field right after `scenario` — replace:
  ```python
      scenario = models.ForeignKey('Scenario', on_delete=models.CASCADE, related_name='proposals', db_index=True)
      phase = models.ForeignKey('Phase', on_delete=models.CASCADE, related_name='proposals')
  ```
  with:
  ```python
      scenario = models.ForeignKey('Scenario', on_delete=models.CASCADE, related_name='proposals', db_index=True)
      generation_run = models.ForeignKey(
          'ProposalGenerationRun', on_delete=models.CASCADE, null=True, blank=True, related_name='proposals',
      )
      phase = models.ForeignKey('Phase', on_delete=models.CASCADE, related_name='proposals')
  ```

  Note: `transaction` and `models.Q` are already available in this file — `from django.db import transaction` is imported at `models.py:9`, and `models.Q` is reachable through the existing `from django.db import models` at `models.py:1`. No new imports needed.

- [ ] **Step 4: Generate the schema migration**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py makemigrations authoringtool --settings=faithDev.settings_test
  ```
  Expected: a single new migration creating `ProposalGenerationRun` and adding the `generation_run` field to `ActivityProposal`. No interactive prompt should appear (the new field is nullable). Note the exact filename Django generated — you'll need it for Step 5.

- [ ] **Step 5: Create and fill in the data migration**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py makemigrations authoringtool --empty -n backfill_proposal_generation_runs --settings=faithDev.settings_test
  ```
  This creates an empty migration file. Open it and replace its entire contents with:
  ```python
  from django.db import migrations


  def backfill_generation_runs(apps, schema_editor):
      ActivityProposal = apps.get_model('authoringtool', 'ActivityProposal')
      ProposalGenerationRun = apps.get_model('authoringtool', 'ProposalGenerationRun')

      scenario_ids = ActivityProposal.objects.filter(
          generation_run__isnull=True
      ).values_list('scenario_id', flat=True).distinct()

      for scenario_id in scenario_ids:
          proposals = ActivityProposal.objects.filter(
              scenario_id=scenario_id, generation_run__isnull=True
          ).order_by('created_at')
          first_proposal = proposals.first()
          if first_proposal is None:
              continue
          run = ProposalGenerationRun.objects.create(
              scenario_id=scenario_id,
              created_by_id=first_proposal.scenario.created_by_id,
              is_current=True,
          )
          # created_at has auto_now_add=True, which silently ignores any
          # value passed to create() — override it afterwards via
          # .update(), which bypasses the auto_now_add pre-save behavior,
          # so the backfilled run's timestamp matches the oldest proposal
          # it covers instead of "now".
          ProposalGenerationRun.objects.filter(pk=run.pk).update(created_at=first_proposal.created_at)
          proposals.update(generation_run_id=run.pk)


  def noop_reverse(apps, schema_editor):
      pass


  class Migration(migrations.Migration):

      dependencies = [
          ('authoringtool', '0057_proposalgenerationrun_activityproposal_generation_run'),  # replace with the exact filename from Step 4
      ]

      operations = [
          migrations.RunPython(backfill_generation_runs, noop_reverse),
      ]
  ```
  Replace the `dependencies` tuple's migration name with whatever Step 4 actually generated.

- [ ] **Step 6: Apply the migrations**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py migrate authoringtool --settings=faithDev.settings_test
  ```
  Expected: both new migrations apply cleanly (the test database starts empty, so the backfill migration's loop runs zero times here — that's expected and fine; its correctness for pre-existing production data is a logical property of the code, not something the empty test DB can exercise).

- [ ] **Step 7: Register in admin**

  In `Trust-AI-Platform/authoringtool/admin.py`, add `ProposalGenerationRun` to the import block at line 16 — replace:
  ```python
      ActivityFlag, ActivityProposal, ActivityProposalEditEvent, QValue, UserProposalReview, Language,
  ```
  with:
  ```python
      ActivityFlag, ActivityProposal, ActivityProposalEditEvent, ProposalGenerationRun, QValue, UserProposalReview, Language,
  ```

  Then, immediately before the `# ─── ActivityProposal ───` section comment (currently at line 555), insert:
  ```python
  # ─── ProposalGenerationRun ─────────────────────────────────────────────────────

  @admin.register(ProposalGenerationRun)
  class ProposalGenerationRunAdmin(admin.ModelAdmin):
      list_display = ('id', 'scenario', 'created_by', 'created_at', 'is_current')
      list_filter = ('is_current', 'created_at')
      search_fields = ('scenario__name',)
      raw_id_fields = ('scenario', 'created_by')
      readonly_fields = ('created_at',)
      date_hierarchy = 'created_at'


  ```

- [ ] **Step 8: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.ProposalGenerationRunModelTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (4 tests).

- [ ] **Step 9: Commit**

  ```bash
  git status --short -- Trust-AI-Platform/authoringtool/migrations/
  ```
  Confirm only the two new migrations from Steps 4-5 are untracked (no other stray untracked migrations should be swept in).
  ```bash
  git add Trust-AI-Platform/authoringtool/models.py Trust-AI-Platform/authoringtool/admin.py Trust-AI-Platform/authoringtool/tests.py Trust-AI-Platform/authoringtool/migrations/
  git commit -m "Add ProposalGenerationRun model, backfill migration, and admin registration"
  ```

---

### Task 2: Restrict generation to the scenario creator; wire runs into the task; fix cloned-scenario scoping

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/views.py`
- Modify: `Trust-AI-Platform/authoringtool/tasks.py`
- Modify: `Trust-AI-Platform/authoringtool/tests.py`

**Interfaces:**
- Consumes: `ProposalGenerationRun.start_new()` (Task 1)
- Produces: `get_accepted_reviews_for_personal_scenario(original_scenario, user)` in `tasks.py` — consumed by nothing outside this task currently, but isolated specifically so Task 3 or future work can reuse it without re-deriving the correct scoping.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/authoringtool/tests.py`:
  ```python
  class TriggerLlmContextTaskPermissionTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.owner = User.objects.create_user('gen_owner', password='pass')
          self.other_teacher = User.objects.create_user('gen_other', password='pass')
          self.admin = User.objects.create_user('gen_admin', password='pass', is_staff=True)
          g, _ = Group.objects.get_or_create(name='teachers')
          self.owner.groups.add(g)
          self.other_teacher.groups.add(g)
          self.admin.groups.add(g)
          self.scenario = Scenario.objects.create(
              name='Gen Perm Scenario', created_by=self.owner, updated_by=self.owner
          )

      def test_non_owner_teacher_forbidden(self):
          self.client.login(username='gen_other', password='pass')
          url = reverse('generate_llm_context', args=[self.scenario.id])
          response = self.client.post(url)
          self.assertEqual(response.status_code, 403)

      def test_non_teacher_forbidden(self):
          non_teacher = User.objects.create_user('gen_notteacher', password='pass')
          self.client.login(username='gen_notteacher', password='pass')
          url = reverse('generate_llm_context', args=[self.scenario.id])
          response = self.client.post(url)
          self.assertEqual(response.status_code, 403)

      def test_owner_can_trigger(self):
          from unittest.mock import patch
          self.client.login(username='gen_owner', password='pass')
          url = reverse('generate_llm_context', args=[self.scenario.id])
          with patch('authoringtool.views.generate_llm_context_for_scenario.delay') as mock_delay:
              mock_delay.return_value.id = 'fake-task-id'
              response = self.client.post(url)
          self.assertEqual(response.status_code, 200)
          mock_delay.assert_called_once_with(self.scenario.id, force_rebuild=False, triggered_by_id=self.owner.id)

      def test_admin_can_trigger(self):
          from unittest.mock import patch
          self.client.login(username='gen_admin', password='pass')
          url = reverse('generate_llm_context', args=[self.scenario.id])
          with patch('authoringtool.views.generate_llm_context_for_scenario.delay') as mock_delay:
              mock_delay.return_value.id = 'fake-task-id'
              response = self.client.post(url)
          self.assertEqual(response.status_code, 200)
          mock_delay.assert_called_once()


  class ProposalListViewCurrentRunScopingTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.user = User.objects.create_user('scope_owner', password='pass')
          g, _ = Group.objects.get_or_create(name='teachers')
          self.user.groups.add(g)
          self.client.login(username='scope_owner', password='pass')

          self.scenario = Scenario.objects.create(
              name='Scope Scenario', created_by=self.user, updated_by=self.user
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

          self.old_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.old_proposal = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.old_run, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='old', translated_action='old',
              json_action='{}', json_translated_action='{}',
          )

          self.current_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.current_proposal = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.current_run, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='current', translated_action='current',
              json_action='{}', json_translated_action='{}',
          )

      def test_proposal_list_only_shows_current_run(self):
          url = reverse('proposal_list', args=[self.scenario.id])
          response = self.client.get(url)
          proposals = list(response.context['proposals'])
          self.assertEqual(proposals, [self.current_proposal])

      def test_old_run_is_archived(self):
          self.old_run.refresh_from_db()
          self.current_run.refresh_from_db()
          self.assertFalse(self.old_run.is_current)
          self.assertTrue(self.current_run.is_current)


  class AcceptedReviewsForPersonalScenarioScopingTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('accepted_scope_owner', password='pass')
          self.scenario = Scenario.objects.create(
              name='Accepted Scope Scenario', created_by=self.user, updated_by=self.user
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

          self.old_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.old_proposal = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.old_run, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='old', translated_action='old',
              json_action='{}', json_translated_action='{}',
          )
          self.old_review = UserProposalReview.objects.create(
              proposal=self.old_proposal, user=self.user, status='accepted',
          )

          self.current_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.current_proposal = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.current_run, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='current', translated_action='current',
              json_action='{}', json_translated_action='{}',
          )
          self.current_review = UserProposalReview.objects.create(
              proposal=self.current_proposal, user=self.user, status='accepted',
          )

      def test_only_current_run_accepted_reviews_are_returned(self):
          from authoringtool.tasks import get_accepted_reviews_for_personal_scenario
          reviews = list(get_accepted_reviews_for_personal_scenario(self.scenario, self.user))
          self.assertEqual(reviews, [self.current_review])
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.TriggerLlmContextTaskPermissionTests authoringtool.tests.ProposalListViewCurrentRunScopingTests authoringtool.tests.AcceptedReviewsForPersonalScenarioScopingTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `TriggerLlmContextTaskPermissionTests` fails because there's no permission check yet (non-owner gets 200, not 403); `ProposalListViewCurrentRunScopingTests` fails because the view currently returns both proposals, not just the current-run one; `AcceptedReviewsForPersonalScenarioScopingTests` fails with `ImportError` (the function doesn't exist yet).

- [ ] **Step 3: Add the permission gate to `trigger_llm_context_task`**

  In `Trust-AI-Platform/authoringtool/views.py`, replace:
  ```python
  @login_required
  def trigger_llm_context_task(request, scenario_id):
      if request.method == "POST":
          try:
              scenario = Scenario.objects.get(id=scenario_id)
              force = request.GET.get("force", "false").lower() == "true"
              task = generate_llm_context_for_scenario.delay(scenario.id, force_rebuild=force)
              return JsonResponse({"status": "started", "task_id": task.id})
          except Scenario.DoesNotExist:
              return JsonResponse({"error": "Scenario not found"}, status=404)
      return JsonResponse({"error": "Invalid request"}, status=400)
  ```
  with:
  ```python
  @group_required('teachers')
  def trigger_llm_context_task(request, scenario_id):
      if request.method == "POST":
          try:
              scenario = Scenario.objects.get(id=scenario_id)
              if scenario.created_by != request.user and not is_admin_user(request.user):
                  return JsonResponse({"error": "You don't own this scenario."}, status=403)
              force = request.GET.get("force", "false").lower() == "true"
              task = generate_llm_context_for_scenario.delay(scenario.id, force_rebuild=force, triggered_by_id=request.user.id)
              return JsonResponse({"status": "started", "task_id": task.id})
          except Scenario.DoesNotExist:
              return JsonResponse({"error": "Scenario not found"}, status=404)
      return JsonResponse({"error": "Invalid request"}, status=400)
  ```

  Note: `@group_required('teachers')` already wraps `@login_required` internally (see its definition at `views.py:49-59`), so this is a strict tightening — every request that passed before (any logged-in user) must now also be in the `teachers` group and pass the new ownership check.

- [ ] **Step 4: Scope `proposal_list_view` to the current run**

  In `Trust-AI-Platform/authoringtool/views.py`, replace:
  ```python
      # 1. Fetch all shared proposals for the scenario
      proposals = ActivityProposal.objects.filter(scenario=myScenario)\
          .select_related('activity', 'phase', 'scenario')\
          .prefetch_related('flag', 'categories_in_risk')\
          .order_by('-created_at')
  ```
  with:
  ```python
      # 1. Fetch all shared proposals for the scenario's current generation run
      proposals = ActivityProposal.objects.filter(scenario=myScenario, generation_run__is_current=True)\
          .select_related('activity', 'phase', 'scenario')\
          .prefetch_related('flag', 'categories_in_risk')\
          .order_by('-created_at')
  ```

- [ ] **Step 5: Wire `ProposalGenerationRun` into `generate_llm_context_for_scenario`**

  In `Trust-AI-Platform/authoringtool/tasks.py`, add `ProposalGenerationRun` to the models import — replace:
  ```python
  from .models import Scenario, Phase, Activity, UserAnswer, Answer, QuestionBunch, ActivityType, NextQuestionLogic, EvQuestionBranching, ActivityFlag, ActivityProposal, CategoryTag, QValue, UserProposalReview
  ```
  with:
  ```python
  from .models import Scenario, Phase, Activity, UserAnswer, Answer, QuestionBunch, ActivityType, NextQuestionLogic, EvQuestionBranching, ActivityFlag, ActivityProposal, CategoryTag, QValue, UserProposalReview, ProposalGenerationRun
  ```

  Then update the task's signature — replace:
  ```python
  def generate_llm_context_for_scenario(scenario_id, force_rebuild=False):
      scenario = Scenario.objects.get(id=scenario_id)
  ```
  with:
  ```python
  def generate_llm_context_for_scenario(scenario_id, force_rebuild=False, triggered_by_id=None):
      scenario = Scenario.objects.get(id=scenario_id)
      triggered_by = User.objects.filter(id=triggered_by_id).first() if triggered_by_id else scenario.created_by
  ```

  Then replace the delete block — replace:
  ```python
      # 5️⃣ FLAG-DRIVEN PROPOSALS
      # — clean out old proposals & reviews —
      ActivityProposal.objects.filter(scenario=scenario).delete()
      print(f"Deleted Proposals")
      UserProposalReview.objects.filter(proposal__scenario=scenario).delete()
      print(f"Deleted User Proposals")
  ```
  with:
  ```python
      # 5️⃣ FLAG-DRIVEN PROPOSALS
      # — archive the previous run (if any) and start a fresh current one —
      generation_run = ProposalGenerationRun.start_new(scenario, triggered_by)
      print(f"Started new ProposalGenerationRun {generation_run.id} for scenario {scenario_id}")
  ```

  Then add `generation_run` to the proposal creation call — replace:
  ```python
          prop = ActivityProposal.objects.create(
              scenario=scenario,
              phase=phase,
              activity=activity,
              proposal_type=proposal_type,
              suggested_action=raw,
              translated_action=translated_raw,
              json_action=json.dumps(structured, ensure_ascii=False),
              json_translated_action=translated_structured_json,
              status='new',
          )
  ```
  with:
  ```python
          prop = ActivityProposal.objects.create(
              scenario=scenario,
              generation_run=generation_run,
              phase=phase,
              activity=activity,
              proposal_type=proposal_type,
              suggested_action=raw,
              translated_action=translated_raw,
              json_action=json.dumps(structured, ensure_ascii=False),
              json_translated_action=translated_structured_json,
              status='new',
          )
  ```

  Note: `User` (the Django auth model) is already imported in `tasks.py` (`from django.contrib.auth.models import User`, line 10) — no new import needed for the `triggered_by` resolution line.

- [ ] **Step 6: Extract and fix the accepted-reviews query**

  In `Trust-AI-Platform/authoringtool/tasks.py`, add a small standalone function immediately before `def apply_proposals_to_cloned_scenario`:
  ```python
  def get_accepted_reviews_for_personal_scenario(original_scenario, user):
      return UserProposalReview.objects.filter(
          user=user,
          proposal__scenario=original_scenario,
          proposal__generation_run__is_current=True,
          status='accepted'
      ).select_related('proposal', 'proposal__activity', 'proposal__phase')


  ```
  Then, inside `apply_proposals_to_cloned_scenario`, replace its existing inline query — replace:
  ```python
      accepted_reviews = UserProposalReview.objects.filter(
          user=user,
          proposal__scenario=original_scenario,
          status='accepted'
      ).select_related('proposal', 'proposal__activity', 'proposal__phase')
  ```
  with:
  ```python
      accepted_reviews = get_accepted_reviews_for_personal_scenario(original_scenario, user)
  ```

- [ ] **Step 7: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.TriggerLlmContextTaskPermissionTests authoringtool.tests.ProposalListViewCurrentRunScopingTests authoringtool.tests.AcceptedReviewsForPersonalScenarioScopingTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (7 tests: `TriggerLlmContextTaskPermissionTests` has 4, `ProposalListViewCurrentRunScopingTests` has 2, `AcceptedReviewsForPersonalScenarioScopingTests` has 1).

  Then the broader authoringtool suite covering everything touched so far:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.ProposalGenerationRunModelTests authoringtool.tests.TriggerLlmContextTaskPermissionTests authoringtool.tests.ProposalListViewCurrentRunScopingTests authoringtool.tests.AcceptedReviewsForPersonalScenarioScopingTests authoringtool.tests.ActivityProposalEditEventModelTests authoringtool.tests.EditProposalJsonViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` — this confirms the pre-existing proposal-edit tests (which create `ActivityProposal` rows directly, without a `generation_run`) still pass unaffected by the new nullable field.

- [ ] **Step 8: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/views.py Trust-AI-Platform/authoringtool/tasks.py Trust-AI-Platform/authoringtool/tests.py
  git commit -m "Restrict proposal generation to scenario creator; preserve run history instead of deleting"
  ```

---

### Task 3: Read-only proposal history UI

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/views.py`
- Modify: `Trust-AI-Platform/authoringtool/urls.py`
- Modify: `Trust-AI-Platform/authoringtool/tests.py`
- Create: `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_history.html`
- Create: `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_history_run_detail.html`

**Interfaces:**
- Consumes: `ProposalGenerationRun` (Task 1), the now-current-run-scoped `ActivityProposal`/`UserProposalReview` data (Task 2)
- Produces: URL names `proposal_history` (`scenarios/<int:scenario_id>/proposals/history/`), `proposal_history_run_detail` (`scenarios/<int:scenario_id>/proposals/history/<int:run_id>/`).

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/authoringtool/tests.py`:
  ```python
  class ProposalHistoryViewTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.user = User.objects.create_user('history_user', password='pass')
          g, _ = Group.objects.get_or_create(name='teachers')
          self.user.groups.add(g)
          self.client.login(username='history_user', password='pass')

          self.scenario = Scenario.objects.create(
              name='History Scenario', created_by=self.user, updated_by=self.user
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

          self.old_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.old_proposal_accepted = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.old_run, phase=self.phase, activity=self.activity,
              proposal_type='revise', suggested_action='Old accepted proposal', translated_action='x',
              json_action='{}', json_translated_action='{}',
          )
          UserProposalReview.objects.create(
              proposal=self.old_proposal_accepted, user=self.user, status='accepted',
          )
          self.old_proposal_rejected = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.old_run, phase=self.phase, activity=self.activity,
              proposal_type='skip', suggested_action='Old rejected proposal', translated_action='x',
              json_action='{}', json_translated_action='{}',
          )
          UserProposalReview.objects.create(
              proposal=self.old_proposal_rejected, user=self.user, status='rejected', rejection_reasons=['not relevant'],
          )

          self.current_run = ProposalGenerationRun.start_new(self.scenario, self.user)
          self.current_proposal = ActivityProposal.objects.create(
              scenario=self.scenario, generation_run=self.current_run, phase=self.phase, activity=self.activity,
              proposal_type='create', suggested_action='Current proposal', translated_action='x',
              json_action='{}', json_translated_action='{}',
          )

      def test_history_index_lists_only_past_runs(self):
          url = reverse('proposal_history', args=[self.scenario.id])
          response = self.client.get(url)
          run_summaries = response.context['run_summaries']
          self.assertEqual(len(run_summaries), 1)
          self.assertEqual(run_summaries[0]['run'], self.old_run)

      def test_history_index_shows_decision_counts(self):
          url = reverse('proposal_history', args=[self.scenario.id])
          response = self.client.get(url)
          summary = response.context['run_summaries'][0]
          self.assertEqual(summary['accepted'], 1)
          self.assertEqual(summary['rejected'], 1)
          self.assertEqual(summary['total'], 2)

      def test_run_detail_shows_that_runs_proposals_only(self):
          url = reverse('proposal_history_run_detail', args=[self.scenario.id, self.old_run.id])
          response = self.client.get(url)
          self.assertContains(response, 'Old accepted proposal')
          self.assertContains(response, 'Old rejected proposal')
          self.assertNotContains(response, 'Current proposal')

      def test_run_detail_shows_rejection_reasons(self):
          url = reverse('proposal_history_run_detail', args=[self.scenario.id, self.old_run.id])
          response = self.client.get(url)
          self.assertContains(response, 'not relevant')

      def test_run_from_wrong_scenario_404s(self):
          other_scenario = Scenario.objects.create(
              name='Other History Scenario', created_by=self.user, updated_by=self.user
          )
          url = reverse('proposal_history_run_detail', args=[other_scenario.id, self.old_run.id])
          response = self.client.get(url)
          self.assertEqual(response.status_code, 404)

      def test_login_required_for_history(self):
          self.client.logout()
          url = reverse('proposal_history', args=[self.scenario.id])
          response = self.client.get(url)
          self.assertEqual(response.status_code, 302)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.ProposalHistoryViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` (the URL names don't exist yet).

- [ ] **Step 3: Add the two views**

  In `Trust-AI-Platform/authoringtool/views.py`, append at the end of the file:
  ```python

  @login_required
  def proposal_history_view(request, scenario_id):
      myScenario = get_object_or_404(Scenario, id=scenario_id)
      past_runs = ProposalGenerationRun.objects.filter(
          scenario=myScenario, is_current=False
      ).order_by('-created_at')

      run_summaries = []
      for run in past_runs:
          reviews = UserProposalReview.objects.filter(
              user=request.user, proposal__generation_run=run
          )
          total = run.proposals.count()
          accepted = sum(1 for r in reviews if r.status == 'accepted')
          rejected = sum(1 for r in reviews if r.status == 'rejected')
          decided_ids = {r.proposal_id for r in reviews if r.status in ('accepted', 'rejected')}
          never_decided = total - len(decided_ids)
          run_summaries.append({
              'run': run,
              'total': total,
              'accepted': accepted,
              'rejected': rejected,
              'never_decided': never_decided,
          })

      return render(request, 'authoringtool/proposal_history.html', {
          'myScenario': myScenario,
          'run_summaries': run_summaries,
      })


  @login_required
  def proposal_history_run_detail_view(request, scenario_id, run_id):
      myScenario = get_object_or_404(Scenario, id=scenario_id)
      run = get_object_or_404(ProposalGenerationRun, id=run_id, scenario=myScenario)

      proposals = run.proposals.select_related('activity', 'phase')\
          .prefetch_related('flag', 'categories_in_risk')\
          .order_by('-created_at')
      user_reviews = {
          review.proposal_id: review
          for review in UserProposalReview.objects.filter(user=request.user, proposal__generation_run=run)
      }

      return render(request, 'authoringtool/proposal_history_run_detail.html', {
          'myScenario': myScenario,
          'run': run,
          'proposals': proposals,
          'user_reviews': user_reviews,
      })
  ```

  Note: `ProposalGenerationRun` is already imported in this file's `.models` import block from earlier work in this plan... actually it isn't yet — `views.py`'s model import (line 4) currently lists `ActivityProposal, ActivityProposalEditEvent, UserProposalReview` but not `ProposalGenerationRun` (Task 1/2 only added it to `models.py`, `admin.py`, and `tasks.py`, not `views.py`). Add it here — replace:
  ```python
  from .models import Scenario, Phase, ActivityType, Activity, Answer, AnswerFeedback, NextQuestionLogic, QuestionBunch, EvQuestionBranching, Simulation, UserAnswer, UserScenarioScore, SchoolDepartment, ExperimentLL, RemoteLabSession, VRARExperiment, ActivityProposal, ActivityProposalEditEvent, UserProposalReview, Language, Subject
  ```
  with:
  ```python
  from .models import Scenario, Phase, ActivityType, Activity, Answer, AnswerFeedback, NextQuestionLogic, QuestionBunch, EvQuestionBranching, Simulation, UserAnswer, UserScenarioScore, SchoolDepartment, ExperimentLL, RemoteLabSession, VRARExperiment, ActivityProposal, ActivityProposalEditEvent, ProposalGenerationRun, UserProposalReview, Language, Subject
  ```

- [ ] **Step 4: Add the URLs**

  In `Trust-AI-Platform/authoringtool/urls.py`, replace:
  ```python
      path('scenarios/<int:scenario_id>/proposals/', views.proposal_list_view, name='proposal_list'),
  ```
  with:
  ```python
      path('scenarios/<int:scenario_id>/proposals/', views.proposal_list_view, name='proposal_list'),
      path('scenarios/<int:scenario_id>/proposals/history/', views.proposal_history_view, name='proposal_history'),
      path('scenarios/<int:scenario_id>/proposals/history/<int:run_id>/', views.proposal_history_run_detail_view, name='proposal_history_run_detail'),
  ```

- [ ] **Step 5: Create the history index template**

  Create `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_history.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — Proposal History — {{ myScenario.name }}</title>{% endblock %}
  {% block atcontent %}

  <style>
    .pr-hero {
      background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
      border-radius: 14px; padding: 26px 30px 20px; color: #fff;
      margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
    }
    .pr-hero-icon {
      background: rgba(255,255,255,0.18); border-radius: 10px;
      width: 50px; height: 50px; display:flex; align-items:center;
      justify-content:center; font-size:22px; flex-shrink:0;
    }
    .pr-hero .breadcrumb { background:none; margin:10px 0 0; padding:0; font-size:12px; }
    .pr-hero .breadcrumb-item+.breadcrumb-item::before { color:rgba(255,255,255,0.5); }
    .pr-hero .breadcrumb-item a { color:rgba(255,255,255,0.72); text-decoration:none; }
    .pr-hero .breadcrumb-item a:hover { color:#fff; }
    .pr-hero .breadcrumb-item.active { color:rgba(255,255,255,0.92); }
    .hero-btn-ghost { background:rgba(255,255,255,0.15); color:#fff; border:1px solid rgba(255,255,255,0.3); font-weight:600; font-size:13px; border-radius:8px; padding:7px 16px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; white-space:nowrap; }
    .hero-btn-ghost:hover { background:rgba(255,255,255,0.25); color:#fff; }
    @media (max-width:575.98px) {
      .pr-hero { padding:14px 16px 12px; }
      .pr-hero > .d-flex { flex-wrap:wrap; }
      .pr-hero-icon { display:none; }
      .pr-hero h2 { font-size:15px !important; }
    }

    .run-card {
      border-radius:12px; border:1px solid #e8edf5; background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,0.04); margin-bottom:12px;
      padding:16px 20px; display:flex; align-items:center; justify-content:space-between;
      gap:14px; flex-wrap:wrap; text-decoration:none; color:inherit; transition:box-shadow .15s;
    }
    .run-card:hover { box-shadow:0 4px 16px rgba(26,86,219,0.09); border-color:#c5d0f0; }
    .run-card-date { font-weight:700; color:#012970; font-size:14px; }
    .run-card-meta { font-size:12px; color:#888; margin-top:2px; }
    .run-stats { display:flex; gap:8px; flex-wrap:wrap; }
    .run-stat { font-size:12px; font-weight:600; padding:3px 10px; border-radius:12px; }
    .run-stat.accepted { background:#dcfce7; color:#166534; }
    .run-stat.rejected { background:#fee2e2; color:#991b1b; }
    .run-stat.pending  { background:#fef3c7; color:#92400e; }
  </style>

  <main id="main" class="main">
    <div class="pr-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="pr-hero-icon"><i class="bi bi-clock-history"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Proposals</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">Proposal History</h2>
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ myScenario.name }}</div>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'proposal_list' myScenario.id %}">Proposals</a></li>
            <li class="breadcrumb-item active">History</li>
          </ol></nav>
        </div>
        <div class="flex-shrink-0 d-flex gap-2 align-items-start" style="padding-top:4px;">
          <a href="{% url 'proposal_list' myScenario.id %}" class="hero-btn-ghost">
            <i class="bi bi-arrow-left"></i> Back
          </a>
        </div>
      </div>
    </div>

    <section class="section">
      {% for summary in run_summaries %}
      <a href="{% url 'proposal_history_run_detail' myScenario.id summary.run.id %}" class="run-card">
        <div>
          <div class="run-card-date">{{ summary.run.created_at|date:"d M Y, H:i" }}</div>
          <div class="run-card-meta">
            Triggered by {{ summary.run.created_by.get_full_name|default:summary.run.created_by.username|default:"Unknown" }}
            · {{ summary.total }} proposal{{ summary.total|pluralize }}
          </div>
        </div>
        <div class="run-stats">
          <span class="run-stat accepted">{{ summary.accepted }} accepted</span>
          <span class="run-stat rejected">{{ summary.rejected }} rejected</span>
          <span class="run-stat pending">{{ summary.never_decided }} never decided</span>
        </div>
      </a>
      {% empty %}
      <div class="text-center text-muted py-5">
        <i class="bi bi-clock-history" style="font-size:2.5rem;color:#d1d9e0;"></i>
        <p class="mt-2 mb-0">No past generations yet.</p>
        <p class="small">History appears here after you regenerate proposals for this scenario.</p>
      </div>
      {% endfor %}
    </section>
  </main>
  {% endblock %}
  ```

- [ ] **Step 6: Create the run-detail template**

  Create `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_history_run_detail.html`:
  ```html
  {% extends "main.html" %}
  {% load group_tags %}
  {% block page_title %}<title>Trust AI Lab — Proposal History — {{ run.created_at|date:"d M Y" }} — {{ myScenario.name }}</title>{% endblock %}
  {% block atcontent %}

  <style>
    .pr-hero {
      background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
      border-radius: 14px; padding: 26px 30px 20px; color: #fff;
      margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
    }
    .pr-hero-icon {
      background: rgba(255,255,255,0.18); border-radius: 10px;
      width: 50px; height: 50px; display:flex; align-items:center;
      justify-content:center; font-size:22px; flex-shrink:0;
    }
    .pr-hero .breadcrumb { background:none; margin:10px 0 0; padding:0; font-size:12px; }
    .pr-hero .breadcrumb-item+.breadcrumb-item::before { color:rgba(255,255,255,0.5); }
    .pr-hero .breadcrumb-item a { color:rgba(255,255,255,0.72); text-decoration:none; }
    .pr-hero .breadcrumb-item a:hover { color:#fff; }
    .pr-hero .breadcrumb-item.active { color:rgba(255,255,255,0.92); }
    .hero-btn-ghost { background:rgba(255,255,255,0.15); color:#fff; border:1px solid rgba(255,255,255,0.3); font-weight:600; font-size:13px; border-radius:8px; padding:7px 16px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; white-space:nowrap; }
    .hero-btn-ghost:hover { background:rgba(255,255,255,0.25); color:#fff; }
    @media (max-width:575.98px) {
      .pr-hero { padding:14px 16px 12px; }
      .pr-hero > .d-flex { flex-wrap:wrap; }
      .pr-hero-icon { display:none; }
      .pr-hero h2 { font-size:15px !important; }
    }

    .proposal-card {
      border-radius:12px; border:1px solid #e8edf5; background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,0.04); margin-bottom:12px;
      padding:16px 20px;
    }
    .proposal-card[data-status="accepted"] { border-left:4px solid #16a34a; }
    .proposal-card[data-status="rejected"] { border-left:4px solid #dc2626; }
    .proposal-card[data-status="new"]      { border-left:4px solid #d97706; }
    .proposal-card-title { font-weight:700; color:#012970; font-size:14px; margin:0; }
    .proposal-card-meta { font-size:12px; color:#888; margin-top:2px; }
    .status-badge { font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px; text-transform:uppercase; letter-spacing:.3px; white-space:nowrap; }
    .status-badge.accepted { background:#dcfce7; color:#166534; }
    .status-badge.rejected { background:#fee2e2; color:#991b1b; }
    .status-badge.new      { background:#fef3c7; color:#92400e; }
    .proposal-card-body { font-size:13px; color:#333; margin-top:10px; line-height:1.5; white-space:pre-wrap; }
    .rejection-reasons { margin-top:8px; font-size:12px; color:#991b1b; }
  </style>

  <main id="main" class="main">
    <div class="pr-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="pr-hero-icon"><i class="bi bi-clock-history"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Proposals</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">{{ run.created_at|date:"d M Y, H:i" }}</h2>
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ myScenario.name }}</div>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'proposal_list' myScenario.id %}">Proposals</a></li>
            <li class="breadcrumb-item"><a href="{% url 'proposal_history' myScenario.id %}">History</a></li>
            <li class="breadcrumb-item active">{{ run.created_at|date:"d M Y" }}</li>
          </ol></nav>
        </div>
        <div class="flex-shrink-0 d-flex gap-2 align-items-start" style="padding-top:4px;">
          <a href="{% url 'proposal_history' myScenario.id %}" class="hero-btn-ghost">
            <i class="bi bi-arrow-left"></i> Back
          </a>
        </div>
      </div>
    </div>

    <section class="section">
      {% for proposal in proposals %}
      {% with review=user_reviews|dict_get:proposal.id %}
      <div class="proposal-card" data-status="{{ review.status|default:'new' }}">
        <div class="d-flex align-items-start justify-content-between gap-2 flex-wrap">
          <h6 class="proposal-card-title">{{ proposal.get_proposal_type_display }} — {{ proposal.activity.name }}</h6>
          <span class="status-badge {{ review.status|default:'new' }}">
            {% if review.status == 'accepted' %}Accepted{% elif review.status == 'rejected' %}Rejected{% else %}Never decided{% endif %}
          </span>
        </div>
        <div class="proposal-card-meta">
          {{ proposal.phase.name }}
          {% if review.status == 'accepted' or review.status == 'rejected' %} · Decided {{ review.reviewed_at|date:"d M Y, H:i" }}{% endif %}
          {% if review.was_edited %} · Edited {{ review.edit_count }} time{{ review.edit_count|pluralize }}{% endif %}
        </div>
        <div class="proposal-card-body">{{ proposal.suggested_action }}</div>
        {% if review.status == 'rejected' and review.rejection_reasons %}
        <div class="rejection-reasons">
          <strong>Rejection reasons:</strong> {{ review.rejection_reasons|join:", " }}
        </div>
        {% endif %}
      </div>
      {% endwith %}
      {% empty %}
      <div class="text-center text-muted py-5">
        <i class="bi bi-inbox" style="font-size:2.5rem;color:#d1d9e0;"></i>
        <p class="mt-2 mb-0">No proposals in this run.</p>
      </div>
      {% endfor %}
    </section>
  </main>
  {% endblock %}
  ```

  Note: `{% load group_tags %}` gives access to the `dict_get` filter (`Trust-AI-Platform/templatetags/group_tags.py:14-15`, `def dict_get(d, key): return d.get(key)`) — the same project-level filter `proposal_list.html` already uses for the identical dict-keyed-by-proposal-id lookup pattern (`proposal_list.html:302`). `review` resolves to `None` for a proposal this user never reviewed (an archived run's proposals only have reviews for users who visited while it was current) — Django templates resolve attribute access on `None` to an empty string rather than raising, so `{{ review.status|default:'new' }}` correctly falls back to the "never decided" styling/badge with no template error.

- [ ] **Step 7: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.ProposalHistoryViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (6 tests).

  Then the full set of tests this plan added, across all three tasks:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test authoringtool.tests.ProposalGenerationRunModelTests authoringtool.tests.TriggerLlmContextTaskPermissionTests authoringtool.tests.ProposalListViewCurrentRunScopingTests authoringtool.tests.AcceptedReviewsForPersonalScenarioScopingTests authoringtool.tests.ProposalHistoryViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (17 tests: 4 + 4 + 2 + 1 + 6). The exact total isn't load-bearing — "OK with zero failures" is what matters; recount from the actual class list above if the number ever looks off.

- [ ] **Step 8: Manually verify (if a real dev environment is available)**

  Same caveat as prior features on this branch — needs a live Postgres-backed dev environment with Ollama/RAG available. If available: as a scenario's creator, regenerate proposals twice (so at least one past run exists), then visit `/scenarios/<id>/proposals/history/`, confirm the past run appears with correct counts, click into it, confirm the read-only cards show the right titles/decisions/rejection-reasons with no action buttons. As a non-creator teacher, confirm you can still view the history (per the deliberate access-parity decision) but cannot reach the "Generate" button's underlying endpoint (403 on direct POST).

  If unavailable, Step 7's automated tests are the load-bearing verification.

- [ ] **Step 9: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/views.py Trust-AI-Platform/authoringtool/urls.py Trust-AI-Platform/authoringtool/tests.py Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_history.html Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_history_run_detail.html
  git commit -m "Add read-only proposal generation history UI"
  ```
