# Feedback Forms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admin-built feedback forms (multiple-choice + free-text questions) shown to teachers after creating a personalized scenario and to students after finishing a scenario via the Plato chatbot, with a staff-only management UI (counts, view/edit/delete forms, delete responses) and XLSX/comma-CSV export.

**Architecture:** Six sequential tasks. Task 1 scaffolds the new `feedback` app with all four models. Task 2 builds the shared submit endpoint and applicability/serialization helpers. Tasks 3–4 build the staff-only management UI (form CRUD, then responses + exports). Task 5 wires the teacher trigger (session flag → modal on the proposals page). Task 6 wires the student trigger (Rasa custom payload → JS relay chain → modal in the scenario view, teacher-gated).

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Bootstrap 5 · openpyxl · Rasa custom actions (RasaFaith, same repo) · SQLite (dev/test)

## Global Constraints

- **Responsive & mobile-first (explicit requirement):** every new template works on phones (≥320px) and tablets (≥768px). Management pages follow the established hero pattern with the `@media (max-width: 575.98px)` breakpoint convention; tables wrapped in `.table-responsive`; modals are standard Bootstrap (already responsive); no new fixed-pixel widths on outer containers; the scenario-assignment checkbox list scrolls inside a `max-height` container.
- **Audience enforcement is server-side, not just UI:** the submit endpoint rejects `student`-audience submissions from users in the `teachers` group with 403, and `teacher`-audience submissions from non-teachers with 403 — hiding the modal from teachers in the student view is defense-in-depth on top of this, not the enforcement itself.
- **`assign_to_all` is a live rule** covering future scenarios; `excluded_scenarios` only consulted when it's True, `included_scenarios` only when False — exactly as `FeedbackForm.applies_to()` encodes in the spec.
- One response per (form, user, scenario), DB-enforced via `UniqueConstraint`; duplicate submissions return a friendly JSON error, not a 500.
- All model `Meta.ordering` values carry a `pk` tiebreaker from the start (`['-created_on', '-pk']` etc.) — the `auto_now_add`-tie flakiness fix applied proactively (fourth time on this branch; never wait for the flaky test).
- `staff_required` is duplicated locally into `feedback/views.py` from `accounts/admin_views.py:12-19` — this codebase's established per-app-duplication convention (`group_required` is duplicated in 5 apps the same way). Same for a local `group_required` needed by nothing here — NOT needed; only `staff_required` and `login_required` are used in this app.
- CSV export uses the **comma** delimiter — a deliberate deviation from `usergroups/views.py:456`'s `delimiter=';'` precedent, explicitly required by the spec ("csv with comma delimiter").
- XLSX export follows `authoringtool/views.py:2563-2580`'s pattern exactly: openpyxl workbook → `io.BytesIO()` → `HttpResponse` with content type `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` and a `Content-Disposition: attachment` header.
- The old `MultilingualQuestion`/`MultilingualAnswer` system is untouched — no imports from it, no changes to it.
- **Two diverged, git-tracked copies of `chat.js` exist** and BOTH must receive the same Task 6 edit: `data/static/chatbot_static/js/components/chat.js` (production-served: docker-compose mounts `./data/static` as the container's staticfiles dir and nginx serves it read-only) and `Trust-AI-Platform/staticfiles/chatbot_static/js/components/chat.js` (served in local dev via `faithDev/urls.py:21`'s `static(STATIC_URL, document_root=STATIC_ROOT)` mapping). The files have drifted from each other, so apply the same anchored insertion to each — do NOT copy one file over the other. The `Nginx Configuration/` folder's copies are deployment artifacts and are NOT touched.
- The Rasa change (Task 6) is one added `dispatcher.utter_message(json_message=...)` line in `ActionEndScenario` — the same `json_message` serialization sibling actions already use for `activity_id` (`RasaFaith/RasaFaith/actions/actions.py:174-176`), which the REST channel exposes as `msg.custom` to `chat.js`. No NLU/domain/stories changes, no retraining; deploy requires a Rasa action-server restart only.
- Tests that hit `create_personal_scenario` MUST mock `authoringtool.views.apply_user_proposals_to_new_scenario.delay` — `CELERY_TASK_ALWAYS_EAGER=True` in test settings means an unmocked `.delay()` would synchronously run the real scenario-cloning task in-process (same class of hazard as the proposal-generation tests fixed earlier on this branch; mock proactively, don't rediscover it).
- Migration file names below use placeholder numbers — confirm the actual filename `makemigrations` generates and use it.

---

### Task 1: `feedback` app scaffold, models, migration, admin, registration

**Files:**
- Create: `Trust-AI-Platform/feedback/__init__.py`
- Create: `Trust-AI-Platform/feedback/apps.py`
- Create: `Trust-AI-Platform/feedback/models.py`
- Create: `Trust-AI-Platform/feedback/admin.py`
- Create: `Trust-AI-Platform/feedback/views.py` (placeholder, filled by later tasks)
- Create: `Trust-AI-Platform/feedback/urls.py` (empty urlpatterns, filled by later tasks)
- Create: `Trust-AI-Platform/feedback/tests.py`
- Create: `Trust-AI-Platform/feedback/migrations/__init__.py`
- Create (via `makemigrations`): `Trust-AI-Platform/feedback/migrations/0001_initial.py`
- Modify: `Trust-AI-Platform/faithDev/settings.py`
- Modify: `Trust-AI-Platform/faithDev/urls.py`

**Interfaces:**
- Produces: `FeedbackForm` (with `applies_to(scenario)` method), `FeedbackQuestion`, `FeedbackResponse`, `FeedbackAnswer` models — consumed by every later task. App registered in `INSTALLED_APPS` and at URL prefix `feedback/`.

- [ ] **Step 1: Create the app skeleton**

  Create `Trust-AI-Platform/feedback/__init__.py` and `Trust-AI-Platform/feedback/migrations/__init__.py` (both empty).

  Create `Trust-AI-Platform/feedback/apps.py`:
  ```python
  from django.apps import AppConfig


  class FeedbackConfig(AppConfig):
      default_auto_field = 'django.db.models.BigAutoField'
      name = 'feedback'
  ```

  Create `Trust-AI-Platform/feedback/views.py`:
  ```python
  # Views are added by the management-UI and submit-endpoint tasks.
  ```

  Create `Trust-AI-Platform/feedback/urls.py`:
  ```python
  from django.urls import path
  from . import views

  urlpatterns = []
  ```

- [ ] **Step 2: Register the app**

  In `Trust-AI-Platform/faithDev/settings.py`, replace:
  ```python
      'home',
      'messaging',
  ]
  ```
  with:
  ```python
      'home',
      'messaging',
      'feedback',
  ]
  ```

  In `Trust-AI-Platform/faithDev/urls.py`, replace:
  ```python
      path('messaging/', include('messaging.urls')),
  ```
  with:
  ```python
      path('messaging/', include('messaging.urls')),
      path('feedback/', include('feedback.urls')),
  ```

- [ ] **Step 3: Write the failing tests**

  Create `Trust-AI-Platform/feedback/tests.py`:
  ```python
  import json

  from django.contrib.auth.models import Group, User
  from django.db import IntegrityError, transaction
  from django.test import Client, TestCase
  from django.urls import reverse

  from authoringtool.models import Scenario
  from feedback.models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse


  class FeedbackFormAppliesToTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('fb_admin', password='pass', is_staff=True)
          self.scenario_a = Scenario.objects.create(name='FB Scenario A', created_by=self.user, updated_by=self.user)
          self.scenario_b = Scenario.objects.create(name='FB Scenario B', created_by=self.user, updated_by=self.user)

      def test_assign_to_all_applies_everywhere(self):
          form = FeedbackForm.objects.create(title='All', audience='student', assign_to_all=True, created_by=self.user)
          self.assertTrue(form.applies_to(self.scenario_a))
          self.assertTrue(form.applies_to(self.scenario_b))

      def test_assign_to_all_respects_exclusions(self):
          form = FeedbackForm.objects.create(title='All minus B', audience='student', assign_to_all=True, created_by=self.user)
          form.excluded_scenarios.add(self.scenario_b)
          self.assertTrue(form.applies_to(self.scenario_a))
          self.assertFalse(form.applies_to(self.scenario_b))

      def test_explicit_inclusion_mode(self):
          form = FeedbackForm.objects.create(title='Only A', audience='student', assign_to_all=False, created_by=self.user)
          form.included_scenarios.add(self.scenario_a)
          self.assertTrue(form.applies_to(self.scenario_a))
          self.assertFalse(form.applies_to(self.scenario_b))

      def test_inactive_form_never_applies(self):
          form = FeedbackForm.objects.create(title='Off', audience='student', assign_to_all=True, is_active=False, created_by=self.user)
          self.assertFalse(form.applies_to(self.scenario_a))


  class FeedbackResponseConstraintTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('fb_responder', password='pass')
          self.scenario = Scenario.objects.create(name='FB Constraint Scenario', created_by=self.user, updated_by=self.user)
          self.form = FeedbackForm.objects.create(title='F', audience='student', created_by=self.user)

      def test_one_response_per_form_user_scenario(self):
          FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)
          with self.assertRaises(IntegrityError):
              with transaction.atomic():
                  FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)

      def test_one_answer_per_question_per_response(self):
          question = FeedbackQuestion.objects.create(form=self.form, text='Q1', question_type='text')
          response = FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)
          FeedbackAnswer.objects.create(response=response, question=question, answer_text='a')
          with self.assertRaises(IntegrityError):
              with transaction.atomic():
                  FeedbackAnswer.objects.create(response=response, question=question, answer_text='b')

      def test_questions_ordered_by_order_field(self):
          FeedbackQuestion.objects.create(form=self.form, text='Second', question_type='text', order=2)
          FeedbackQuestion.objects.create(form=self.form, text='First', question_type='text', order=1)
          texts = list(self.form.questions.values_list('text', flat=True))
          self.assertEqual(texts, ['First', 'Second'])
  ```

- [ ] **Step 4: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `ModuleNotFoundError`/`ImportError` around `feedback.models` (models don't exist yet).

- [ ] **Step 5: Create the models**

  Create `Trust-AI-Platform/feedback/models.py`:
  ```python
  from django.contrib.auth.models import User
  from django.db import models


  class FeedbackForm(models.Model):
      AUDIENCE_CHOICES = [('teacher', 'Teacher'), ('student', 'Student')]

      title = models.CharField(max_length=255)
      description = models.TextField(blank=True)
      audience = models.CharField(max_length=16, choices=AUDIENCE_CHOICES)
      is_active = models.BooleanField(default=True)
      assign_to_all = models.BooleanField(default=True)
      included_scenarios = models.ManyToManyField('authoringtool.Scenario', blank=True, related_name='included_feedback_forms')
      excluded_scenarios = models.ManyToManyField('authoringtool.Scenario', blank=True, related_name='excluded_feedback_forms')
      created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='feedback_forms')
      created_on = models.DateTimeField(auto_now_add=True)
      updated_on = models.DateTimeField(auto_now=True)

      class Meta:
          ordering = ['-created_on', '-pk']

      def __str__(self):
          return f"{self.title} ({self.get_audience_display()})"

      def applies_to(self, scenario):
          if not self.is_active:
              return False
          if self.assign_to_all:
              return not self.excluded_scenarios.filter(pk=scenario.pk).exists()
          return self.included_scenarios.filter(pk=scenario.pk).exists()


  class FeedbackQuestion(models.Model):
      TYPE_CHOICES = [('choice', 'Multiple Choice'), ('text', 'Free Text')]

      form = models.ForeignKey(FeedbackForm, on_delete=models.CASCADE, related_name='questions')
      text = models.CharField(max_length=500)
      question_type = models.CharField(max_length=16, choices=TYPE_CHOICES)
      options = models.JSONField(default=list, blank=True)
      is_required = models.BooleanField(default=True)
      order = models.PositiveIntegerField(default=0)

      class Meta:
          ordering = ['order', 'pk']

      def __str__(self):
          return f"{self.text[:50]} ({self.form.title})"


  class FeedbackResponse(models.Model):
      form = models.ForeignKey(FeedbackForm, on_delete=models.CASCADE, related_name='responses')
      user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedback_responses')
      scenario = models.ForeignKey('authoringtool.Scenario', on_delete=models.CASCADE, related_name='feedback_responses')
      submitted_at = models.DateTimeField(auto_now_add=True)

      class Meta:
          ordering = ['-submitted_at', '-pk']
          constraints = [
              models.UniqueConstraint(fields=['form', 'user', 'scenario'], name='unique_feedback_response'),
          ]

      def __str__(self):
          return f"{self.user.username} -> {self.form.title} ({self.scenario.name})"


  class FeedbackAnswer(models.Model):
      response = models.ForeignKey(FeedbackResponse, on_delete=models.CASCADE, related_name='answers')
      question = models.ForeignKey(FeedbackQuestion, on_delete=models.CASCADE, related_name='answers')
      answer_text = models.TextField(blank=True)

      class Meta:
          constraints = [
              models.UniqueConstraint(fields=['response', 'question'], name='unique_feedback_answer_per_question'),
          ]

      def __str__(self):
          return f"{self.question.text[:30]}: {self.answer_text[:30]}"
  ```

- [ ] **Step 6: Generate and apply the migration**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py makemigrations feedback --settings=faithDev.settings_test
  "../djangofaithvenv/Scripts/python.exe" manage.py migrate feedback --settings=faithDev.settings_test
  ```
  Expected: `0001_initial.py` creating all four models, applied cleanly.

- [ ] **Step 7: Register in Django admin**

  Create `Trust-AI-Platform/feedback/admin.py`:
  ```python
  from django.contrib import admin

  from .models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse


  class FeedbackQuestionInline(admin.TabularInline):
      model = FeedbackQuestion
      extra = 0


  @admin.register(FeedbackForm)
  class FeedbackFormAdmin(admin.ModelAdmin):
      list_display = ('id', 'title', 'audience', 'is_active', 'assign_to_all', 'created_by', 'created_on')
      list_filter = ('audience', 'is_active', 'assign_to_all')
      search_fields = ('title',)
      filter_horizontal = ('included_scenarios', 'excluded_scenarios')
      readonly_fields = ('created_on', 'updated_on')
      inlines = [FeedbackQuestionInline]


  @admin.register(FeedbackResponse)
  class FeedbackResponseAdmin(admin.ModelAdmin):
      list_display = ('id', 'form', 'user', 'scenario', 'submitted_at')
      list_filter = ('form', 'submitted_at')
      search_fields = ('user__username', 'form__title', 'scenario__name')
      raw_id_fields = ('form', 'user', 'scenario')
      readonly_fields = ('submitted_at',)


  @admin.register(FeedbackAnswer)
  class FeedbackAnswerAdmin(admin.ModelAdmin):
      list_display = ('id', 'response', 'question', 'answer_text')
      search_fields = ('answer_text', 'question__text')
      raw_id_fields = ('response', 'question')
  ```

- [ ] **Step 8: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (7 tests).

- [ ] **Step 9: Commit**

  ```bash
  git status --short -- Trust-AI-Platform/feedback/
  ```
  Confirm only the new app's files are untracked.
  ```bash
  git add Trust-AI-Platform/feedback/ Trust-AI-Platform/faithDev/settings.py Trust-AI-Platform/faithDev/urls.py
  git commit -m "Add feedback app with form, question, response, and answer models"
  ```
  Note: `git add Trust-AI-Platform/feedback/` is the one permitted directory-level add here because the directory is brand new and Step 9's `git status` check confirms it contains only this task's files (excluding any `__pycache__` — if `git status` shows `.pyc` files under `feedback/`, add the files by exact path instead).

---

### Task 2: Applicability helpers + shared submit endpoint

**Files:**
- Create: `Trust-AI-Platform/feedback/utils.py`
- Modify: `Trust-AI-Platform/feedback/views.py`
- Modify: `Trust-AI-Platform/feedback/urls.py`
- Modify: `Trust-AI-Platform/feedback/tests.py`

**Interfaces:**
- Consumes: the four models (Task 1)
- Produces: `feedback.utils.get_applicable_form(scenario, audience)` (newest applicable active form or `None`), `feedback.utils.user_has_responded(form, user, scenario)` (bool), `feedback.utils.serialize_form(form)` (dict with `id`, `title`, `description`, `questions` list) — consumed by Tasks 5 and 6. URL name `feedback_submit` (`feedback/submit/<int:form_id>/<int:scenario_id>/`, POST JSON) — consumed by both trigger modals.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/feedback/tests.py`:
  ```python
  class FeedbackSubmitEndpointTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.teacher = User.objects.create_user('fb_teacher', password='pass')
          self.teacher.groups.add(teachers)
          self.student = User.objects.create_user('fb_student', password='pass')
          self.scenario = Scenario.objects.create(name='FB Submit Scenario', created_by=self.teacher, updated_by=self.teacher)

          self.student_form = FeedbackForm.objects.create(title='Student form', audience='student', created_by=self.teacher)
          self.q_choice = FeedbackQuestion.objects.create(
              form=self.student_form, text='Useful?', question_type='choice',
              options=['Yes', 'No'], is_required=True, order=1,
          )
          self.q_text = FeedbackQuestion.objects.create(
              form=self.student_form, text='Comments?', question_type='text',
              is_required=False, order=2,
          )
          self.teacher_form = FeedbackForm.objects.create(title='Teacher form', audience='teacher', created_by=self.teacher)
          self.tq = FeedbackQuestion.objects.create(
              form=self.teacher_form, text='Proposals good?', question_type='choice',
              options=['Yes', 'No'], is_required=True, order=1,
          )

      def _submit(self, form, answers):
          url = reverse('feedback_submit', args=[form.id, self.scenario.id])
          return self.client.post(url, json.dumps({'answers': answers}), content_type='application/json')

      def test_student_can_submit_student_form(self):
          self.client.login(username='fb_student', password='pass')
          r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes', str(self.q_text.id): 'Great tool'})
          self.assertTrue(r.json()['success'])
          response = FeedbackResponse.objects.get(form=self.student_form, user=self.student, scenario=self.scenario)
          self.assertEqual(response.answers.count(), 2)

      def test_teacher_blocked_from_student_form(self):
          self.client.login(username='fb_teacher', password='pass')
          r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
          self.assertEqual(r.status_code, 403)
          self.assertFalse(FeedbackResponse.objects.filter(form=self.student_form).exists())

      def test_student_blocked_from_teacher_form(self):
          self.client.login(username='fb_student', password='pass')
          r = self._submit(self.teacher_form, {str(self.tq.id): 'Yes'})
          self.assertEqual(r.status_code, 403)

      def test_form_not_applicable_to_scenario_blocked(self):
          self.student_form.assign_to_all = False
          self.student_form.save()
          self.client.login(username='fb_student', password='pass')
          r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
          self.assertEqual(r.status_code, 403)

      def test_missing_required_answer_rejected(self):
          self.client.login(username='fb_student', password='pass')
          r = self._submit(self.student_form, {str(self.q_text.id): 'only optional'})
          self.assertEqual(r.status_code, 400)
          self.assertFalse(FeedbackResponse.objects.filter(form=self.student_form).exists())

      def test_choice_answer_must_be_valid_option(self):
          self.client.login(username='fb_student', password='pass')
          r = self._submit(self.student_form, {str(self.q_choice.id): 'Maybe'})
          self.assertEqual(r.status_code, 400)

      def test_duplicate_submission_returns_friendly_error(self):
          self.client.login(username='fb_student', password='pass')
          self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
          r = self._submit(self.student_form, {str(self.q_choice.id): 'No'})
          self.assertEqual(r.status_code, 400)
          self.assertIn('already', r.json()['error'].lower())
          self.assertEqual(FeedbackResponse.objects.filter(form=self.student_form).count(), 1)

      def test_get_method_not_allowed(self):
          self.client.login(username='fb_student', password='pass')
          url = reverse('feedback_submit', args=[self.student_form.id, self.scenario.id])
          r = self.client.get(url)
          self.assertEqual(r.status_code, 405)


  class FeedbackUtilsTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('fb_utils', password='pass')
          self.scenario = Scenario.objects.create(name='FB Utils Scenario', created_by=self.user, updated_by=self.user)

      def test_get_applicable_form_returns_newest_applicable(self):
          from feedback.utils import get_applicable_form
          older = FeedbackForm.objects.create(title='Older', audience='student', created_by=self.user)
          newer = FeedbackForm.objects.create(title='Newer', audience='student', created_by=self.user)
          self.assertEqual(get_applicable_form(self.scenario, 'student'), newer)

      def test_get_applicable_form_skips_wrong_audience_and_inactive(self):
          from feedback.utils import get_applicable_form
          FeedbackForm.objects.create(title='Teacher only', audience='teacher', created_by=self.user)
          FeedbackForm.objects.create(title='Inactive', audience='student', is_active=False, created_by=self.user)
          self.assertIsNone(get_applicable_form(self.scenario, 'student'))

      def test_serialize_form_shape(self):
          from feedback.utils import serialize_form
          form = FeedbackForm.objects.create(title='S', description='D', audience='student', created_by=self.user)
          FeedbackQuestion.objects.create(form=form, text='Q', question_type='choice', options=['A', 'B'], order=1)
          data = serialize_form(form)
          self.assertEqual(data['title'], 'S')
          self.assertEqual(len(data['questions']), 1)
          self.assertEqual(data['questions'][0]['options'], ['A', 'B'])
          self.assertEqual(data['questions'][0]['type'], 'choice')
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.FeedbackSubmitEndpointTests feedback.tests.FeedbackUtilsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` (`feedback_submit` doesn't exist) and `ImportError` (`feedback.utils` doesn't exist).

- [ ] **Step 3: Create the helpers**

  Create `Trust-AI-Platform/feedback/utils.py`:
  ```python
  from .models import FeedbackForm, FeedbackResponse


  def get_applicable_form(scenario, audience):
      """Newest active form for this audience that applies to the scenario, or None."""
      for form in FeedbackForm.objects.filter(audience=audience, is_active=True):
          if form.applies_to(scenario):
              return form
      return None


  def user_has_responded(form, user, scenario):
      return FeedbackResponse.objects.filter(form=form, user=user, scenario=scenario).exists()


  def serialize_form(form):
      return {
          'id': form.id,
          'title': form.title,
          'description': form.description,
          'questions': [
              {
                  'id': q.id,
                  'text': q.text,
                  'type': q.question_type,
                  'options': q.options,
                  'required': q.is_required,
              }
              for q in form.questions.all()
          ],
      }
  ```
  (`FeedbackForm.Meta.ordering` is `['-created_on', '-pk']`, so plain iteration already visits newest first — "newest applicable" needs no extra ordering.)

- [ ] **Step 4: Create the submit view and URL**

  Replace the entire contents of `Trust-AI-Platform/feedback/views.py` with:
  ```python
  import json

  from django.contrib.auth.decorators import login_required
  from django.db import transaction
  from django.http import HttpResponseForbidden, JsonResponse
  from django.shortcuts import get_object_or_404
  from django.views.decorators.http import require_POST
  from functools import wraps

  from authoringtool.models import Scenario
  from .models import FeedbackAnswer, FeedbackForm, FeedbackResponse


  def staff_required(view_func):
      @wraps(view_func)
      @login_required
      def _wrapped(request, *args, **kwargs):
          if not (request.user.is_staff or request.user.is_superuser):
              return HttpResponseForbidden("Access denied.")
          return view_func(request, *args, **kwargs)
      return _wrapped


  @require_POST
  @login_required
  def submit_feedback(request, form_id, scenario_id):
      form = get_object_or_404(FeedbackForm, id=form_id)
      scenario = get_object_or_404(Scenario, id=scenario_id)

      is_teacher = request.user.groups.filter(name='teachers').exists()
      if form.audience == 'student' and is_teacher:
          return JsonResponse({'success': False, 'error': 'Student forms cannot be submitted by teachers.'}, status=403)
      if form.audience == 'teacher' and not is_teacher:
          return JsonResponse({'success': False, 'error': 'Teacher forms can only be submitted by teachers.'}, status=403)
      if not form.applies_to(scenario):
          return JsonResponse({'success': False, 'error': 'This form does not apply to this scenario.'}, status=403)

      if FeedbackResponse.objects.filter(form=form, user=request.user, scenario=scenario).exists():
          return JsonResponse({'success': False, 'error': 'You have already submitted this form.'}, status=400)

      try:
          payload = json.loads(request.body or '{}')
      except json.JSONDecodeError:
          return JsonResponse({'success': False, 'error': 'Invalid JSON.'}, status=400)
      answers = payload.get('answers') or {}

      questions = list(form.questions.all())
      for question in questions:
          raw = (answers.get(str(question.id)) or '').strip()
          if question.is_required and not raw:
              return JsonResponse({'success': False, 'error': f'Question "{question.text}" is required.'}, status=400)
          if raw and question.question_type == 'choice' and raw not in question.options:
              return JsonResponse({'success': False, 'error': f'Invalid option for "{question.text}".'}, status=400)

      with transaction.atomic():
          response = FeedbackResponse.objects.create(form=form, user=request.user, scenario=scenario)
          for question in questions:
              raw = (answers.get(str(question.id)) or '').strip()
              if raw:
                  FeedbackAnswer.objects.create(response=response, question=question, answer_text=raw)

      return JsonResponse({'success': True})
  ```

  Replace the contents of `Trust-AI-Platform/feedback/urls.py` with:
  ```python
  from django.urls import path
  from . import views

  urlpatterns = [
      path('submit/<int:form_id>/<int:scenario_id>/', views.submit_feedback, name='feedback_submit'),
  ]
  ```

  Note: `staff_required` is defined here now (duplicated from `accounts/admin_views.py:12-19` per the per-app-duplication convention) even though nothing in this task uses it — Task 3's management views need it and defining it alongside the app's first real view keeps Task 3's diff focused on its own views. It is exempt from "no unused code" for exactly one task.

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (18 tests: 7 from Task 1 + 11 new).

- [ ] **Step 6: Commit**

  ```bash
  git add Trust-AI-Platform/feedback/utils.py Trust-AI-Platform/feedback/views.py Trust-AI-Platform/feedback/urls.py Trust-AI-Platform/feedback/tests.py
  git commit -m "Add feedback applicability helpers and shared submit endpoint"
  ```

---

### Task 3: Management UI — forms list, create/edit, delete, dashboard link

**Files:**
- Modify: `Trust-AI-Platform/feedback/views.py`
- Modify: `Trust-AI-Platform/feedback/urls.py`
- Modify: `Trust-AI-Platform/feedback/tests.py`
- Create: `Trust-AI-Platform/feedback/templates/feedback/form_list.html`
- Create: `Trust-AI-Platform/feedback/templates/feedback/form_edit.html`
- Modify: `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`

**Interfaces:**
- Consumes: models (Task 1), `staff_required` (Task 2)
- Produces: URL names `feedback_form_list` (`feedback/manage/`), `feedback_form_create` (`feedback/manage/create/`), `feedback_form_edit` (`feedback/manage/<int:form_id>/edit/`), `feedback_form_delete` (`feedback/manage/<int:form_id>/delete/`) — `feedback_form_list` is linked from the admin dashboard; Task 4 adds its responses/export URLs alongside these.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/feedback/tests.py`:
  ```python
  class FeedbackManagementViewTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.staff = User.objects.create_user('fb_staff', password='pass', is_staff=True)
          self.plain = User.objects.create_user('fb_plain', password='pass')
          self.scenario = Scenario.objects.create(name='FB Manage Scenario', created_by=self.staff, updated_by=self.staff)

      def test_non_staff_forbidden(self):
          self.client.login(username='fb_plain', password='pass')
          r = self.client.get(reverse('feedback_form_list'))
          self.assertEqual(r.status_code, 403)

      def test_staff_sees_form_list_with_response_counts(self):
          form = FeedbackForm.objects.create(title='Counted', audience='student', created_by=self.staff)
          FeedbackResponse.objects.create(form=form, user=self.plain, scenario=self.scenario)
          self.client.login(username='fb_staff', password='pass')
          r = self.client.get(reverse('feedback_form_list'))
          self.assertContains(r, 'Counted')
          self.assertEqual(r.context['forms'][0].response_count, 1)

      def test_create_form_with_questions(self):
          self.client.login(username='fb_staff', password='pass')
          r = self.client.post(reverse('feedback_form_create'), {
              'title': 'New Form',
              'description': 'Desc',
              'audience': 'teacher',
              'is_active': 'on',
              'assign_to_all': 'on',
              'scenarios': [str(self.scenario.id)],
              'questions_json': json.dumps([
                  {'text': 'Useful?', 'type': 'choice', 'options': ['Yes', 'No'], 'required': True},
                  {'text': 'Comments', 'type': 'text', 'options': [], 'required': False},
              ]),
          })
          self.assertRedirects(r, reverse('feedback_form_list'))
          form = FeedbackForm.objects.get(title='New Form')
          self.assertEqual(form.questions.count(), 2)
          self.assertTrue(form.assign_to_all)
          self.assertEqual(form.excluded_scenarios.count(), 0)

      def test_create_assign_to_all_unchecked_scenario_becomes_exclusion(self):
          other = Scenario.objects.create(name='FB Other Scenario', created_by=self.staff, updated_by=self.staff)
          self.client.login(username='fb_staff', password='pass')
          self.client.post(reverse('feedback_form_create'), {
              'title': 'Partial', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
              'scenarios': [str(self.scenario.id)],  # `other` left unchecked -> excluded
              'questions_json': json.dumps([{'text': 'Q', 'type': 'text', 'options': [], 'required': True}]),
          })
          form = FeedbackForm.objects.get(title='Partial')
          self.assertTrue(form.applies_to(self.scenario))
          self.assertFalse(form.applies_to(other))

      def test_create_without_assign_to_all_checked_scenarios_are_inclusions(self):
          other = Scenario.objects.create(name='FB Incl Scenario', created_by=self.staff, updated_by=self.staff)
          self.client.login(username='fb_staff', password='pass')
          self.client.post(reverse('feedback_form_create'), {
              'title': 'Incl', 'audience': 'student', 'is_active': 'on',
              'scenarios': [str(self.scenario.id)],
              'questions_json': json.dumps([{'text': 'Q', 'type': 'text', 'options': [], 'required': True}]),
          })
          form = FeedbackForm.objects.get(title='Incl')
          self.assertFalse(form.assign_to_all)
          self.assertTrue(form.applies_to(self.scenario))
          self.assertFalse(form.applies_to(other))

      def test_edit_replaces_questions(self):
          form = FeedbackForm.objects.create(title='Editable', audience='student', created_by=self.staff)
          FeedbackQuestion.objects.create(form=form, text='Old Q', question_type='text', order=1)
          self.client.login(username='fb_staff', password='pass')
          self.client.post(reverse('feedback_form_edit', args=[form.id]), {
              'title': 'Editable v2', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
              'questions_json': json.dumps([{'text': 'New Q', 'type': 'text', 'options': [], 'required': True}]),
          })
          form.refresh_from_db()
          self.assertEqual(form.title, 'Editable v2')
          self.assertEqual(list(form.questions.values_list('text', flat=True)), ['New Q'])

      def test_delete_form(self):
          form = FeedbackForm.objects.create(title='Doomed', audience='student', created_by=self.staff)
          self.client.login(username='fb_staff', password='pass')
          r = self.client.post(reverse('feedback_form_delete', args=[form.id]))
          self.assertRedirects(r, reverse('feedback_form_list'))
          self.assertFalse(FeedbackForm.objects.filter(id=form.id).exists())

      def test_delete_requires_post(self):
          form = FeedbackForm.objects.create(title='Get-safe', audience='student', created_by=self.staff)
          self.client.login(username='fb_staff', password='pass')
          r = self.client.get(reverse('feedback_form_delete', args=[form.id]))
          self.assertEqual(r.status_code, 405)
          self.assertTrue(FeedbackForm.objects.filter(id=form.id).exists())

      def test_create_rejects_choice_question_without_options(self):
          self.client.login(username='fb_staff', password='pass')
          r = self.client.post(reverse('feedback_form_create'), {
              'title': 'Bad', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
              'questions_json': json.dumps([{'text': 'Q', 'type': 'choice', 'options': [], 'required': True}]),
          })
          self.assertEqual(r.status_code, 200)  # re-rendered with error, not redirected
          self.assertFalse(FeedbackForm.objects.filter(title='Bad').exists())
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.FeedbackManagementViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` (the management URL names don't exist yet).

- [ ] **Step 3: Add the management views**

  In `Trust-AI-Platform/feedback/views.py`, extend the import block — replace:
  ```python
  from django.http import HttpResponseForbidden, JsonResponse
  from django.shortcuts import get_object_or_404
  ```
  with:
  ```python
  from django.contrib import messages
  from django.db.models import Count
  from django.http import HttpResponseForbidden, JsonResponse
  from django.shortcuts import get_object_or_404, redirect, render
  ```

  Then append at the end of the file:
  ```python

  def _parse_questions_json(raw):
      """Returns (questions_list, error_message). Valid shape: list of
      {text, type in {choice,text}, options list, required bool}."""
      try:
          data = json.loads(raw or '[]')
      except json.JSONDecodeError:
          return None, 'Invalid question data.'
      if not isinstance(data, list) or not data:
          return None, 'At least one question is required.'
      cleaned = []
      for item in data:
          text = (item.get('text') or '').strip()
          qtype = item.get('type')
          options = item.get('options') or []
          if not text:
              return None, 'Every question needs text.'
          if qtype not in ('choice', 'text'):
              return None, 'Invalid question type.'
          options = [str(o).strip() for o in options if str(o).strip()]
          if qtype == 'choice' and len(options) < 2:
              return None, f'Question "{text}" needs at least two options.'
          cleaned.append({
              'text': text, 'type': qtype, 'options': options if qtype == 'choice' else [],
              'required': bool(item.get('required')),
          })
      return cleaned, None


  def _save_form_from_post(request, form=None):
      """Shared create/edit POST handling. Returns (form, error_message)."""
      title = (request.POST.get('title') or '').strip()
      if not title:
          return None, 'Title is required.'
      audience = request.POST.get('audience')
      if audience not in ('teacher', 'student'):
          return None, 'Invalid audience.'
      questions, error = _parse_questions_json(request.POST.get('questions_json'))
      if error:
          return None, error

      assign_to_all = request.POST.get('assign_to_all') == 'on'
      checked_ids = set()
      for raw_id in request.POST.getlist('scenarios'):
          if raw_id.isdigit():
              checked_ids.add(int(raw_id))

      with transaction.atomic():
          if form is None:
              form = FeedbackForm(created_by=request.user)
          form.title = title
          form.description = (request.POST.get('description') or '').strip()
          form.audience = audience
          form.is_active = request.POST.get('is_active') == 'on'
          form.assign_to_all = assign_to_all
          form.save()

          all_ids = set(Scenario.objects.values_list('id', flat=True))
          if assign_to_all:
              form.excluded_scenarios.set(all_ids - checked_ids)
              form.included_scenarios.clear()
          else:
              form.included_scenarios.set(checked_ids & all_ids)
              form.excluded_scenarios.clear()

          form.questions.all().delete()
          for index, q in enumerate(questions):
              FeedbackQuestion.objects.create(
                  form=form, text=q['text'], question_type=q['type'],
                  options=q['options'], is_required=q['required'], order=index,
              )
      return form, None


  @staff_required
  def feedback_form_list(request):
      forms = FeedbackForm.objects.annotate(
          question_count=Count('questions', distinct=True),
          response_count=Count('responses', distinct=True),
      )
      return render(request, 'feedback/form_list.html', {'forms': forms})


  @staff_required
  def feedback_form_create(request):
      if request.method == 'POST':
          form, error = _save_form_from_post(request)
          if error is None:
              messages.success(request, 'Feedback form created.')
              return redirect('feedback_form_list')
          return render(request, 'feedback/form_edit.html', {
              'form_obj': None, 'error': error, 'scenarios': Scenario.objects.order_by('name'),
              'questions_json': request.POST.get('questions_json') or '[]',
              'posted': request.POST,
          })
      return render(request, 'feedback/form_edit.html', {
          'form_obj': None, 'scenarios': Scenario.objects.order_by('name'), 'questions_json': '[]',
      })


  @staff_required
  def feedback_form_edit(request, form_id):
      form = get_object_or_404(FeedbackForm, id=form_id)
      if request.method == 'POST':
          _, error = _save_form_from_post(request, form=form)
          if error is None:
              messages.success(request, 'Feedback form updated.')
              return redirect('feedback_form_list')
          return render(request, 'feedback/form_edit.html', {
              'form_obj': form, 'error': error, 'scenarios': Scenario.objects.order_by('name'),
              'questions_json': request.POST.get('questions_json') or '[]',
              'posted': request.POST,
          })
      questions_json = json.dumps([
          {'text': q.text, 'type': q.question_type, 'options': q.options, 'required': q.is_required}
          for q in form.questions.all()
      ])
      return render(request, 'feedback/form_edit.html', {
          'form_obj': form, 'scenarios': Scenario.objects.order_by('name'), 'questions_json': questions_json,
      })


  @require_POST
  @staff_required
  def feedback_form_delete(request, form_id):
      form = get_object_or_404(FeedbackForm, id=form_id)
      form.delete()
      messages.success(request, 'Feedback form deleted.')
      return redirect('feedback_form_list')
  ```

  Note: `require_POST`, `Scenario`, and `transaction` are all already imported at the top of the file from Task 2 — do not re-import any of them.

  In `Trust-AI-Platform/feedback/urls.py`, replace:
  ```python
  urlpatterns = [
      path('submit/<int:form_id>/<int:scenario_id>/', views.submit_feedback, name='feedback_submit'),
  ]
  ```
  with:
  ```python
  urlpatterns = [
      path('submit/<int:form_id>/<int:scenario_id>/', views.submit_feedback, name='feedback_submit'),
      path('manage/', views.feedback_form_list, name='feedback_form_list'),
      path('manage/create/', views.feedback_form_create, name='feedback_form_create'),
      path('manage/<int:form_id>/edit/', views.feedback_form_edit, name='feedback_form_edit'),
      path('manage/<int:form_id>/delete/', views.feedback_form_delete, name='feedback_form_delete'),
  ]
  ```

- [ ] **Step 4: Create the forms-list template**

  Create `Trust-AI-Platform/feedback/templates/feedback/form_list.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — Feedback Forms</title>{% endblock %}
  {% block atcontent %}

  <style>
    .fb-hero {
      background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
      border-radius: 14px; padding: 26px 30px 20px; color: #fff;
      margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
    }
    .fb-hero-icon {
      background: rgba(255,255,255,0.18); border-radius: 10px;
      width: 50px; height: 50px; display:flex; align-items:center;
      justify-content:center; font-size:22px; flex-shrink:0;
    }
    .fb-hero .breadcrumb { background:none; margin:10px 0 0; padding:0; font-size:12px; }
    .fb-hero .breadcrumb-item+.breadcrumb-item::before { color:rgba(255,255,255,0.5); }
    .fb-hero .breadcrumb-item a { color:rgba(255,255,255,0.72); text-decoration:none; }
    .fb-hero .breadcrumb-item.active { color:rgba(255,255,255,0.92); }
    .hero-btn-solid { background:#fff; color:#1a56db; border:none; font-weight:600; font-size:13px; border-radius:8px; padding:7px 16px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; white-space:nowrap; cursor:pointer; }
    .hero-btn-solid:hover { background:#eef3ff; color:#1a56db; }
    @media (max-width: 575.98px) {
      .fb-hero { padding: 14px 16px 12px; }
      .fb-hero > .d-flex { flex-wrap: wrap; }
      .fb-hero-icon { display: none; }
      .fb-hero h2 { font-size: 15px !important; }
    }
    .audience-badge { font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px; text-transform:uppercase; }
    .audience-badge.teacher { background:#ede9fe; color:#7c3aed; }
    .audience-badge.student { background:#cffafe; color:#0891b2; }
    .inactive-badge { font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px; background:#f3f4f6; color:#6b7280; }
    .action-btn { background:none; border:none; padding:4px 6px; border-radius:4px; cursor:pointer; font-size:15px; line-height:1; color:#333; text-decoration:none; }
    .action-btn:hover { background:#f0f4ff; }
    .action-btn.remove { color:#888; }
    .action-btn.remove:hover { color:#c62828; background:#ffebee; }
    .action-form { display:inline; }
  </style>

  <main id="main" class="main">
    <div class="fb-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="fb-hero-icon"><i class="bi bi-clipboard-check-fill"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Administration</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">Feedback Forms</h2>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'admin_dashboard' %}">User Management</a></li>
            <li class="breadcrumb-item active">Feedback Forms</li>
          </ol></nav>
        </div>
        <div class="flex-shrink-0 d-flex gap-2 align-items-start" style="padding-top:4px;">
          <a href="{% url 'feedback_form_create' %}" class="hero-btn-solid">
            <i class="bi bi-plus-lg"></i> New Form
          </a>
        </div>
      </div>
    </div>

    <section class="section">
      <div class="card" style="border-radius:12px; overflow:hidden;">
        <div class="table-responsive">
          <table class="table table-hover mb-0">
            <thead>
              <tr>
                <th>Title</th>
                <th>Audience</th>
                <th>Status</th>
                <th class="text-center">Questions</th>
                <th class="text-center">Responses</th>
                <th>Created</th>
                <th class="text-end">Actions</th>
              </tr>
            </thead>
            <tbody>
              {% for form in forms %}
              <tr>
                <td style="font-weight:600; color:#012970;">{{ form.title }}</td>
                <td><span class="audience-badge {{ form.audience }}">{{ form.get_audience_display }}</span></td>
                <td>{% if form.is_active %}<span class="badge bg-success">Active</span>{% else %}<span class="inactive-badge">Inactive</span>{% endif %}</td>
                <td class="text-center">{{ form.question_count }}</td>
                <td class="text-center">{{ form.response_count }}</td>
                <td class="text-muted small">{{ form.created_on|date:"d M Y" }}</td>
                <td class="text-end" style="white-space:nowrap;">
                  <a href="{% url 'feedback_form_responses' form.id %}" class="action-btn" title="View responses"><i class="bi bi-inbox"></i></a>
                  <a href="{% url 'feedback_form_edit' form.id %}" class="action-btn" title="Edit"><i class="bi bi-pencil"></i></a>
                  <form method="post" action="{% url 'feedback_form_delete' form.id %}" class="action-form" onsubmit="return confirm('Delete this form and ALL its responses?');">
                    {% csrf_token %}
                    <button type="submit" class="action-btn remove" title="Delete"><i class="bi bi-trash"></i></button>
                  </form>
                </td>
              </tr>
              {% empty %}
              <tr><td colspan="7" class="text-center text-muted py-5">
                <i class="bi bi-clipboard-check" style="font-size:2rem;color:#d1d9e0;display:block;margin-bottom:8px;"></i>
                No feedback forms yet.
              </td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  {% endblock %}
  ```

  Note: this template references `feedback_form_responses`, which Task 4 creates. To keep this task independently green, Step 3's URL additions must ALSO include a placeholder for it now — add this line to `urls.py`'s urlpatterns in this task (Task 4 replaces the placeholder view with the real one):
  ```python
      path('manage/<int:form_id>/responses/', views.feedback_form_responses, name='feedback_form_responses'),
  ```
  and append this placeholder view to `views.py` in this task:
  ```python
  @staff_required
  def feedback_form_responses(request, form_id):
      form = get_object_or_404(FeedbackForm, id=form_id)
      return render(request, 'feedback/form_responses.html', {'form_obj': form, 'responses': form.responses.select_related('user', 'scenario')})
  ```
  and create a minimal `Trust-AI-Platform/feedback/templates/feedback/form_responses.html` placeholder (Task 4 replaces it entirely):
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — Responses — {{ form_obj.title }}</title>{% endblock %}
  {% block atcontent %}
  <main id="main" class="main">
    <section class="section">
      <div class="card"><div class="card-body">
        <h5>{{ form_obj.title }} — {{ responses|length }} response(s)</h5>
        <p class="text-muted">Full responses view arrives in the next task.</p>
        <a href="{% url 'feedback_form_list' %}">Back</a>
      </div></div>
    </section>
  </main>
  {% endblock %}
  ```

- [ ] **Step 5: Create the create/edit template**

  Create `Trust-AI-Platform/feedback/templates/feedback/form_edit.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — {% if form_obj %}Edit{% else %}New{% endif %} Feedback Form</title>{% endblock %}
  {% block atcontent %}

  <style>
    .fb-hero {
      background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
      border-radius: 14px; padding: 26px 30px 20px; color: #fff;
      margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
    }
    .fb-hero-icon { background: rgba(255,255,255,0.18); border-radius: 10px; width: 50px; height: 50px; display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0; }
    .fb-hero .breadcrumb { background:none; margin:10px 0 0; padding:0; font-size:12px; }
    .fb-hero .breadcrumb-item+.breadcrumb-item::before { color:rgba(255,255,255,0.5); }
    .fb-hero .breadcrumb-item a { color:rgba(255,255,255,0.72); text-decoration:none; }
    .fb-hero .breadcrumb-item.active { color:rgba(255,255,255,0.92); }
    .hero-btn-ghost { background:rgba(255,255,255,0.15); color:#fff; border:1.5px solid rgba(255,255,255,0.4); font-weight:600; font-size:13px; border-radius:8px; padding:6px 14px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; white-space:nowrap; }
    .hero-btn-ghost:hover { background:rgba(255,255,255,0.25); color:#fff; }
    @media (max-width: 575.98px) {
      .fb-hero { padding: 14px 16px 12px; }
      .fb-hero > .d-flex { flex-wrap: wrap; }
      .fb-hero-icon { display: none; }
      .fb-hero h2 { font-size: 15px !important; }
    }
    .form-card { max-width: 860px; margin: 0 auto; }
    .field-label { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 5px; }
    .question-row { border:1px solid #e8edf5; border-radius:10px; padding:14px; margin-bottom:10px; background:#fafbff; }
    .question-row .remove-q { color:#c62828; background:none; border:none; cursor:pointer; }
    .scenario-box { max-height: 260px; overflow-y: auto; border:1px solid #e8edf5; border-radius:10px; padding:12px 14px; }
    .warn-banner { background:#fef3c7; color:#92400e; border-radius:8px; padding:10px 14px; font-size:13px; margin-bottom:14px; }
  </style>

  <main id="main" class="main">
    <div class="fb-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="fb-hero-icon"><i class="bi bi-clipboard-plus-fill"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Administration</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">{% if form_obj %}Edit Feedback Form{% else %}New Feedback Form{% endif %}</h2>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'feedback_form_list' %}">Feedback Forms</a></li>
            <li class="breadcrumb-item active">{% if form_obj %}Edit{% else %}New{% endif %}</li>
          </ol></nav>
        </div>
        <div class="flex-shrink-0 d-flex gap-2 align-items-start" style="padding-top:4px;">
          <a href="{% url 'feedback_form_list' %}" class="hero-btn-ghost"><i class="bi bi-arrow-left"></i> Back</a>
        </div>
      </div>
    </div>

    <section class="section">
      <div class="card form-card">
        <div class="card-body p-4">
          {% if form_obj and form_obj.responses.exists %}
          <div class="warn-banner"><i class="bi bi-exclamation-triangle-fill me-1"></i>
            This form already has responses. Changing or removing questions affects how historical answers line up in exports.
          </div>
          {% endif %}
          {% if error %}
          <div class="alert alert-danger py-2" style="font-size:13.5px;">{{ error }}</div>
          {% endif %}

          <form method="post" id="feedbackFormEditor">
            {% csrf_token %}
            <div class="mb-3">
              <label class="field-label">Title</label>
              <input type="text" name="title" class="form-control" required
                     value="{% if posted %}{{ posted.title }}{% elif form_obj %}{{ form_obj.title }}{% endif %}">
            </div>
            <div class="mb-3">
              <label class="field-label">Description (shown above the questions)</label>
              <textarea name="description" class="form-control" rows="2">{% if posted %}{{ posted.description }}{% elif form_obj %}{{ form_obj.description }}{% endif %}</textarea>
            </div>
            <div class="row g-3 mb-3">
              <div class="col-12 col-md-6">
                <label class="field-label">Audience</label>
                <select name="audience" class="form-select">
                  <option value="teacher" {% if form_obj and form_obj.audience == 'teacher' %}selected{% endif %}>Teachers (after creating a personalized scenario)</option>
                  <option value="student" {% if form_obj and form_obj.audience == 'student' %}selected{% endif %}>Students (after finishing a scenario)</option>
                </select>
              </div>
              <div class="col-12 col-md-6 d-flex align-items-end">
                <div class="form-check">
                  <input class="form-check-input" type="checkbox" name="is_active" id="isActive"
                         {% if not form_obj or form_obj.is_active %}checked{% endif %}>
                  <label class="form-check-label" for="isActive" style="font-size:13.5px;">Active (shown to users)</label>
                </div>
              </div>
            </div>

            <hr>
            <div class="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-2">
              <label class="field-label mb-0">Questions</label>
              <button type="button" class="btn btn-sm btn-outline-primary" id="addQuestionBtn"><i class="bi bi-plus-lg"></i> Add question</button>
            </div>
            <div id="questionList"></div>
            <input type="hidden" name="questions_json" id="questionsJson">

            <hr>
            <label class="field-label">Scenario assignment</label>
            <div class="form-check mb-2">
              <input class="form-check-input" type="checkbox" name="assign_to_all" id="assignAll"
                     {% if not form_obj or form_obj.assign_to_all %}checked{% endif %}>
              <label class="form-check-label" for="assignAll" style="font-size:13.5px;">
                Assign to <strong>all scenarios</strong> (includes scenarios created in the future)
              </label>
            </div>
            <div class="text-muted small mb-2" id="scenarioHint"></div>
            <div class="scenario-box">
              {% for scenario in scenarios %}
              <div class="form-check">
                <input class="form-check-input scenario-cb" type="checkbox" name="scenarios" value="{{ scenario.id }}" id="sc{{ scenario.id }}"
                  {% if form_obj %}
                    {% if form_obj.assign_to_all %}
                      {% if scenario not in form_obj.excluded_scenarios.all %}checked{% endif %}
                    {% else %}
                      {% if scenario in form_obj.included_scenarios.all %}checked{% endif %}
                    {% endif %}
                  {% endif %}>
                <label class="form-check-label" for="sc{{ scenario.id }}" style="font-size:13.5px;">{{ scenario.name }}</label>
              </div>
              {% empty %}
              <div class="text-muted small">No scenarios exist yet.</div>
              {% endfor %}
            </div>

            <div class="text-end mt-4">
              <button type="submit" class="btn btn-primary">
                <i class="bi bi-check-lg me-1"></i>{% if form_obj %}Save Changes{% else %}Create Form{% endif %}
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  </main>

  {{ questions_json|json_script:"initialQuestions" }}
  <script>
  document.addEventListener('DOMContentLoaded', function () {
    const list = document.getElementById('questionList');
    const addBtn = document.getElementById('addQuestionBtn');
    const hiddenInput = document.getElementById('questionsJson');
    const assignAll = document.getElementById('assignAll');
    const hint = document.getElementById('scenarioHint');
    const editorForm = document.getElementById('feedbackFormEditor');

    function updateHint() {
      hint.textContent = assignAll.checked
        ? 'All scenarios are covered. Uncheck a scenario below to EXCLUDE it (new scenarios are always covered).'
        : 'Only the CHECKED scenarios below get this form.';
    }
    assignAll.addEventListener('change', function () {
      updateHint();
      if (assignAll.checked) {
        document.querySelectorAll('.scenario-cb').forEach(function (cb) { cb.checked = true; });
      }
    });
    updateHint();

    function questionRow(q) {
      const row = document.createElement('div');
      row.className = 'question-row';
      row.innerHTML =
        '<div class="d-flex gap-2 align-items-start flex-wrap">' +
        '  <input type="text" class="form-control flex-grow-1 q-text" placeholder="Question text" style="min-width:200px;">' +
        '  <select class="form-select q-type" style="width:auto;">' +
        '    <option value="choice">Multiple choice</option>' +
        '    <option value="text">Free text</option>' +
        '  </select>' +
        '  <div class="form-check align-self-center">' +
        '    <input class="form-check-input q-required" type="checkbox" checked>' +
        '    <label class="form-check-label" style="font-size:12.5px;">Required</label>' +
        '  </div>' +
        '  <button type="button" class="remove-q" title="Remove question"><i class="bi bi-x-lg"></i></button>' +
        '</div>' +
        '<div class="mt-2 q-options-wrap">' +
        '  <label style="font-size:12px;color:#666;">Options (one per line)</label>' +
        '  <textarea class="form-control q-options" rows="3" placeholder="Very useful\nSomewhat useful\nNot useful"></textarea>' +
        '</div>';
      row.querySelector('.q-text').value = q.text || '';
      row.querySelector('.q-type').value = q.type || 'choice';
      row.querySelector('.q-required').checked = q.required !== false;
      row.querySelector('.q-options').value = (q.options || []).join('\n');
      function syncOptionsVisibility() {
        row.querySelector('.q-options-wrap').style.display =
          row.querySelector('.q-type').value === 'choice' ? '' : 'none';
      }
      row.querySelector('.q-type').addEventListener('change', syncOptionsVisibility);
      row.querySelector('.remove-q').addEventListener('click', function () { row.remove(); });
      syncOptionsVisibility();
      return row;
    }

    addBtn.addEventListener('click', function () {
      list.appendChild(questionRow({}));
    });

    const initial = JSON.parse(JSON.parse(document.getElementById('initialQuestions').textContent));
    if (initial.length === 0) {
      list.appendChild(questionRow({}));
    } else {
      initial.forEach(function (q) { list.appendChild(questionRow(q)); });
    }

    editorForm.addEventListener('submit', function () {
      const questions = [];
      list.querySelectorAll('.question-row').forEach(function (row) {
        questions.push({
          text: row.querySelector('.q-text').value,
          type: row.querySelector('.q-type').value,
          options: row.querySelector('.q-options').value.split('\n').map(function (s) { return s.trim(); }).filter(Boolean),
          required: row.querySelector('.q-required').checked,
        });
      });
      hiddenInput.value = JSON.stringify(questions);
    });
  });
  </script>
  {% endblock %}
  ```

  Note on `JSON.parse(JSON.parse(...))`: the view passes `questions_json` as a JSON *string*; `json_script` then JSON-encodes that string again for safe embedding. The double parse unwraps both layers. This is deliberate — do not "fix" it to a single parse.

- [ ] **Step 6: Add the dashboard link**

  In `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`, replace:
  ```html
      <div class="admin-hero">
        <div class="d-flex align-items-start gap-3">
          <div class="admin-hero-icon"><i class="bi bi-shield-lock-fill"></i></div>
          <div class="flex-grow-1">
            <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Administration</div>
            <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">User Management</h2>
            <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ stats.total }} user{{ stats.total|pluralize }} · {{ stats.active }} active</div>
            <nav><ol class="breadcrumb">
              <li class="breadcrumb-item active">User Management</li>
            </ol></nav>
          </div>
        </div>
      </div>
  ```
  with:
  ```html
      <div class="admin-hero">
        <div class="d-flex align-items-start gap-3">
          <div class="admin-hero-icon"><i class="bi bi-shield-lock-fill"></i></div>
          <div class="flex-grow-1">
            <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Administration</div>
            <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">User Management</h2>
            <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ stats.total }} user{{ stats.total|pluralize }} · {{ stats.active }} active</div>
            <nav><ol class="breadcrumb">
              <li class="breadcrumb-item active">User Management</li>
            </ol></nav>
          </div>
          <div class="flex-shrink-0 d-flex gap-2 align-items-start" style="padding-top:4px;">
            <a href="{% url 'feedback_form_list' %}" style="background:rgba(255,255,255,0.15);color:#fff;border:1.5px solid rgba(255,255,255,0.4);font-weight:600;font-size:13px;border-radius:8px;padding:6px 14px;display:inline-flex;align-items:center;gap:6px;text-decoration:none;white-space:nowrap;">
              <i class="bi bi-clipboard-check"></i> Feedback Forms
            </a>
          </div>
        </div>
      </div>
  ```

  Note: the exact indentation of this block in the live file may differ from what's shown (verify against the file at `admin_dashboard.html:77-89` before replacing; match the file's own indentation).

- [ ] **Step 7: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (27 tests: 18 prior + 9 new).

- [ ] **Step 8: Commit**

  ```bash
  git add Trust-AI-Platform/feedback/views.py Trust-AI-Platform/feedback/urls.py Trust-AI-Platform/feedback/tests.py Trust-AI-Platform/feedback/templates/feedback/form_list.html Trust-AI-Platform/feedback/templates/feedback/form_edit.html Trust-AI-Platform/feedback/templates/feedback/form_responses.html Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html
  git commit -m "Add feedback form management UI: list, create/edit, delete"
  ```

---

### Task 4: Management UI — responses page, response deletion, XLSX/CSV exports

**Files:**
- Modify: `Trust-AI-Platform/feedback/views.py`
- Modify: `Trust-AI-Platform/feedback/urls.py`
- Modify: `Trust-AI-Platform/feedback/tests.py`
- Modify (full replace): `Trust-AI-Platform/feedback/templates/feedback/form_responses.html`

**Interfaces:**
- Consumes: models, `staff_required`, the Task 3 placeholder responses view (replaced here)
- Produces: URL names `feedback_response_delete` (`feedback/manage/response/<int:response_id>/delete/`), `feedback_form_export_xlsx` (`feedback/manage/<int:form_id>/export/xlsx/`), `feedback_form_export_csv` (`feedback/manage/<int:form_id>/export/csv/`).

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/feedback/tests.py`:
  ```python
  class FeedbackResponsesAndExportTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.staff = User.objects.create_user('fb_exp_staff', password='pass', is_staff=True)
          self.responder = User.objects.create_user('fb_exp_user', password='pass', first_name='Resp', last_name='Onder')
          self.scenario = Scenario.objects.create(name='FB Export Scenario', created_by=self.staff, updated_by=self.staff)
          self.form = FeedbackForm.objects.create(title='Export form', audience='student', created_by=self.staff)
          self.q1 = FeedbackQuestion.objects.create(form=self.form, text='Useful?', question_type='choice', options=['Yes', 'No'], order=1)
          self.q2 = FeedbackQuestion.objects.create(form=self.form, text='Comments', question_type='text', order=2)
          self.response = FeedbackResponse.objects.create(form=self.form, user=self.responder, scenario=self.scenario)
          FeedbackAnswer.objects.create(response=self.response, question=self.q1, answer_text='Yes')
          FeedbackAnswer.objects.create(response=self.response, question=self.q2, answer_text='Nice tool')
          self.client.login(username='fb_exp_staff', password='pass')

      def test_responses_page_lists_answers(self):
          r = self.client.get(reverse('feedback_form_responses', args=[self.form.id]))
          self.assertContains(r, 'fb_exp_user')
          self.assertContains(r, 'Nice tool')

      def test_delete_response(self):
          r = self.client.post(reverse('feedback_response_delete', args=[self.response.id]))
          self.assertRedirects(r, reverse('feedback_form_responses', args=[self.form.id]))
          self.assertFalse(FeedbackResponse.objects.filter(id=self.response.id).exists())
          self.assertFalse(FeedbackAnswer.objects.exists())

      def test_delete_response_requires_post(self):
          r = self.client.get(reverse('feedback_response_delete', args=[self.response.id]))
          self.assertEqual(r.status_code, 405)

      def test_csv_export_uses_comma_and_contains_answers(self):
          r = self.client.get(reverse('feedback_form_export_csv', args=[self.form.id]))
          self.assertEqual(r['Content-Type'], 'text/csv')
          content = r.content.decode('utf-8')
          header = content.splitlines()[0]
          self.assertIn('Username,', header)
          self.assertIn('Useful?', header)
          self.assertIn('Nice tool', content)

      def test_xlsx_export_contains_answers(self):
          import io
          import openpyxl
          r = self.client.get(reverse('feedback_form_export_xlsx', args=[self.form.id]))
          self.assertEqual(r['Content-Type'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
          wb = openpyxl.load_workbook(io.BytesIO(r.content))
          ws = wb.active
          rows = list(ws.iter_rows(values_only=True))
          self.assertEqual(rows[0][0], 'Username')
          self.assertIn('Useful?', rows[0])
          self.assertIn('Yes', rows[1])
          self.assertIn('Nice tool', rows[1])

      def test_exports_blocked_for_non_staff(self):
          self.client.logout()
          self.client.login(username='fb_exp_user', password='pass')
          r = self.client.get(reverse('feedback_form_export_csv', args=[self.form.id]))
          self.assertEqual(r.status_code, 403)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.FeedbackResponsesAndExportTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` for `feedback_response_delete`/`feedback_form_export_csv`/`feedback_form_export_xlsx` (the responses-page test may pass already via the placeholder — that's fine, the failures that matter are the missing URL names).

- [ ] **Step 3: Replace the placeholder responses view and add delete/export views**

  In `Trust-AI-Platform/feedback/views.py`, replace the Task 3 placeholder:
  ```python
  @staff_required
  def feedback_form_responses(request, form_id):
      form = get_object_or_404(FeedbackForm, id=form_id)
      return render(request, 'feedback/form_responses.html', {'form_obj': form, 'responses': form.responses.select_related('user', 'scenario')})
  ```
  with:
  ```python
  @staff_required
  def feedback_form_responses(request, form_id):
      form = get_object_or_404(FeedbackForm, id=form_id)
      questions = list(form.questions.all())
      responses = form.responses.select_related('user', 'scenario').prefetch_related('answers__question')
      response_rows = []
      for response in responses:
          answer_map = {a.question_id: a.answer_text for a in response.answers.all()}
          response_rows.append({
              'response': response,
              'answers': [(q, answer_map.get(q.id, '')) for q in questions],
          })
      return render(request, 'feedback/form_responses.html', {
          'form_obj': form,
          'questions': questions,
          'response_rows': response_rows,
      })


  @require_POST
  @staff_required
  def feedback_response_delete(request, response_id):
      response = get_object_or_404(FeedbackResponse, id=response_id)
      form_id = response.form_id
      response.delete()
      messages.success(request, 'Response deleted.')
      return redirect('feedback_form_responses', form_id=form_id)


  def _export_rows(form):
      """Header row + one row per response, question columns in question order."""
      questions = list(form.questions.all())
      header = ['Username', 'Scenario', 'Submitted'] + [q.text for q in questions]
      rows = [header]
      responses = form.responses.select_related('user', 'scenario').prefetch_related('answers')
      for response in responses:
          answer_map = {a.question_id: a.answer_text for a in response.answers.all()}
          rows.append(
              [response.user.username, response.scenario.name, response.submitted_at.strftime('%Y-%m-%d %H:%M')]
              + [answer_map.get(q.id, '') for q in questions]
          )
      return rows


  @staff_required
  def feedback_form_export_csv(request, form_id):
      import csv
      form = get_object_or_404(FeedbackForm, id=form_id)
      http_response = HttpResponse(content_type='text/csv')
      http_response['Content-Disposition'] = f'attachment; filename="feedback_form_{form.id}_responses.csv"'
      writer = csv.writer(http_response)  # default delimiter: comma (deliberate; spec requirement)
      for row in _export_rows(form):
          writer.writerow(row)
      return http_response


  @staff_required
  def feedback_form_export_xlsx(request, form_id):
      import io
      import openpyxl
      form = get_object_or_404(FeedbackForm, id=form_id)
      wb = openpyxl.Workbook()
      ws = wb.active
      ws.title = 'Responses'
      for row in _export_rows(form):
          ws.append(row)
      buf = io.BytesIO()
      wb.save(buf)
      buf.seek(0)
      http_response = HttpResponse(
          buf.read(),
          content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      )
      http_response['Content-Disposition'] = f'attachment; filename="feedback_form_{form.id}_responses.xlsx"'
      return http_response
  ```

  Also extend the top import block — replace:
  ```python
  from django.http import HttpResponseForbidden, JsonResponse
  ```
  with:
  ```python
  from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
  ```

  In `Trust-AI-Platform/feedback/urls.py`, replace:
  ```python
      path('manage/<int:form_id>/responses/', views.feedback_form_responses, name='feedback_form_responses'),
  ]
  ```
  with:
  ```python
      path('manage/<int:form_id>/responses/', views.feedback_form_responses, name='feedback_form_responses'),
      path('manage/response/<int:response_id>/delete/', views.feedback_response_delete, name='feedback_response_delete'),
      path('manage/<int:form_id>/export/csv/', views.feedback_form_export_csv, name='feedback_form_export_csv'),
      path('manage/<int:form_id>/export/xlsx/', views.feedback_form_export_xlsx, name='feedback_form_export_xlsx'),
  ]
  ```

- [ ] **Step 4: Replace the responses template**

  Replace the entire contents of `Trust-AI-Platform/feedback/templates/feedback/form_responses.html` with:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — Responses — {{ form_obj.title }}</title>{% endblock %}
  {% block atcontent %}

  <style>
    .fb-hero {
      background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
      border-radius: 14px; padding: 26px 30px 20px; color: #fff;
      margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
    }
    .fb-hero-icon { background: rgba(255,255,255,0.18); border-radius: 10px; width: 50px; height: 50px; display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0; }
    .fb-hero .breadcrumb { background:none; margin:10px 0 0; padding:0; font-size:12px; }
    .fb-hero .breadcrumb-item+.breadcrumb-item::before { color:rgba(255,255,255,0.5); }
    .fb-hero .breadcrumb-item a { color:rgba(255,255,255,0.72); text-decoration:none; }
    .fb-hero .breadcrumb-item.active { color:rgba(255,255,255,0.92); }
    .hero-btn-ghost { background:rgba(255,255,255,0.15); color:#fff; border:1.5px solid rgba(255,255,255,0.4); font-weight:600; font-size:13px; border-radius:8px; padding:6px 14px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; white-space:nowrap; }
    .hero-btn-ghost:hover { background:rgba(255,255,255,0.25); color:#fff; }
    .hero-btn-solid { background:#fff; color:#1a56db; border:none; font-weight:600; font-size:13px; border-radius:8px; padding:7px 16px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; white-space:nowrap; cursor:pointer; }
    .hero-btn-solid:hover { background:#eef3ff; color:#1a56db; }
    @media (max-width: 575.98px) {
      .fb-hero { padding: 14px 16px 12px; }
      .fb-hero > .d-flex { flex-wrap: wrap; }
      .fb-hero-icon { display: none; }
      .fb-hero .d-flex.flex-shrink-0 { flex-shrink: 1 !important; width: 100%; justify-content: flex-start !important; margin-top: 10px; }
      .fb-hero h2 { font-size: 15px !important; }
    }
    .response-card { border:1px solid #e8edf5; border-radius:12px; background:#fff; margin-bottom:10px; overflow:hidden; }
    .response-header { padding:12px 16px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; cursor:pointer; }
    .response-header:hover { background:#f8faff; }
    .response-meta { font-size:12px; color:#888; }
    .answer-row { padding:8px 16px; border-top:1px solid #f0f4ff; font-size:13.5px; }
    .answer-q { font-weight:600; color:#012970; }
    .action-btn { background:none; border:none; padding:4px 6px; border-radius:4px; cursor:pointer; font-size:15px; line-height:1; color:#888; }
    .action-btn:hover { color:#c62828; background:#ffebee; }
    .action-form { display:inline; margin-left:auto; }
  </style>

  <main id="main" class="main">
    <div class="fb-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="fb-hero-icon"><i class="bi bi-inbox-fill"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Feedback Forms</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">{{ form_obj.title }}</h2>
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ response_rows|length }} response{{ response_rows|length|pluralize }}</div>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'feedback_form_list' %}">Feedback Forms</a></li>
            <li class="breadcrumb-item active">Responses</li>
          </ol></nav>
        </div>
        <div class="flex-shrink-0 d-flex gap-2 align-items-start flex-wrap" style="padding-top:4px;">
          <a href="{% url 'feedback_form_export_xlsx' form_obj.id %}" class="hero-btn-solid"><i class="bi bi-file-earmark-excel"></i> XLSX</a>
          <a href="{% url 'feedback_form_export_csv' form_obj.id %}" class="hero-btn-solid"><i class="bi bi-filetype-csv"></i> CSV</a>
          <a href="{% url 'feedback_form_list' %}" class="hero-btn-ghost"><i class="bi bi-arrow-left"></i> Back</a>
        </div>
      </div>
    </div>

    <section class="section">
      {% for row in response_rows %}
      <div class="response-card">
        <div class="response-header" data-bs-toggle="collapse" data-bs-target="#resp{{ row.response.id }}">
          <i class="bi bi-chevron-down" style="color:#888;"></i>
          <div>
            <div style="font-weight:600; font-size:14px; color:#012970;">{{ row.response.user.get_full_name|default:row.response.user.username }}</div>
            <div class="response-meta">{{ row.response.scenario.name }} · {{ row.response.submitted_at|date:"d M Y, H:i" }}</div>
          </div>
          <form method="post" action="{% url 'feedback_response_delete' row.response.id %}" class="action-form" onsubmit="return confirm('Delete this response?');" onclick="event.stopPropagation();">
            {% csrf_token %}
            <button type="submit" class="action-btn" title="Delete response"><i class="bi bi-trash"></i></button>
          </form>
        </div>
        <div class="collapse" id="resp{{ row.response.id }}">
          {% for question, answer in row.answers %}
          <div class="answer-row">
            <span class="answer-q">{{ question.text }}</span><br>
            {% if answer %}{{ answer }}{% else %}<span class="text-muted">—</span>{% endif %}
          </div>
          {% endfor %}
        </div>
      </div>
      {% empty %}
      <div class="text-center text-muted py-5">
        <i class="bi bi-inbox" style="font-size:2.5rem;color:#d1d9e0;"></i>
        <p class="mt-2 mb-0">No responses yet.</p>
      </div>
      {% endfor %}
    </section>
  </main>
  {% endblock %}
  ```

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (33 tests: 27 prior + 6 new).

- [ ] **Step 6: Commit**

  ```bash
  git add Trust-AI-Platform/feedback/views.py Trust-AI-Platform/feedback/urls.py Trust-AI-Platform/feedback/tests.py Trust-AI-Platform/feedback/templates/feedback/form_responses.html
  git commit -m "Add feedback responses page, response deletion, and XLSX/CSV exports"
  ```

---

### Task 5: Teacher trigger — session flag + modal on the proposals page

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/views.py` (`create_personal_scenario`, `proposal_list_view`)
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_list.html`
- Modify: `Trust-AI-Platform/feedback/tests.py`

**Interfaces:**
- Consumes: `get_applicable_form`, `user_has_responded`, `serialize_form` (Task 2), `feedback_submit` URL (Task 2)
- Produces: no new interfaces.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/feedback/tests.py`:
  ```python
  class TeacherFeedbackTriggerTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.teacher = User.objects.create_user('fb_trig_teacher', password='pass')
          self.teacher.groups.add(teachers)
          self.scenario = Scenario.objects.create(name='FB Trigger Scenario', created_by=self.teacher, updated_by=self.teacher)
          self.form = FeedbackForm.objects.create(title='Post-creation form', audience='teacher', created_by=self.teacher)
          FeedbackQuestion.objects.create(form=self.form, text='Good proposals?', question_type='choice', options=['Yes', 'No'], order=1)
          self.client.login(username='fb_trig_teacher', password='pass')

      def _create_personal(self):
          from unittest.mock import patch
          with patch('authoringtool.views.apply_user_proposals_to_new_scenario.delay') as mock_delay:
              return self.client.get(reverse('create_personal_scenario', args=[self.scenario.id]), follow=True)

      def test_modal_context_present_after_creation(self):
          r = self._create_personal()
          self.assertIsNotNone(r.context['feedback_form_json'])
          self.assertContains(r, 'feedbackModal')

      def test_no_modal_without_creation_flow(self):
          r = self.client.get(reverse('proposal_list', args=[self.scenario.id]))
          self.assertIsNone(r.context['feedback_form_json'])

      def test_no_modal_when_already_responded(self):
          FeedbackResponse.objects.create(form=self.form, user=self.teacher, scenario=self.scenario)
          r = self._create_personal()
          self.assertIsNone(r.context['feedback_form_json'])

      def test_no_modal_when_no_applicable_form(self):
          self.form.is_active = False
          self.form.save()
          r = self._create_personal()
          self.assertIsNone(r.context['feedback_form_json'])

      def test_session_flag_consumed_after_one_render(self):
          self._create_personal()
          r = self.client.get(reverse('proposal_list', args=[self.scenario.id]))
          self.assertIsNone(r.context['feedback_form_json'])
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.TeacherFeedbackTriggerTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `feedback_form_json` isn't in `proposal_list_view`'s context yet (KeyError on `r.context['feedback_form_json']`).

- [ ] **Step 3: Set the session flag in `create_personal_scenario`**

  In `Trust-AI-Platform/authoringtool/views.py`, replace:
  ```python
  @login_required
  def create_personal_scenario(request, scenario_id):
      print(f"SCENARIO IS : {scenario_id}")
      apply_user_proposals_to_new_scenario.delay(scenario_id, request.user.id)
      messages.success(request, "Your personalized scenario is being created. It will appear in your scenarios shortly.")
      return redirect('proposal_list', scenario_id=scenario_id)
  ```
  with:
  ```python
  @login_required
  def create_personal_scenario(request, scenario_id):
      print(f"SCENARIO IS : {scenario_id}")
      apply_user_proposals_to_new_scenario.delay(scenario_id, request.user.id)
      messages.success(request, "Your personalized scenario is being created. It will appear in your scenarios shortly.")
      request.session['feedback_prompt_scenario_id'] = scenario_id
      return redirect('proposal_list', scenario_id=scenario_id)
  ```

- [ ] **Step 4: Pop the flag and build the modal context in `proposal_list_view`**

  In `Trust-AI-Platform/authoringtool/views.py`, in `proposal_list_view`, replace:
  ```python
      return render(request, 'authoringtool/proposal_list.html', {
          'proposals':         proposals,
          'myScenario':        myScenario,
          'user_reviews':      user_reviews,
          'show_create_button': show_create_button,
          'total_count':       total,
          'accepted_count':    accepted_count,
          'rejected_count':    rejected_count,
          'pending_count':     pending_count,
      })
  ```
  with:
  ```python
      feedback_form_json = None
      if request.session.pop('feedback_prompt_scenario_id', None) == myScenario.id:
          from feedback.utils import get_applicable_form, serialize_form, user_has_responded
          fb_form = get_applicable_form(myScenario, 'teacher')
          if fb_form and not user_has_responded(fb_form, request.user, myScenario):
              feedback_form_json = _json.dumps(serialize_form(fb_form), ensure_ascii=False)

      return render(request, 'authoringtool/proposal_list.html', {
          'proposals':         proposals,
          'myScenario':        myScenario,
          'user_reviews':      user_reviews,
          'show_create_button': show_create_button,
          'total_count':       total,
          'accepted_count':    accepted_count,
          'rejected_count':    rejected_count,
          'pending_count':     pending_count,
          'feedback_form_json': feedback_form_json,
      })
  ```
  Note: `_json` is the alias `proposal_list_view` already imports mid-function (`import json as _json` a few lines above this render call) — reuse it, don't add a new import. The `feedback.utils` import is function-level, matching this codebase's cross-app import precedent (`accounts/views.py` → `organization.models`).

- [ ] **Step 5: Add the modal to `proposal_list.html`**

  In `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_list.html`, immediately before the final `{% endblock %}` line, insert:
  ```html
  {% if feedback_form_json %}
  {{ feedback_form_json|json_script:"feedbackFormData" }}
  <div class="modal fade" id="feedbackModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable">
      <div class="modal-content" style="border-radius:14px;">
        <div class="modal-header" style="background:linear-gradient(135deg,#1a56db 0%,#1e3a8a 100%);color:#fff;border-radius:14px 14px 0 0;">
          <h5 class="modal-title" id="feedbackModalTitle" style="font-size:16px;font-weight:700;"></h5>
          <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <p class="text-muted" id="feedbackModalDesc" style="font-size:13.5px;"></p>
          <form id="feedbackModalForm">{% csrf_token %}<div id="feedbackModalQuestions"></div></form>
          <div class="text-danger small" id="feedbackModalError" style="display:none;"></div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Skip</button>
          <button type="button" class="btn btn-primary btn-sm" id="feedbackModalSubmit"><i class="bi bi-send me-1"></i>Submit</button>
        </div>
      </div>
    </div>
  </div>
  <script>
  document.addEventListener('DOMContentLoaded', function () {
    const formData = JSON.parse(JSON.parse(document.getElementById('feedbackFormData').textContent));
    document.getElementById('feedbackModalTitle').textContent = formData.title;
    document.getElementById('feedbackModalDesc').textContent = formData.description || '';
    const wrap = document.getElementById('feedbackModalQuestions');

    formData.questions.forEach(function (q) {
      const block = document.createElement('div');
      block.className = 'mb-3';
      const label = document.createElement('div');
      label.style.cssText = 'font-size:13.5px;font-weight:600;color:#012970;margin-bottom:6px;';
      label.textContent = q.text + (q.required ? ' *' : '');
      block.appendChild(label);
      if (q.type === 'choice') {
        q.options.forEach(function (opt, i) {
          const div = document.createElement('div');
          div.className = 'form-check';
          const input = document.createElement('input');
          input.className = 'form-check-input';
          input.type = 'radio';
          input.name = 'fbq_' + q.id;
          input.id = 'fbq_' + q.id + '_' + i;
          input.value = opt;
          const lab = document.createElement('label');
          lab.className = 'form-check-label';
          lab.setAttribute('for', input.id);
          lab.style.fontSize = '13.5px';
          lab.textContent = opt;
          div.appendChild(input);
          div.appendChild(lab);
          block.appendChild(div);
        });
      } else {
        const ta = document.createElement('textarea');
        ta.className = 'form-control';
        ta.rows = 3;
        ta.name = 'fbq_' + q.id;
        block.appendChild(ta);
      }
      wrap.appendChild(block);
    });

    document.getElementById('feedbackModalSubmit').addEventListener('click', function () {
      const answers = {};
      formData.questions.forEach(function (q) {
        const els = document.getElementsByName('fbq_' + q.id);
        if (q.type === 'choice') {
          for (const el of els) { if (el.checked) answers[q.id] = el.value; }
        } else if (els.length) {
          answers[q.id] = els[0].value;
        }
      });
      const csrfToken = document.querySelector('#feedbackModalForm [name=csrfmiddlewaretoken]').value;
      fetch('{% url "feedback_submit" 0 myScenario.id %}'.replace('/0/', '/' + formData.id + '/'), {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken, 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: answers }),
      })
      .then(function (r) { return r.json().then(function (data) { return { ok: r.ok, data: data }; }); })
      .then(function (res) {
        if (res.ok && res.data.success) {
          bootstrap.Modal.getInstance(document.getElementById('feedbackModal')).hide();
        } else {
          const err = document.getElementById('feedbackModalError');
          err.textContent = res.data.error || 'Something went wrong.';
          err.style.display = '';
        }
      });
    });

    new bootstrap.Modal(document.getElementById('feedbackModal')).show();
  });
  </script>
  {% endif %}
  ```

  Note on the URL construction: `{% url %}` needs literal args at template-render time, but the form id lives in the JSON payload. Rendering the URL with a `0` placeholder for `form_id` and string-replacing `'/0/'` with the real id keeps the URL reversing server-side (no hardcoded path) while letting JS fill the dynamic part. `myScenario.id` renders directly since it IS known at render time. The replace targets `'/0/'` (with slashes) so it can only match the placeholder segment.

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.TeacherFeedbackTriggerTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (5 tests).

  Then the full feedback suite plus the authoringtool suite (this task touched `authoringtool/views.py`):
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback authoringtool -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK`, zero failures.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/views.py Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_list.html Trust-AI-Platform/feedback/tests.py
  git commit -m "Show teacher feedback form modal after creating a personalized scenario"
  ```

---

### Task 6: Student trigger — Rasa signal, JS relay chain, gated modal in the scenario view

**Files:**
- Modify: `RasaFaith/RasaFaith/actions/actions.py` (`ActionEndScenario.run`)
- Modify: `data/static/chatbot_static/js/components/chat.js`
- Modify: `Trust-AI-Platform/staticfiles/chatbot_static/js/components/chat.js` (same edit, second diverged copy — see Global Constraints)
- Modify: `Trust-AI-Platform/studentview/templates/studentview/chatBot.html`
- Modify: `Trust-AI-Platform/studentview/views.py` (`scenario_viewer`)
- Modify: `Trust-AI-Platform/studentview/templates/studentview/scenarioView.html`
- Modify: `Trust-AI-Platform/feedback/tests.py`

**Interfaces:**
- Consumes: `get_applicable_form`, `user_has_responded`, `serialize_form`, `feedback_submit` URL (Task 2)
- Produces: postMessage type `scenarioEnded` (chatbot iframe → parent page); Rasa custom payload `{'scenario_ended': True}`.

- [ ] **Step 1: Write the failing tests (Django-side only — the JS/Rasa relay is not unit-testable here)**

  Append to `Trust-AI-Platform/feedback/tests.py`:
  ```python
  class StudentFeedbackTriggerTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.teacher = User.objects.create_user('fb_sv_teacher', password='pass')
          self.teacher.groups.add(teachers)
          self.student = User.objects.create_user('fb_sv_student', password='pass')
          self.scenario = Scenario.objects.create(name='FB SV Scenario', created_by=self.teacher, updated_by=self.teacher)
          from authoringtool.models import ActivityType, Phase, Activity
          phase = Phase.objects.create(name='P1', scenario=self.scenario, created_by=self.teacher, updated_by=self.teacher)
          atype = ActivityType.objects.create(name='Explanation', created_by=self.teacher, updated_by=self.teacher)
          Activity.objects.create(name='A1', text='x', scenario=self.scenario, phase=phase,
                                  activity_type=atype, created_by=self.teacher, updated_by=self.teacher)
          self.form = FeedbackForm.objects.create(title='Post-scenario form', audience='student', created_by=self.teacher)
          FeedbackQuestion.objects.create(form=self.form, text='Fun?', question_type='choice', options=['Yes', 'No'], order=1)

      def test_student_gets_feedback_form_in_context(self):
          self.client.login(username='fb_sv_student', password='pass')
          r = self.client.get(reverse('studentView', args=[self.scenario.id]))
          self.assertIsNotNone(r.context['feedback_form_json'])
          self.assertContains(r, 'feedbackModal')

      def test_teacher_gets_no_feedback_form(self):
          self.client.login(username='fb_sv_teacher', password='pass')
          r = self.client.get(reverse('studentView', args=[self.scenario.id]))
          self.assertIsNone(r.context['feedback_form_json'])
          self.assertNotContains(r, 'id="feedbackModal"')

      def test_already_responded_student_gets_no_form(self):
          FeedbackResponse.objects.create(form=self.form, user=self.student, scenario=self.scenario)
          self.client.login(username='fb_sv_student', password='pass')
          r = self.client.get(reverse('studentView', args=[self.scenario.id]))
          self.assertIsNone(r.context['feedback_form_json'])

      def test_no_applicable_form_gives_none(self):
          self.form.is_active = False
          self.form.save()
          self.client.login(username='fb_sv_student', password='pass')
          r = self.client.get(reverse('studentView', args=[self.scenario.id]))
          self.assertIsNone(r.context['feedback_form_json'])
  ```

  Note: check the actual URL name for `scenario_viewer` in `studentview/urls.py` before running — the plan assumes `studentView` (used by the dead-code redirect at `studentview/views.py:33`); if it differs, use the real name in these tests.

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.StudentFeedbackTriggerTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `feedback_form_json` isn't in `scenario_viewer`'s context yet.

- [ ] **Step 3: Add the form context to `scenario_viewer`**

  In `Trust-AI-Platform/studentview/views.py`, replace:
  ```python
  @login_required
  def scenario_viewer(request, scenario_id):
      scenario = get_object_or_404(Scenario, pk=scenario_id)
      first_activity = scenario.phases.first().activities.first() if scenario.phases.exists() else None
      user = request.user
      is_teacher = user.groups.filter(name="teachers").exists()
      if not first_activity:
          return render(request, 'error_page.html', {'error': 'No activities found in this scenario.'})
      return render(request, 'studentview/scenarioView.html', {'activity': first_activity, 'myScenario': scenario, 'user': user, 'is_teacher': is_teacher})
  ```
  with:
  ```python
  @login_required
  def scenario_viewer(request, scenario_id):
      scenario = get_object_or_404(Scenario, pk=scenario_id)
      first_activity = scenario.phases.first().activities.first() if scenario.phases.exists() else None
      user = request.user
      is_teacher = user.groups.filter(name="teachers").exists()
      if not first_activity:
          return render(request, 'error_page.html', {'error': 'No activities found in this scenario.'})

      feedback_form_json = None
      if not is_teacher:
          from feedback.utils import get_applicable_form, serialize_form, user_has_responded
          fb_form = get_applicable_form(scenario, 'student')
          if fb_form and not user_has_responded(fb_form, user, scenario):
              feedback_form_json = json.dumps(serialize_form(fb_form), ensure_ascii=False)

      return render(request, 'studentview/scenarioView.html', {
          'activity': first_activity, 'myScenario': scenario, 'user': user,
          'is_teacher': is_teacher, 'feedback_form_json': feedback_form_json,
      })
  ```
  Note: `json` is already imported at the top of `studentview/views.py` (line 11) — no new import needed.

- [ ] **Step 4: Emit the signal from Rasa**

  In `RasaFaith/RasaFaith/actions/actions.py`, replace:
  ```python
  class ActionEndScenario(Action):
      def name(self):
          return "action_end_scenario"

      def run(self, dispatcher, tracker, domain):
          user_locale = tracker.get_slot("locale") or \
                        tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
          if _is_greek(user_locale):
              dispatcher.utter_message(text="Ευχαριστούμε για την συμμετοχή σου!")
          else:
              dispatcher.utter_message(text="Thank you for participating!")
          return [AllSlotsReset()]
  ```
  with:
  ```python
  class ActionEndScenario(Action):
      def name(self):
          return "action_end_scenario"

      def run(self, dispatcher, tracker, domain):
          user_locale = tracker.get_slot("locale") or \
                        tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
          if _is_greek(user_locale):
              dispatcher.utter_message(text="Ευχαριστούμε για την συμμετοχή σου!")
          else:
              dispatcher.utter_message(text="Thank you for participating!")
          dispatcher.utter_message(json_message={'scenario_ended': True})
          return [AllSlotsReset()]
  ```
  (`json_message` is the same serialization sibling actions use for `activity_id` — `actions.py:174-176` — which the REST channel exposes to the frontend as `msg.custom`.)

- [ ] **Step 5: Relay in `chat.js` (BOTH copies)**

  In `data/static/chatbot_static/js/components/chat.js`, inside the `botResponse.forEach((msg) => { ... })` block, replace:
  ```javascript
      if (msg.custom?.activity_id !== undefined) {
        window.dispatchEvent(new CustomEvent('activityIdReceived', {
          detail: { activityId: msg.custom.activity_id }
        }));
      }
  ```
  with:
  ```javascript
      if (msg.custom?.activity_id !== undefined) {
        window.dispatchEvent(new CustomEvent('activityIdReceived', {
          detail: { activityId: msg.custom.activity_id }
        }));
      }
      if (msg.custom?.scenario_ended) {
        window.dispatchEvent(new CustomEvent('scenarioEnded'));
      }
  ```

  Then apply the SAME insertion to `Trust-AI-Platform/staticfiles/chatbot_static/js/components/chat.js`. The two files have drifted from each other — find the equivalent `msg.custom?.activity_id` dispatch block in that copy and add the same two-line `scenario_ended` check after it. If that copy's structure differs so much that no equivalent block exists, report it rather than guessing.

- [ ] **Step 6: Relay in `chatBot.html`**

  In `Trust-AI-Platform/studentview/templates/studentview/chatBot.html`, replace:
  ```javascript
      window.addEventListener('activityIdReceived', function(event) {
        parent.postMessage({
          type:       'activityIdReceived',
          activityId: event.detail.activityId,
        }, window.location.origin);
      });
  ```
  with:
  ```javascript
      window.addEventListener('activityIdReceived', function(event) {
        parent.postMessage({
          type:       'activityIdReceived',
          activityId: event.detail.activityId,
        }, window.location.origin);
      });

      // Relay scenario-completion signal to parent page
      window.addEventListener('scenarioEnded', function() {
        parent.postMessage({ type: 'scenarioEnded' }, window.location.origin);
      });
  ```

- [ ] **Step 7: Listen and show the modal in `scenarioView.html`**

  In `Trust-AI-Platform/studentview/templates/studentview/scenarioView.html`, in the unified message listener, replace:
  ```javascript
            if (data.type === 'activityIdReceived') {
                const readyButton = document.getElementById('readyButton');
                if (readyButton) readyButton.style.display = 'none';
                latestPendulumData = { "Pendulum 1": null, "Pendulum 2": null };
                sendPendulumDataToIframe();
                fetchAndDisplayActivity(data.activityId);
            }
  ```
  with:
  ```javascript
            if (data.type === 'activityIdReceived') {
                const readyButton = document.getElementById('readyButton');
                if (readyButton) readyButton.style.display = 'none';
                latestPendulumData = { "Pendulum 1": null, "Pendulum 2": null };
                sendPendulumDataToIframe();
                fetchAndDisplayActivity(data.activityId);
            }
            if (data.type === 'scenarioEnded') {
                const modalEl = document.getElementById('feedbackModal');
                if (modalEl) new bootstrap.Modal(modalEl).show();
            }
  ```

  Then, immediately before the template's final `{% endblock %}` (or closing `</body>`-equivalent block end — match the file's actual structure), insert the same modal used in Task 5, adapted for this page (only differences: no auto-show on load — it opens only from the `scenarioEnded` handler above — and the scenario id comes from `myScenario.id` exactly as in Task 5):
  ```html
  {% if feedback_form_json %}
  {{ feedback_form_json|json_script:"feedbackFormData" }}
  <div class="modal fade" id="feedbackModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable">
      <div class="modal-content" style="border-radius:14px;">
        <div class="modal-header" style="background:linear-gradient(135deg,#1a56db 0%,#1e3a8a 100%);color:#fff;border-radius:14px 14px 0 0;">
          <h5 class="modal-title" id="feedbackModalTitle" style="font-size:16px;font-weight:700;"></h5>
          <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <p class="text-muted" id="feedbackModalDesc" style="font-size:13.5px;"></p>
          <form id="feedbackModalForm">{% csrf_token %}<div id="feedbackModalQuestions"></div></form>
          <div class="text-danger small" id="feedbackModalError" style="display:none;"></div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Skip</button>
          <button type="button" class="btn btn-primary btn-sm" id="feedbackModalSubmit"><i class="bi bi-send me-1"></i>Submit</button>
        </div>
      </div>
    </div>
  </div>
  <script>
  document.addEventListener('DOMContentLoaded', function () {
    const formData = JSON.parse(JSON.parse(document.getElementById('feedbackFormData').textContent));
    document.getElementById('feedbackModalTitle').textContent = formData.title;
    document.getElementById('feedbackModalDesc').textContent = formData.description || '';
    const wrap = document.getElementById('feedbackModalQuestions');

    formData.questions.forEach(function (q) {
      const block = document.createElement('div');
      block.className = 'mb-3';
      const label = document.createElement('div');
      label.style.cssText = 'font-size:13.5px;font-weight:600;color:#012970;margin-bottom:6px;';
      label.textContent = q.text + (q.required ? ' *' : '');
      block.appendChild(label);
      if (q.type === 'choice') {
        q.options.forEach(function (opt, i) {
          const div = document.createElement('div');
          div.className = 'form-check';
          const input = document.createElement('input');
          input.className = 'form-check-input';
          input.type = 'radio';
          input.name = 'fbq_' + q.id;
          input.id = 'fbq_' + q.id + '_' + i;
          input.value = opt;
          const lab = document.createElement('label');
          lab.className = 'form-check-label';
          lab.setAttribute('for', input.id);
          lab.style.fontSize = '13.5px';
          lab.textContent = opt;
          div.appendChild(input);
          div.appendChild(lab);
          block.appendChild(div);
        });
      } else {
        const ta = document.createElement('textarea');
        ta.className = 'form-control';
        ta.rows = 3;
        ta.name = 'fbq_' + q.id;
        block.appendChild(ta);
      }
      wrap.appendChild(block);
    });

    document.getElementById('feedbackModalSubmit').addEventListener('click', function () {
      const answers = {};
      formData.questions.forEach(function (q) {
        const els = document.getElementsByName('fbq_' + q.id);
        if (q.type === 'choice') {
          for (const el of els) { if (el.checked) answers[q.id] = el.value; }
        } else if (els.length) {
          answers[q.id] = els[0].value;
        }
      });
      const csrfToken = document.querySelector('#feedbackModalForm [name=csrfmiddlewaretoken]').value;
      fetch('{% url "feedback_submit" 0 myScenario.id %}'.replace('/0/', '/' + formData.id + '/'), {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken, 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: answers }),
      })
      .then(function (r) { return r.json().then(function (data) { return { ok: r.ok, data: data }; }); })
      .then(function (res) {
        if (res.ok && res.data.success) {
          bootstrap.Modal.getInstance(document.getElementById('feedbackModal')).hide();
        } else {
          const err = document.getElementById('feedbackModalError');
          err.textContent = res.data.error || 'Something went wrong.';
          err.style.display = '';
        }
      });
    });
  });
  </script>
  {% endif %}
  ```

  Note: yes, this modal block is a near-verbatim duplicate of Task 5's (only the auto-show line differs). This is deliberate per-page duplication in the same spirit as the codebase's per-app helper duplication — the two pages have different base layouts and lifecycles, and a shared include would be the first-ever cross-app template include in this codebase. If the task reviewer flags this as duplication worth extracting, that's a legitimate discussion — record it rather than pre-empting it.

- [ ] **Step 8: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.StudentFeedbackTriggerTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (4 tests).

  Then the full feedback suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (42 tests total across all six tasks — recount from the actual run; zero failures is what matters).

- [ ] **Step 9: Manually verify (if a real dev environment is available)**

  Needs a live environment with Rasa + the action server running. If available: as a student, complete a scenario through Plato until `action_end_scenario` fires — confirm the modal opens, submit, confirm the response appears on the staff responses page and in both exports; replay the scenario and confirm the modal does NOT reappear. As a teacher, run the same scenario in the student view and confirm no modal ever appears (and that a direct POST to the submit endpoint returns 403). Restart the Rasa action server before testing (the new `json_message` line requires it).

  If unavailable: the Django-side tests are the load-bearing verification for context gating and submission; the Rasa→JS relay chain specifically CANNOT be verified in this sandbox and must be flagged as unverified, not assumed correct.

- [ ] **Step 10: Commit**

  ```bash
  git add RasaFaith/RasaFaith/actions/actions.py data/static/chatbot_static/js/components/chat.js Trust-AI-Platform/staticfiles/chatbot_static/js/components/chat.js Trust-AI-Platform/studentview/templates/studentview/chatBot.html Trust-AI-Platform/studentview/views.py Trust-AI-Platform/studentview/templates/studentview/scenarioView.html Trust-AI-Platform/feedback/tests.py
  git commit -m "Show student feedback form when Plato ends a scenario, teacher-gated"
  ```
