# Feedback Forms — Design Spec

## Goal

Let site admins build feedback forms (multiple-choice and free-text questions) that are shown to teachers right after they create a personalized scenario from AI proposals, and to students right after they finish a scenario through the Plato chatbot. Admins can manage forms and responses (view counts, edit, delete, delete individual responses) and export all answers as XLSX or comma-delimited CSV for analysis. Teachers testing a scenario as a student never see the student forms. Everything must be responsive on phones (≥320px) and tablets (≥768px).

## Scope

- New Django app: `Trust-AI-Platform/feedback/` (models, views, urls, admin, tests, templates) — matching this codebase's one-app-per-concern convention (`messaging`, `organization`, `usergroups`, `home`).
- `Trust-AI-Platform/faithDev/settings.py` + `faithDev/urls.py` — register the app and its URL prefix.
- `Trust-AI-Platform/authoringtool/views.py` (`create_personal_scenario`, `proposal_list_view`) — teacher trigger via session flag.
- `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_list.html` — teacher form modal.
- `Trust-AI-Platform/studentview/views.py` (`scenario_viewer`) + `studentview/templates/studentview/scenarioView.html` — student form modal.
- `Trust-AI-Platform/data/static/chatbot_static/js/components/chat.js` + the chatbot iframe template (`chatBot.html`) — relay the new `scenario_ended` signal.
- `RasaFaith/RasaFaith/actions/actions.py` (`ActionEndScenario`) — emit the `scenario_ended` custom payload (one added `dispatcher.utter_message` line). Deploy note: requires a Rasa action-server restart; no NLU retraining.
- `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html` — link to the new Feedback Forms management page.

## What Already Exists (confirmed by reading the code)

- A half-dead survey system (`MultilingualQuestion`/`MultilingualAnswer` in `authoringtool/models.py:439-488`): working models, a working submit endpoint, a CSV export — but the student-facing form UI is unreachable dead code (`studentview/views.py:33` redirects unconditionally before the render). **This system stays untouched** — the new feature is built separately; removing/migrating the old one is out of scope.
- No scenario-completion signal exists anywhere. Rasa's `ActionEndScenario` (`RasaFaith/RasaFaith/actions/actions.py:650-661`) utters a thank-you text and resets slots — nothing machine-readable reaches the web page. The relay pattern to build on: Rasa custom payloads → `chat.js` inspects `msg.custom` and dispatches `CustomEvent`s (`chat.js:194-209`, the `activityIdReceived` pattern) → `chatBot.html` relays to the parent via postMessage → `scenarioView.html` listens (`scenarioView.html:100-114`).
- `scenario_viewer` already computes `is_teacher = user.groups.filter(name="teachers").exists()` and passes it to `scenarioView.html` (`studentview/views.py:26,29`).
- `create_personal_scenario` (`authoringtool/views.py:2473-2477`) queues the clone task and redirects to `proposal_list` with a success toast — the natural place for the teacher modal.
- Export precedents: XLSX via openpyxl + BytesIO + `HttpResponse` with the xlsx content type (`authoringtool/views.py:2563-2580`); CSV via `csv.writer(response)` (`usergroups/views.py:438-456` — note it uses `delimiter=';'`; the new export deliberately uses the default comma per this feature's requirement).
- Staff-only management page precedent: `staff_required` decorator (`accounts/admin_views.py:12-19`, checks `is_staff or is_superuser`, returns `HttpResponseForbidden`) used by the admin dashboard. The new management views duplicate this decorator locally in `feedback/views.py` per the established per-app-duplication convention (`group_required` is duplicated in 5 apps the same way).

## Models (all in `feedback/models.py`)

```python
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

    def applies_to(self, scenario):
        if not self.is_active:
            return False
        if self.assign_to_all:
            return not self.excluded_scenarios.filter(pk=scenario.pk).exists()
        return self.included_scenarios.filter(pk=scenario.pk).exists()
```

- `assign_to_all=True` is a **live rule**: future scenarios are covered automatically; `excluded_scenarios` is the opt-out list (only consulted in this mode).
- `assign_to_all=False`: only `included_scenarios` are covered (only consulted in this mode).
- Ordering carries the `-pk` tiebreaker from the start (same `auto_now_add`-tie flakiness fix applied proactively three times before on this branch).

```python
class FeedbackQuestion(models.Model):
    TYPE_CHOICES = [('choice', 'Multiple Choice'), ('text', 'Free Text')]
    form = models.ForeignKey(FeedbackForm, on_delete=models.CASCADE, related_name='questions')
    text = models.CharField(max_length=500)
    question_type = models.CharField(max_length=16, choices=TYPE_CHOICES)
    options = models.JSONField(default=list, blank=True)  # list of strings; only for 'choice'
    is_required = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order', 'pk']
```

```python
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
```

```python
class FeedbackAnswer(models.Model):
    response = models.ForeignKey(FeedbackResponse, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(FeedbackQuestion, on_delete=models.CASCADE, related_name='answers')
    answer_text = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['response', 'question'], name='unique_feedback_answer_per_question'),
        ]
```

- Choice answers store the selected option string in `answer_text` — one flat text field keeps the export trivially flat too. Deleting a question cascades its answers (acceptable: forms are meant to be finalized before collection; a warning appears in the edit UI when a form already has responses).

## Admin Management UI (staff-only, outside Django admin)

All views gated by a locally-duplicated `staff_required`. URL prefix `feedback/manage/`. Linked from the admin dashboard (`accounts/admin_dashboard.html`) with a "Feedback Forms" card/button.

- **Forms list** (`feedback_form_list`): every form with audience badge, active toggle state, question count, response count, created date. Actions per row: View responses, Edit, Delete (POST, confirm dialog).
- **Create/Edit form** (`feedback_form_create` / `feedback_form_edit`): title, description, audience, active checkbox; a JS-driven question editor (add/remove/reorder questions; per-question type selector; options editor shown only for choice type); the scenario assignment block — an "Assign to all scenarios (includes future ones)" master checkbox; when checked, the scenario list becomes exclusion checkboxes ("uncheck to exclude" — all checked by default); when unchecked, it becomes inclusion checkboxes (all unchecked by default). Editing a form that already has responses shows a warning banner (changing/removing questions affects historical answer alignment) but is allowed.
- **Responses page** (`feedback_form_responses`): table of responses (user, scenario, submitted date), each expandable/linked to its answers; per-response Delete button (POST, confirm — the "for testing purposes" requirement); export buttons: **Download XLSX** and **Download CSV**.
- **Exports** (`feedback_form_export_xlsx` / `feedback_form_export_csv`): one row per response; columns: username, scenario name, submitted date, then one column per question (headers = question text, in question order). CSV uses the **comma** delimiter (deliberate deviation from `usergroups`' semicolon precedent — explicitly requested). XLSX via openpyxl in-memory, same response-construction pattern as `download_template`.

## Teacher Trigger

1. `create_personal_scenario` (`authoringtool/views.py`) additionally sets `request.session['feedback_prompt_scenario_id'] = scenario_id` before redirecting (only when an applicable active teacher-audience form exists for that scenario — checked via a small helper imported from the `feedback` app; cross-app import at function level, matching the `accounts/views.py` → `organization.models` precedent).
2. `proposal_list_view` pops the session key (`.pop()`, so it fires once per creation) and, if set and the user hasn't already responded, passes the applicable form (+ its questions) into the template context.
3. `proposal_list.html` renders a Bootstrap modal auto-opened on load when the context is present: description, questions (radio groups for choice, textarea for text), Submit + Skip. Submit POSTs via fetch to the `feedback` app's submit endpoint; Skip just closes (the modal reappears on the next scenario creation until the form is answered for that scenario — the unique constraint governs "answered").

## Student Trigger

1. **Rasa** (`ActionEndScenario.run`): add `dispatcher.utter_message(json_message={"custom": {"scenario_ended": True}})` alongside the existing thank-you text. (Exact serialization to match how activity_id custom payloads are already emitted by sibling actions in this file — the implementer copies the working sibling pattern rather than inventing one.)
2. **`chat.js`**: in the existing `botResponse.forEach` custom-payload inspection block, add: if `msg.custom?.scenario_ended`, dispatch `new CustomEvent('scenarioEnded')` — identical shape to the `activityIdReceived` dispatch right above it.
3. **`chatBot.html`**: relay `scenarioEnded` to the parent window via postMessage, same as its existing `activityIdReceived` relay.
4. **`scenarioView.html`**: listen for the `scenarioEnded` message; if `is_teacher` is false and the server said a form is applicable, open the feedback modal. The applicable form (+ questions + already-answered flag) is embedded in the page context by `scenario_viewer` at load time (no extra fetch needed at completion time).
5. **Server-side enforcement** (defense in depth, not just UI hiding): the submit endpoint rejects submissions to a `student`-audience form from any user in the `teachers` group with 403, and rejects `teacher`-audience submissions from non-teachers likewise. It also re-validates `form.applies_to(scenario)`, required questions answered, and choice answers being one of the question's options.

## Submit Endpoint (shared by both triggers)

`POST feedback/submit/<form_id>/<scenario_id>/` — `@login_required`, JSON in/out (fetch + CSRF header, matching the org-chat send endpoint's pattern). Creates the `FeedbackResponse` + `FeedbackAnswer`s in one transaction. Duplicate submission (unique constraint) returns a friendly already-submitted JSON error rather than a 500.

## Responsive & Mobile-Friendly (phones ≥320px, tablets ≥768px)

- All new management pages follow the established hero/breadcrumb pattern with the existing `@media (max-width: 575.98px)` mobile breakpoint convention used across this branch; tables on the forms-list and responses pages are wrapped in `.table-responsive` (horizontal scroll within the card, never the page).
- Both feedback modals use Bootstrap's standard modal (already responsive); question option rows stack vertically; no new fixed-pixel widths on outer containers.
- The scenario-assignment checkbox list scrolls within a max-height container on small screens rather than growing unbounded.

## What Does NOT Change

- The old `MultilingualQuestion`/`MultilingualAnswer` system — models, admin, dead-code view, CSV export all untouched.
- The proposals/history features just shipped — untouched except the two teacher-trigger integration points named above.
- Rasa NLU/domain/stories — only the one custom action gains one line; no retraining, action-server restart only.
- No form versioning, no anonymous responses, no multilingual question text, no per-question analytics UI — out of scope for this first pass (the export covers analysis).
