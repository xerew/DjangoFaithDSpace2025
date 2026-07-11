# CSV Per-Activity Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Include per-activity scores & timing" checkbox to the authoring tool dashboard that, when checked, appends per-activity type, time, and score columns to the downloaded CSV.

**Architecture:** Three-layer change — checkbox in the HTML template reads its state and appends `include_activity_detail=1` to the Celery task URL; the Django view passes the flag through to the task; the Celery task builds extra columns (one triple per activity: Type, Time, Score) when the flag is true.

**Tech Stack:** Django 4, Celery, Python `csv` / `io.StringIO`, Bootstrap Icons, vanilla JS

---

## File Map

| File | Change |
|---|---|
| `Trust-AI-Platform/authoringtool/templates/authoringtool/index.html` | Add checkbox before download button; read it in JS click handler |
| `Trust-AI-Platform/authoringtool/views.py` | Read `include_activity_detail` GET param; pass to `.delay()` |
| `Trust-AI-Platform/authoringtool/tasks.py` | Add `include_activity_detail=False` param; append per-activity columns when True |
| `Trust-AI-Platform/authoringtool/tests.py` | Unit test for per-activity header/row logic |

---

### Task 1: Add checkbox to the UI and wire it into the JS fetch URL

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/index.html` (lines ~289–316)

- [ ] **Step 1: Add the checkbox element before the download button**

Replace this block in `index.html`:

```html
			  <button type="button" id="generateDownloadButton" style="background:none;border:1.5px solid #16a34a;border-radius:8px;padding:7px 18px;font-size:13px;font-weight:600;color:#16a34a;cursor:pointer;display:inline-flex;align-items:center;gap:5px;">
                    <i class="bi bi-download"></i> Generate &amp; Download CSV
                  </button>
```

With:

```html
			  <label style="display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:500;color:#374151;cursor:pointer;user-select:none;">
                    <input type="checkbox" id="includeActivityDetail" style="width:15px;height:15px;cursor:pointer;">
                    Include per-activity scores &amp; timing
                  </label>
                  <button type="button" id="generateDownloadButton" style="background:none;border:1.5px solid #16a34a;border-radius:8px;padding:7px 18px;font-size:13px;font-weight:600;color:#16a34a;cursor:pointer;display:inline-flex;align-items:center;gap:5px;">
                    <i class="bi bi-download"></i> Generate &amp; Download CSV
                  </button>
```

- [ ] **Step 2: Read the checkbox in the JS click handler and append flag to URL**

In the same `<script>` block, after the line:
```js
const groupSelect = document.getElementById("groupSelect"); // Multi-select dropdown
```

Add:
```js
const includeActivityDetail = document.getElementById("includeActivityDetail");
```

Then replace the URL construction line:
```js
const url = `/authoringtool/student_performance_metrics/${scenarioId}/?start_date=${startDate}&end_date=${endDate}&group_ids=${selectedGroups.join(",")}`;
```

With:
```js
const activityDetail = includeActivityDetail.checked ? '1' : '0';
const url = `/authoringtool/student_performance_metrics/${scenarioId}/?start_date=${startDate}&end_date=${endDate}&group_ids=${selectedGroups.join(",")}&include_activity_detail=${activityDetail}`;
```

- [ ] **Step 3: Verify the checkbox renders and the URL param is appended**

Open the authoring tool dashboard in a browser. Open DevTools → Network tab. Check the checkbox, click "Generate & Download CSV", and confirm the request URL contains `include_activity_detail=1`. Uncheck, repeat — confirm it shows `include_activity_detail=0`.

- [ ] **Step 4: Commit**

```bash
git add Trust-AI-Platform/authoringtool/templates/authoringtool/index.html
git commit -m "Add include-per-activity-detail checkbox to CSV download UI"
```

---

### Task 2: Pass the flag through the Django view to the Celery task

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/views.py` (around line 2168)

- [ ] **Step 1: Read the flag and pass it to `.delay()`**

Find the `student_performance_metrics` view (around line 2168). Replace:

```python
def student_performance_metrics(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_student_performance_metrics.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})
```

With:

```python
def student_performance_metrics(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    include_activity_detail = request.GET.get('include_activity_detail', '0') == '1'
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []

    result = compute_student_performance_metrics.delay(
        scenario_id, group_ids, start_date, end_date, include_activity_detail
    )
    return JsonResponse({'task_id': result.id})
```

- [ ] **Step 2: Commit**

```bash
git add Trust-AI-Platform/authoringtool/views.py
git commit -m "Pass include_activity_detail flag from view to Celery task"
```

---

### Task 3: Extend the Celery task to emit per-activity columns

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/tasks.py` (function `compute_student_performance_metrics`, line ~1387)

- [ ] **Step 1: Add the flag parameter and broaden select_related**

Change the task signature from:
```python
def compute_student_performance_metrics(scenario_id, group_ids, start_date, end_date):
```
To:
```python
def compute_student_performance_metrics(scenario_id, group_ids, start_date, end_date, include_activity_detail=False):
```

Change the `all_acts` query from:
```python
all_acts = list(Activity.objects.filter(phase__scenario=scenario).select_related('phase'))
```
To:
```python
all_acts = list(Activity.objects.filter(phase__scenario=scenario).select_related('phase', 'activity_type').order_by('id'))
```

The added `.order_by('id')` and `select_related('activity_type')` ensure deterministic column order and avoid N+1 queries on `activity.activity_type` when the flag is on.

- [ ] **Step 2: Append per-activity data to each user's row when the flag is on**

After the existing per-phase loop that builds each user's `row` (after `row.append(final_categorization)`), add:

```python
        if include_activity_detail:
            for phase in phases:
                for activity in sorted(phase_activities_map[phase.id], key=lambda a: a.id):
                    act_type = activity.activity_type.name if activity.activity_type else ''
                    ua = last_answers_dict.get((user.id, activity.id))
                    timing = ua.timing if ua and ua.timing is not None else ''
                    if activity.is_evaluatable and ua and ua.answer:
                        score = ua.answer.answer_weight
                    else:
                        score = ''
                    row.extend([act_type, timing, score])
```

This block goes inside the `for user in valid_users:` loop, after `row.append(final_categorization)` and before `csv_data.append(row)`.

- [ ] **Step 3: Extend the CSV header with per-activity columns when the flag is on**

After the existing header build (after `header.append('Final Categorization')`), add:

```python
    if include_activity_detail:
        for phase in phases:
            for activity in sorted(phase_activities_map[phase.id], key=lambda a: a.id):
                prefix = f"{phase.name} > {activity.name}"
                header.extend([f"{prefix} Type", f"{prefix} Time (s)", f"{prefix} Score"])
```

- [ ] **Step 4: Commit**

```bash
git add Trust-AI-Platform/authoringtool/tasks.py
git commit -m "Add per-activity type/time/score columns to student performance CSV"
```

---

### Task 4: Write and run a unit test for the per-activity column logic

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/tests.py`

- [ ] **Step 1: Write the test**

Replace the contents of `Trust-AI-Platform/authoringtool/tests.py` with:

```python
from django.test import TestCase
from unittest.mock import patch, MagicMock
import io, csv


class ComputeStudentPerformanceMetricsActivityDetailTest(TestCase):
    """
    Verifies that compute_student_performance_metrics appends the correct
    per-activity columns (Type, Time (s), Score) when include_activity_detail=True.

    We call the task synchronously (not via Celery) by importing the underlying
    function directly. We patch get_object_or_404 and all ORM calls to avoid
    needing a real database.
    """

    def _make_activity(self, id, name, activity_type_name, is_evaluatable):
        act = MagicMock()
        act.id = id
        act.name = name
        act.is_evaluatable = is_evaluatable
        act.is_primary_ev = False
        act.activity_type = MagicMock()
        act.activity_type.name = activity_type_name
        act.phase_id = 1
        return act

    def _make_ua(self, timing, answer_weight):
        ua = MagicMock()
        ua.timing = timing
        if answer_weight is not None:
            ua.answer = MagicMock()
            ua.answer.answer_weight = answer_weight
        else:
            ua.answer = None
        return ua

    @patch('authoringtool.tasks.get_last_answers')
    @patch('authoringtool.tasks.QuestionBunch')
    @patch('authoringtool.tasks.Answer')
    @patch('authoringtool.tasks.UserAnswer')
    @patch('authoringtool.tasks.Activity')
    @patch('authoringtool.tasks.Phase')
    @patch('authoringtool.tasks.User')
    @patch('authoringtool.tasks.get_object_or_404')
    def test_activity_detail_columns_appended(
        self, mock_404, mock_user_cls, mock_phase_cls, mock_activity_cls,
        mock_ua_cls, mock_answer_cls, mock_qb_cls, mock_get_last_answers
    ):
        from authoringtool.tasks import compute_student_performance_metrics

        # Scenario + phase
        scenario = MagicMock(); scenario.id = 1; scenario.name = "Test Scenario"
        phase = MagicMock(); phase.id = 1; phase.name = "Phase 1"
        mock_404.return_value = scenario
        mock_phase_cls.objects.filter.return_value.order_by.return_value = [phase]

        # Two activities: one evaluatable, one not
        act1 = self._make_activity(1, "Read Text",    "Info",       False)
        act2 = self._make_activity(2, "Quiz",         "Evaluatable", True)
        act1.phase = phase; act2.phase = phase

        mock_activity_cls.objects.filter.return_value.select_related.return_value.order_by.return_value = [act1, act2]
        mock_activity_cls.objects.filter.return_value.order_by.return_value.first.return_value = act1
        mock_activity_cls.objects.filter.return_value.select_related.return_value = MagicMock(
            __iter__=lambda s: iter([act1, act2]),
            filter=MagicMock(return_value=MagicMock(
                values_list=MagicMock(return_value=[]),
            ))
        )

        # One valid user
        user = MagicMock(); user.id = 99; user.username = "student1"
        mock_user_cls.objects.filter.return_value.distinct.return_value.exclude.return_value = [user]

        # Last answers: act1 has timing only, act2 has timing + score
        ua1 = self._make_ua(timing=30,  answer_weight=None)
        ua2 = self._make_ua(timing=120, answer_weight=4)
        mock_get_last_answers.return_value.filter.return_value.filter.return_value = MagicMock(
            values_list=MagicMock(return_value=[99]),
            filter=MagicMock(return_value=MagicMock(
                select_related=MagicMock(return_value=[
                    MagicMock(user_id=99, activity_id=1, **{'answer': ua1.answer, 'timing': ua1.timing}),
                    MagicMock(user_id=99, activity_id=2, **{'answer': ua2.answer, 'timing': ua2.timing}),
                ])
            ))
        )

        mock_qb_cls.objects.filter.return_value = []
        mock_answer_cls.objects.filter.return_value.values.return_value.annotate.return_value = []
        mock_ua_cls.objects.filter.return_value.filter.return_value.filter.return_value.values.return_value.annotate.return_value = []
        mock_ua_cls.objects.filter.return_value.filter.return_value.values.return_value.annotate.return_value = []

        result = compute_student_performance_metrics(
            scenario_id=1, group_ids=[], start_date=None, end_date=None,
            include_activity_detail=True
        )

        self.assertIn('csv_content', result)
        reader = list(csv.reader(io.StringIO(result['csv_content'])))
        header = reader[0]

        self.assertIn("Phase 1 > Read Text Type",    header)
        self.assertIn("Phase 1 > Read Text Time (s)", header)
        self.assertIn("Phase 1 > Read Text Score",   header)
        self.assertIn("Phase 1 > Quiz Type",         header)
        self.assertIn("Phase 1 > Quiz Time (s)",     header)
        self.assertIn("Phase 1 > Quiz Score",        header)
```

- [ ] **Step 2: Run the test**

```bash
cd Trust-AI-Platform
python manage.py test authoringtool.tests.ComputeStudentPerformanceMetricsActivityDetailTest -v 2
```

Expected: the test may need adjustment depending on how deep the mocking goes — the goal is to confirm the header column names are generated correctly. If ORM mocking is too entangled, skip to the manual end-to-end test in Task 5 and note this for future test infrastructure.

- [ ] **Step 3: Commit**

```bash
git add Trust-AI-Platform/authoringtool/tests.py
git commit -m "Add unit test for per-activity CSV column generation"
```

---

### Task 5: End-to-end manual verification

- [ ] **Step 1: On the server, pull and rebuild**

```bash
git pull
sudo docker compose up -d --build web celery
```

- [ ] **Step 2: Open the authoring tool dashboard in a browser**

Navigate to `/authoringtool/` and select a scenario that has student activity data.

- [ ] **Step 3: Download WITHOUT checkbox checked**

Click "Generate & Download CSV". Open the file. Confirm the columns are:
`User ID, Username, Scenario Name, [Phase] Categorization, [Phase] Start Time, [Phase] Time, [Phase] Score, …, Final Categorization`
— no per-activity columns present.

- [ ] **Step 4: Download WITH checkbox checked**

Check "Include per-activity scores & timing", click the button again. Open the file. Confirm:
- Columns appear after "Final Categorization"
- Each activity has three columns: `[Phase] > [Activity] Type`, `[Phase] > [Activity] Time (s)`, `[Phase] > [Activity] Score`
- Non-evaluatable activities have blank Score cells
- Activities with no recorded answer have blank Time cells

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "Verify per-activity CSV detail feature end-to-end"
git push
```