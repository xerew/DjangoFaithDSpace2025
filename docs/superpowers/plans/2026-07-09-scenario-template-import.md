# Scenario Template Import — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow teachers to upload a pre-formatted `.xlsx` file from the "Create New Scenario" modal to create a complete scenario (phases, activities, answers, routing, evaluation) in one atomic operation.

**Architecture:** A `ScenarioImporter` service class in `authoringtool/importer.py` handles parse → validate → create. Two new views (`import_scenario`, `download_template`) wire it to HTTP. The scenarios list page gains a modal with two tabs (Upload Template / Manual Setup).

**Tech Stack:** Django 4.x, openpyxl (Excel read/write), markdown2 (markdown→HTML), Bootstrap 5 modal + tabs (already in use), fetch() for AJAX upload.

## Global Constraints

- All new views in `authoringtool/` use `@group_required('teachers')` — consistent with all existing authoring views.
- `import_scenario` must be `@require_POST`.
- Import is fully atomic: no DB writes if any error exists.
- All errors are collected before returning — never stop at the first error.
- `shared_secret` and `consumer_key` are never touched by this feature (the importer links existing labs by name only).
- Tests run with `python manage.py test authoringtool --settings=faithDev.settings_test`.
- The test settings use SQLite in-memory. Do NOT set `age_of_students` in tests (it is a Postgres-only `IntegerRangeField`). Do NOT create `QuestionBunch` in SQLite tests (it uses `ArrayField`). Test the evaluation/QuestionBunch path via the importer's in-memory logic only (mock the DB write or skip that specific assertion).
- New dependencies: `openpyxl` and `markdown2` — add both to `Trust-AI-Platform/requirements.txt`.
- Do not modify any existing view or URL. Only add new ones.

---

### Task 1: Dependencies + `importer.py` — Parse & Validate

**Files:**
- Modify: `Trust-AI-Platform/requirements.txt`
- Create: `Trust-AI-Platform/authoringtool/importer.py`
- Modify: `Trust-AI-Platform/authoringtool/tests.py`

**Interfaces:**
- Produces: `ScenarioImporter(file_obj, user)` class with `.run()` → `(scenario | None, errors_list)`
- `errors_list` items: `{'sheet': str, 'row': int, 'column': str, 'message': str}`

---

- [ ] **Step 1: Add dependencies to requirements.txt**

Open `Trust-AI-Platform/requirements.txt` and append:
```
openpyxl>=3.1.0
markdown2>=2.4.0
```

- [ ] **Step 2: Write the failing tests for parse + validate**

Add to `Trust-AI-Platform/authoringtool/tests.py`:

```python
import io
import json
import openpyxl
from django.test import TestCase, Client
from django.contrib.auth.models import User, Group
from django.urls import reverse

from authoringtool.models import (
    Scenario, Phase, Activity, Answer, AnswerFeedback,
    NextQuestionLogic, ActivityType,
)


def make_xlsx(
    scenario_name='Test Scenario',
    phases=None,
    activities=None,
    answers=None,
    routing=None,
    evaluation=None,
    missing_sheet=None,
):
    """Build a minimal valid .xlsx in memory. Each argument is a list of tuples (row values)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'README'
    ws['A1'] = 'Instructions'

    sheets = {
        'Scenario': (
            ['Name', 'Description', 'Visibility'],
            [[scenario_name, '', 'private']],
        ),
        'Phases': (
            ['Phase Name', 'Description', 'Video URL'],
            phases or [['Phase 1', '', '']],
        ),
        'Activities': (
            ['Activity Name', 'Phase Name', 'Text', 'Activity Type', 'Helper',
             'Is Evaluatable', 'Is Primary Evaluation', 'Must Wait', 'Score Limit',
             'Simulation Name', 'Remote Lab Name', 'VR Lab Name', 'Image URL', 'Video URL'],
            activities or [['Act 1', 'Phase 1', 'Hello world', '', '', 'No', 'No', 'No', '', '', '', '', '', '']],
        ),
        'Answers': (
            ['Activity Name', 'Answer Text', 'Is Correct', 'Answer Weight',
             'Image URL', 'Video URL', 'Feedback Text', 'Feedback Image URL', 'Feedback Video URL'],
            answers or [],
        ),
        'Next Activity': (
            ['Source Activity Name', 'Answer Text', 'Next Activity Name'],
            routing or [],
        ),
        'Evaluation': (
            ['Primary Activity Name', 'Grouped Activities', 'High Branch Activity',
             'High Branch Feedback', 'Mid Branch Activity', 'Mid Branch Feedback',
             'Low Branch Activity', 'Low Branch Feedback'],
            evaluation or [],
        ),
    }

    for name, (header, rows) in sheets.items():
        if name == missing_sheet:
            continue
        ws2 = wb.create_sheet(name)
        ws2.append(header)
        for row in rows:
            ws2.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


class ImporterValidationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('teacher', password='pass')

    def _run(self, **kwargs):
        from authoringtool.importer import ScenarioImporter
        buf = make_xlsx(**kwargs)
        importer = ScenarioImporter(buf, self.user)
        importer._parse()
        if not importer.errors:
            importer._validate()
        return importer.errors

    def test_valid_file_no_errors(self):
        errors = self._run()
        self.assertEqual(errors, [])

    def test_missing_sheet_reported(self):
        errors = self._run(missing_sheet='Activities')
        messages = [e['message'] for e in errors]
        self.assertTrue(any('Activities' in m for m in messages))

    def test_missing_scenario_name(self):
        errors = self._run(scenario_name='')
        messages = [e['message'] for e in errors]
        self.assertTrue(any('Name' in m or 'required' in m.lower() for m in messages))

    def test_duplicate_scenario_name(self):
        Scenario.objects.create(name='Test Scenario', created_by=self.user, updated_by=self.user)
        errors = self._run(scenario_name='Test Scenario')
        messages = [e['message'] for e in errors]
        self.assertTrue(any('already exists' in m for m in messages))

    def test_invalid_visibility(self):
        # build with bad visibility by patching scenario_data after parse
        from authoringtool.importer import ScenarioImporter
        buf = make_xlsx()
        importer = ScenarioImporter(buf, self.user)
        importer._parse()
        importer.scenario_data['Visibility'] = 'invalid'
        importer._load_db_lookups()
        importer._validate_scenario()
        self.assertTrue(any('Visibility' in e['column'] for e in importer.errors))

    def test_activity_references_unknown_phase(self):
        errors = self._run(
            activities=[['Act 1', 'NonexistentPhase', 'Hello', '', '', 'No', 'No', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('Phase' in e['message'] for e in errors))

    def test_duplicate_activity_names(self):
        errors = self._run(
            activities=[
                ['Act 1', 'Phase 1', 'Hello', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
                ['Act 1', 'Phase 1', 'Dupe', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
            ],
        )
        self.assertTrue(any('Duplicate' in e['message'] for e in errors))

    def test_primary_ev_without_evaluatable(self):
        errors = self._run(
            activities=[['Act 1', 'Phase 1', 'Hello', '', '', 'No', 'Yes', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('Is Evaluatable' in e['message'] or 'Is Primary' in e['message'] for e in errors))

    def test_evaluatable_without_evaluation_row(self):
        errors = self._run(
            activities=[['Quiz', 'Phase 1', 'Q?', '', '', 'Yes', 'Yes', 'No', '', '', '', '', '', '']],
            answers=[['Quiz', 'Option A', 'Yes', '1', '', '', '', '', '']],
            evaluation=[],  # no evaluation row
        )
        self.assertTrue(any('Evaluation' in e['message'] or 'evaluatable' in e['message'].lower() for e in errors))

    def test_answer_references_unknown_activity(self):
        errors = self._run(
            answers=[['NonexistentAct', 'Option A', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('NonexistentAct' in e['message'] for e in errors))

    def test_routing_references_unknown_next_activity(self):
        errors = self._run(
            routing=[['Act 1', '', 'GhostActivity']],
        )
        self.assertTrue(any('GhostActivity' in e['message'] for e in errors))

    def test_routing_answer_not_found(self):
        errors = self._run(
            answers=[['Act 1', 'Option A', 'Yes', '1', '', '', '', '', '']],
            routing=[['Act 1', 'Wrong Answer Text', '']],
        )
        self.assertTrue(any('Wrong Answer Text' in e['message'] for e in errors))

    def test_invalid_boolean_field(self):
        errors = self._run(
            activities=[['Act 1', 'Phase 1', 'Hello', '', '', 'maybe', 'No', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('Yes' in e['message'] or 'No' in e['message'] for e in errors))
```

- [ ] **Step 3: Run tests — verify they all fail**

```
cd Trust-AI-Platform
python manage.py test authoringtool.tests.ImporterValidationTest --settings=faithDev.settings_test -v 2
```

Expected: `ImportError: No module named 'authoringtool.importer'` (or similar)

- [ ] **Step 4: Create `authoringtool/importer.py` with parse + validate**

Create `Trust-AI-Platform/authoringtool/importer.py`:

```python
import openpyxl
import markdown2
from django.utils.html import strip_tags

from .models import (
    Scenario, Simulation, ExperimentLL, VRARExperiment, ActivityType,
)

# ── Column definitions ────────────────────────────────────────────────────

REQUIRED_SHEETS = ['README', 'Scenario', 'Phases', 'Activities', 'Answers', 'Next Activity', 'Evaluation']

SCENARIO_REQUIRED = ['Name']
PHASES_REQUIRED = ['Phase Name']
ACTIVITIES_REQUIRED = ['Activity Name', 'Phase Name', 'Text']
ANSWERS_REQUIRED = ['Activity Name', 'Answer Text']
ROUTING_REQUIRED = ['Source Activity Name']
EVALUATION_REQUIRED = ['Primary Activity Name', 'Grouped Activities']

BOOL_COLS_ACTIVITIES = ('Is Evaluatable', 'Is Primary Evaluation', 'Must Wait')
BOOL_COLS_ANSWERS = ('Is Correct',)


def _to_bool(value, default=False):
    v = str(value).strip().lower()
    if v == 'yes':
        return True
    if v == 'no':
        return False
    return default


def _to_float(value):
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _to_int(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _md(text):
    if not text:
        return ''
    return markdown2.markdown(str(text), extras=['fenced-code-blocks', 'tables'])


# ── Main class ────────────────────────────────────────────────────────────

class ScenarioImporter:
    def __init__(self, file_obj, user):
        self.file_obj = file_obj
        self.user = user
        self.errors = []

        self.scenario_data = {}
        self.phases = []
        self.activities = []
        self.activity_map = {}   # name -> dict
        self.answers = []
        self.answer_map = {}     # (activity_name, answer_text) -> dict
        self.routing = []
        self.evaluation = []

        self._simulations = {}
        self._remote_labs = {}
        self._vr_labs = {}
        self._activity_types = {}

    def run(self):
        """Parse, validate, create. Returns (Scenario | None, errors list)."""
        self._parse()
        if not self.errors:
            self._validate()
        if not self.errors:
            return self._create(), []
        return None, self.errors

    def _add_error(self, sheet, row, column, message):
        self.errors.append({'sheet': sheet, 'row': row, 'column': column, 'message': message})

    # ── Parse ─────────────────────────────────────────────────────────────

    def _parse(self):
        try:
            wb = openpyxl.load_workbook(self.file_obj, read_only=True, data_only=True)
        except Exception:
            self._add_error('File', 0, '-', 'Cannot read file. Ensure it is a valid .xlsx file.')
            return

        sheet_index = {ws.title.lower(): ws.title for ws in wb.worksheets}
        missing = [s for s in REQUIRED_SHEETS if s.lower() not in sheet_index]
        for s in missing:
            self._add_error('File', 0, '-', f'Missing required sheet: "{s}"')
        if self.errors:
            return

        def ws(name):
            return wb[sheet_index[name.lower()]]

        self.scenario_data = self._parse_single_row(ws('Scenario'), 'Scenario', SCENARIO_REQUIRED)
        self.phases = self._parse_rows(ws('Phases'), 'Phases', PHASES_REQUIRED)
        self.activities = self._parse_rows(ws('Activities'), 'Activities', ACTIVITIES_REQUIRED)
        self.activity_map = {a['Activity Name']: a for a in self.activities if a.get('Activity Name')}
        self.answers = self._parse_rows(ws('Answers'), 'Answers', ANSWERS_REQUIRED)
        self.answer_map = {
            (a['Activity Name'], a['Answer Text']): a
            for a in self.answers
            if a.get('Activity Name') and a.get('Answer Text')
        }
        self.routing = self._parse_rows(ws('Next Activity'), 'Next Activity', ROUTING_REQUIRED)
        self.evaluation = self._parse_rows(ws('Evaluation'), 'Evaluation', EVALUATION_REQUIRED)

    def _read_headers(self, ws, sheet_name, required_cols):
        rows = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        if not rows or all(v is None for v in rows[0]):
            self._add_error(sheet_name, 1, '-', 'Header row is empty.')
            return None
        headers = {}
        for i, h in enumerate(rows[0]):
            if h is not None:
                headers[str(h).replace('*', '').strip()] = i
        missing = [c for c in required_cols if c not in headers]
        for c in missing:
            self._add_error(sheet_name, 1, c, f'Required column "{c}" not found in header row.')
        return None if missing else headers

    def _parse_rows(self, ws, sheet_name, required_cols):
        headers = self._read_headers(ws, sheet_name, required_cols)
        if headers is None:
            return []
        rows = []
        for row_num, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            vals = [str(v).strip() if v is not None else '' for v in row]
            if all(v == '' for v in vals):
                continue
            d = {col: (vals[idx] if idx < len(vals) else '') for col, idx in headers.items()}
            d['_row'] = row_num
            rows.append(d)
        return rows

    def _parse_single_row(self, ws, sheet_name, required_cols):
        headers = self._read_headers(ws, sheet_name, required_cols)
        if headers is None:
            return {}
        data_rows = list(ws.iter_rows(min_row=2, max_row=2, values_only=True))
        if not data_rows:
            return {}
        vals = [str(v).strip() if v is not None else '' for v in data_rows[0]]
        d = {col: (vals[idx] if idx < len(vals) else '') for col, idx in headers.items()}
        d['_row'] = 2
        return d

    # ── Validate ──────────────────────────────────────────────────────────

    def _validate(self):
        self._load_db_lookups()
        self._validate_scenario()
        self._validate_phases()
        self._validate_activities()
        self._validate_answers()
        self._validate_routing()
        self._validate_evaluation()

    def _load_db_lookups(self):
        self._simulations = {s.name: s for s in Simulation.objects.all()}
        self._remote_labs = {l.name: l for l in ExperimentLL.objects.all()}
        self._vr_labs = {v.name: v for v in VRARExperiment.objects.all()}
        self._activity_types = {at.name: at for at in ActivityType.objects.all()}

    def _validate_scenario(self):
        if not self.scenario_data:
            self._add_error('Scenario', 2, 'Name', 'Scenario data row is missing.')
            return
        name = self.scenario_data.get('Name', '').strip()
        if not name:
            self._add_error('Scenario', 2, 'Name', 'Scenario Name is required.')
        elif Scenario.objects.filter(name=name).exists():
            self._add_error('Scenario', 2, 'Name', f'A scenario named "{name}" already exists.')
        vis = self.scenario_data.get('Visibility', '').strip().lower()
        if vis and vis not in ('private', 'org', 'public'):
            self._add_error('Scenario', 2, 'Visibility', 'Visibility must be "private", "org", or "public".')
        for col in ('Age Min', 'Age Max', 'Suggested Time (min)'):
            v = self.scenario_data.get(col, '').strip()
            if v and _to_int(v) is None:
                self._add_error('Scenario', 2, col, f'"{col}" must be a whole number.')

    def _validate_phases(self):
        if not self.phases:
            self._add_error('Phases', 2, 'Phase Name', 'At least one phase is required.')

    def _validate_activities(self):
        if not self.activities:
            self._add_error('Activities', 2, 'Activity Name', 'At least one activity is required.')
            return
        phase_names = {p['Phase Name'] for p in self.phases if p.get('Phase Name')}
        seen = set()
        for a in self.activities:
            row = a['_row']
            name = a.get('Activity Name', '').strip()
            if not name:
                self._add_error('Activities', row, 'Activity Name', 'Activity Name is required.')
            elif name in seen:
                self._add_error('Activities', row, 'Activity Name', f'Duplicate activity name: "{name}".')
            else:
                seen.add(name)
            if not a.get('Text', '').strip():
                self._add_error('Activities', row, 'Text', 'Text is required.')
            phase = a.get('Phase Name', '').strip()
            if not phase:
                self._add_error('Activities', row, 'Phase Name', 'Phase Name is required.')
            elif phase not in phase_names:
                self._add_error('Activities', row, 'Phase Name', f'Phase "{phase}" not found in Phases sheet.')
            for col in BOOL_COLS_ACTIVITIES:
                v = a.get(col, '').strip().lower()
                if v and v not in ('yes', 'no'):
                    self._add_error('Activities', row, col, f'"{col}" must be "Yes" or "No".')
            if (_to_bool(a.get('Is Primary Evaluation', 'No'))
                    and not _to_bool(a.get('Is Evaluatable', 'No'))):
                self._add_error('Activities', row, 'Is Primary Evaluation',
                                'Is Primary Evaluation = Yes requires Is Evaluatable = Yes.')
            v = a.get('Score Limit', '').strip()
            if v and _to_float(v) is None:
                self._add_error('Activities', row, 'Score Limit', 'Score Limit must be a number.')
            for col, lookup in (
                ('Simulation Name', self._simulations),
                ('Remote Lab Name', self._remote_labs),
                ('VR Lab Name', self._vr_labs),
            ):
                v = a.get(col, '').strip()
                if v and v not in lookup:
                    self._add_error('Activities', row, col, f'"{v}" not found in the database.')
            v = a.get('Activity Type', '').strip()
            if v and v not in self._activity_types:
                self._add_error('Activities', row, 'Activity Type', f'Activity type "{v}" not found in the database.')

    def _validate_answers(self):
        for ans in self.answers:
            row = ans['_row']
            act_name = ans.get('Activity Name', '').strip()
            if act_name not in self.activity_map:
                self._add_error('Answers', row, 'Activity Name',
                                f'Activity "{act_name}" not found in Activities sheet.')
            for col in BOOL_COLS_ANSWERS:
                v = ans.get(col, '').strip().lower()
                if v and v not in ('yes', 'no'):
                    self._add_error('Answers', row, col, f'"{col}" must be "Yes" or "No".')
            v = ans.get('Answer Weight', '').strip()
            if v and _to_int(v) is None:
                self._add_error('Answers', row, 'Answer Weight', 'Answer Weight must be a whole number.')

    def _validate_routing(self):
        seen_pairs = set()
        for r in self.routing:
            row = r['_row']
            src = r.get('Source Activity Name', '').strip()
            if not src:
                self._add_error('Next Activity', row, 'Source Activity Name',
                                'Source Activity Name is required.')
                continue
            if src not in self.activity_map:
                self._add_error('Next Activity', row, 'Source Activity Name',
                                f'Activity "{src}" not found in Activities sheet.')
                continue
            ans_text = r.get('Answer Text', '').strip()
            pair = (src, ans_text)
            if pair in seen_pairs:
                self._add_error('Next Activity', row, 'Answer Text',
                                f'Duplicate routing rule for activity "{src}" / '
                                f'answer "{ans_text or "(default)"}".')
            seen_pairs.add(pair)
            if ans_text and (src, ans_text) not in self.answer_map:
                self._add_error('Next Activity', row, 'Answer Text',
                                f'Answer "{ans_text}" not found for activity "{src}".')
            next_act = r.get('Next Activity Name', '').strip()
            if next_act and next_act not in self.activity_map:
                self._add_error('Next Activity', row, 'Next Activity Name',
                                f'Activity "{next_act}" not found in Activities sheet.')

    def _validate_evaluation(self):
        evaluatable = {
            a['Activity Name']
            for a in self.activities
            if _to_bool(a.get('Is Evaluatable', 'No'))
        }
        ev_seen = set()
        for ev in self.evaluation:
            row = ev['_row']
            primary = ev.get('Primary Activity Name', '').strip()
            if not primary:
                self._add_error('Evaluation', row, 'Primary Activity Name',
                                'Primary Activity Name is required.')
                continue
            if primary not in self.activity_map:
                self._add_error('Evaluation', row, 'Primary Activity Name',
                                f'Activity "{primary}" not found in Activities sheet.')
                continue
            if primary not in evaluatable:
                self._add_error('Evaluation', row, 'Primary Activity Name',
                                f'Activity "{primary}" is not marked Is Evaluatable = Yes.')
            ev_seen.add(primary)
            grouped_raw = ev.get('Grouped Activities', '').strip()
            if not grouped_raw:
                self._add_error('Evaluation', row, 'Grouped Activities',
                                'Grouped Activities is required.')
            else:
                for g in [x.strip() for x in grouped_raw.split(',') if x.strip()]:
                    if g not in self.activity_map:
                        self._add_error('Evaluation', row, 'Grouped Activities',
                                        f'Activity "{g}" not found in Activities sheet.')
            for col in ('High Branch Activity', 'Mid Branch Activity', 'Low Branch Activity'):
                v = ev.get(col, '').strip()
                if v and v not in self.activity_map:
                    self._add_error('Evaluation', row, col,
                                    f'Activity "{v}" not found in Activities sheet.')
        for name in evaluatable:
            if name not in ev_seen:
                a = self.activity_map[name]
                self._add_error('Activities', a.get('_row', '?'), 'Is Evaluatable',
                                f'Activity "{name}" is marked Is Evaluatable = Yes '
                                f'but has no row in the Evaluation sheet.')

    def _create(self):
        raise NotImplementedError('Implemented in Task 2')
```

- [ ] **Step 5: Run tests — verify they all pass**

```
python manage.py test authoringtool.tests.ImporterValidationTest --settings=faithDev.settings_test -v 2
```

Expected: All 12 tests pass.

- [ ] **Step 6: Commit**

```bash
git add Trust-AI-Platform/requirements.txt Trust-AI-Platform/authoringtool/importer.py Trust-AI-Platform/authoringtool/tests.py
git commit -m "feat: add ScenarioImporter parse+validate; add openpyxl+markdown2 deps"
```

---

### Task 2: `importer.py` — Create Stage

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/importer.py` (replace `_create` stub)
- Modify: `Trust-AI-Platform/authoringtool/tests.py` (add `ImporterCreationTest`)

**Interfaces:**
- Consumes: `ScenarioImporter` from Task 1 (parse + validate already on the class)
- Produces: `ScenarioImporter.run()` returns a saved `Scenario` instance on success

---

- [ ] **Step 1: Write failing tests for creation**

Add to `authoringtool/tests.py`:

```python
class ImporterCreationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('teacher2', password='pass')

    def _import(self, **kwargs):
        from authoringtool.importer import ScenarioImporter
        buf = make_xlsx(**kwargs)
        importer = ScenarioImporter(buf, self.user)
        return importer.run()

    def test_creates_scenario(self):
        scenario, errors = self._import(scenario_name='Created Scenario')
        self.assertEqual(errors, [])
        self.assertIsNotNone(scenario)
        self.assertTrue(Scenario.objects.filter(name='Created Scenario').exists())

    def test_creates_phases(self):
        scenario, errors = self._import(
            phases=[['Phase A', 'First phase', ''], ['Phase B', '', '']],
            activities=[
                ['Act 1', 'Phase A', 'Hello', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
                ['Act 2', 'Phase B', 'World', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
            ],
        )
        self.assertEqual(errors, [])
        self.assertEqual(Phase.objects.filter(scenario=scenario).count(), 2)
        phase_names = list(Phase.objects.filter(scenario=scenario).values_list('name', flat=True))
        self.assertIn('Phase A', phase_names)
        self.assertIn('Phase B', phase_names)

    def test_creates_activities_with_correct_phase(self):
        scenario, errors = self._import(
            activities=[['MyAct', 'Phase 1', '**Bold text**', '', '', 'No', 'No', 'No', '', '', '', '', '', '']],
        )
        self.assertEqual(errors, [])
        act = Activity.objects.get(scenario=scenario)
        self.assertEqual(act.name, 'MyAct')
        self.assertIn('<strong>', act.text)  # markdown converted
        self.assertNotIn('<strong>', act.plain_text)  # tags stripped

    def test_creates_answers_with_feedback(self):
        scenario, errors = self._import(
            answers=[
                ['Act 1', 'Option A', 'Yes', '2', '', '', 'Great job!', '', ''],
                ['Act 1', 'Option B', 'No', '0', '', '', '', '', ''],
            ],
        )
        self.assertEqual(errors, [])
        act = Activity.objects.get(scenario=scenario)
        self.assertEqual(act.answers.count(), 2)
        correct = act.answers.get(is_correct=True)
        self.assertEqual(correct.answer_weight, 2)
        self.assertTrue(correct.feedbacks.exists())
        self.assertIn('Great', correct.feedbacks.first().text)

    def test_creates_next_question_logic_default(self):
        scenario, errors = self._import(
            activities=[
                ['Act 1', 'Phase 1', 'First', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
                ['Act 2', 'Phase 1', 'Second', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
            ],
            routing=[['Act 1', '', 'Act 2']],
        )
        self.assertEqual(errors, [])
        act1 = Activity.objects.get(scenario=scenario, name='Act 1')
        act2 = Activity.objects.get(scenario=scenario, name='Act 2')
        logic = NextQuestionLogic.objects.get(activity=act1, answer__isnull=True)
        self.assertEqual(logic.next_activity, act2)

    def test_creates_next_question_logic_per_answer(self):
        scenario, errors = self._import(
            activities=[
                ['Q', 'Phase 1', 'Question?', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
                ['Good', 'Phase 1', 'Well done!', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
                ['Bad', 'Phase 1', 'Try again!', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
            ],
            answers=[
                ['Q', 'Correct', 'Yes', '1', '', '', '', '', ''],
                ['Q', 'Wrong', 'No', '0', '', '', '', '', ''],
            ],
            routing=[
                ['Q', 'Correct', 'Good'],
                ['Q', 'Wrong', 'Bad'],
            ],
        )
        self.assertEqual(errors, [])
        q_act = Activity.objects.get(scenario=scenario, name='Q')
        correct_ans = q_act.answers.get(is_correct=True)
        logic = NextQuestionLogic.objects.get(activity=q_act, answer=correct_ans)
        self.assertEqual(logic.next_activity.name, 'Good')

    def test_created_by_is_set(self):
        scenario, errors = self._import()
        self.assertEqual(errors, [])
        self.assertEqual(scenario.created_by, self.user)
        self.assertEqual(scenario.updated_by, self.user)
```

- [ ] **Step 2: Run failing tests**

```
python manage.py test authoringtool.tests.ImporterCreationTest --settings=faithDev.settings_test -v 2
```

Expected: `NotImplementedError: Implemented in Task 2`

- [ ] **Step 3: Replace `_create` stub in `importer.py`**

In `Trust-AI-Platform/authoringtool/importer.py`, replace the `_create` method:

```python
    def _create(self):
        from django.db import transaction
        from django.utils.html import strip_tags
        from .models import (
            Phase, Activity, Answer, AnswerFeedback,
            NextQuestionLogic, QuestionBunch, EvQuestionBranching,
        )

        with transaction.atomic():
            # 1. Scenario
            scenario = Scenario.objects.create(
                name=self.scenario_data['Name'].strip(),
                description=_md(self.scenario_data.get('Description', '')),
                learning_goals=_md(self.scenario_data.get('Learning Goals', '')),
                language=self.scenario_data.get('Language', ''),
                subject_domains=self.scenario_data.get('Subject Domains', ''),
                suggested_learning_time=_to_int(self.scenario_data.get('Suggested Time (min)', '')) or None,
                video_url=self.scenario_data.get('Video URL', '') or None,
                visibility_status=self.scenario_data.get('Visibility', '').strip().lower() or 'private',
                created_by=self.user,
                updated_by=self.user,
            )

            # 2. Phases (row order preserved)
            phase_obj_map = {}
            for ph in self.phases:
                obj = Phase.objects.create(
                    name=ph['Phase Name'],
                    description=ph.get('Description', ''),
                    video_url=ph.get('Video URL', '') or None,
                    scenario=scenario,
                    created_by=self.user,
                    updated_by=self.user,
                )
                phase_obj_map[ph['Phase Name']] = obj

            # 3. Activities (row order preserved)
            activity_obj_map = {}
            for act in self.activities:
                html = _md(act.get('Text', ''))
                obj = Activity.objects.create(
                    name=act['Activity Name'],
                    text=html,
                    plain_text=strip_tags(html),
                    is_evaluatable=_to_bool(act.get('Is Evaluatable', 'No')),
                    is_primary_ev=_to_bool(act.get('Is Primary Evaluation', 'No')),
                    must_wait=_to_bool(act.get('Must Wait', 'No')),
                    score_limit=_to_float(act.get('Score Limit', '')) or 0.0,
                    helper=act.get('Helper', ''),
                    scenario=scenario,
                    phase=phase_obj_map[act['Phase Name']],
                    activity_type=self._activity_types.get(act.get('Activity Type', '').strip()),
                    simulation=self._simulations.get(act.get('Simulation Name', '').strip()),
                    experiment_ll=self._remote_labs.get(act.get('Remote Lab Name', '').strip()),
                    vr_ar_experiment=self._vr_labs.get(act.get('VR Lab Name', '').strip()),
                    created_by=self.user,
                    updated_by=self.user,
                )
                activity_obj_map[act['Activity Name']] = obj

            # 4. Answers + AnswerFeedback
            answer_obj_map = {}
            for ans in self.answers:
                ans_html = _md(ans.get('Answer Text', ''))
                ans_obj = Answer.objects.create(
                    activity=activity_obj_map[ans['Activity Name']],
                    text=ans_html,
                    is_correct=_to_bool(ans.get('Is Correct', 'No')),
                    answer_weight=_to_int(ans.get('Answer Weight', '')) or 0,
                    vid_url=ans.get('Video URL', '') or None,
                    created_by=self.user,
                    updated_by=self.user,
                )
                answer_obj_map[(ans['Activity Name'], ans['Answer Text'])] = ans_obj
                fb_text = ans.get('Feedback Text', '').strip()
                if fb_text:
                    AnswerFeedback.objects.create(
                        answer=ans_obj,
                        text=_md(fb_text),
                        vid_url=ans.get('Feedback Video URL', '') or None,
                        created_by=self.user,
                        updated_by=self.user,
                    )

            # 5. NextQuestionLogic
            for r in self.routing:
                src = activity_obj_map[r['Source Activity Name'].strip()]
                ans_text = r.get('Answer Text', '').strip()
                ans_obj = (
                    answer_obj_map.get((r['Source Activity Name'].strip(), ans_text))
                    if ans_text else None
                )
                next_name = r.get('Next Activity Name', '').strip()
                NextQuestionLogic.objects.create(
                    activity=src,
                    answer=ans_obj,
                    next_activity=activity_obj_map.get(next_name),
                )

            # 6. QuestionBunch + EvQuestionBranching (Postgres ArrayField — skipped in SQLite tests)
            for ev in self.evaluation:
                primary = activity_obj_map[ev['Primary Activity Name'].strip()]
                grouped = [x.strip() for x in ev['Grouped Activities'].split(',') if x.strip()]
                grouped_ids = [activity_obj_map[n].id for n in grouped]
                QuestionBunch.objects.create(
                    activity_primary=primary,
                    activity_ids=grouped_ids,
                )
                EvQuestionBranching.objects.create(
                    activity=primary,
                    next_question_on_high=activity_obj_map.get(ev.get('High Branch Activity', '').strip()),
                    next_question_on_high_feedback=_md(ev.get('High Branch Feedback', '')),
                    next_question_on_mid=activity_obj_map.get(ev.get('Mid Branch Activity', '').strip()),
                    next_question_on_mid_feedback=_md(ev.get('Mid Branch Feedback', '')),
                    next_question_on_low=activity_obj_map.get(ev.get('Low Branch Activity', '').strip()),
                    next_question_on_low_feedback=_md(ev.get('Low Branch Feedback', '')),
                )

        return scenario
```

- [ ] **Step 4: Run tests — all pass**

```
python manage.py test authoringtool.tests.ImporterCreationTest --settings=faithDev.settings_test -v 2
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Trust-AI-Platform/authoringtool/importer.py Trust-AI-Platform/authoringtool/tests.py
git commit -m "feat: add ScenarioImporter create stage"
```

---

### Task 3: Template Generator + Views + URLs

**Files:**
- Create: `Trust-AI-Platform/authoringtool/template_generator.py`
- Modify: `Trust-AI-Platform/authoringtool/views.py` (append 2 views)
- Modify: `Trust-AI-Platform/authoringtool/urls.py` (append 2 patterns)
- Modify: `Trust-AI-Platform/authoringtool/tests.py` (add 2 test classes)

**Interfaces:**
- Consumes: `ScenarioImporter` from Tasks 1+2; `group_required` decorator from `views.py`
- Produces:
  - `GET /authoringtool/template/download/` → `.xlsx` file download
  - `POST /authoringtool/import/` → JSON `{success, errors}` or `{success, scenario_id, redirect}`

---

- [ ] **Step 1: Write failing tests**

Add to `authoringtool/tests.py`:

```python
from django.core.files.uploadedfile import SimpleUploadedFile


class TemplateDownloadTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('teacher3', password='pass')
        g = Group.objects.create(name='teachers')
        self.user.groups.add(g)
        self.client.login(username='teacher3', password='pass')

    def test_download_requires_login(self):
        self.client.logout()
        r = self.client.get(reverse('download_template'))
        self.assertNotEqual(r.status_code, 200)

    def test_download_returns_xlsx(self):
        r = self.client.get(reverse('download_template'))
        self.assertEqual(r.status_code, 200)
        self.assertIn('spreadsheetml', r['Content-Type'])

    def test_download_has_all_sheets(self):
        import openpyxl, io
        r = self.client.get(reverse('download_template'))
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        for sheet in ['README', 'Scenario', 'Phases', 'Activities', 'Answers', 'Next Activity', 'Evaluation']:
            self.assertIn(sheet, wb.sheetnames)


class ImportViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('teacher4', password='pass')
        g, _ = Group.objects.get_or_create(name='teachers')
        self.user.groups.add(g)
        self.client.login(username='teacher4', password='pass')

    def _upload(self, buf, filename='template.xlsx'):
        f = SimpleUploadedFile(filename, buf.read(),
                               content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        return self.client.post(reverse('import_scenario'), {'template_file': f})

    def test_requires_login(self):
        self.client.logout()
        buf = make_xlsx(scenario_name='X')
        r = self._upload(buf)
        self.assertNotEqual(r.status_code, 200)

    def test_valid_file_creates_scenario(self):
        buf = make_xlsx(scenario_name='Imported Scenario')
        r = self._upload(buf)
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertTrue(Scenario.objects.filter(name='Imported Scenario').exists())

    def test_valid_file_returns_redirect_url(self):
        buf = make_xlsx(scenario_name='Redir Test')
        r = self._upload(buf)
        data = json.loads(r.content)
        self.assertIn('redirect', data)
        self.assertIn(str(data['scenario_id']), data['redirect'])

    def test_invalid_file_returns_errors(self):
        buf = make_xlsx(scenario_name='')  # missing required name
        r = self._upload(buf)
        data = json.loads(r.content)
        self.assertFalse(data['success'])
        self.assertIsInstance(data['errors'], list)
        self.assertGreater(len(data['errors']), 0)

    def test_non_xlsx_rejected(self):
        f = SimpleUploadedFile('file.csv', b'name,val', content_type='text/csv')
        r = self.client.post(reverse('import_scenario'), {'template_file': f})
        data = json.loads(r.content)
        self.assertFalse(data['success'])

    def test_get_not_allowed(self):
        r = self.client.get(reverse('import_scenario'))
        self.assertEqual(r.status_code, 405)

    def test_duplicate_scenario_name_returns_error(self):
        Scenario.objects.create(name='Dupe', created_by=self.user, updated_by=self.user)
        buf = make_xlsx(scenario_name='Dupe')
        r = self._upload(buf)
        data = json.loads(r.content)
        self.assertFalse(data['success'])
        self.assertTrue(any('already exists' in e['message'] for e in data['errors']))
```

- [ ] **Step 2: Run tests — they fail**

```
python manage.py test authoringtool.tests.TemplateDownloadTest authoringtool.tests.ImportViewTest --settings=faithDev.settings_test -v 2
```

Expected: `NoReverseMatch` for `download_template` and `import_scenario`

- [ ] **Step 3: Create `authoringtool/template_generator.py`**

Create `Trust-AI-Platform/authoringtool/template_generator.py`:

```python
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

_HEADER_FONT = Font(bold=True, color='FFFFFF')
_HEADER_FILL = PatternFill('solid', fgColor='1D4ED8')
_REQ_FILL = PatternFill('solid', fgColor='DC2626')
_BOOL_DV = DataValidation(type='list', formula1='"Yes,No"', showDropDown=False, showErrorMessage=True)


def _write_sheet(wb, title, columns, bool_cols=(), example_rows=()):
    """columns: list of (name, required_bool)"""
    ws = wb.create_sheet(title)
    for col_idx, (name, required) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=f'{name}*' if required else name)
        cell.font = _HEADER_FONT
        cell.fill = _REQ_FILL if required else _HEADER_FILL
        cell.alignment = Alignment(horizontal='center')
        ws.column_dimensions[cell.column_letter].width = max(len(name) + 4, 16)

    col_name_to_idx = {name: i + 1 for i, (name, _) in enumerate(columns)}
    dv = DataValidation(type='list', formula1='"Yes,No"', showDropDown=False)
    ws.add_data_validation(dv)
    for bool_col in bool_cols:
        if bool_col in col_name_to_idx:
            idx = col_name_to_idx[bool_col]
            letter = ws.cell(row=1, column=idx).column_letter
            dv.sqref = f'{letter}2:{letter}200'

    for row_data in example_rows:
        ws.append(row_data)

    return ws


def _write_readme(ws):
    ws['A1'] = 'SCENARIO TEMPLATE — INSTRUCTIONS'
    ws['A1'].font = Font(bold=True, size=14)
    lines = [
        '',
        'SHEETS AND THEIR PURPOSE',
        '  Scenario  — One row of scenario metadata (name, description, etc.)',
        '  Phases    — One row per phase; row order = phase order in the scenario',
        '  Activities — One row per activity; row order = activity order within each phase',
        '  Answers   — One row per answer (multiple rows per activity are fine)',
        '  Next Activity — One row per routing rule (which activity comes after which)',
        '  Evaluation — One row per evaluatable activity (scoring bunches + score branches)',
        '',
        'REFERENCING RULES',
        '  Activities reference Phases by Phase Name — must match exactly.',
        '  Answers, Next Activity, and Evaluation reference Activities by Activity Name.',
        '  Activity names must be unique within the file.',
        '  TIP: Copy-paste names from the Activities sheet to avoid typos.',
        '',
        'TEXT FIELDS',
        '  Activity Text and Answer Text support Markdown formatting:',
        '    **bold**   _italic_   # Heading   - bullet list   [Link](https://...)',
        '  Markdown is converted to HTML automatically on import.',
        '',
        'BOOLEAN FIELDS (Is Evaluatable, Is Correct, etc.)',
        '  Use the dropdown: Yes or No (case-insensitive).',
        '',
        'NEXT ACTIVITY SHEET',
        '  Leave "Answer Text" blank for an unconditional (default) next activity.',
        '  Leave "Next Activity Name" blank to end the scenario at that activity.',
        '  Example:',
        '    Source Activity | Answer Text | Next Activity',
        '    Welcome         |             | Quiz 1         (default route)',
        '    Quiz 1          | Correct     | Well Done      (per-answer route)',
        '    Quiz 1          | Wrong       | Try Again',
        '',
        'EVALUATION SHEET',
        '  Primary Activity Name: an activity with Is Evaluatable = Yes.',
        '  Grouped Activities: comma-separated activity names that form the scoring group.',
        '    Include the primary activity itself in this list.',
        '  High/Mid/Low Branch Activity: where students go based on their score.',
        '',
        'COLUMNS MARKED WITH *',
        '  Red headers are required. Leave others blank if not needed.',
    ]
    for i, line in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=line)
    ws.column_dimensions['A'].width = 80


def generate_blank_template():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'README'
    _write_readme(ws)

    _write_sheet(wb, 'Scenario', [
        ('Name', True), ('Description', False), ('Learning Goals', False),
        ('Language', False), ('Subject Domains', False), ('Age Min', False),
        ('Age Max', False), ('Suggested Time (min)', False), ('Video URL', False),
        ('Visibility', False),
    ], example_rows=[
        ['My Scenario', 'A brief description', 'Students will learn...', 'English',
         'Physics,STEM', '14', '18', '90', '', 'private'],
    ])

    _write_sheet(wb, 'Phases', [
        ('Phase Name', True), ('Description', False), ('Video URL', False),
    ], example_rows=[
        ['Introduction', 'Students explore the topic', ''],
        ['Experiment', '', ''],
        ['Evaluation', '', ''],
    ])

    _write_sheet(wb, 'Activities', [
        ('Activity Name', True), ('Phase Name', True), ('Text', True),
        ('Activity Type', False), ('Helper', False), ('Is Evaluatable', False),
        ('Is Primary Evaluation', False), ('Must Wait', False), ('Score Limit', False),
        ('Simulation Name', False), ('Remote Lab Name', False), ('VR Lab Name', False),
        ('Image URL', False), ('Video URL', False),
    ], bool_cols=['Is Evaluatable', 'Is Primary Evaluation', 'Must Wait'], example_rows=[
        ['Welcome', 'Introduction', 'Welcome to this scenario! **Read carefully.**',
         '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
        ['Quiz 1', 'Evaluation', 'What is the boiling point of water?',
         'Question', '', 'Yes', 'Yes', 'No', '', '', '', '', '', ''],
    ])

    _write_sheet(wb, 'Answers', [
        ('Activity Name', True), ('Answer Text', True), ('Is Correct', False),
        ('Answer Weight', False), ('Image URL', False), ('Video URL', False),
        ('Feedback Text', False), ('Feedback Image URL', False), ('Feedback Video URL', False),
    ], bool_cols=['Is Correct'], example_rows=[
        ['Quiz 1', '100°C', 'Yes', '1', '', '', 'Correct! Water boils at 100°C.', '', ''],
        ['Quiz 1', '50°C', 'No', '0', '', '', 'Not quite. Try again!', '', ''],
    ])

    _write_sheet(wb, 'Next Activity', [
        ('Source Activity Name', True), ('Answer Text', False), ('Next Activity Name', False),
    ], example_rows=[
        ['Welcome', '', 'Quiz 1'],
        ['Quiz 1', '100°C', 'Well Done'],
        ['Quiz 1', '50°C', 'Try Again'],
    ])

    _write_sheet(wb, 'Evaluation', [
        ('Primary Activity Name', True), ('Grouped Activities', True),
        ('High Branch Activity', False), ('High Branch Feedback', False),
        ('Mid Branch Activity', False), ('Mid Branch Feedback', False),
        ('Low Branch Activity', False), ('Low Branch Feedback', False),
    ], example_rows=[
        ['Quiz 1', 'Quiz 1,Quiz 2', 'Advanced', 'Great job!', 'Standard', 'Good effort!', 'Review', 'Keep trying!'],
    ])

    return wb
```

- [ ] **Step 4: Add the two views to `authoringtool/views.py`**

Append at the end of the file (after all existing views):

```python
@group_required('teachers')
def download_template(request):
    import io
    from .template_generator import generate_blank_template
    wb = generate_blank_template()
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    response = HttpResponse(
        buf.read(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    response['Content-Disposition'] = 'attachment; filename="scenario_template.xlsx"'
    return response


@require_POST
@group_required('teachers')
def import_scenario(request):
    from .importer import ScenarioImporter
    uploaded = request.FILES.get('template_file')
    if not uploaded:
        return JsonResponse({'success': False, 'errors': [
            {'sheet': 'File', 'row': 0, 'column': '-', 'message': 'No file uploaded.'}
        ]})
    if not uploaded.name.endswith('.xlsx'):
        return JsonResponse({'success': False, 'errors': [
            {'sheet': 'File', 'row': 0, 'column': '-', 'message': 'Only .xlsx files are supported.'}
        ]})
    importer = ScenarioImporter(uploaded, request.user)
    scenario, errors = importer.run()
    if errors:
        return JsonResponse({'success': False, 'errors': errors})
    return JsonResponse({
        'success': True,
        'scenario_id': scenario.id,
        'redirect': reverse('viewScenario', args=[scenario.id]),
    })
```

- [ ] **Step 5: Add URL patterns to `authoringtool/urls.py`**

Append to the `urlpatterns` list:

```python
    path('template/download/', views.download_template, name='download_template'),
    path('import/', views.import_scenario, name='import_scenario'),
```

- [ ] **Step 6: Run tests — all pass**

```
python manage.py test authoringtool.tests.TemplateDownloadTest authoringtool.tests.ImportViewTest --settings=faithDev.settings_test -v 2
```

Expected: All 10 tests pass.

- [ ] **Step 7: Run all authoringtool tests to catch regressions**

```
python manage.py test authoringtool --settings=faithDev.settings_test -v 2
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add Trust-AI-Platform/authoringtool/template_generator.py \
        Trust-AI-Platform/authoringtool/views.py \
        Trust-AI-Platform/authoringtool/urls.py \
        Trust-AI-Platform/authoringtool/tests.py
git commit -m "feat: add template download + import_scenario views and URLs"
```

---

### Task 4: UI Modal in `scenarios.html`

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/scenarios.html`

**Interfaces:**
- Consumes: `{% url 'import_scenario' %}`, `{% url 'download_template' %}`, `{% url 'createScenario' %}`
- No new Python code. Visual test only.

---

- [ ] **Step 1: Change "Create New" button to a modal trigger**

Find in `scenarios.html` (~line 155):
```html
<a href="{% url 'createScenario' %}" class="hero-btn-solid">
  <i class="bi bi-plus-lg"></i> Create New
</a>
```

Replace with:
```html
<button class="hero-btn-solid" data-bs-toggle="modal" data-bs-target="#createScenarioModal">
  <i class="bi bi-plus-lg"></i> Create New
</button>
```

- [ ] **Step 2: Add the modal HTML**

Add the following block immediately before the closing `</div>` of the main container (or just before `{% endblock %}`). Find the end of the page body and insert:

```html
<!-- ── Create Scenario Modal ── -->
<div class="modal fade" id="createScenarioModal" tabindex="-1" aria-labelledby="createScenarioModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-dialog-centered" style="max-width:520px;">
    <div class="modal-content" style="border-radius:14px;overflow:hidden;">
      <div class="modal-header" style="background:#1D4ED8;padding:16px 24px;">
        <h5 class="modal-title" id="createScenarioModalLabel" style="color:#fff;font-size:16px;font-weight:700;margin:0;">
          <i class="bi bi-plus-circle me-2"></i>Create New Scenario
        </h5>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
      </div>

      <!-- Tab navigation -->
      <ul class="nav nav-tabs px-3 pt-3" id="createScenarioTabs" role="tablist" style="border-bottom:1px solid #e5e7eb;">
        <li class="nav-item" role="presentation">
          <button class="nav-link active" id="upload-tab" data-bs-toggle="tab" data-bs-target="#uploadPane"
                  type="button" role="tab" style="font-size:14px;font-weight:600;">
            <i class="bi bi-upload me-1"></i>Upload Template
          </button>
        </li>
        <li class="nav-item" role="presentation">
          <button class="nav-link" id="manual-tab" data-bs-toggle="tab" data-bs-target="#manualPane"
                  type="button" role="tab" style="font-size:14px;font-weight:600;">
            <i class="bi bi-pencil-square me-1"></i>Manual Setup
          </button>
        </li>
      </ul>

      <div class="tab-content">
        <!-- Upload Template pane -->
        <div class="tab-pane fade show active" id="uploadPane" role="tabpanel">
          <div class="modal-body" style="padding:24px;">
            <!-- Error list (hidden by default) -->
            <div id="importErrors" class="d-none mb-3">
              <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px 16px;max-height:240px;overflow-y:auto;">
                <p style="font-size:13px;font-weight:700;color:#dc2626;margin:0 0 8px;">
                  <i class="bi bi-exclamation-circle me-1"></i>Fix these errors and re-upload:
                </p>
                <ul id="importErrorList" style="font-size:12.5px;color:#7f1d1d;margin:0;padding-left:16px;"></ul>
              </div>
            </div>

            <!-- Drop zone -->
            <label for="templateFileInput"
                   style="display:block;border:2px dashed #93c5fd;border-radius:10px;padding:32px 16px;text-align:center;cursor:pointer;background:#eff6ff;transition:background 0.15s;"
                   id="dropZoneLabel">
              <i class="bi bi-file-earmark-excel" style="font-size:2rem;color:#2563eb;display:block;margin-bottom:8px;"></i>
              <span style="font-size:14px;color:#1e40af;font-weight:600;" id="dropZoneText">
                Drop your .xlsx file here or click to browse
              </span>
            </label>
            <input type="file" id="templateFileInput" accept=".xlsx" class="d-none">

            <!-- Download link -->
            <p class="mt-3 mb-0" style="font-size:13px;text-align:center;color:#6b7280;">
              <a href="{% url 'download_template' %}" style="color:#2563eb;font-weight:600;">
                <i class="bi bi-download me-1"></i>Download blank template
              </a>
              &nbsp;·&nbsp; Fill it in, then upload it here.
            </p>
          </div>
          <div class="modal-footer" style="border-top:1px solid #f3f4f6;padding:12px 24px;justify-content:space-between;">
            <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Cancel</button>
            <button type="button" class="btn btn-primary btn-sm" id="importBtn" style="min-width:100px;">
              <span id="importBtnText"><i class="bi bi-upload me-1"></i>Import</span>
              <span id="importBtnSpinner" class="d-none">
                <span class="spinner-border spinner-border-sm me-1"></span>Importing…
              </span>
            </button>
          </div>
        </div>

        <!-- Manual Setup pane -->
        <div class="tab-pane fade" id="manualPane" role="tabpanel">
          <div class="modal-body" style="padding:32px 24px;text-align:center;">
            <i class="bi bi-pencil-square" style="font-size:2.5rem;color:#6b7280;display:block;margin-bottom:12px;"></i>
            <p style="font-size:14px;color:#374151;margin-bottom:20px;">
              Build your scenario step-by-step using the authoring tool.
            </p>
            <a href="{% url 'createScenario' %}" class="btn btn-primary btn-sm">
              <i class="bi bi-arrow-right me-1"></i>Start Manual Setup
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Add the import JS**

Add the following `<script>` block just before `{% endblock %}` (or at the bottom of the existing inline `<script>` block):

```html
<script>
(function () {
  const fileInput = document.getElementById('templateFileInput');
  const dropLabel = document.getElementById('dropZoneLabel');
  const dropText  = document.getElementById('dropZoneText');
  const importBtn = document.getElementById('importBtn');
  const btnText   = document.getElementById('importBtnText');
  const spinner   = document.getElementById('importBtnSpinner');
  const errBox    = document.getElementById('importErrors');
  const errList   = document.getElementById('importErrorList');

  if (!fileInput) return;  // guard: this script only runs on the scenarios page

  // Show selected filename in drop zone
  fileInput.addEventListener('change', function () {
    if (fileInput.files.length) {
      dropText.textContent = fileInput.files[0].name;
      dropLabel.style.background = '#d1fae5';
      dropLabel.style.borderColor = '#6ee7b7';
    }
  });

  // Reset errors when modal opens
  document.getElementById('createScenarioModal').addEventListener('show.bs.modal', function () {
    errBox.classList.add('d-none');
    errList.innerHTML = '';
    fileInput.value = '';
    dropText.textContent = 'Drop your .xlsx file here or click to browse';
    dropLabel.style.background = '#eff6ff';
    dropLabel.style.borderColor = '#93c5fd';
  });

  importBtn.addEventListener('click', function () {
    errBox.classList.add('d-none');
    errList.innerHTML = '';

    const file = fileInput.files[0];
    if (!file) {
      _showErrors([{sheet: 'File', row: 0, column: '-', message: 'Please select a file first.'}]);
      return;
    }

    // Show spinner
    btnText.classList.add('d-none');
    spinner.classList.remove('d-none');
    importBtn.disabled = true;

    const fd = new FormData();
    fd.append('template_file', file);
    fd.append('csrfmiddlewaretoken', '{{ csrf_token }}');

    fetch('{% url "import_scenario" %}', {method: 'POST', body: fd})
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          window.location.href = data.redirect;
        } else {
          _showErrors(data.errors || []);
        }
      })
      .catch(() => {
        _showErrors([{sheet: 'Network', row: 0, column: '-', message: 'Request failed. Please try again.'}]);
      })
      .finally(() => {
        btnText.classList.remove('d-none');
        spinner.classList.add('d-none');
        importBtn.disabled = false;
      });
  });

  function _showErrors(errors) {
    errList.innerHTML = '';
    errors.forEach(function (e) {
      const li = document.createElement('li');
      li.style.marginBottom = '4px';
      const loc = e.row > 0
        ? `Sheet "${e.sheet}", row ${e.row}, column "${e.column}"`
        : `Sheet "${e.sheet}"`;
      li.textContent = `${loc}: ${e.message}`;
      errList.appendChild(li);
    });
    errBox.classList.remove('d-none');
  }
}());
</script>
```

- [ ] **Step 4: Also update the "create a new one" inline link**

Find (~line 311):
```html
<a href="{% url 'createScenario' %}" style="color:#1a56db;">create a new one</a>
```

Replace with:
```html
<a href="#" style="color:#1a56db;" data-bs-toggle="modal" data-bs-target="#createScenarioModal">create a new one</a>
```

- [ ] **Step 5: Manual smoke test**

Start the development server and:
1. Navigate to `/authoringtool/scenarios/`
2. Click "Create New" — modal opens with two tabs
3. Click "Download blank template" — `.xlsx` downloads with 7 sheets and example rows
4. Switch to "Manual Setup" tab — "Start Manual Setup" button goes to `/authoringtool/scenarios/createScenario/`
5. Fill in the template, save, upload — spinner appears, then redirects to the new scenario
6. Upload a file with a blank Name field — error list appears with the correct message

- [ ] **Step 6: Run full test suite to confirm no regressions**

```
python manage.py test authoringtool accounts --settings=faithDev.settings_test -v 2
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add Trust-AI-Platform/authoringtool/templates/authoringtool/scenarios.html
git commit -m "feat: add create scenario modal with template upload and manual setup tabs"
```
