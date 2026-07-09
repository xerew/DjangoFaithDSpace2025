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
