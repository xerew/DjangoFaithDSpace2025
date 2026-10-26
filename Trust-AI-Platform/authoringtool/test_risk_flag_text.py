import csv
import importlib
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from django.apps import apps as django_apps
from django.contrib.auth.models import Group, User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import translation

from authoringtool.models import Activity, ActivityFlag, ActivityType, Phase, Scenario
from authoringtool.risk_flag_text import (
    FLAG_TYPES, REASONS, display_flag, parse_reason, reason_text,
)
from authoringtool.tasks import calculate_activities_in_risk, get_data_insight
from authoringtool.utils import get_scenario_evidence_cache_paths

# Every FIXTURES entry below was captured from the code as it was before this
# change: the Reason from running calculate_activities_in_risk on the
# SCENARIOS_ROWS metrics, the insight from running get_data_insight on it.


# activity -> (type, [(category, % wrong, avg time)])
SCENARIOS_ROWS = {
    'Q v-shape': ('Question', [('High', 10.0, 20.0), ('Moderate', 60.0, 21.0), ('Low', 20.0, 22.0)]),
    'Q reversal': ('Question', [('High', 50.0, 20.0), ('Moderate', 60.0, 21.0), ('Low', 40.0, 22.0)]),
    'Q high worse': ('Question', [('High', 80.0, 20.0), ('Moderate', 50.0, 21.0), ('Low', 60.0, 22.0)]),
    'Q time wrong': ('Question', [('High', 30.0, 15.0), ('Moderate', 60.0, 20.0), ('Low', 100.0, 45.5)]),
    'Q time right': ('Question', [('High', 0.0, 40.0), ('Moderate', 30.0, 15.0), ('Low', 40.0, 12.0)]),
    'Q systemic': ('Question', [('High', 60.0, 20.0), ('Moderate', 70.0, 21.0), ('Low', 80.0, 22.0)]),
    'E fast': ('Explanation', [('High', 0.0, 60.0), ('Moderate', 0.0, 62.0), ('Low', 0.0, 10.0)]),
    'E slow': ('Explanation', [('High', 0.0, 20.0), ('Moderate', 0.0, 21.0), ('Low', 0.0, 55.3)]),
}


def build_metrics(path):
    fields = ['Phase', 'Activity', 'Type', 'Category', 'Total', 'Correct',
              'Wrong', '% Correct', '% Wrong', 'Avg Time', 'Timing Total']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for act, (typ, cats) in SCENARIOS_ROWS.items():
            for cat, wrong, t in cats:
                w.writerow({'Phase': 'P', 'Activity': act, 'Type': typ,
                            'Category': cat, 'Total': 20, 'Correct': 0,
                            'Wrong': 0, '% Correct': 100 - wrong,
                            '% Wrong': wrong, 'Avg Time': t,
                            'Timing Total': 20})


FIXTURES = [{'activity': 'Q v-shape',
  'category': 'Moderate',
  'flag': 'Extreme correctness pattern',
  'reason': 'Unexpected V-shape: Moderate group (60.0% wrong) performs worse than both High '
            '(10.0%) and Low (20.0%) groups. Activity may be poorly calibrated for mid-range '
            'learners.',
  'code': 'v_shape',
  'params': {'moderate': '60.0', 'high': '10.0', 'low': '20.0'},
  'insight': ('Moderate group performance is misaligned with peers, signaling a difficulty '
              'mismatch.',
              'Introduce a **bridging activity**—e.g., a conceptual animation or '
              'mini-experiment—targeted to this group.')},
 {'activity': 'Q v-shape',
  'category': 'Moderate',
  'flag': 'Dynamic correctness threshold',
  'reason': "Dynamic correctness: Moderate group's %Wrong (60.0%) exceeds mean+std threshold "
            '(51.60%) with a category spread of 50.0pp. This group is struggling significantly '
            'more than the others on this activity.',
  'code': 'dynamic_correctness',
  'params': {'group': 'Moderate', 'wrong': '60.0', 'threshold': '51.60', 'spread': '50.0'},
  'insight': ('Moderate group error rate deviates significantly (60.0%).',
              'Recommend **revising** the activity wording or adding a short worked example to '
              'scaffold understanding.')},
 {'activity': 'Q reversal',
  'category': 'Moderate',
  'flag': 'Extreme correctness pattern',
  'reason': 'Low group (40.0% wrong) outperforms Moderate (60.0% wrong) — unexpected reversal in '
            'performance order. Activity content or answer choices may not align with the expected '
            'difficulty level.',
  'code': 'low_beats_moderate',
  'params': {'low': '40.0', 'moderate': '60.0'},
  'insight': ('Moderate group performance is misaligned with peers, signaling a difficulty '
              'mismatch.',
              'Introduce a **bridging activity**—e.g., a conceptual animation or '
              'mini-experiment—targeted to this group.')},
 {'activity': 'Q reversal',
  'category': 'Moderate',
  'flag': 'Dynamic correctness threshold',
  'reason': "Dynamic correctness: Moderate group's %Wrong (60.0%) exceeds mean+std threshold "
            '(58.16%) with a category spread of 20.0pp. This group is struggling significantly '
            'more than the others on this activity.',
  'code': 'dynamic_correctness',
  'params': {'group': 'Moderate', 'wrong': '60.0', 'threshold': '58.16', 'spread': '20.0'},
  'insight': ('Moderate group error rate deviates significantly (60.0%).',
              'Recommend **revising** the activity wording or adding a short worked example to '
              'scaffold understanding.')},
 {'activity': 'Q high worse',
  'category': 'High',
  'flag': 'Extreme correctness pattern',
  'reason': 'High group (80.0% wrong) performs worse than Moderate (50.0% wrong). High-category '
            'students should outperform Moderate — this reversal may indicate activity '
            'miscalibration or a mismatch between the activity and the scoring rubric.',
  'code': 'high_worse_than_moderate',
  'params': {'high': '80.0', 'moderate': '50.0'},
  'insight': ('High group performance is misaligned with peers, signaling a difficulty mismatch.',
              'Introduce a **bridging activity**—e.g., a conceptual animation or '
              'mini-experiment—targeted to this group.')},
 {'activity': 'Q high worse',
  'category': 'High',
  'flag': 'Dynamic correctness threshold',
  'reason': "Dynamic correctness: High group's %Wrong (80.0%) exceeds mean+std threshold (75.81%) "
            'with a category spread of 30.0pp. This group is struggling significantly more than '
            'the others on this activity.',
  'code': 'dynamic_correctness',
  'params': {'group': 'High', 'wrong': '80.0', 'threshold': '75.81', 'spread': '30.0'},
  'insight': ('High group error rate is very high (80.0%), indicating a conceptual gap.',
              'Suggest **creating** interactive explanation covering the missing concept before '
              'this activity with an informational video or data.')},
 {'activity': 'Q time wrong',
  'category': 'Low',
  'flag': 'Engagement risk (too slow, question)',
  'reason': "Engagement risk: Low group's Avg Time (45.5s) is above mean+std (26.8s ± 13.4s, "
            'CV=0.50). Significantly slower than the other groups — possible confusion or '
            'disengagement.',
  'code': 'engagement_slow',
  'params': {'group': 'Low', 't': '45.5', 'mean': '26.8', 'std': '13.4', 'cv': '0.50'},
  'insight': ('Low group spends 45.5s, well above average, suggesting confusion or cognitive '
              'overload.',
              'Add a concise **video walkthrough** or **annotated image** explaining each step, to '
              'reduce load.')},
 {'activity': 'Q time wrong',
  'category': 'Low',
  'flag': 'Timing discrepancy under extreme correctness',
  'reason': 'Low group answered everything incorrectly (100% wrong) yet spent 45.5s — ≥1.5x and '
            '≥10.0s more than the other groups (15.0s, 20.0s). Prolonged time with total failure '
            'indicates active confusion or cognitive overload, not disengagement.',
  'code': 'timing_all_wrong',
  'params': {'group': 'Low',
             'time': '45.5',
             'ratio': '1.5',
             'gap': '10.0',
             'other1': '15.0',
             'other2': '20.0'},
  'insight': ('Low group answered everything incorrectly yet spent disproportionately long on this '
              'activity. Prolonged time with total failure indicates active confusion or cognitive '
              'overload, not disengagement.',
              'Add a **worked example** or **conceptual video** before this activity to reduce '
              'cognitive load, and consider breaking it into smaller steps.')},
 {'activity': 'Q time right',
  'category': 'High',
  'flag': 'Engagement risk (too slow, question)',
  'reason': "Engagement risk: High group's Avg Time (40.0s) is above mean+std (22.3s ± 12.6s, "
            'CV=0.56). Significantly slower than the other groups — possible confusion or '
            'disengagement.',
  'code': 'engagement_slow',
  'params': {'group': 'High', 't': '40.0', 'mean': '22.3', 'std': '12.6', 'cv': '0.56'},
  'insight': ('High group spends 40.0s, well above average, suggesting confusion or cognitive '
              'overload.',
              'Add a concise **video walkthrough** or **annotated image** explaining each step, to '
              'reduce load.')},
 {'activity': 'Q time right',
  'category': 'High',
  'flag': 'Timing discrepancy under extreme correctness',
  'reason': 'High group answered everything correctly (0% wrong) yet spent 40.0s — ≥1.5x and '
            '≥10.0s more than the other groups (15.0s, 12.0s). Perfect outcome with excessive time '
            "may indicate effortful retrieval at the boundary of the group's ability range, rather "
            'than confident mastery.',
  'code': 'timing_all_right',
  'params': {'group': 'High',
             'time': '40.0',
             'ratio': '1.5',
             'gap': '10.0',
             'other1': '15.0',
             'other2': '12.0'},
  'insight': ('High group answered everything correctly yet spent disproportionately long. Perfect '
              'outcome with excessive time may indicate effortful retrieval at the boundary of the '
              "group's ability range, rather than confident mastery.",
              'Consider **revising** the activity to streamline decision points, or **creating** a '
              'faster-paced variant to build fluency for this group.')},
 {'activity': 'Q systemic',
  'category': 'All',
  'flag': 'Systemic failure',
  'reason': 'All performance categories individually exceed the PISA Low-performance boundary '
            '(50.3% wrong): High 60.0%, Moderate 70.0%, Low 80.0% wrong. All groups score below '
            'the Low-performance threshold on this activity, indicating a content-level failure '
            'independent of learner category — review the question wording, answer options, or '
            'prerequisite content.',
  'code': 'systemic_failure',
  'params': {'base': '50.3', 'high': '60.0', 'moderate': '70.0', 'low': '80.0'},
  'insight': ('All performance groups exceed the PISA Low-performance boundary on this activity, '
              'indicating a content-level failure independent of learner category.',
              '**Revise** the question wording, answer options, or prerequisite content — this '
              'activity is too difficult for all groups in the current instructional sequence.')},
 {'activity': 'E fast',
  'category': 'Low',
  'flag': 'Engagement risk (too fast, non-question)',
  'reason': "Engagement risk: Low group's Avg Time (10.0s) is below mean-std (44.0s ± 24.1s, "
            'CV=0.55). Suspiciously fast relative to other groups — may indicate guessing or '
            'skipping.',
  'code': 'engagement_fast',
  'params': {'group': 'Low', 't': '10.0', 'mean': '44.0', 'std': '24.1', 'cv': '0.55'},
  'insight': ('Low group spends only 10.0s, suggesting skimming or that the activity is trivial.',
              'Either **skip** this activity for this group or **revise** by adding reflective '
              'questions or richer multimedia (e.g., an image-based quiz).')},
 {'activity': 'E slow',
  'category': 'Low',
  'flag': 'Engagement risk (too slow, non-question)',
  'reason': "Engagement risk: Low group's Avg Time (55.3s) is above mean+std (32.1s ± 16.4s, "
            'CV=0.51). Significantly slower than the other groups — possible confusion or '
            'disengagement.',
  'code': 'engagement_slow',
  'params': {'group': 'Low', 't': '55.3', 'mean': '32.1', 'std': '16.4', 'cv': '0.51'},
  'insight': ('Low group spends 55.3s, well above average, suggesting confusion or cognitive '
              'overload.',
              'Add a concise **video walkthrough** or **annotated image** explaining each step, to '
              'reduce load.')}]


def _flag(fx, **extra):
    return SimpleNamespace(
        flag_type=fx['flag'], flag_reason=fx['reason'],
        category=fx['category'], activity=SimpleNamespace(name=fx['activity']),
        **extra,
    )


class ReasonTextTests(SimpleTestCase):
    def test_nine_codes(self):
        self.assertEqual(set(REASONS), {
            'engagement_slow', 'engagement_fast', 'v_shape',
            'low_beats_moderate', 'high_worse_than_moderate',
            'timing_all_wrong', 'timing_all_right', 'systemic_failure',
            'dynamic_correctness',
        })
        self.assertEqual(len(FLAG_TYPES), 6)

    def test_english_is_byte_identical(self):
        for fx in FIXTURES:
            with self.subTest(fx['code'], reason=fx['reason'][:30]):
                self.assertEqual(
                    reason_text(fx['code'], fx['params'], 'en'), fx['reason'])
                with translation.override('el'):
                    self.assertEqual(
                        reason_text(fx['code'], fx['params'], 'en'),
                        fx['reason'])

    def test_parse_round_trips(self):
        for fx in FIXTURES:
            with self.subTest(fx['code'], reason=fx['reason'][:30]):
                self.assertEqual(
                    parse_reason(fx['flag'], fx['reason']),
                    (fx['code'], fx['params']))

    def test_unparseable_reason_displays_unchanged(self):
        self.assertIsNone(parse_reason('Systemic failure', 'something else'))
        self.assertIsNone(parse_reason('Unknown type', FIXTURES[0]['reason']))
        self.assertEqual(
            display_flag('Systemic failure', 'something else'),
            ('Systemic failure', 'something else'))

    def test_display_in_english_matches_stored_text(self):
        for fx in FIXTURES:
            self.assertEqual(
                display_flag(fx['flag'], fx['reason']),
                (fx['flag'], fx['reason']))
            self.assertEqual(
                display_flag(fx['flag'], 'stale', fx['code'], fx['params']),
                (fx['flag'], fx['reason']))

    def test_group_names_use_interface_language_outside_english(self):
        fx = FIXTURES[6]  # engagement_slow, Low group
        with patch('localization.reference_labels.ui_label',
                   side_effect=lambda v: 'GR-' + v) as label:
            with translation.override('el'):
                text = reason_text(fx['code'], fx['params'])
            self.assertIn('GR-Low', text)
            label.reset_mock()
            self.assertEqual(
                reason_text(fx['code'], fx['params'], 'en'), fx['reason'])
            label.assert_not_called()

    def test_insight_numbers_unchanged(self):
        for fx in FIXTURES:
            with self.subTest(fx['code'], reason=fx['reason'][:30]):
                self.assertEqual(get_data_insight(_flag(fx)), fx['insight'])
                self.assertEqual(
                    get_data_insight(_flag(
                        fx, reason_code=fx['code'],
                        reason_params=fx['params'])),
                    fx['insight'])


class TaskStoresCodeTests(TestCase):
    def test_task_flags_match_fixtures_and_store_code(self):
        owner = User.objects.create_user('rf_owner')
        scenario = Scenario.objects.create(
            name='RF', created_by=owner, updated_by=owner)
        phase = Phase.objects.create(
            name='P', scenario=scenario, created_by=owner, updated_by=owner)
        atype = ActivityType.objects.create(
            name='RF type', created_by=owner, updated_by=owner)
        for name in SCENARIOS_ROWS:
            Activity.objects.create(
                name=name, scenario=scenario, phase=phase,
                activity_type=atype, created_by=owner, updated_by=owner)
        with tempfile.TemporaryDirectory() as root:
            with override_settings(AI_METRICS_CACHE_ROOT=root):
                paths = get_scenario_evidence_cache_paths(scenario, 'local')
                build_metrics(paths['metrics'])
                calculate_activities_in_risk.run(
                    scenario.id, evidence_scope='local')
                with open(paths['flags'], newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(FIXTURES))
        for row, fx in zip(rows, FIXTURES):
            self.assertEqual(row['Flag'], fx['flag'])
            self.assertEqual(row['Reason'], fx['reason'])
            # The CSV keeps its English columns; no extra columns.
            self.assertEqual(
                list(row), ['Activity', 'Category', 'Flag', 'Reason'])
        stored = {
            (f.activity.name, f.flag_type, f.reason_code): f
            for f in ActivityFlag.objects.all()
        }
        self.assertEqual(len(stored), len(FIXTURES))
        for fx in FIXTURES:
            f = stored[(fx['activity'], fx['flag'], fx['code'])]
            self.assertEqual(f.flag_reason, fx['reason'])
            self.assertEqual(f.reason_params, fx['params'])


class BackfillTests(TestCase):
    def test_backfill_sets_code_and_params(self):
        migration = importlib.import_module(
            'authoringtool.migrations.0077_backfill_flag_reasons')
        owner = User.objects.create_user('bf_owner')
        scenario = Scenario.objects.create(
            name='BF', created_by=owner, updated_by=owner)
        phase = Phase.objects.create(
            name='P', scenario=scenario, created_by=owner, updated_by=owner)
        atype = ActivityType.objects.create(
            name='BF type', created_by=owner, updated_by=owner)
        act = Activity.objects.create(
            name='A', scenario=scenario, phase=phase, activity_type=atype,
            created_by=owner, updated_by=owner)
        ids = {}
        for fx in FIXTURES:
            ids[ActivityFlag.objects.create(
                activity=act, category=fx['category'], flag_type=fx['flag'],
                flag_reason=fx['reason']).id] = fx
        odd = ActivityFlag.objects.create(
            activity=act, category='High', flag_type='Systemic failure',
            flag_reason='hand written')
        migration.backfill(django_apps, None)
        for pk, fx in ids.items():
            f = ActivityFlag.objects.get(pk=pk)
            self.assertEqual(f.reason_code, fx['code'])
            self.assertEqual(f.reason_params, fx['params'])
            self.assertEqual(f.flag_reason, fx['reason'])
        odd.refresh_from_db()
        self.assertEqual(odd.reason_code, '')


class MetricsPageTests(TestCase):
    def setUp(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user('rf_page_owner')
        self.owner.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='RF page', visibility_status='private',
            created_by=self.owner, updated_by=self.owner)
        self.client.force_login(self.owner)

    def _write_flags(self):
        paths = get_scenario_evidence_cache_paths(self.scenario, 'compatible')
        fx = FIXTURES[10]
        with open(paths['flags'], 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(
                f, fieldnames=['Activity', 'Category', 'Flag', 'Reason'])
            w.writeheader()
            w.writerow({'Activity': 'Q systemic', 'Category': 'All',
                        'Flag': fx['flag'], 'Reason': fx['reason']})

    def test_metrics_page_shows_display_text(self):
        with tempfile.TemporaryDirectory() as root:
            with override_settings(AI_METRICS_CACHE_ROOT=root):
                self._write_flags()
                url = reverse('ai_metrics', args=[self.scenario.id])
                with patch('authoringtool.views.display_flag',
                           return_value=('SHOWN FLAG', 'SHOWN REASON')):
                    response = self.client.get(url)
                self.assertContains(response, 'SHOWN FLAG')
                self.assertContains(response, 'SHOWN REASON')
                plain = self.client.get(url)
                self.assertContains(plain, FIXTURES[10]['flag'])
                self.assertContains(plain, 'PISA Low-performance boundary')

    def test_status_rows_carry_display_keys(self):
        with tempfile.TemporaryDirectory() as root:
            with override_settings(AI_METRICS_CACHE_ROOT=root):
                self._write_flags()
                fake = SimpleNamespace(
                    state='SUCCESS',
                    result={'scenario_id': self.scenario.id,
                            'evidence_scope': 'compatible',
                            'evidence_language': ''})
                with patch('authoringtool.views.AsyncResult',
                           return_value=fake):
                    response = self.client.get(
                        reverse('risk_flags_status', args=['t1']))
        row = json.loads(response.content)['data'][0]
        self.assertEqual(row['Flag'], FIXTURES[10]['flag'])
        self.assertEqual(row['Reason'], FIXTURES[10]['reason'])
        self.assertEqual(row['FlagDisplay'], FIXTURES[10]['flag'])
        self.assertEqual(row['ReasonDisplay'], FIXTURES[10]['reason'])
