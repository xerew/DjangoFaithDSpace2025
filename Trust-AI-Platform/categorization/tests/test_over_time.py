"""Over time: models trained only on earlier students, tested month by month."""

from datetime import datetime, timedelta, timezone as dt_timezone
from io import BytesIO
from unittest import mock

import numpy as np
import openpyxl
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from authoringtool.performance_formula import categorize, phase_weights, weighted_percentage
from categorization import over_time
from categorization.dataset import AnswerEvent
from categorization.models import TrainingRun
from categorization.report import over_time_view
from categorization.tests.test_dashboard import counts, results
from categorization.tests.test_features import make_run


def dated_run(index, when, weights):
    """A 4-phase run started at `when`; phase k answered with weights[k-1] of 2."""
    answers = [
        AnswerEvent(activity_id=100 + pos, phase_position=pos,
                    created_on=when + timedelta(minutes=pos), active_seconds=30 + pos,
                    chosen_weight=weight, is_last=True)
        for pos, weight in enumerate(weights, start=1)
    ]
    phase_scores = [(weight, 2) for weight in weights]
    run = make_run(answers, phase_scores, implementation_id=index, user_id=index,
                   label=categorize(weighted_percentage(phase_scores, phase_weights(4))))
    return run


def months_of_runs(per_month, seed=0):
    """Runs spread over months; per_month is [(year, month, count), ...]."""
    rng = np.random.default_rng(seed)
    runs = []
    for year, month, count in per_month:
        for day in range(count):
            index = len(runs) + 1
            when = datetime(year, month, 1 + day % 27, 9, 0)
            runs.append(dated_run(index, when, [int(w) for w in rng.integers(0, 3, size=4)]))
    return runs


class PeriodTests(SimpleTestCase):
    def test_small_months_join_the_next(self):
        runs = months_of_runs([(2024, 5, 40), (2024, 6, 5), (2024, 7, 10), (2024, 8, 30), (2024, 9, 3)])
        found = [(first, last, len(part)) for first, last, part in over_time.periods(runs, min_runs=30)]
        # The last 3 runs of September are too few on their own: joined with the period before.
        self.assertEqual(found, [((2024, 5), (2024, 5), 40), ((2024, 6), (2024, 9), 48)])

    def test_months_follow_athens_time(self):
        # 22:30 UTC on 31 March is 01:30 on 1 April in Athens.
        late = dated_run(1, datetime(2025, 3, 31, 22, 30, tzinfo=dt_timezone.utc), [2, 2, 2, 2])
        found = over_time.periods([late], min_runs=1)
        self.assertEqual(found[0][0], (2025, 4))

    def test_runs_are_dated_by_their_first_answer(self):
        run = dated_run(1, datetime(2025, 3, 31, 23, 50), [2, 2, 2, 2])
        self.assertEqual(over_time.started_on(run), datetime(2025, 3, 31, 23, 51))


class EvaluateTests(SimpleTestCase):
    def setUp(self):
        self.runs = months_of_runs([(2024, 9, 60), (2024, 10, 60), (2024, 11, 60), (2025, 2, 60)])

    def test_each_period_is_tested_on_a_model_of_the_past_only(self):
        seen = []
        real = over_time.build_norms

        def recording(runs):
            seen.append(max(over_time.started_on(run) for run in runs))
            return real(runs)

        with mock.patch.object(over_time, 'build_norms', side_effect=recording):
            result = over_time.evaluate(self.runs)
        periods = result['periods']
        self.assertEqual([p['start'] for p in periods], ['2024-09', '2024-10', '2024-11', '2025-02'])
        # The first month has nothing before it.
        self.assertNotIn('final', periods[0])
        self.assertEqual([p['trained_on'] for p in periods], [0, 60, 120, 180])
        for period, latest in zip(periods[1:], seen):
            year, month = map(int, period['start'].split('-'))
            self.assertLess(latest, datetime(year, month, 1))

    def test_results_for_final_and_checkpoints(self):
        period = over_time.evaluate(self.runs)['periods'][-1]
        self.assertEqual(period['tested'], 60)
        self.assertEqual(period['final']['xgb']['runs'], 60)
        self.assertEqual(sorted(period['checkpoints']), ['1', '2', '3'])
        checkpoint = period['checkpoints']['1']
        self.assertEqual(checkpoint['xgb']['runs'], checkpoint['baseline']['runs'])
        self.assertIsNotNone(checkpoint['baseline']['macro_f1'])

    def test_progress_is_reported(self):
        messages = []
        over_time.evaluate(self.runs, progress=messages.append)
        self.assertEqual(messages[-1], 'Over time: period 4 / 4')

    def test_too_little_history_is_skipped(self):
        runs = months_of_runs([(2024, 9, 30), (2024, 10, 30)])
        periods = over_time.evaluate(runs)['periods']
        # 30 earlier runs are fewer than the minimum to train on.
        self.assertNotIn('final', periods[1])


def stored():
    period = {'xgb': {'runs': 40, 'accuracy': 0.5, 'macro_f1': 0.4},
              'baseline': {'runs': 40, 'accuracy': 0.45, 'macro_f1': 0.38}}
    return {'min_period_runs': 30, 'periods': [
        {'start': '2024-09', 'end': '2024-09', 'trained_on': 0, 'students_trained': 0, 'tested': 60},
        {'start': '2024-10', 'end': '2024-12', 'trained_on': 60, 'students_trained': 50, 'tested': 70,
         'final': {'xgb': {'runs': 70, 'accuracy': 0.9, 'macro_f1': 0.85}},
         'checkpoints': {'1': period, '2': period}},
    ]}


class ViewTests(SimpleTestCase):
    def test_defaults_to_after_phase_one(self):
        view = over_time_view({'over_time': stored()})
        self.assertEqual(view['options'], ['1', '2', 'final'])
        self.assertEqual(view['picked'], '1')
        self.assertEqual([row['evaluated'] for row in view['rows']], [False, True])
        self.assertEqual(view['rows'][1]['label'], 'Oct 2024 – Dec 2024')
        self.assertEqual(view['chart']['xgb_f1'], [None, 0.4])

    def test_final_has_no_formula_line(self):
        view = over_time_view({'over_time': stored()}, 'final')
        self.assertIsNone(view['rows'][1]['baseline'])
        self.assertEqual(view['chart']['xgb_accuracy'], [None, 0.9])
        self.assertEqual(view['chart']['baseline_accuracy'], [None, None])

    def test_older_runs_have_nothing(self):
        self.assertIsNone(over_time_view({}))


class OverTimeTabTests(TestCase):
    url = reverse('categorization_dashboard')

    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)
        self.client.force_login(self.staff)

    def run_with(self, data):
        return TrainingRun.objects.create(
            started_by=self.staff, status='finished', started_at=timezone.now(),
            finished_at=timezone.now(), results=data, counts=counts(),
        )

    def test_tab_shows_periods(self):
        self.run_with({**results(), 'over_time': stored()})
        response = self.client.get(self.url, {'tab': 'time'})
        self.assertContains(response, 'Oct 2024 – Dec 2024')
        self.assertContains(response, 'Over time')
        self.assertEqual(response.context['over_time']['picked'], '1')

    def test_final_prediction_page(self):
        # The final categorization has no formula line: its cells show a dash.
        self.run_with({**results(), 'over_time': stored()})
        response = self.client.get(self.url, {'tab': 'time', 'prediction': 'final'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['over_time']['picked'], 'final')
        self.assertContains(response, '90.0%')

    def test_over_time_error_is_shown(self):
        self.run_with({**results(), 'over_time': {'error': 'ValueError: boom'}})
        response = self.client.get(self.url, {'tab': 'time'})
        self.assertContains(response, 'The over-time test failed: ValueError: boom')

    def test_run_trained_before_this_tab(self):
        self.run_with(results())
        response = self.client.get(self.url, {'tab': 'time'})
        self.assertContains(response, 'Train the models again to see this tab.')

    def test_export(self):
        run = self.run_with({**results(), 'over_time': stored()})
        response = self.client.get(reverse('categorization_export', args=[run.id, 'time']))
        self.assertEqual(response.status_code, 200)
        book = openpyxl.load_workbook(BytesIO(response.content))
        self.assertEqual(book.sheetnames, ['Over time'])
        # Header plus one row per evaluated period and prediction (2 checkpoints + final).
        self.assertEqual(book['Over time'].max_row, 4)
