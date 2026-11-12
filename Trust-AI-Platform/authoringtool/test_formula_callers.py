import csv
import io

from django.test import TestCase

from authoringtool.tasks import (
    compute_final_performance,
    compute_student_performance_metrics,
)
from categorization.tests.fixtures import (
    answer_all,
    make_scenario,
    make_student,
    make_teacher,
    three_students,
)


def csv_rows(scenario):
    content = compute_student_performance_metrics(scenario.id, None, None, None)['csv_content']
    return {row['Username']: row for row in csv.DictReader(io.StringIO(content))}


class ExistingCallersUnchangedTests(TestCase):
    """Pinned before the formula moved; must hold before and after."""

    def setUp(self):
        self.ns = make_scenario(make_teacher())
        self.high, self.moderate, self.low = three_students(self.ns)

    def test_chart_counts_unchanged(self):
        data = compute_final_performance(self.ns.scenario.id, None, None, None)
        self.assertEqual(data['waterfall_data'], [
            {'name': 'Total', 'value': 3},
            {'name': 'High', 'value': 1},
            {'name': 'Moderate', 'value': 1},
            {'name': 'Low', 'value': 1},
        ])

    def test_csv_categories_unchanged(self):
        rows = csv_rows(self.ns.scenario)
        expected = {
            'st_high': ('High', ['High', 'High', 'High', 'High']),
            'st_moderate': ('Moderate', ['High', 'Moderate', 'Moderate', 'High']),
            'st_low': ('Low', ['Low', 'Moderate', 'Low', 'Low']),
        }
        for username, (final, phases) in expected.items():
            row = rows[username]
            self.assertEqual(row['Final Categorization'], final, username)
            self.assertEqual(
                [row[f'Phase {n} Categorization'] for n in range(1, 5)], phases, username,
            )

    def test_csv_sixth_phase_gets_no_weight(self):
        # Built before anyone answers: answering an activity added after the
        # student's scenario version marks the run suspect.
        ns = make_scenario(self.ns.teacher, name='Six phases', phases=6)
        student = make_student('six_high')
        answer_all(student, ns, [2, 2, 2, 2, 1, 0])
        # Weights 20/20/30/15/15 over phases 1-5 give 92.5% (High). Phase 6
        # has no weight; at 15% its 0% would give 80.4% (Moderate).
        self.assertEqual(csv_rows(ns.scenario)['six_high']['Final Categorization'], 'High')
