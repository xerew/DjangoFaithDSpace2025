import os
import tempfile
from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from categorization import training
from categorization.models import TrainingRun
from categorization.tasks import run_training
from categorization.tests.fixtures import answer_all, make_scenario, make_student, make_teacher

WEIGHT_PATTERNS = [[2, 2, 2, 2], [2, 1, 1, 2], [0, 1, 0, 0], [2, 2, 1, 2], [1, 1, 1, 1], [0, 0, 1, 0]]


class TrainingJobTests(TestCase):
    def setUp(self):
        self.teacher = make_teacher()
        self.admin = User.objects.create_user('job_admin', password='pass', is_staff=True)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        overrides = override_settings(CATEGORIZATION_MODEL_DIR=self.folder.name)
        overrides.enable()
        self.addCleanup(overrides.disable)

    def students(self, ns, count, prefix):
        for index in range(count):
            answer_all(make_student(f'{prefix}{index}'), ns, WEIGHT_PATTERNS[index % len(WEIGHT_PATTERNS)])

    def test_job_with_too_little_data_fails_cleanly(self):
        self.students(make_scenario(self.teacher), 3, 'few')
        run = TrainingRun.start(self.admin)
        run_training(run.id)
        run.refresh_from_db()
        self.assertEqual(run.status, 'failed')
        self.assertTrue(run.error.startswith('Not enough runs to train'), run.error)
        self.assertIsNotNone(run.finished_at)

    def test_job_success_fills_run(self):
        self.students(make_scenario(self.teacher, name='First'), 6, 'a')
        second = make_scenario(self.teacher, name='Second')
        self.students(second, 6, 'b')
        answer_all(make_student('dropout'), second, [2, 1, None, None])
        run = TrainingRun.start(self.admin)
        with mock.patch.object(training, 'MIN_LABELLED_RUNS', 5):
            run_training(run.id)
        run.refresh_from_db()
        self.assertEqual(run.status, 'finished', run.error)
        # The student who stopped early is counted too.
        self.assertEqual(run.counts['labelled'], 13)
        self.assertEqual(run.counts['finished'], 12)
        self.assertEqual(run.counts['unfinished'], 1)
        self.assertEqual(run.counts['students'], 13)
        self.assertEqual({s['name']: s['finished'] for s in run.counts['scenarios']},
                         {'First': 6, 'Second': 6})
        self.assertEqual(len(run.counts['scenarios']), 2)
        self.assertEqual(run.results['final']['overall']['runs'], run.counts['labelled'])
        self.assertTrue(run.results['early']['checkpoints'])
        self.assertIn('xgboost', run.versions)
        self.assertIn('fast_answer_rule', run.parameters)
        for path in (run.final_model_path, run.early_model_path):
            self.assertTrue(os.path.exists(path), path)
            self.assertTrue(path.startswith(self.folder.name))

    def test_over_time_error_keeps_the_trained_models(self):
        self.students(make_scenario(self.teacher, name='First'), 6, 'a')
        self.students(make_scenario(self.teacher, name='Second'), 6, 'b')
        run = TrainingRun.start(self.admin)
        with mock.patch.object(training, 'MIN_LABELLED_RUNS', 5), \
             mock.patch('categorization.tasks.over_time.evaluate', side_effect=ValueError('boom')):
            run_training(run.id)
        run.refresh_from_db()
        self.assertEqual(run.status, 'finished', run.error)
        self.assertEqual(run.results['over_time'], {'error': 'ValueError: boom'})
        self.assertTrue(os.path.exists(run.final_model_path))

    def test_unexpected_error_marks_failed(self):
        run = TrainingRun.start(self.admin)
        with mock.patch('categorization.tasks.load_runs', side_effect=RuntimeError('boom')):
            run_training(run.id)
        run.refresh_from_db()
        self.assertEqual(run.status, 'failed')
        self.assertEqual(run.error, 'RuntimeError: boom')
