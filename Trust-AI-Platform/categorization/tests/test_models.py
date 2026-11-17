from datetime import timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from categorization.models import TrainingAlreadyRunning, TrainingRun


class TrainingRunStartTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user('ml_admin', password='pass', is_staff=True)

    def test_start_creates_queued_run(self):
        run = TrainingRun.start(self.admin)
        self.assertEqual(run.status, 'queued')
        self.assertEqual(run.started_by, self.admin)

    def test_second_start_while_running_raises(self):
        run = TrainingRun.start(self.admin)
        run.status = 'running'
        run.save()
        with self.assertRaises(TrainingAlreadyRunning):
            TrainingRun.start(self.admin)

    def test_stale_running_run_is_failed_and_new_run_starts(self):
        old = TrainingRun.start(self.admin)
        TrainingRun.objects.filter(pk=old.pk).update(
            status='running', created_at=timezone.now() - timedelta(hours=7),
        )
        new = TrainingRun.start(self.admin)
        old.refresh_from_db()
        self.assertEqual(old.status, 'failed')
        self.assertEqual(old.error, 'Stopped without finishing (the worker may have restarted).')
        self.assertEqual(new.status, 'queued')

    def test_finished_runs_do_not_block(self):
        run = TrainingRun.start(self.admin)
        run.status = 'finished'
        run.save()
        self.assertEqual(TrainingRun.start(self.admin).status, 'queued')
