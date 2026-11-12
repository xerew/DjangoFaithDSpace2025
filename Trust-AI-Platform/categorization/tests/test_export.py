from io import BytesIO

import openpyxl
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from categorization.models import TrainingRun
from categorization.tests.test_dashboard import counts, results


class ExportTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)
        self.run = TrainingRun.objects.create(
            started_by=self.staff, status='finished', started_at=timezone.now(),
            finished_at=timezone.now(), results=results(), counts=counts(),
        )
        self.client.force_login(self.staff)

    def workbook(self, tab, **query):
        response = self.client.get(reverse('categorization_export', args=[self.run.id, tab]), query)
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'categorization-run-{self.run.id}-{tab}.xlsx', response['Content-Disposition'])
        return openpyxl.load_workbook(BytesIO(response.content))

    def test_final_export_sheets(self):
        book = self.workbook('final')
        self.assertEqual(book.sheetnames, [
            'Summary', 'Confusion matrix', 'Disagreements', 'Importance', 'Per scenario',
        ])
        # Header row plus the one disagreement the page shows.
        self.assertEqual(book['Disagreements'].max_row, 2)
        self.assertEqual(book['Disagreements']['A2'].value, 'student2')

    def test_early_export_sheets(self):
        book = self.workbook('early')
        self.assertEqual(book.sheetnames, [
            'Checkpoints', 'Confusion matrix', 'Importance', 'Unseen scenarios',
        ])
        self.assertEqual(book['Checkpoints'].cell(row=book['Checkpoints'].max_row, column=1).value,
                         'After phase 1')

    def test_export_respects_filters(self):
        book = self.workbook('final', scenario='2')
        self.assertEqual(book['Disagreements'].max_row, 1)
        values = [row for row in book['Summary'].iter_rows(values_only=True)]
        self.assertIn(('Scenario filter', 'Beta'), values)
        self.assertIn(('Agreement with PISA', 1.0), values)

    def test_teacher_cannot_export(self):
        self.client.force_login(User.objects.create_user('teacher', password='pass'))
        response = self.client.get(reverse('categorization_export', args=[self.run.id, 'final']))
        self.assertEqual(response.status_code, 403)

    def test_unfinished_run_404(self):
        running = TrainingRun.objects.create(status='running')
        for run_id, tab in ((running.id, 'final'), (self.run.id, 'other'), (99999, 'final')):
            response = self.client.get(reverse('categorization_export', args=[run_id, tab]))
            self.assertEqual(response.status_code, 404, (run_id, tab))
