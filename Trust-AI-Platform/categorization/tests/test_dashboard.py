import re
from unittest import mock

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse
from datetime import timedelta

from django.utils import timezone

from categorization.models import TrainingRun

EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')


def prediction(impl, scenario_id, name, pisa, model, confidence=0.9, pct=90.0, language='English'):
    return {
        'implementation_id': impl, 'user_id': impl, 'username': f'student{impl}',
        'scenario_id': scenario_id, 'scenario_name': name, 'language': language,
        'group_key': scenario_id, 'is_current_version': True, 'pisa': pisa, 'model': model,
        'confidence': confidence, 'final_percentage': pct,
    }


def empty_metrics(runs=4):
    per_class = {label: {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 1}
                 for label in ('Low', 'Moderate', 'High')}
    return {'runs': runs, 'accuracy': 0.75, 'macro_f1': 0.7, 'per_class': per_class,
            'confusion': [[1, 0, 0], [1, 0, 0], [0, 0, 2]]}


def results():
    return {
        'final': {
            'overall': empty_metrics(),
            'by_version': {'current': empty_metrics(), 'older': empty_metrics(0)},
            'importances': [{'input': 'p1_pct', 'mean_abs_shap': 0.5}],
            'loso': [{'group_key': 1, 'scenario_names': 'Alpha', 'runs': 40,
                      'accuracy': 0.8, 'macro_f1': 0.7}],
            'predictions': [
                prediction(1, 1, 'Alpha', 'High', 'High'),
                prediction(2, 1, 'Alpha', 'Moderate', 'Low', confidence=0.7, pct=55.0),
                prediction(3, 2, 'Beta', 'High', 'High', language='Greek'),
                prediction(4, 2, 'Beta', 'Low', 'Low', pct=20.0, language='Greek'),
            ],
        },
        'early': {
            'checkpoints': [],
            'importances': {'1': [{'input': 'p1_pct', 'mean_abs_shap': 0.4}]},
            'loso': [{'group_key': 1, 'scenario_names': 'Alpha', 'runs': 40, 'accuracy': 0.6,
                      'macro_f1': 0.5, 'by_checkpoint': {'1': {'runs': 40, 'accuracy': 0.6,
                                                               'baseline_accuracy': 0.5}}}],
            'predictions': [
                {'implementation_id': 1, 'scenario_id': 1, 'scenario_name': 'Alpha',
                 'language': 'English', 'group_key': 1, 'is_current_version': True,
                 'checkpoint': 1, 'truth': 'High', 'model': 'High', 'baseline': 'Moderate'},
                {'implementation_id': 3, 'scenario_id': 2, 'scenario_name': 'Beta',
                 'language': 'Greek', 'group_key': 2, 'is_current_version': True,
                 'checkpoint': 1, 'truth': 'High', 'model': 'Moderate', 'baseline': 'High'},
            ],
        },
        'seconds': {'total': 12.5, 'loso': 4.0},
    }


def counts():
    return {'students': 4, 'runs_total': 5, 'labelled': 4, 'unfinished': 1, 'route_end': 4,
            'last_phase': 0, 'current_version': 4, 'older_version': 0,
            'scenarios': [{'id': 1, 'name': 'Alpha', 'language': 'English', 'runs': 3, 'labelled': 2},
                          {'id': 2, 'name': 'Beta', 'language': 'Greek', 'runs': 2, 'labelled': 2}]}


class DashboardTests(TestCase):
    url = reverse('categorization_dashboard')

    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)
        self.teacher = User.objects.create_user('teacher', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))

    def finished_run(self):
        return TrainingRun.objects.create(
            started_by=self.staff, status='finished', started_at=timezone.now(),
            finished_at=timezone.now(), results=results(), counts=counts(),
            parameters={'fast_answer_rule': 'Fast rule text.'},
        )

    def test_staff_sees_page(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_superuser_sees_page(self):
        self.client.force_login(User.objects.create_superuser('root', 'root@example.com', 'pass'))
        self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_teacher_gets_403(self):
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(self.url).status_code, 403)

    def test_anonymous_redirected_to_admin_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith(reverse('admin:login')))

    def test_empty_state(self):
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertContains(response, 'No training has finished yet.')
        self.assertContains(response, 'Train models')

    def test_train_starts_job(self):
        self.client.force_login(self.staff)
        with mock.patch('categorization.views.train_models_task.delay') as delay:
            response = self.client.post(reverse('categorization_train'))
        self.assertRedirects(response, self.url)
        run = TrainingRun.objects.get()
        self.assertEqual(run.status, 'queued')
        delay.assert_called_once_with(run.id)

    def test_teacher_cannot_train(self):
        self.client.force_login(self.teacher)
        with mock.patch('categorization.views.train_models_task.delay') as delay:
            self.assertEqual(self.client.post(reverse('categorization_train')).status_code, 403)
        delay.assert_not_called()

    def test_train_while_running_shows_message(self):
        TrainingRun.objects.create(status='running')
        self.client.force_login(self.staff)
        with mock.patch('categorization.views.train_models_task.delay') as delay:
            response = self.client.post(reverse('categorization_train'), follow=True)
        delay.assert_not_called()
        self.assertEqual(TrainingRun.objects.count(), 1)
        self.assertContains(response, 'A training run is already in progress.')

    def test_filters_recompute_metrics(self):
        self.finished_run()
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertEqual(response.context['final']['overall']['accuracy'], 0.75)
        self.assertContains(response, '75.0%')
        self.assertContains(response, 'student2')
        response = self.client.get(self.url, {'scenario': '2'})
        self.assertEqual(response.context['final']['overall']['accuracy'], 1.0)
        self.assertEqual(response.context['final']['disagreements'], [])
        self.assertNotContains(response, 'student2')

    def test_disagreement_row_details(self):
        self.finished_run()
        self.client.force_login(self.staff)
        row = self.client.get(self.url).context['final']['disagreements'][0]
        self.assertEqual(row['username'], 'student2')
        self.assertAlmostEqual(row['threshold_distance'], 5.3)

    def test_early_tab(self):
        self.finished_run()
        self.client.force_login(self.staff)
        response = self.client.get(self.url, {'tab': 'early'})
        early = response.context['early']
        self.assertEqual(early['picked'], 1)
        self.assertEqual(early['rows'][0]['xgb']['accuracy'], 0.5)
        self.assertEqual(early['rows'][0]['baseline']['accuracy'], 0.5)
        self.assertEqual(early['loso'][0]['baseline_accuracy'], 0.5)

    def test_lost_run_does_not_block_train_button(self):
        lost = TrainingRun.objects.create(status='running')
        TrainingRun.objects.filter(pk=lost.pk).update(created_at=timezone.now() - timedelta(hours=7))
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertIsNone(response.context['active'])
        lost.refresh_from_db()
        self.assertEqual(lost.status, 'failed')
        self.assertContains(response, TrainingRun.STALE_ERROR)

    def test_status_expires_lost_run(self):
        lost = TrainingRun.objects.create(status='queued')
        TrainingRun.objects.filter(pk=lost.pk).update(created_at=timezone.now() - timedelta(hours=7))
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(reverse('categorization_status')).json()['status'], 'failed')

    def test_run_dropdown_does_not_load_results(self):
        self.finished_run()
        self.finished_run()
        self.client.force_login(self.staff)
        runs = list(self.client.get(self.url).context['finished_runs'])
        self.assertEqual(len(runs), 2)
        self.assertIn('results', runs[0].get_deferred_fields())

    def test_status_json(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(reverse('categorization_status')).json(), {})
        run = TrainingRun.objects.create(status='running', progress='Final model: cross-validation 2 / 5')
        self.assertEqual(self.client.get(reverse('categorization_status')).json(), {
            'id': run.id, 'status': 'running', 'progress': 'Final model: cross-validation 2 / 5',
            'error': '',
        })

    def test_failed_run_error_is_shown(self):
        TrainingRun.objects.create(status='failed', error='Not enough finished runs to train: 3 found.')
        self.client.force_login(self.staff)
        self.assertContains(self.client.get(self.url), 'Not enough finished runs to train: 3 found.')

    def test_header_link_present(self):
        self.client.force_login(self.staff)
        self.assertContains(self.client.get(reverse('admin:index')), self.url)

    def test_no_emoji_in_template(self):
        self.finished_run()
        self.client.force_login(self.staff)
        for tab in ('final', 'early'):
            html = self.client.get(self.url, {'tab': tab}).content.decode()
            self.assertIsNone(EMOJI.search(html), tab)
