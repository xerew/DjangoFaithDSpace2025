"""Chatbot intent evaluation: job record, report, page, Excel and worker helpers."""

import importlib.util
import os
import tempfile
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import openpyxl
from django.conf import settings
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from categorization import chatbot_report
from categorization.models import ChatbotEvaluation, EvaluationAlreadyRunning, TrainingRun

LOG_TAIL = '\n'.join([
    '2026-11-12 10:12:56 INFO     rasa.model_testing  - CV evaluation (n=3)',
    '2026-11-12 10:12:56 INFO     rasa.nlu.test  - train Accuracy: 0.988 (0.008)',
    '2026-11-12 10:12:56 INFO     rasa.nlu.test  - test Accuracy: 0.776 (0.019)',
    '2026-11-12 10:12:56 INFO     rasa.nlu.test  - test F1-score: 0.761 (0.032)',
    '2026-11-12 10:12:56 INFO     rasa.nlu.test  - test Precision: 0.777 (0.054)',
])


def report():
    return {
        'greet': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 2},
        'deny': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
        'affirm': {'precision': 1.0, 'recall': 0.5, 'f1-score': 0.667, 'support': 2},
        'accuracy': 0.6,
        'macro avg': {'precision': 0.833, 'recall': 0.667, 'f1-score': 0.722, 'support': 5},
        'weighted avg': {'precision': 0.8, 'recall': 0.6, 'f1-score': 0.667, 'support': 5},
    }


def prediction(text, intent, predicted, confidence):
    return {'text': text, 'intent': intent,
            'intent_prediction': {'name': predicted, 'confidence': confidence}}


def errors():
    return [
        prediction('hello there', 'greet', 'affirm', 0.9),
        prediction('=1+1', 'affirm', 'restart', 0.2),
    ]


def successes():
    return [
        prediction('hi', 'greet', 'greet', 0.95),
        prediction('no', 'deny', 'deny', 0.99),
        prediction('yes', 'affirm', 'affirm', 0.55),
    ]


def finished_evaluation(**fields):
    values = dict(
        status='finished', folds=3, started_at=timezone.now() - timedelta(minutes=31),
        finished_at=timezone.now(), report=report(), errors=errors(), successes=successes(),
        log_tail=LOG_TAIL, nlu_text='version: "3.1"\nnlu: []\n', nlu_sha256='ab' * 32,
        rasa_version='3.6.0', fallback_threshold=0.3,
    )
    values.update(fields)
    return ChatbotEvaluation.objects.create(**values)


class EvaluationModelTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)

    def test_start_queues_one_run_at_a_time(self):
        evaluation = ChatbotEvaluation.start(self.staff, folds=5)
        self.assertEqual((evaluation.status, evaluation.folds), ('queued', 5))
        with self.assertRaises(EvaluationAlreadyRunning):
            ChatbotEvaluation.start(self.staff, folds=3)

    def test_lost_run_expires_after_three_hours(self):
        lost = ChatbotEvaluation.start(self.staff, folds=3)
        ChatbotEvaluation.objects.filter(pk=lost.pk).update(created_at=timezone.now() - timedelta(hours=4))
        ChatbotEvaluation.start(self.staff, folds=3)
        lost.refresh_from_db()
        self.assertEqual(lost.status, 'failed')

    def test_independent_of_categorization_training(self):
        TrainingRun.objects.create(status='running')
        self.assertEqual(ChatbotEvaluation.start(self.staff, folds=3).status, 'queued')


class ReportTests(SimpleTestCase):
    def build(self, **fields):
        values = dict(report=report(), errors=errors(), successes=successes(), log_tail=LOG_TAIL,
                      fallback_threshold=0.3, folds=3)
        values.update(fields)
        return chatbot_report.build(ChatbotEvaluation(**values))

    def test_per_intent_worst_first_with_wrong_counts(self):
        rows = self.build()['per_intent']
        self.assertEqual([row['intent'] for row in rows], ['greet', 'affirm', 'deny'])
        self.assertEqual({row['intent']: row['wrong'] for row in rows}, {'greet': 1, 'affirm': 1, 'deny': 0})

    def test_confusion_matrix(self):
        confusion = self.build()['confusion']
        # Intents tested, then predicted-only labels.
        self.assertEqual(confusion['rows'], ['affirm', 'deny', 'greet'])
        self.assertEqual(confusion['columns'], ['affirm', 'deny', 'greet', 'restart'])
        self.assertEqual(confusion['matrix'], [[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0]])

    def test_summary(self):
        summary = self.build()['summary']
        self.assertEqual(summary['examples'], 5)
        self.assertEqual(summary['intents'], 3)
        self.assertAlmostEqual(summary['accuracy'], 0.6)
        self.assertAlmostEqual(summary['macro_f1'], 0.722)
        self.assertEqual(summary['wrong'], 2)
        # 0.9 is above the 0.3 fallback; 0.2 would be caught by it.
        self.assertEqual(summary['wrong_above_fallback'], 1)
        self.assertEqual(summary['per_fold']['test Accuracy'], (0.776, 0.019))
        self.assertEqual(summary['per_fold']['test F1-score'], (0.761, 0.032))

    def test_wrong_predictions_highest_confidence_first(self):
        wrong = self.build()['wrong']
        self.assertEqual([row['text'] for row in wrong], ['hello there', '=1+1'])
        self.assertFalse(wrong[0]['below_fallback'])
        self.assertTrue(wrong[1]['below_fallback'])

    def test_confidence_buckets(self):
        confidence = self.build()['confidence']
        self.assertEqual(len(confidence['labels']), 10)
        self.assertEqual(sum(confidence['right']), 3)
        self.assertEqual(sum(confidence['wrong']), 2)
        self.assertEqual(confidence['right'][9], 2)  # 0.95 and 0.99
        self.assertEqual(confidence['wrong'][2], 1)  # 0.2
        self.assertEqual(confidence['fallback'], 0.3)

    def test_empty_results(self):
        built = self.build(report={}, errors=[], successes=[], log_tail='')
        self.assertEqual(built['per_intent'], [])
        self.assertEqual(built['summary']['examples'], 0)
        self.assertEqual(built['summary']['per_fold'], {})

    def test_missing_prediction_name(self):
        built = self.build(errors=[prediction('???', 'greet', None, 0.1)])
        self.assertIn(chatbot_report.NO_INTENT, built['confusion']['columns'])


class ChatbotPageTests(TestCase):
    url = reverse('categorization_dashboard')

    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)
        self.teacher = User.objects.create_user('teacher', password='pass')

    def page(self, **query):
        return self.client.get(self.url, {'tab': 'chatbot', **query})

    def test_teacher_is_refused_everywhere(self):
        evaluation = finished_evaluation()
        self.client.force_login(self.teacher)
        self.assertEqual(self.page().status_code, 403)
        self.assertEqual(self.client.post(reverse('categorization_chatbot_run'), {'folds': 3}).status_code, 403)
        self.assertEqual(self.client.get(reverse('categorization_chatbot_status')).status_code, 403)
        export = reverse('categorization_chatbot_export', args=[evaluation.id])
        self.assertEqual(self.client.get(export).status_code, 403)
        self.assertFalse(ChatbotEvaluation.objects.filter(status='queued').exists())

    def test_empty_state(self):
        self.client.force_login(self.staff)
        response = self.page()
        self.assertContains(response, 'No chatbot evaluation has finished yet.')
        self.assertContains(response, 'Run evaluation')

    def test_tabs_link_each_other(self):
        self.client.force_login(self.staff)
        self.assertContains(self.client.get(self.url), '?tab=chatbot')
        self.assertContains(self.page(), 'tab=final')

    def test_run_queues_and_returns_to_tab(self):
        self.client.force_login(self.staff)
        response = self.client.post(reverse('categorization_chatbot_run'), {'folds': '5'})
        self.assertRedirects(response, self.url + '?tab=chatbot')
        evaluation = ChatbotEvaluation.objects.get()
        self.assertEqual((evaluation.status, evaluation.folds, evaluation.started_by), ('queued', 5, self.staff))

    def test_run_rejects_other_fold_counts(self):
        self.client.force_login(self.staff)
        for folds in ('2', '10', 'x', ''):
            self.client.post(reverse('categorization_chatbot_run'), {'folds': folds})
        self.assertFalse(ChatbotEvaluation.objects.exists())

    def test_run_needs_post(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(reverse('categorization_chatbot_run')).status_code, 405)

    def test_second_run_while_active_shows_message(self):
        ChatbotEvaluation.objects.create(status='running', folds=3)
        self.client.force_login(self.staff)
        response = self.client.post(reverse('categorization_chatbot_run'), {'folds': '3'}, follow=True)
        self.assertEqual(ChatbotEvaluation.objects.count(), 1)
        self.assertContains(response, 'A chatbot evaluation is already in progress.')

    def test_queued_run_waits_for_worker(self):
        ChatbotEvaluation.objects.create(status='queued', folds=3)
        self.client.force_login(self.staff)
        self.assertContains(self.page(), 'Waiting for the chatbot evaluation worker')

    def test_status_json(self):
        self.client.force_login(self.staff)
        url = reverse('categorization_chatbot_status')
        self.assertEqual(self.client.get(url).json(), {})
        evaluation = ChatbotEvaluation.objects.create(status='running', folds=3, progress='Fold 2 / 3')
        self.assertEqual(self.client.get(url).json(), {
            'id': evaluation.id, 'status': 'running', 'progress': 'Fold 2 / 3', 'error': '',
        })

    def test_finished_evaluation_is_shown(self):
        finished_evaluation()
        self.client.force_login(self.staff)
        response = self.page()
        self.assertContains(response, '60.0%')
        self.assertContains(response, 'hello there')
        self.assertContains(response, 'Download Excel')
        self.assertEqual(response.context['chatbot']['summary']['wrong'], 2)

    def test_pick_an_earlier_evaluation(self):
        older = finished_evaluation(report={**report(), 'accuracy': 0.4})
        finished_evaluation()
        self.client.force_login(self.staff)
        self.assertAlmostEqual(self.page().context['chatbot']['summary']['accuracy'], 0.6)
        self.assertAlmostEqual(self.page(evaluation=older.id).context['chatbot']['summary']['accuracy'], 0.4)

    def test_dropdown_does_not_load_results(self):
        finished_evaluation()
        self.client.force_login(self.staff)
        evaluations = list(self.page().context['evaluations'])
        self.assertIn('report', evaluations[0].get_deferred_fields())

    def test_failed_evaluation_error_is_shown(self):
        ChatbotEvaluation.objects.create(status='failed', folds=3, error='Rasa stopped with exit code 1.')
        self.client.force_login(self.staff)
        self.assertContains(self.page(), 'Rasa stopped with exit code 1.')

    def test_final_tab_does_not_show_chatbot_section(self):
        finished_evaluation()
        self.client.force_login(self.staff)
        self.assertNotContains(self.client.get(self.url), 'hello there')


class ChatbotExportTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)
        self.client.force_login(self.staff)

    def test_workbook(self):
        evaluation = finished_evaluation()
        response = self.client.get(reverse('categorization_chatbot_export', args=[evaluation.id]))
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'chatbot-intents-{evaluation.id}.xlsx', response['Content-Disposition'])
        book = openpyxl.load_workbook(BytesIO(response.content))
        self.assertEqual(book.sheetnames, [
            'Summary', 'Per intent', 'Confusion matrix', 'Wrong predictions', 'All predictions',
        ])
        self.assertEqual(book['Wrong predictions'].max_row, 3)
        self.assertEqual(book['All predictions'].max_row, 6)
        # A sentence that looks like a formula stays text.
        cells = [c for row in book['Wrong predictions'].iter_rows() for c in row if c.value == '=1+1']
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].data_type, 's')

    def test_unfinished_404(self):
        running = ChatbotEvaluation.objects.create(status='running', folds=3)
        for evaluation_id in (running.id, 99999):
            response = self.client.get(reverse('categorization_chatbot_export', args=[evaluation_id]))
            self.assertEqual(response.status_code, 404)


def load_worker():
    path = Path(settings.BASE_DIR).parent / 'deploy' / 'rasa-eval' / 'worker.py'
    spec = importlib.util.spec_from_file_location('rasa_eval_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkerHelperTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.worker = load_worker()

    def test_writes_only_columns_the_model_has(self):
        self.assertEqual(self.worker.TABLE, ChatbotEvaluation._meta.db_table)
        columns = {field.column for field in ChatbotEvaluation._meta.concrete_fields}
        self.assertLessEqual(set(self.worker.RESULT_COLUMNS), columns)

    def test_fallback_threshold(self):
        threshold = self.worker.fallback_threshold
        self.assertEqual(threshold({'pipeline': None}), 0.3)
        self.assertEqual(threshold({}), 0.3)
        self.assertEqual(threshold({'pipeline': [
            {'name': 'DIETClassifier'}, {'name': 'FallbackClassifier', 'threshold': 0.45},
        ]}), 0.45)

    def test_progress_counts_folds(self):
        progress = self.worker.Progress(folds=3)
        diet = "INFO rasa.engine.training.hooks  - Starting to train component 'DIETClassifier'."
        self.assertIsNone(progress.read('some other line'))
        self.assertEqual(progress.read(diet), 'Fold 1 / 3: training')
        self.assertIsNone(progress.read(diet.replace('Starting', 'Finished')))
        self.assertEqual(progress.read(diet), 'Fold 2 / 3: training')
        self.assertEqual(progress.read('INFO rasa.nlu.test  - Accumulated test folds intent evaluation results:'),
                         'Writing the results')

    def test_collect_results(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'intent_report.json').write_text('{"accuracy": 0.5}', encoding='utf-8')
            (out / 'intent_errors.json').write_text('[{"text": "γεια"}]', encoding='utf-8')
            collected = self.worker.collect_results(out)
        self.assertEqual(collected['report'], {'accuracy': 0.5})
        self.assertEqual(collected['errors'], [{'text': 'γεια'}])
        self.assertEqual(collected['successes'], [])

    def test_rasa_command(self):
        command = self.worker.rasa_command(5)
        self.assertEqual(command[:3], ['rasa', 'test', 'nlu'])
        self.assertIn('--cross-validation', command)
        self.assertEqual(command[command.index('--folds') + 1], '5')
        self.assertIn('--successes', command)
        self.assertNotIn('--enable-api', command)

    def test_worker_file_is_mounted_path(self):
        self.assertTrue(os.path.exists(Path(settings.BASE_DIR).parent / 'deploy' / 'rasa-eval' / 'worker.py'))
