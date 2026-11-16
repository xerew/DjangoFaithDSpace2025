"""Progress of proposal generation, remembered per scenario.

The job reports its stages to the shared cache; the Metrics & AI page reads
them by scenario, so a reload or another teacher sees the same progress and
Generate stays locked while a run is going.
"""
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from authoringtool.machine_translation.health import AIServerUnavailable
from authoringtool.proposal_progress import JobProgress, job_status, reserve
from authoringtool.models import Scenario
from authoringtool.tasks import ProposalRunAborted
from authoringtool import test_proposal_run_safety as run_safety
from authoringtool.test_ai_server_banner import server_up

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
NOW = 'authoringtool.proposal_progress.timezone.now'


@override_settings(CACHES=LOCMEM)
class JobProgressTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_no_job_is_none(self):
        self.assertIsNone(job_status(5))

    def test_stages_report_percent_summaries_and_the_current_item(self):
        progress = JobProgress(5, started_by='Eva')
        progress.stage('check')
        progress.stage('pdfs')
        progress.summary('Read the PDFs (2 files)')
        progress.stage('activities', total=4)
        progress.step(2, '"Pendulum length"')

        status = job_status(5)

        self.assertTrue(status['running'])
        self.assertEqual(status['started_by'], 'Eva')
        self.assertEqual(status['stage'], 'activities')
        self.assertEqual(status['percent'], 18)  # 2 + 4 + 25 * 2/4
        self.assertEqual(status['detail'], '"Pendulum length"')
        stages = status['stages']
        self.assertEqual(len(stages), 8)
        self.assertEqual(stages[0]['status'], 'done')
        self.assertEqual(stages[1]['summary'], 'Read the PDFs (2 files)')
        self.assertEqual(stages[2]['status'], 'active')
        self.assertEqual((stages[2]['done'], stages[2]['total']), (2, 4))
        self.assertEqual(stages[3]['status'], 'pending')

    def test_counted_stage_summary_gives_the_count(self):
        progress = JobProgress(5)
        progress.stage('activities', total=3)
        progress.step(3)
        progress.stage('overview')

        self.assertEqual(job_status(5)['stages'][2]['summary'], 'Summarised activities (3 of 3)')

    def test_stage_with_nothing_to_do_says_up_to_date(self):
        progress = JobProgress(5)
        progress.stage('activities', total=0)
        progress.stage('overview')

        self.assertIn('already up to date', job_status(5)['stages'][2]['summary'])

    def test_finished_job_fills_the_bar_and_unlocks(self):
        progress = JobProgress(5)
        progress.stage('proposals', total=2)
        progress.finish(True, '2 new proposals are ready.')

        status = job_status(5)

        self.assertEqual(status['state'], 'done')
        self.assertFalse(status['running'])
        self.assertEqual(status['percent'], 100)
        self.assertEqual(status['message'], '2 new proposals are ready.')
        self.assertTrue(all(stage['status'] == 'done' for stage in status['stages']))

    def test_failed_job_keeps_where_it_stopped(self):
        progress = JobProgress(5)
        progress.stage('check')
        progress.finish(False, "The AI server isn't answering (502 Bad Gateway).")

        status = job_status(5)

        self.assertEqual(status['state'], 'failed')
        self.assertFalse(status['running'])
        self.assertLess(status['percent'], 100)
        self.assertEqual(status['stages'][0]['status'], 'failed')
        self.assertIn('502', status['message'])

    def test_job_silent_for_half_an_hour_no_longer_locks(self):
        start = timezone.now()
        with patch(NOW, return_value=start):
            JobProgress(5).stage('proposals', total=9)
        with patch(NOW, return_value=start + timedelta(minutes=31)):
            status = job_status(5)

        self.assertEqual(status['state'], 'stuck')
        self.assertFalse(status['running'])

    def test_reserve_locks_until_the_job_ends(self):
        self.assertTrue(reserve(5, 'Eva'))
        self.assertFalse(reserve(5, 'Nikos'))
        self.assertEqual(job_status(5)['started_by'], 'Eva')

        JobProgress(5).finish(True, 'done')

        self.assertTrue(reserve(5, 'Nikos'))

    def test_job_keeps_the_reserved_starter(self):
        reserve(5, 'Eva')

        JobProgress(5, started_by='someone else').stage('check')

        self.assertEqual(job_status(5)['started_by'], 'Eva')


@override_settings(CACHES=LOCMEM)
class ProgressViewTests(TestCase):
    def setUp(self):
        cache.clear()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user('progress_owner', first_name='Eva', last_name='P')
        self.owner.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.client.force_login(self.owner)
        self.generate_url = reverse('generate_llm_context', args=[self.scenario.id])
        self.status_url = reverse('proposal_job_status', args=[self.scenario.id])
        self.metrics_url = reverse('ai_metrics', args=[self.scenario.id])

    def test_second_generate_is_refused_while_one_runs(self):
        server_up()
        with patch('authoringtool.views.generate_llm_context_for_scenario.delay') as delay:
            delay.return_value.id = 'task-1'
            first = self.client.post(self.generate_url)
            second = self.client.post(self.generate_url)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertIn('Eva P', second.json()['error'])
        self.assertEqual(delay.call_count, 1)
        self.assertEqual(job_status(self.scenario.id)['task_id'], 'task-1')

    def test_status_endpoint_returns_the_progress(self):
        reserve(self.scenario.id, 'Eva P')
        JobProgress(self.scenario.id).stage('pdfs')

        data = self.client.get(self.status_url).json()

        self.assertTrue(data['running'])
        self.assertEqual(data['stage'], 'pdfs')
        self.assertEqual(len(data['stages']), 8)

    def test_status_endpoint_without_a_job(self):
        self.assertEqual(self.client.get(self.status_url).json(), {'state': 'none', 'running': False})

    def test_status_endpoint_needs_view_rights(self):
        self.client.force_login(User.objects.create_user('progress_stranger'))

        self.assertEqual(self.client.get(self.status_url).status_code, 403)

    def test_metrics_page_shows_the_running_job_and_locks_generate(self):
        server_up()
        reserve(self.scenario.id, 'Eva P')

        response = self.client.get(self.metrics_url)

        self.assertContains(response, 'id="proposal-progress"')
        self.assertTrue(response.context['proposal_job']['running'])
        content = response.content.decode()
        tag = content[content.index('id="btn-generate-context"'):]
        self.assertIn('disabled', tag[:tag.index('>')])

    def test_metrics_page_without_a_job_leaves_generate_open(self):
        server_up()

        response = self.client.get(self.metrics_url)

        self.assertIsNone(response.context['proposal_job'])
        content = response.content.decode()
        tag = content[content.index('id="btn-generate-context"'):]
        self.assertNotIn('disabled', tag[:tag.index('>')])


@override_settings(CACHES=LOCMEM)
class JobReportsProgressTests(run_safety.GenerationJobFixture):
    def setUp(self):
        cache.clear()
        super().setUp()

    def test_finished_job_reports_every_stage(self):
        for activity in self.activities:
            self.flag(activity)

        self.run_job(proposals=[self.written(), self.written()])

        status = job_status(self.scenario.id)
        self.assertEqual(status['state'], 'done')
        self.assertEqual(status['percent'], 100)
        summaries = {stage['key']: stage['summary'] for stage in status['stages']}
        self.assertEqual(summaries['risk'], 'Found 2 activities at risk')
        self.assertEqual(summaries['proposals'], 'Wrote 2 proposals')
        self.assertIn('2 new proposals', status['message'])

    def test_failed_job_reports_its_reason(self):
        self.flag(self.activities[0])

        with self.assertRaises(ProposalRunAborted):
            self.run_job(probe=AIServerUnavailable("The AI server isn't answering (502 Bad Gateway)."))

        status = job_status(self.scenario.id)
        self.assertEqual(status['state'], 'failed')
        self.assertIn('502 Bad Gateway', status['message'])

    def test_current_proposal_names_the_activity_and_the_action(self):
        self.flag(self.activities[0])
        seen = []

        def write(*args, **kwargs):
            seen.append(job_status(self.scenario.id))
            return self.written()

        self.run_job(proposals=write)

        during = seen[0]
        self.assertEqual(during['stage'], 'proposals')
        self.assertIn('"Q1"', during['detail'])
        self.assertIn('writing', during['detail'])
        self.assertRegex(during['detail'], r'(CREATE|REVISE|SKIP)')
