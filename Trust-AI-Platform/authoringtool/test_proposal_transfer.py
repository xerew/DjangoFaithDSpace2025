"""Proposals generated on a local copy of the platform (from a database dump)
are exported to a file and imported on the server.

The dump keeps the server's ids, so scenarios, phases and activities are
matched by id, and checked by name in case the server changed since.
"""
import json
from datetime import timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from authoringtool.models import (
    Activity,
    ActivityFlag,
    ActivityProposal,
    ActivityType,
    CategoryTag,
    Phase,
    ProposalGenerationRun,
    ProposalStructuralFailure,
    Scenario,
)
from authoringtool.proposal_transfer import FORMAT, export_runs, import_runs


class TransferFixture(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user('transfer_owner')
        self.scenario = Scenario.objects.create(
            name='Pendulum', created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='P', scenario=self.scenario, created_by=self.owner, updated_by=self.owner,
        )
        question = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        self.activities = [
            Activity.objects.create(
                name=f'Q{n}', text='q', phase=self.phase, scenario=self.scenario,
                activity_type=question, created_by=self.owner, updated_by=self.owner,
            )
            for n in (1, 2)
        ]
        self.old_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, is_current=True, created_by=self.owner,
        )
        ProposalGenerationRun.objects.filter(pk=self.old_run.pk).update(
            created_at=timezone.now() - timedelta(days=10),
        )
        self.old_run.refresh_from_db()

    def make_local_run(self, activities=None):
        """What a Generate on the local copy leaves behind."""
        run = ProposalGenerationRun.start_new(
            self.scenario, self.owner, evidence_summary={'implementation_count': 300},
        )
        tag, _ = CategoryTag.objects.get_or_create(name='Low')
        for activity in activities or self.activities:
            flag = ActivityFlag.objects.create(
                activity=activity, scenario=self.scenario, phase=self.phase,
                category='Low', flag_type='Systemic failure', flag_reason='r',
                evidence_signature='sig',
            )
            proposal = ActivityProposal.objects.create(
                scenario=self.scenario, generation_run=run, phase=self.phase,
                activity=activity, proposal_type='create',
                suggested_action='{"action": "create"}', translated_action='{"t": 1}',
                json_action='{"action": "create"}', json_translated_action='{"t": 1}',
                llm_model='qwen3.6:35b', prompt_version='2026-11-27',
                selection_policy='weighted_exploration', selection_probability=0.7,
                action_probabilities={'create': 0.7, 'revise': 0.3},
                selection_contexts=[['Systemic failure', 'Low']],
            )
            proposal.flag.set([flag])
            proposal.categories_in_risk.add(tag)
        ProposalStructuralFailure.objects.create(
            scenario=self.scenario, generation_run=run, activity=self.activities[0],
            selected_action='revise', stage='generation', errors=['bad json'],
        )
        return run

    def exported(self, run):
        """The file contents, after the local rows are gone (as on the server)."""
        data = json.loads(json.dumps(export_runs([run])))
        ProposalStructuralFailure.objects.filter(generation_run=run).delete()
        ActivityFlag.objects.filter(proposals__generation_run=run).delete()
        run.proposals.all().delete()
        run.delete()
        self.old_run.refresh_from_db()
        ProposalGenerationRun.objects.filter(pk=self.old_run.pk).update(is_current=True)
        return data

    def current(self):
        return ProposalGenerationRun.objects.get(scenario=self.scenario, is_current=True)


class ExportTests(TransferFixture):
    def test_file_names_its_format_and_carries_everything(self):
        run = self.make_local_run()
        data = export_runs([run])
        self.assertEqual(data['format'], FORMAT)
        [item] = data['runs']
        self.assertEqual(item['scenario_id'], self.scenario.id)
        self.assertEqual(item['scenario_name'], 'Pendulum')
        self.assertEqual(len(item['proposals']), 2)
        proposal = item['proposals'][0]
        self.assertEqual(proposal['activity_name'], 'Q1')
        self.assertEqual(proposal['categories'], ['Low'])
        self.assertEqual(proposal['flags'][0]['flag_type'], 'Systemic failure')
        self.assertEqual(proposal['action_probabilities'], {'create': 0.7, 'revise': 0.3})
        self.assertEqual(len(item['failures']), 1)
        json.dumps(data)  # serialisable

    def test_command_writes_current_runs_since_a_time(self):
        run = self.make_local_run()
        with TemporaryDirectory() as folder:
            path = Path(folder, 'out.json')
            since = (timezone.now() - timedelta(days=1)).isoformat()
            call_command('export_proposals', '--output', str(path), '--since', since,
                         stdout=StringIO())
            data = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual([r['id'] for r in data['runs']], [run.id])


class ImportTests(TransferFixture):
    def test_import_replaces_the_current_run_like_a_finished_generate(self):
        data = self.exported(self.make_local_run())
        report = import_runs(data)

        new = self.current()
        self.assertNotEqual(new, self.old_run)
        self.old_run.refresh_from_db()
        self.assertFalse(self.old_run.is_current)
        self.assertEqual(new.proposals.count(), 2)
        proposal = new.proposals.get(activity=self.activities[0])
        self.assertEqual(proposal.status, 'new')
        self.assertEqual(proposal.llm_model, 'qwen3.6:35b')
        self.assertEqual(proposal.selection_probability, 0.7)
        self.assertEqual(proposal.flag.get().flag_type, 'Systemic failure')
        self.assertEqual(proposal.flag.get().activity, self.activities[0])
        self.assertEqual(list(proposal.categories_in_risk.values_list('name', flat=True)), ['Low'])
        self.assertEqual(new.evidence_summary['implementation_count'], 300)
        self.assertEqual(new.structural_failures.count(), 1)
        self.assertEqual(report.imported, 1)

    def test_times_are_kept(self):
        run = self.make_local_run()
        created = run.created_at
        data = self.exported(run)
        import_runs(data)
        self.assertEqual(self.current().created_at, created)

    def test_dry_run_changes_nothing(self):
        data = self.exported(self.make_local_run())
        report = import_runs(data, dry_run=True)
        self.assertEqual(self.current(), self.old_run)
        self.assertEqual(ProposalGenerationRun.objects.count(), 1)
        self.assertEqual(report.imported, 1)  # what would happen

    def test_run_already_on_the_server_is_skipped(self):
        run = self.make_local_run()
        report = import_runs(json.loads(json.dumps(export_runs([run]))))
        self.assertEqual(report.imported, 0)
        self.assertIn('already on this server', '\n'.join(report.lines))
        self.assertEqual(ProposalGenerationRun.objects.count(), 2)

    def test_proposal_whose_activity_changed_is_skipped(self):
        data = self.exported(self.make_local_run())
        self.activities[1].name = 'Renamed on the server'
        self.activities[1].save()
        report = import_runs(data)
        self.assertEqual(self.current().proposals.count(), 1)
        self.assertIn('Q2', '\n'.join(report.lines))

    def test_proposal_whose_activity_is_gone_is_skipped(self):
        data = self.exported(self.make_local_run())
        gone = self.activities[1].id
        Activity.objects.filter(id=gone).delete()
        import_runs(data)
        self.assertEqual(self.current().proposals.count(), 1)

    def test_run_left_with_no_proposals_is_not_imported(self):
        data = self.exported(self.make_local_run())
        Activity.objects.filter(id__in=[a.id for a in self.activities]).update(name='x')
        report = import_runs(data)
        self.assertEqual(self.current(), self.old_run)
        self.assertEqual(report.imported, 0)

    def test_newer_proposals_on_the_server_win_unless_forced(self):
        data = self.exported(self.make_local_run())
        ProposalGenerationRun.objects.filter(pk=self.old_run.pk).update(
            created_at=timezone.now() + timedelta(hours=1),
        )
        report = import_runs(data)
        self.assertEqual(self.current(), self.old_run)
        self.assertIn('newer', '\n'.join(report.lines))
        import_runs(data, force=True)
        self.assertNotEqual(self.current(), self.old_run)

    def test_scenario_with_a_running_job_is_skipped(self):
        data = self.exported(self.make_local_run())
        with patch('authoringtool.proposal_transfer.job_status',
                   return_value={'running': True}):
            report = import_runs(data)
        self.assertEqual(self.current(), self.old_run)
        self.assertIn('job is running', '\n'.join(report.lines))

    def test_unknown_scenario_is_skipped(self):
        data = self.exported(self.make_local_run())
        data['runs'][0]['scenario_id'] = 999999
        report = import_runs(data)
        self.assertEqual(report.imported, 0)

    def test_wrong_file_is_refused(self):
        with self.assertRaises(ValueError):
            import_runs({'format': 'something else', 'runs': []})

    def test_command_reads_the_file(self):
        data = self.exported(self.make_local_run())
        with TemporaryDirectory() as folder:
            path = Path(folder, 'in.json')
            path.write_text(json.dumps(data), encoding='utf-8')
            out = StringIO()
            call_command('import_proposals', str(path), stdout=out)
        self.assertIn('Pendulum', out.getvalue())
        self.assertNotEqual(self.current(), self.old_run)
