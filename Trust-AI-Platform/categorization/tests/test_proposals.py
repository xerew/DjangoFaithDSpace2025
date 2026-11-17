"""AI proposals tab: how often teachers accept the AI activity proposals."""

from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import openpyxl
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import ActivityProposal, Scenario, UserProposalReview
from categorization import proposals_report

UTC = ZoneInfo('UTC')


class ProposalData(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('staff', password='pass', is_staff=True)
        self.teachers = [User.objects.create_user(f'teacher{i}') for i in range(3)]
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='public', created_by=self.staff, updated_by=self.staff,
        )
        self.other = Scenario.objects.create(
            name='Optics', visibility_status='public', created_by=self.staff, updated_by=self.staff,
        )

    def proposal(self, when, proposal_type='revise', policy='', model='', scenario=None):
        proposal = ActivityProposal.objects.create(
            scenario=scenario or self.scenario, proposal_type=proposal_type,
            selection_policy=policy, llm_model=model, suggested_action='Do this',
        )
        ActivityProposal.objects.filter(pk=proposal.pk).update(created_at=when)
        return proposal

    def decide(self, proposal, *statuses, reasons=None, feedback='pedagogical'):
        # bulk_create: no reward signal, which is not what is tested here.
        UserProposalReview.objects.bulk_create([
            UserProposalReview(proposal=proposal, user=teacher, status=status,
                               rejection_reasons=(reasons or []) if status == 'rejected' else [],
                               feedback_type=feedback if status == 'rejected' else 'pedagogical')
            for teacher, status in zip(self.teachers, statuses)
        ])


class BuildTests(ProposalData):
    def test_counts_every_teacher_decision(self):
        july = self.proposal(datetime(2025, 7, 10, 9, tzinfo=UTC))
        self.decide(july, 'accepted', 'accepted', 'rejected')
        september = self.proposal(datetime(2025, 9, 3, 9, tzinfo=UTC), proposal_type='create',
                                  policy='weighted_exploration', model='qwen', scenario=self.other)
        self.decide(september, 'rejected', 'new')
        view = proposals_report.build()
        s = view['summary']
        self.assertEqual((s['proposals'], s['decisions'], s['accepted'], s['rejected']), (2, 4, 2, 2))
        self.assertEqual(s['acceptance'], 0.5)
        self.assertEqual(s['undecided'], 1)
        self.assertEqual(s['undecided_share'], 0.2)
        self.assertEqual(s['reviewers'], 3)
        self.assertEqual(s['proposals_decided'], 2)
        self.assertEqual([m['month'] for m in view['months']], ['2025-07', '2025-09'])
        self.assertAlmostEqual(view['months'][0]['acceptance'], 2 / 3)
        self.assertEqual(view['months'][1]['acceptance'], 0.0)
        self.assertEqual(view['chart']['decisions'], [3, 1])
        self.assertEqual(view['chart']['undecided'], [0, 1])

    def test_month_in_platform_time(self):
        # 22:30 UTC on 31 July is 01:30 on 1 August in Athens.
        self.proposal(datetime(2025, 7, 31, 22, 30, tzinfo=UTC))
        self.assertEqual(proposals_report.build()['months'][0]['month'], '2025-08')

    def test_month_without_decisions_has_no_rate(self):
        proposal = self.proposal(datetime(2025, 7, 10, tzinfo=UTC))
        self.decide(proposal, 'new')
        view = proposals_report.build()
        self.assertIsNone(view['months'][0]['acceptance'])
        self.assertIsNone(view['summary']['acceptance'])

    def test_by_kind(self):
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted')
        self.decide(self.proposal(datetime(2025, 7, 2, tzinfo=UTC), proposal_type='skip',
                                  policy='weighted_exploration', model='qwen'), 'rejected')
        kinds = {kind['title']: {row['label']: row['acceptance'] for row in kind['rows']}
                 for kind in proposals_report.build()['kinds']}
        self.assertEqual(kinds['By proposal type'], {'Revise Activity': 1.0, 'Skip Activity': 0.0})
        self.assertEqual(kinds['By how the action was chosen'], {
            'Chosen by the learning bandit': 0.0,
            'Before the bandit (no selection policy recorded)': 1.0,
        })
        self.assertEqual(kinds['By language model'], {'Not recorded': 1.0, 'qwen': 0.0})

    def test_unreviewed_proposals_are_counted_everywhere(self):
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted')
        self.proposal(datetime(2025, 7, 2, tzinfo=UTC), proposal_type='create', scenario=self.other)
        view = proposals_report.build()
        types = {row['label']: row for row in view['kinds'][0]['rows']}
        self.assertEqual(types['Create New Activity']['proposals'], 1)
        self.assertIsNone(types['Create New Activity']['acceptance'])
        self.assertEqual(sum(row['proposals'] for row in view['scenarios']), 2)
        self.assertEqual(view['months'][0]['proposals'], 2)

    def test_teacher_agreement(self):
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted', 'accepted', 'accepted')
        split = self.proposal(datetime(2025, 7, 2, tzinfo=UTC))
        self.decide(split, 'accepted', 'rejected', 'new')
        self.decide(self.proposal(datetime(2025, 7, 3, tzinfo=UTC)), 'accepted')  # one teacher only
        agreement = proposals_report.build()['agreement']
        self.assertEqual(agreement['proposals'], 2)
        self.assertEqual(agreement['unanimous'], 1)
        # 3 agreeing pairs on the first, 1 disagreeing pair on the split one.
        self.assertEqual(agreement['pairs'], 4)
        self.assertEqual(agreement['pairwise'], 0.75)
        self.assertEqual([row['proposal_id'] for row in agreement['split']], [split.id])

    def test_rejection_reasons_and_structural(self):
        proposal = self.proposal(datetime(2025, 7, 1, tzinfo=UTC))
        self.decide(proposal, 'rejected', reasons=['structural_invalid', 'poor_content'], feedback='structural')
        view = proposals_report.build()
        self.assertEqual(view['summary']['structural'], 1)
        self.assertEqual({row['code'] for row in view['reasons']}, {'structural_invalid', 'poor_content'})
        self.assertEqual(view['decisions'][0]['reasons'][0], 'Malformed or structurally invalid proposal')

    def test_nothing_yet(self):
        view = proposals_report.build()
        self.assertEqual(view['summary']['proposals'], 0)
        self.assertEqual(view['months'], [])


class ProposalsTabTests(ProposalData):
    url = reverse('categorization_dashboard')

    def test_tab(self):
        self.client.force_login(self.staff)
        self.decide(self.proposal(datetime(2025, 7, 10, tzinfo=UTC)), 'accepted', 'rejected')
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Teachers and the AI activity proposals')
        self.assertContains(response, 'Jul 2025')
        self.assertContains(response, 'id="proposals-chart-data"')
        self.assertContains(response, reverse('categorization_proposals_export'))

    def test_empty_tab(self):
        self.client.force_login(self.staff)
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertContains(response, 'No AI proposals yet.')
        self.assertNotContains(response, 'id="proposals-chart-data"')

    def test_staff_only(self):
        self.client.force_login(self.teachers[0])
        self.assertEqual(self.client.get(self.url, {'tab': 'proposals'}).status_code, 403)
        self.assertEqual(self.client.get(reverse('categorization_proposals_export')).status_code, 403)

    def test_export(self):
        self.client.force_login(self.staff)
        proposal = self.proposal(datetime(2025, 7, 10, tzinfo=UTC))
        self.decide(proposal, 'accepted', 'rejected', reasons=['poor_content'])
        response = self.client.get(reverse('categorization_proposals_export'))
        self.assertEqual(response.status_code, 200)
        book = openpyxl.load_workbook(BytesIO(response.content))
        self.assertEqual(book.sheetnames, ['Summary', 'By month', 'By kind', 'By scenario', 'Rejection reasons',
                                           'Split decisions', 'Decisions'])
        decisions = book['Decisions']
        self.assertEqual(decisions.max_row, 3)  # header + 2 decisions
        self.assertEqual(decisions.cell(row=2, column=6).value, '2025-07-10 03:00')  # Athens time
