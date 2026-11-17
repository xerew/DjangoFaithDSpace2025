"""AI proposals tab: how often teachers accept the AI activity proposals."""

from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import openpyxl
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import (
    Activity, ActivityProposal, ActivityType, Phase, ProposalGenerationRun, Scenario, UserProposalReview,
)
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
        kind = ActivityType.objects.create(name='Explanation', created_by=self.staff, updated_by=self.staff)
        self.activities = {}
        for scenario in (self.scenario, self.other):
            phase = Phase.objects.create(name='Phase', scenario=scenario, created_by=self.staff, updated_by=self.staff)
            self.activities[scenario.id] = Activity.objects.create(
                name='Activity', text='Text', plain_text='Text', scenario=scenario, phase=phase,
                activity_type=kind, created_by=self.staff, updated_by=self.staff,
            )

    def proposal(self, when, proposal_type='revise', policy='', model='', scenario=None, run=None, detached=False):
        scenario = scenario or self.scenario
        proposal = ActivityProposal.objects.create(
            scenario=scenario, proposal_type=proposal_type, generation_run=run,
            activity=None if detached else self.activities[scenario.id],
            selection_policy=policy, llm_model=model, suggested_action='Do this',
        )
        ActivityProposal.objects.filter(pk=proposal.pk).update(created_at=when)
        return proposal

    def decide(self, proposal, *statuses, reasons=None, feedback='pedagogical', users=None):
        # bulk_create: no reward signal, which is not what is tested here.
        UserProposalReview.objects.bulk_create([
            UserProposalReview(proposal=proposal, user=user, status=status,
                               rejection_reasons=(reasons or []) if status == 'rejected' else [],
                               feedback_type=feedback if status == 'rejected' else 'pedagogical')
            for user, status in zip(users or self.teachers, statuses)
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
        # The scenario has no generation runs: its proposal can still be decided.
        self.assertEqual((s['open'], s['closed']), (1, 0))
        self.assertEqual(s['reviewers'], 3)
        self.assertEqual(s['proposals_decided'], 2)
        # August has no proposals and is kept as a gap.
        self.assertEqual([m['month'] for m in view['months']], ['2025-07', '2025-08', '2025-09'])
        self.assertAlmostEqual(view['months'][0]['acceptance'], 2 / 3)
        self.assertIsNone(view['months'][1]['acceptance'])
        self.assertEqual(view['months'][2]['acceptance'], 0.0)
        self.assertEqual(view['chart']['decisions'], [3, 0, 1])
        self.assertEqual(view['chart']['open'], [0, 0, 1])

    def test_still_open_or_closed(self):
        old = ProposalGenerationRun.objects.create(scenario=self.scenario, created_by=self.staff, is_current=False)
        current = ProposalGenerationRun.objects.create(scenario=self.scenario, created_by=self.staff, is_current=True)
        when = datetime(2025, 7, 1, tzinfo=UTC)
        self.decide(self.proposal(when, run=current), 'new')                 # on the review page
        self.decide(self.proposal(when, run=old), 'new')                     # replaced by a newer run
        self.decide(self.proposal(when), 'new')                              # before runs, but a run exists now
        self.decide(self.proposal(when, run=current, detached=True), 'new')  # activity removed
        self.decide(self.proposal(when, scenario=self.other), 'new')         # scenario without runs
        s = proposals_report.build()['summary']
        self.assertEqual((s['open'], s['closed']), (2, 3))

    def test_teachers_only(self):
        proposal = self.proposal(datetime(2025, 7, 1, tzinfo=UTC))
        self.decide(proposal, 'accepted', 'rejected', users=[self.staff, self.teachers[0]])
        everyone = proposals_report.build()['summary']
        self.assertEqual((everyone['decisions'], everyone['staff_decisions']), (2, 1))
        teachers = proposals_report.build('teachers')['summary']
        self.assertEqual((teachers['decisions'], teachers['acceptance'], teachers['reviewers']), (1, 0.0, 1))
        self.assertEqual(teachers['who'], 'teachers')

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
        # 4 of the 5 decisions accepted: chance agreement 0.8^2 + 0.2^2.
        self.assertAlmostEqual(agreement['chance'], 0.68)
        self.assertEqual([row['proposal_id'] for row in agreement['split']], [split.id])

    def test_rejection_reasons_and_structural(self):
        proposal = self.proposal(datetime(2025, 7, 1, tzinfo=UTC))
        self.decide(proposal, 'rejected', reasons=['structural_invalid', 'poor_content'], feedback='structural')
        view = proposals_report.build()
        self.assertEqual(view['summary']['structural'], 1)
        self.assertEqual({row['code'] for row in view['reasons']}, {'structural_invalid', 'poor_content'})
        self.assertEqual(view['decisions'][0]['reasons'][0], 'Malformed or structurally invalid proposal')
        self.assertEqual(view['decisions'][0]['decision'], 'Rejected')
        self.assertEqual(view['decisions'][0]['feedback_type'], 'Structural / malformed')

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
        self.assertContains(response, reverse('categorization_proposals_export') + '?who=all')

    def test_agreement_sentence(self):
        self.client.force_login(self.staff)
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted', 'accepted')
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertContains(response, '1 proposal was decided by two or more reviewers')
        self.assertContains(response, 'Pairs of reviewers who decided the same proposal: 1; 100.0% of them decided alike.')
        self.assertContains(response, 'On these shared proposals 100.0% of the decisions were acceptances')
        self.assertContains(response, 'so this agreement says nothing beyond chance')

    def test_mixed_agreement_has_no_warning(self):
        self.client.force_login(self.staff)
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted', 'rejected')
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertContains(response, '0.0% of them decided alike')
        self.assertContains(response, 'would agree 50.0% of the time by chance')
        self.assertNotContains(response, 'says nothing beyond chance')

    def test_undecided_labels_say_the_page_was_opened(self):
        self.client.force_login(self.staff)
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'new')
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertContains(response, 'Opened, not decided (still decidable)')
        self.assertContains(response, 'Opened, not decided (no longer decidable)')
        self.assertNotContains(response, 'Not decided, still open')
        # Months with decisions are joined into one line across empty months.
        self.assertContains(response, 'connectNulls: true')

    def test_teachers_only_page(self):
        self.client.force_login(self.staff)
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted', 'rejected',
                    users=[self.staff, self.teachers[0]])
        response = self.client.get(self.url, {'tab': 'proposals', 'who': 'teachers'})
        self.assertEqual(response.context['proposals']['summary']['decisions'], 1)
        self.assertContains(response, '?who=teachers')
        everyone = self.client.get(self.url, {'tab': 'proposals', 'who': 'bogus'})
        self.assertEqual(everyone.context['proposals']['summary']['who'], 'all')
        self.assertContains(everyone, '1 of these decisions was made by a staff or admin account.')

    def test_teacher_written_names_are_escaped(self):
        self.client.force_login(self.staff)
        Scenario.objects.filter(pk=self.scenario.pk).update(name='<script>alert(1)</script>')
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted')
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertNotContains(response, '<script>alert(1)</script>')
        self.assertContains(response, '&lt;script&gt;alert(1)&lt;/script&gt;')

    def test_empty_tab(self):
        self.client.force_login(self.staff)
        response = self.client.get(self.url, {'tab': 'proposals'})
        self.assertContains(response, 'No AI proposals yet.')
        self.assertNotContains(response, 'id="proposals-chart-data"')

    def test_staff_only(self):
        self.client.force_login(self.teachers[0])
        self.assertEqual(self.client.get(self.url, {'tab': 'proposals'}).status_code, 403)
        self.assertEqual(self.client.get(reverse('categorization_proposals_export')).status_code, 403)

    def test_anonymous_export_goes_to_login(self):
        response = self.client.get(reverse('categorization_proposals_export'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response['Location'])

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
        self.assertEqual(decisions.cell(row=1, column=12).value, 'Last changed')
        self.assertEqual(decisions.cell(row=2, column=9).value, 'Accepted')

    def test_export_keeps_odd_names_as_text(self):
        self.client.force_login(self.staff)
        Scenario.objects.filter(pk=self.scenario.pk).update(name='=cmd|x\x0bpasted')
        self.decide(self.proposal(datetime(2025, 7, 1, tzinfo=UTC)), 'accepted')
        response = self.client.get(reverse('categorization_proposals_export'))
        self.assertEqual(response.status_code, 200)
        cell = openpyxl.load_workbook(BytesIO(response.content))['Decisions'].cell(row=2, column=2)
        self.assertEqual(cell.data_type, 's')
        self.assertEqual(cell.value, '=cmd|xpasted')
