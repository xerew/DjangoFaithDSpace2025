import json

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse

from .models import (
    Activity, ActivityFlag, ActivityProposal, ActivityType, Phase, ProposalGenerationRun,
    QValue, Scenario, UserProposalReview,
)
from .evidence import get_evidence_context
from .tasks import _build_personal_scenario, get_accepted_reviews_for_personal_scenario


class ProposalHistoryVisibilityTests(TestCase):
    def setUp(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user('history_owner')
        self.owner.groups.add(teachers)
        self.other_teacher = User.objects.create_user('history_other')
        self.other_teacher.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='History scenario', visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Phase', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        activity_type = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Activity', text='Original content', plain_text='Original content',
            scenario=self.scenario, phase=self.phase, activity_type=activity_type,
            created_by=self.owner, updated_by=self.owner,
        )
        self.version = self.scenario.ensure_current_version()
        self.client.force_login(self.owner)

    def proposal(self, run=None, text='Earlier recommendation'):
        return ActivityProposal.objects.create(
            scenario=self.scenario, phase=self.phase, activity=self.activity,
            generation_run=run, proposal_type='revise', suggested_action=text,
        )

    def history(self):
        return self.client.get(reverse('proposal_history', args=[self.scenario.id]))

    def test_unversioned_current_run_is_visible_without_changing_it(self):
        run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, created_by=self.owner, is_current=True,
        )
        proposal = self.proposal(run)
        response = self.history()
        self.assertContains(response, 'scenario version not recorded')
        detail_url = reverse('proposal_history_run_detail', args=[self.scenario.id, run.id])
        self.assertContains(response, detail_url)
        detail = self.client.get(detail_url)
        self.assertContains(detail, proposal.suggested_action)
        self.assertContains(detail, 'reference only')
        self.assertNotContains(detail, reverse('accept_proposal', args=[self.scenario.id, proposal.id]))
        run.refresh_from_db()
        self.assertTrue(run.is_current)
        self.assertIsNone(run.scenario_version_id)
        self.assertFalse(UserProposalReview.objects.exists())
        self.assertFalse(QValue.objects.exists())

    def test_current_run_for_previous_version_remains_visible(self):
        run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, scenario_version=self.version,
            created_by=self.owner, is_current=True,
        )
        self.proposal(run)
        self.activity.text = self.activity.plain_text = 'Updated content'
        self.activity.save()
        self.scenario.ensure_current_version()
        response = self.history()
        self.assertContains(response, 'Earlier scenario version')
        self.assertEqual(response.context['run_summaries'][0]['run'].id, run.id)

    def test_history_includes_current_and_archived_runs_and_personal_decisions(self):
        old_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, scenario_version=self.version, is_current=False,
        )
        current_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, scenario_version=self.version, is_current=True,
        )
        old_proposal = self.proposal(old_run)
        self.proposal(current_run)
        UserProposalReview.objects.create(proposal=old_proposal, user=self.owner, status='accepted')
        UserProposalReview.objects.create(proposal=old_proposal, user=self.other_teacher, status='rejected')
        response = self.history()
        summaries = {s['run'].id: s for s in response.context['run_summaries']}
        self.assertEqual(set(summaries), {old_run.id, current_run.id})
        self.assertEqual(summaries[old_run.id]['total'], 1)
        self.assertEqual(summaries[old_run.id]['accepted'], 1)
        self.assertEqual(summaries[old_run.id]['rejected'], 0)
        self.assertContains(response, 'Latest run for this scenario version')

    def test_proposals_without_run_are_browsable_and_scoped_to_scenario(self):
        proposal = self.proposal(text='Legacy visible recommendation')
        UserProposalReview.objects.create(proposal=proposal, user=self.owner, status='accepted')
        other_scenario = Scenario.objects.create(
            name='Other scenario', created_by=self.owner, updated_by=self.owner,
        )
        # Even malformed legacy associations must stay scoped by scenario.
        ActivityProposal.objects.create(
            scenario=other_scenario, phase=self.phase, activity=self.activity,
            proposal_type='revise', suggested_action='Unrelated recommendation',
        )
        response = self.history()
        legacy_url = reverse('proposal_history_legacy', args=[self.scenario.id])
        self.assertContains(response, legacy_url)
        self.assertEqual(response.context['legacy_summary']['total'], 1)
        self.assertEqual(response.context['legacy_summary']['accepted'], 1)
        detail = self.client.get(legacy_url)
        self.assertContains(detail, 'Legacy visible recommendation')
        self.assertContains(detail, 'Accepted')
        self.assertNotContains(detail, 'Unrelated recommendation')
        proposal.refresh_from_db()
        self.assertIsNone(proposal.generation_run_id)

    def test_private_history_and_legacy_details_require_scenario_access(self):
        run = ProposalGenerationRun.objects.create(scenario=self.scenario)
        self.proposal(run)
        self.proposal()
        self.client.force_login(self.other_teacher)
        for name, args in [
            ('proposal_history', [self.scenario.id]),
            ('proposal_history_legacy', [self.scenario.id]),
            ('proposal_history_run_detail', [self.scenario.id, run.id]),
        ]:
            with self.subTest(name=name):
                self.assertEqual(self.client.get(reverse(name, args=args)).status_code, 403)

    def test_scenario_and_metrics_link_history_with_no_implementations(self):
        history_url = reverse('proposal_history', args=[self.scenario.id])
        for name in ['viewScenario', 'ai_metrics']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name, args=[self.scenario.id]))
                self.assertContains(response, history_url)
                self.assertContains(response, 'Proposal History')

    def test_disabled_policy_shows_all_scenario_proposals_and_keeps_decisions(self):
        old_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, scenario_version=self.version, is_current=False,
        )
        self.activity.text = self.activity.plain_text = 'New version content'
        self.activity.save()
        self.scenario.ensure_current_version()
        unversioned_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, is_current=True,
        )
        legacy = self.proposal()
        old = self.proposal(old_run)
        unversioned = self.proposal(unversioned_run)
        review = UserProposalReview.objects.create(
            proposal=old, user=self.owner, status='accepted',
        )
        other_scenario = Scenario.objects.create(
            name='Different scenario', created_by=self.owner, updated_by=self.owner,
        )
        ActivityProposal.objects.create(
            scenario=other_scenario, activity=self.activity, phase=self.phase,
            proposal_type='revise', suggested_action='Other scenario proposal',
        )

        response = self.client.get(reverse('proposal_list', args=[self.scenario.id]))

        self.assertContains(response, 'Showing all saved proposals for this scenario')
        self.assertEqual(
            {p.id for p in response.context['proposals']},
            {legacy.id, old.id, unversioned.id},
        )
        self.assertEqual(response.context['accepted_count'], 1)
        self.assertEqual(response.context['pending_count'], 2)
        review.refresh_from_db()
        self.assertEqual(review.status, 'accepted')
        unversioned_run.refresh_from_db()
        self.assertTrue(unversioned_run.is_current)
        for proposal in [legacy, old, unversioned]:
            self.assertTrue(proposal.is_bandit_reward_eligible())

    def test_enabled_policy_keeps_current_version_filter_for_list_and_application(self):
        self.scenario.use_family_evidence_pooling = True
        self.scenario.save(update_fields=['use_family_evidence_pooling'])
        archived_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, scenario_version=self.version, is_current=False,
        )
        evidence = get_evidence_context(self.scenario, 'compatible')
        current_run = ProposalGenerationRun.start_new(
            self.scenario, self.owner, evidence_scope='compatible',
            evidence_version_ids=evidence['version_ids'], evidence_summary=evidence,
        )
        legacy = self.proposal()
        archived = self.proposal(archived_run)
        current = self.proposal(current_run)
        for proposal in [legacy, archived, current]:
            UserProposalReview.objects.create(
                proposal=proposal, user=self.owner, status='accepted',
            )

        response = self.client.get(reverse('proposal_list', args=[self.scenario.id]))

        self.assertEqual([p.id for p in response.context['proposals']], [current.id])
        self.assertEqual(
            list(get_accepted_reviews_for_personal_scenario(
                self.scenario, self.owner,
            ).values_list('proposal_id', flat=True)),
            [current.id],
        )
        self.assertFalse(legacy.is_bandit_reward_eligible())
        self.assertFalse(archived.is_bandit_reward_eligible())
        self.assertTrue(current.is_bandit_reward_eligible())

    def test_disabled_policy_applies_accepted_reviews_across_runs_for_this_user_only(self):
        run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, is_current=False,
        )
        legacy, archived, rejected, other_teacher = (
            self.proposal(), self.proposal(run), self.proposal(), self.proposal(),
        )
        for proposal in [legacy, archived]:
            UserProposalReview.objects.create(proposal=proposal, user=self.owner, status='accepted')
        UserProposalReview.objects.create(proposal=rejected, user=self.owner, status='rejected')
        UserProposalReview.objects.create(proposal=other_teacher, user=self.other_teacher, status='accepted')
        other_scenario = Scenario.objects.create(
            name='Unrelated scenario', created_by=self.owner, updated_by=self.owner,
        )
        other_proposal = ActivityProposal.objects.create(
            scenario=other_scenario, activity=self.activity, phase=self.phase,
            proposal_type='revise',
        )
        UserProposalReview.objects.create(proposal=other_proposal, user=self.owner, status='accepted')
        reviews = get_accepted_reviews_for_personal_scenario(self.scenario, self.owner)
        self.assertEqual(set(reviews.values_list('proposal_id', flat=True)), {legacy.id, archived.id})

    def test_disabled_policy_legacy_proposal_can_be_accepted_and_applied(self):
        proposal = self.proposal()
        proposal.json_action = json.dumps({
            'action': 'revise', 'activity_name': self.activity.name,
            'activity_type': 'Explanation', 'content': 'Approved older explanation',
            'answers': [], 'insert_location': 'after',
            'explanation': 'Clarifies the activity.',
        })
        proposal.save(update_fields=['json_action'])
        flag = ActivityFlag.objects.create(
            activity=self.activity, scenario=self.scenario, phase=self.phase,
            category='Low', flag_type='Systemic failure', flag_reason='Test',
        )
        proposal.flag.add(flag)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(reverse('accept_proposal', args=[self.scenario.id, proposal.id]))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(QValue.objects.get(action='revise').positive_reward_count, 1)

        _build_personal_scenario(self.scenario.id, self.owner.id)

        clone = Scenario.objects.get(origin_scenario=self.scenario, is_personal=True)
        self.assertEqual(clone.activities.get().plain_text, 'Approved older explanation')
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.plain_text, 'Original content')
