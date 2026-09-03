import tempfile

from django.contrib.auth.models import Group, User
from django.core.exceptions import ValidationError
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.evidence import (
    get_evidence_context,
    get_evidence_implementation_count,
)
from authoringtool.models import (
    Activity,
    ActivityFlag,
    ActivityRevision,
    ActivityType,
    Phase,
    ProposalGenerationRun,
    QValue,
    Scenario,
    ScenarioImplementation,
    UserAnswer,
    UserProposalReview,
    UserScenarioScore,
    ActivityProposal,
    rebuild_q_values_for_context,
)
from authoringtool.tasks import compute_category_metrics_per_phase_activity
from authoringtool.utils import get_scenario_evidence_cache_paths


class PhaseCompletionBase(TestCase):
    def setUp(self):
        self.teacher_group = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user(
            'phase_owner',
            password='pass',
        )
        self.owner.groups.add(self.teacher_group)
        self.student = User.objects.create_user(
            'phase_student',
            password='pass',
        )
        self.scenario = Scenario.objects.create(
            name='Revision-safe pendulum',
            language='English',
            visibility_status='public',
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Explore',
            scenario=self.scenario,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.activity_type = ActivityType.objects.create(
            name='Explanation',
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Read',
            text='Published content',
            plain_text='Published content',
            scenario=self.scenario,
            phase=self.phase,
            activity_type=self.activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.scenario.refresh_from_db()


class RevisionPublishWorkflowTests(PhaseCompletionBase):
    def test_implemented_scenario_requires_draft_and_publishes_new_version(self):
        version_one = self.scenario.ensure_current_version(
            created_by=self.owner,
        )
        UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
        )
        self.client.force_login(self.owner)

        blocked = self.client.get(
            reverse(
                'deletePhase',
                args=[self.scenario.id, self.phase.id],
            )
        )
        self.assertRedirects(
            blocked,
            reverse('updateScenario', args=[self.scenario.id]),
        )
        self.assertTrue(Phase.objects.filter(pk=self.phase.id).exists())

        opened = self.client.post(
            reverse(
                'begin_scenario_revision',
                args=[self.scenario.id],
            )
        )
        self.assertRedirects(
            opened,
            reverse('updateScenario', args=[self.scenario.id]),
        )
        self.scenario.refresh_from_db()
        self.assertTrue(hasattr(self.scenario, 'revision_draft'))

        self.activity.text = 'Draft content'
        self.activity.plain_text = 'Draft content'
        self.activity.save()
        refreshed = self.scenario.refresh_version_if_initialized(
            created_by=self.owner,
        )
        self.assertEqual(refreshed.id, version_one.id)
        version_one.refresh_from_db()
        self.assertEqual(
            version_one.snapshot['content']['phases'][0]['activities'][0][
                'text'
            ],
            'Published content',
        )

        self.client.force_login(self.student)
        paused = self.client.get(
            reverse('studentView', args=[self.scenario.id])
        )
        self.assertEqual(paused.status_code, 503)

        self.client.force_login(self.owner)
        published = self.client.post(
            reverse(
                'publish_scenario_revision',
                args=[self.scenario.id],
            ),
            {'change_summary': 'Clarified the explanation'},
        )
        self.assertRedirects(
            published,
            reverse('viewScenario', args=[self.scenario.id]),
        )
        self.scenario.refresh_from_db()
        self.assertFalse(hasattr(self.scenario, 'revision_draft'))
        self.assertNotEqual(
            self.scenario.current_version_id,
            version_one.id,
        )
        self.assertEqual(
            self.scenario.current_version.change_summary,
            'Clarified the explanation',
        )
        self.assertTrue(
            ActivityRevision.objects.filter(
                scenario_version=self.scenario.current_version,
                activity=self.activity,
            ).exists()
        )

        self.client.force_login(self.student)
        resumed = self.client.get(
            reverse('studentView', args=[self.scenario.id])
        )
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(
            resumed.context['implementation'].scenario_version_id,
            self.scenario.current_version_id,
        )

    def test_published_version_and_activity_revision_are_immutable(self):
        version = self.scenario.ensure_current_version(
            created_by=self.owner,
        )
        revision = ActivityRevision.objects.get(
            scenario_version=version,
            activity=self.activity,
        )

        version.change_summary = 'Metadata may change'
        version.save(update_fields=['change_summary'])
        version.snapshot = {'mutated': True}
        with self.assertRaises(ValidationError):
            version.save(update_fields=['snapshot'])

        revision.snapshot = {'mutated': True}
        with self.assertRaises(ValidationError):
            revision.save(update_fields=['snapshot'])

    def test_invalid_draft_graph_is_not_published(self):
        self.scenario.ensure_current_version(created_by=self.owner)
        UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
        )
        self.scenario.begin_revision_draft(self.owner)
        Activity.objects.create(
            name='Unreachable',
            text='Unreachable',
            plain_text='Unreachable',
            scenario=self.scenario,
            phase=self.phase,
            activity_type=self.activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.client.force_login(self.owner)

        response = self.client.post(
            reverse(
                'publish_scenario_revision',
                args=[self.scenario.id],
            )
        )

        self.assertRedirects(
            response,
            reverse('updateScenario', args=[self.scenario.id]),
        )
        self.scenario.refresh_from_db()
        self.assertTrue(hasattr(self.scenario, 'revision_draft'))


class ImplementationAndActivityLineageTests(PhaseCompletionBase):
    def test_score_and_answers_share_exact_implementation_and_revision(self):
        version = self.scenario.ensure_current_version()
        score = UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
        )
        answer = UserAnswer.objects.create(
            user=self.student,
            activity=self.activity,
            timing=20,
        )

        self.assertEqual(
            score.implementation_id,
            answer.implementation_id,
        )
        self.assertEqual(score.scenario_version_id, version.id)
        self.assertEqual(answer.scenario_version_id, version.id)
        self.assertEqual(
            answer.activity_revision.scenario_version_id,
            version.id,
        )
        self.assertEqual(
            answer.activity_revision.concept_id,
            self.activity.revisions.get(
                scenario_version=version,
            ).concept_id,
        )

    def test_teacher_implementations_never_enter_student_evidence(self):
        version = self.scenario.ensure_current_version()
        UserScenarioScore.objects.create(
            user=self.owner,
            scenario=self.scenario,
        )
        UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
        )

        self.assertEqual(
            ScenarioImplementation.objects.filter(
                scenario_version=version,
            ).count(),
            2,
        )
        self.assertEqual(self.scenario.eligible_implementation_count(), 1)

    def test_legacy_answer_uses_the_legacy_implementation(self):
        score = UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
            version_confidence='legacy_unknown',
        )
        answer = UserAnswer.objects.create(
            user=self.student,
            activity=self.activity,
            version_confidence='legacy_unknown',
        )

        self.assertEqual(
            answer.implementation_id,
            score.implementation_id,
        )
        self.assertIsNone(answer.scenario_version_id)


class FamilyLanguageAnalyticsTests(PhaseCompletionBase):
    def setUp(self):
        super().setUp()
        self.scenario.use_family_evidence_pooling = True
        self.scenario.save(update_fields=['use_family_evidence_pooling'])
        self.version_en = self.scenario.ensure_current_version()
        self.translation = Scenario.objects.create(
            name='Εκκρεμές',
            language='Greek',
            visibility_status='public',
            family=self.scenario.family,
            origin_scenario=self.scenario,
            variant_type='translation',
            created_by=self.owner,
            updated_by=self.owner,
        )
        translation_phase = Phase.objects.create(
            name='Διερεύνηση',
            scenario=self.translation,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.translation_activity = Activity.objects.create(
            name='Ανάγνωση',
            text='Μεταφρασμένο περιεχόμενο',
            plain_text='Μεταφρασμένο περιεχόμενο',
            scenario=self.translation,
            phase=translation_phase,
            activity_type=self.activity_type,
            lineage_key=self.activity.lineage_key,
            concept=self.activity.concept,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.version_el = self.translation.ensure_current_version()
        self.student_el = User.objects.create_user('phase_student_el')
        UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
        )
        UserAnswer.objects.create(
            user=self.student,
            activity=self.activity,
            timing=20,
        )
        UserScenarioScore.objects.create(
            user=self.student_el,
            scenario=self.translation,
        )
        UserAnswer.objects.create(
            user=self.student_el,
            activity=self.translation_activity,
            timing=40,
        )

    def test_language_filter_splits_compatible_family_counts(self):
        self.assertEqual(
            get_evidence_implementation_count(
                self.scenario,
                'compatible',
            ),
            2,
        )
        self.assertEqual(
            get_evidence_implementation_count(
                self.scenario,
                'compatible',
                'English',
            ),
            1,
        )
        self.assertEqual(
            get_evidence_implementation_count(
                self.scenario,
                'compatible',
                'Greek',
            ),
            1,
        )
        context = get_evidence_context(
            self.scenario,
            'compatible',
            'Greek',
        )
        self.assertEqual(context['languages'], ['Greek'])
        self.assertEqual(context['implementation_count'], 1)

    def test_filtered_metrics_have_a_separate_downloadable_csv(self):
        with tempfile.TemporaryDirectory() as cache_root:
            with override_settings(AI_METRICS_CACHE_ROOT=cache_root):
                result = compute_category_metrics_per_phase_activity.run(
                    self.scenario.id,
                    evidence_scope='compatible',
                    evidence_language='Greek',
                )
                path = get_scenario_evidence_cache_paths(
                    self.scenario,
                    'compatible',
                    'Greek',
                )['metrics']
                self.assertEqual(result['evidence_language'], 'Greek')
                self.assertTrue(path.endswith(
                    '_combined_activity_metrics.csv'
                ))
                self.client.force_login(self.owner)
                response = self.client.get(
                    reverse(
                        'download_ai_evidence_csv',
                        args=[self.scenario.id, 'metrics'],
                    ),
                    {'scope': 'compatible', 'language': 'Greek'},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn(
                    'scenario-',
                    response.headers['Content-Disposition'],
                )
                self.assertTrue(b''.join(response.streaming_content))


class BanditEvidenceEligibilityTests(PhaseCompletionBase):
    def setUp(self):
        super().setUp()
        self.scenario.use_family_evidence_pooling = True
        self.scenario.save(update_fields=['use_family_evidence_pooling'])

    def _proposal(self, run):
        flag = ActivityFlag.objects.create(
            activity=self.activity,
            category='Low',
            flag_type='Low correctness',
            flag_reason='Test evidence',
            evidence_scope='compatible',
        )
        proposal = ActivityProposal.objects.create(
            scenario=self.scenario,
            generation_run=run,
            phase=self.phase,
            activity=self.activity,
            proposal_type='skip',
            suggested_action='Skip',
            translated_action='Skip',
            json_action='{"action": "skip"}',
            json_translated_action='{"action": "skip"}',
        )
        proposal.flag.add(flag)
        return proposal

    def test_only_current_compatible_provenance_trains_bandit(self):
        version = self.scenario.ensure_current_version()
        context = get_evidence_context(self.scenario, 'compatible')
        run = ProposalGenerationRun.start_new(
            self.scenario,
            self.owner,
            scenario_version=version,
            evidence_scope='compatible',
            evidence_version_ids=context['version_ids'],
            evidence_summary=context,
        )
        proposal = self._proposal(run)
        self.assertTrue(proposal.is_bandit_reward_eligible())

        review = UserProposalReview.objects.create(
            proposal=proposal,
            user=self.owner,
        )
        with self.captureOnCommitCallbacks(execute=True):
            review.accept()
        q_value = QValue.objects.get(
            flag_type='Low correctness',
            category='Low',
            action='skip',
        )
        self.assertEqual(q_value.reward_count, 1)

        self.activity.text = 'New published definition'
        self.activity.plain_text = 'New published definition'
        self.activity.save()
        self.scenario.ensure_current_version()
        self.assertFalse(proposal.is_bandit_reward_eligible())

        rebuild_q_values_for_context('Low correctness', 'Low')
        q_value.refresh_from_db()
        self.assertEqual(q_value.reward_count, 0)

    def test_local_generation_run_is_never_reward_eligible(self):
        version = self.scenario.ensure_current_version()
        context = get_evidence_context(self.scenario, 'local')
        run = ProposalGenerationRun.start_new(
            self.scenario,
            self.owner,
            scenario_version=version,
            evidence_scope='local',
            evidence_version_ids=context['version_ids'],
            evidence_summary=context,
        )
        proposal = self._proposal(run)

        self.assertFalse(proposal.is_bandit_reward_eligible())
