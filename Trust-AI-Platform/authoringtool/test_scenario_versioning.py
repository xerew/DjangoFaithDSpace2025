from django.contrib.auth.models import User
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TestCase, TransactionTestCase
from django.test.utils import CaptureQueriesContext

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    ProposalGenerationRun,
    Scenario,
    ScenarioVersion,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.utils import get_eligible_user_answers


class ScenarioEvidenceVersionMigrationTests(TransactionTestCase):
    migrate_from = ('authoringtool', '0061_scenario_family')
    migrate_to = ('authoringtool', '0062_scenario_evidence_versioning')

    def setUp(self):
        super().setUp()
        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_from])
        old_apps = executor.loader.project_state([self.migrate_from]).apps

        UserModel = old_apps.get_model('auth', 'User')
        ScenarioModel = old_apps.get_model('authoringtool', 'Scenario')
        PhaseModel = old_apps.get_model('authoringtool', 'Phase')
        ActivityTypeModel = old_apps.get_model(
            'authoringtool',
            'ActivityType',
        )
        ActivityModel = old_apps.get_model('authoringtool', 'Activity')
        UserAnswerModel = old_apps.get_model(
            'authoringtool',
            'UserAnswer',
        )
        UserScenarioScoreModel = old_apps.get_model(
            'authoringtool',
            'UserScenarioScore',
        )

        owner = UserModel.objects.create(
            username='legacy_version_owner',
            password='unused',
        )
        student = UserModel.objects.create(
            username='legacy_version_student',
            password='unused',
        )
        canonical = ScenarioModel.objects.create(
            name='Legacy version canonical',
            language='English',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        translation = ScenarioModel.objects.create(
            name='Legacy version translation',
            language='Greek',
            origin_scenario_id=canonical.id,
            family_id=canonical.family_id,
            variant_type='translation',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        activity_type = ActivityTypeModel.objects.create(
            name='Explanation',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        canonical_phase = PhaseModel.objects.create(
            name='Phase',
            scenario_id=canonical.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        translation_phase = PhaseModel.objects.create(
            name='Phase translated',
            scenario_id=translation.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        canonical_activity = ActivityModel.objects.create(
            name='Read',
            text='Read this',
            plain_text='Read this',
            scenario_id=canonical.id,
            phase_id=canonical_phase.id,
            activity_type_id=activity_type.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        translation_activity = ActivityModel.objects.create(
            name='Read translated',
            text='Translated content',
            plain_text='Translated content',
            scenario_id=translation.id,
            phase_id=translation_phase.id,
            activity_type_id=activity_type.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        legacy_score = UserScenarioScoreModel.objects.create(
            user_id=student.id,
            scenario_id=canonical.id,
        )
        legacy_answer = UserAnswerModel.objects.create(
            user_id=student.id,
            activity_id=canonical_activity.id,
        )
        self.ids = {
            'canonical_activity': canonical_activity.id,
            'translation_activity': translation_activity.id,
            'legacy_score': legacy_score.id,
            'legacy_answer': legacy_answer.id,
        }

        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_to])
        self.apps = executor.loader.project_state([self.migrate_to]).apps

    def test_migration_maps_unchanged_copies_and_quarantines_legacy_data(self):
        ActivityModel = self.apps.get_model('authoringtool', 'Activity')
        UserAnswerModel = self.apps.get_model(
            'authoringtool',
            'UserAnswer',
        )
        UserScenarioScoreModel = self.apps.get_model(
            'authoringtool',
            'UserScenarioScore',
        )

        canonical_activity = ActivityModel.objects.get(
            pk=self.ids['canonical_activity']
        )
        translation_activity = ActivityModel.objects.get(
            pk=self.ids['translation_activity']
        )
        score = UserScenarioScoreModel.objects.get(
            pk=self.ids['legacy_score']
        )
        answer = UserAnswerModel.objects.get(
            pk=self.ids['legacy_answer']
        )

        self.assertEqual(
            canonical_activity.lineage_key,
            translation_activity.lineage_key,
        )
        self.assertEqual(score.version_confidence, 'legacy_unknown')
        self.assertIsNone(score.scenario_version_id)
        self.assertEqual(answer.version_confidence, 'legacy_unknown')
        self.assertIsNone(answer.scenario_version_id)

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate([
            ('authoringtool', '0067_family_review_dashboard_llm')
        ])
        super().tearDown()


class ScenarioEvidenceVersionTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(
            'version_owner',
            password='pass',
        )
        self.student_one = User.objects.create_user('version_student_one')
        self.student_two = User.objects.create_user('version_student_two')
        self.scenario = Scenario.objects.create(
            name='Versioned pendulum',
            language='English',
            created_by=self.owner,
            updated_by=self.owner,
            visibility_status='public',
        )
        self.phase = Phase.objects.create(
            name='Explore',
            scenario=self.scenario,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.activity_type = ActivityType.objects.create(
            name='Versioned Explanation',
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Read the model',
            text='Initial explanation',
            plain_text='Initial explanation',
            scenario=self.scenario,
            phase=self.phase,
            activity_type=self.activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )

    def test_content_and_structure_changes_create_distinct_versions(self):
        version_one = self.scenario.ensure_current_version()
        unchanged = self.scenario.ensure_current_version()
        self.assertEqual(unchanged.id, version_one.id)

        self.activity.text = 'Corrected explanation'
        self.activity.plain_text = 'Corrected explanation'
        self.activity.save()
        version_two = self.scenario.ensure_current_version()

        self.assertEqual(version_two.version_number, 2)
        self.assertEqual(
            version_two.structure_fingerprint,
            version_one.structure_fingerprint,
        )
        self.assertNotEqual(
            version_two.content_fingerprint,
            version_one.content_fingerprint,
        )

        Answer.objects.create(
            activity=self.activity,
            text='A. New structural option',
            is_correct=True,
            answer_weight=3,
            created_by=self.owner,
            updated_by=self.owner,
        )
        version_three = self.scenario.ensure_current_version()

        self.assertEqual(version_three.version_number, 3)
        self.assertNotEqual(
            version_three.structure_fingerprint,
            version_two.structure_fingerprint,
        )
        self.assertEqual(ScenarioVersion.objects.filter(is_current=True).count(), 1)

    def test_version_row_lock_does_not_join_nullable_current_version(self):
        with CaptureQueriesContext(connection) as captured:
            version = self.scenario.ensure_current_version()

        scenario_fetch = next(
            query['sql']
            for query in captured.captured_queries
            if 'FROM "authoringtool_scenario"' in query['sql']
        )
        self.assertNotIn(
            'JOIN "authoringtool_scenarioversion"',
            scenario_fetch,
        )
        self.assertEqual(version.scenario_id, self.scenario.id)

    def test_generated_llm_context_does_not_change_evidence_version(self):
        version = self.scenario.ensure_current_version()
        self.activity.llm_context = 'Generated review'
        self.activity.short_llm_summary = 'Generated summary'
        self.activity.save()

        refreshed = self.scenario.ensure_current_version()

        self.assertEqual(refreshed.id, version.id)

    def test_activity_without_phase_can_be_versioned(self):
        unassigned_activity = Activity.objects.create(
            name='Unassigned activity',
            text='Content',
            plain_text='Content',
            scenario=self.scenario,
            activity_type=self.activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )

        version = self.scenario.ensure_current_version()

        content_activities = [
            activity
            for phase in version.snapshot['content']['phases']
            for activity in phase['activities']
        ]
        self.assertIn(
            str(unassigned_activity.lineage_key),
            [activity['lineage_key'] for activity in content_activities],
        )

    def test_implementations_and_answers_are_pinned_to_current_version(self):
        version_one = self.scenario.ensure_current_version()
        score_one = UserScenarioScore.objects.create(
            user=self.student_one,
            scenario=self.scenario,
        )
        answer_one = UserAnswer.objects.create(
            user=self.student_one,
            activity=self.activity,
            timing=20,
        )
        self.assertEqual(score_one.scenario_version_id, version_one.id)
        self.assertEqual(answer_one.scenario_version_id, version_one.id)
        self.assertEqual(self.scenario.eligible_implementation_count(), 1)
        self.assertEqual(
            get_eligible_user_answers(self.scenario.id).count(),
            1,
        )

        self.activity.text = 'New version content'
        self.activity.save()
        version_two = self.scenario.ensure_current_version()
        self.assertNotEqual(version_two.id, version_one.id)
        self.assertEqual(self.scenario.eligible_implementation_count(), 0)

        score_two = UserScenarioScore.objects.create(
            user=self.student_two,
            scenario=self.scenario,
        )
        self.assertEqual(score_two.scenario_version_id, version_two.id)
        self.assertEqual(self.scenario.eligible_implementation_count(), 1)

        score_two.data_quality_status = 'suspect'
        score_two.save(update_fields=['data_quality_status'])
        self.assertEqual(self.scenario.eligible_implementation_count(), 0)
        self.assertEqual(
            get_eligible_user_answers(self.scenario.id).count(),
            0,
        )

    def test_proposal_generation_run_is_pinned_to_current_version(self):
        version = self.scenario.ensure_current_version()

        run = ProposalGenerationRun.start_new(self.scenario, self.owner)

        self.assertEqual(run.scenario_version_id, version.id)
