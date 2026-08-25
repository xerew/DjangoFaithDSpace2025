from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase


class ActivityAndImplementationLineageMigrationTests(TransactionTestCase):
    migrate_from = ('authoringtool', '0064_scenario_family_discovery')
    migrate_to = ('authoringtool', '0066_revision_publish_workflow')

    def setUp(self):
        super().setUp()
        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_from])
        old_apps = executor.loader.project_state([self.migrate_from]).apps

        User = old_apps.get_model('auth', 'User')
        Scenario = old_apps.get_model('authoringtool', 'Scenario')
        ScenarioFamily = old_apps.get_model(
            'authoringtool',
            'ScenarioFamily',
        )
        Phase = old_apps.get_model('authoringtool', 'Phase')
        ActivityType = old_apps.get_model(
            'authoringtool',
            'ActivityType',
        )
        Activity = old_apps.get_model('authoringtool', 'Activity')
        ScenarioVersion = old_apps.get_model(
            'authoringtool',
            'ScenarioVersion',
        )
        UserScenarioScore = old_apps.get_model(
            'authoringtool',
            'UserScenarioScore',
        )
        UserAnswer = old_apps.get_model('authoringtool', 'UserAnswer')

        owner = User.objects.create(username='migration_owner')
        exact_student = User.objects.create(username='migration_exact')
        legacy_student = User.objects.create(username='migration_legacy')
        scenario = Scenario.objects.create(
            name='Migration scenario',
            language='English',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        family = ScenarioFamily.objects.create(
            title=scenario.name,
            canonical_scenario_id=scenario.id,
            created_by_id=owner.id,
        )
        Scenario.objects.filter(pk=scenario.id).update(
            family_id=family.id,
        )
        scenario.family_id = family.id
        phase = Phase.objects.create(
            name='Phase',
            scenario_id=scenario.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        activity_type = ActivityType.objects.create(
            name='Explanation',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        activity = Activity.objects.create(
            name='Read',
            text='Read',
            plain_text='Read',
            scenario_id=scenario.id,
            phase_id=phase.id,
            activity_type_id=activity_type.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        lineage = str(activity.lineage_key)
        structure_activity = {
            'lineage_key': lineage,
            'activity_type': 'Explanation',
        }
        content_activity = {
            'lineage_key': lineage,
            'name': 'Read',
            'text': 'Read',
        }
        version = ScenarioVersion.objects.create(
            scenario_id=scenario.id,
            version_number=1,
            structure_fingerprint='a' * 64,
            content_fingerprint='b' * 64,
            snapshot={
                'schema': 1,
                'structure': {
                    'schema': 1,
                    'phases': [{
                        'position': 0,
                        'activities': [structure_activity],
                    }],
                },
                'content': {
                    'schema': 1,
                    'phases': [{
                        'position': 0,
                        'activities': [content_activity],
                    }],
                },
            },
            created_by_id=owner.id,
            is_current=True,
        )
        Scenario.objects.filter(pk=scenario.id).update(
            current_version_id=version.id
        )

        exact_score = UserScenarioScore.objects.create(
            user_id=exact_student.id,
            scenario_id=scenario.id,
            scenario_version_id=version.id,
            version_confidence='exact',
        )
        exact_answer = UserAnswer.objects.create(
            user_id=exact_student.id,
            activity_id=activity.id,
            scenario_version_id=version.id,
            version_confidence='exact',
        )
        legacy_score = UserScenarioScore.objects.create(
            user_id=legacy_student.id,
            scenario_id=scenario.id,
            version_confidence='legacy_unknown',
        )
        legacy_answer = UserAnswer.objects.create(
            user_id=legacy_student.id,
            activity_id=activity.id,
            version_confidence='legacy_unknown',
        )
        self.ids = {
            'scenario': scenario.id,
            'activity': activity.id,
            'version': version.id,
            'exact_score': exact_score.id,
            'exact_answer': exact_answer.id,
            'legacy_score': legacy_score.id,
            'legacy_answer': legacy_answer.id,
        }

        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_to])
        self.apps = executor.loader.project_state([self.migrate_to]).apps

    def test_migration_backfills_attempts_activity_revisions_and_legacy_boundary(
        self,
    ):
        Activity = self.apps.get_model('authoringtool', 'Activity')
        ActivityRevision = self.apps.get_model(
            'authoringtool',
            'ActivityRevision',
        )
        ScenarioImplementation = self.apps.get_model(
            'authoringtool',
            'ScenarioImplementation',
        )
        ScenarioVersion = self.apps.get_model(
            'authoringtool',
            'ScenarioVersion',
        )
        UserScenarioScore = self.apps.get_model(
            'authoringtool',
            'UserScenarioScore',
        )
        UserAnswer = self.apps.get_model('authoringtool', 'UserAnswer')

        activity = Activity.objects.get(pk=self.ids['activity'])
        exact_score = UserScenarioScore.objects.get(
            pk=self.ids['exact_score']
        )
        exact_answer = UserAnswer.objects.get(
            pk=self.ids['exact_answer']
        )
        legacy_score = UserScenarioScore.objects.get(
            pk=self.ids['legacy_score']
        )
        legacy_answer = UserAnswer.objects.get(
            pk=self.ids['legacy_answer']
        )
        published = ScenarioVersion.objects.get(pk=self.ids['version'])
        legacy_boundary = ScenarioVersion.objects.get(
            scenario_id=self.ids['scenario'],
            version_number=0,
        )

        self.assertIsNotNone(activity.concept_id)
        self.assertEqual(
            ScenarioImplementation.objects.filter(
                scenario_id=self.ids['scenario'],
            ).count(),
            2,
        )
        self.assertEqual(
            exact_score.implementation_id,
            exact_answer.implementation_id,
        )
        self.assertIsNotNone(exact_answer.activity_revision_id)
        self.assertTrue(
            ActivityRevision.objects.filter(
                pk=exact_answer.activity_revision_id,
                concept_id=activity.concept_id,
                scenario_version_id=published.id,
            ).exists()
        )
        self.assertEqual(
            legacy_score.implementation_id,
            legacy_answer.implementation_id,
        )
        self.assertIsNone(legacy_answer.activity_revision_id)
        self.assertEqual(published.revision_status, 'published')
        self.assertIsNotNone(published.published_at)
        self.assertEqual(legacy_boundary.revision_status, 'legacy')
        self.assertFalse(legacy_boundary.is_current)
        self.assertTrue(legacy_boundary.snapshot['legacy_unknown'])

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate([
            ('authoringtool', '0067_family_review_dashboard_llm')
        ])
        super().tearDown()
