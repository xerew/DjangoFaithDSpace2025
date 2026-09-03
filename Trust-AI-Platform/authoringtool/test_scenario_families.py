from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import Client, TestCase, TransactionTestCase
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    Scenario,
    ScenarioFamily,
    Subject,
    UserScenarioScore,
)
from authoringtool.tasks import _build_personal_scenario


class ScenarioFamilyMigrationTests(TransactionTestCase):
    migrate_from = ('authoringtool', '0060_scenario_start_graph_bandit_controls')
    migrate_to = ('authoringtool', '0061_scenario_family')

    def setUp(self):
        super().setUp()
        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_from])
        old_apps = executor.loader.project_state([self.migrate_from]).apps

        UserModel = old_apps.get_model('auth', 'User')
        ScenarioModel = old_apps.get_model('authoringtool', 'Scenario')
        SubjectModel = old_apps.get_model('authoringtool', 'Subject')

        owner = UserModel.objects.create(
            username='legacy_family_owner',
            password='unused',
        )
        subject = SubjectModel.objects.create(
            name='Legacy family physics',
            category='STEM',
        )
        canonical = ScenarioModel.objects.create(
            name='Legacy pendulum',
            language='English',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        canonical.subjects.add(subject)
        adaptation = ScenarioModel.objects.create(
            name='Legacy pendulum edited',
            language='English',
            origin_scenario_id=canonical.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        translation = ScenarioModel.objects.create(
            name='Legacy pendulum Greek',
            language='Greek',
            origin_scenario_id=canonical.id,
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        self.legacy_ids = {
            'canonical': canonical.id,
            'adaptation': adaptation.id,
            'translation': translation.id,
            'subject': subject.id,
        }

        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_to])
        self.apps = executor.loader.project_state([self.migrate_to]).apps

    def test_origin_chain_is_backfilled_into_one_family(self):
        ScenarioModel = self.apps.get_model('authoringtool', 'Scenario')
        ScenarioFamilyModel = self.apps.get_model(
            'authoringtool',
            'ScenarioFamily',
        )

        canonical = ScenarioModel.objects.get(
            pk=self.legacy_ids['canonical']
        )
        adaptation = ScenarioModel.objects.get(
            pk=self.legacy_ids['adaptation']
        )
        translation = ScenarioModel.objects.get(
            pk=self.legacy_ids['translation']
        )
        family = ScenarioFamilyModel.objects.get()

        self.assertEqual(family.canonical_scenario_id, canonical.id)
        self.assertEqual(canonical.variant_type, 'canonical')
        self.assertEqual(adaptation.variant_type, 'adaptation')
        self.assertEqual(translation.variant_type, 'translation')
        self.assertEqual(canonical.family_id, family.id)
        self.assertEqual(adaptation.family_id, family.id)
        self.assertEqual(translation.family_id, family.id)
        self.assertEqual(
            list(family.subjects.values_list('id', flat=True)),
            [self.legacy_ids['subject']],
        )

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate([
            ('authoringtool', '0067_family_review_dashboard_llm')
        ])
        super().tearDown()


class ScenarioFamilyTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user(
            'family_owner',
            password='pass',
        )
        self.owner.groups.add(teachers)
        self.other_teacher = User.objects.create_user(
            'family_other_teacher',
            password='pass',
        )
        self.other_teacher.groups.add(teachers)
        self.student_one = User.objects.create_user('family_student_one')
        self.student_two = User.objects.create_user('family_student_two')
        self.subject = Subject.objects.create(
            name='Scenario family physics',
            category='STEM',
        )

    def create_canonical(self, **overrides):
        values = {
            'name': 'Family pendulum scenario',
            'language': 'English',
            'created_by': self.owner,
            'updated_by': self.owner,
            'visibility_status': 'public',
            'ai_metrics_min_implementations': 200,
            'use_family_evidence_pooling': True,
        }
        values.update(overrides)
        scenario = Scenario.objects.create(**values)
        scenario.subjects.add(self.subject)
        scenario.family.subjects.set(scenario.subjects.all())
        return scenario

    def test_new_scenario_gets_a_canonical_family(self):
        scenario = self.create_canonical()

        self.assertIsNotNone(scenario.family_id)
        self.assertEqual(scenario.variant_type, 'canonical')
        self.assertEqual(scenario.family.canonical_scenario, scenario)
        self.assertEqual(scenario.family.title, scenario.name)
        self.assertEqual(
            list(scenario.family.subjects.values_list('id', flat=True)),
            [self.subject.id],
        )

    def test_manual_adaptation_copy_preserves_family_and_subjects(self):
        scenario = self.create_canonical()
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('duplicate_scenario', args=[scenario.id]),
            {'variant': 'adaptation'},
        )

        clone = Scenario.objects.get(origin_scenario=scenario)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(clone.family_id, scenario.family_id)
        self.assertEqual(clone.variant_type, 'adaptation')
        self.assertEqual(
            set(clone.subjects.values_list('id', flat=True)),
            {self.subject.id},
        )
        self.assertEqual(ScenarioFamily.objects.count(), 1)

    def test_unrelated_teacher_cannot_copy_private_scenario_by_id(self):
        scenario = self.create_canonical(visibility_status='private')
        self.client.force_login(self.other_teacher)

        response = self.client.get(
            reverse('duplicate_scenario', args=[scenario.id])
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(Scenario.objects.count(), 1)

    def test_deleting_canonical_promotes_a_surviving_variant(self):
        canonical = self.create_canonical()
        translation = Scenario.objects.create(
            name='Family replacement translation',
            language='Greek',
            created_by=self.owner,
            updated_by=self.owner,
            family=canonical.family,
            origin_scenario=canonical,
            variant_type='translation',
        )
        translation.subjects.add(self.subject)
        family = canonical.family

        canonical.delete()

        family.refresh_from_db()
        translation.refresh_from_db()
        self.assertEqual(family.canonical_scenario_id, translation.id)
        self.assertEqual(translation.variant_type, 'canonical')
        self.assertEqual(
            set(family.subjects.values_list('id', flat=True)),
            {self.subject.id},
        )

    def test_deleting_the_last_scenario_removes_its_empty_family(self):
        scenario = self.create_canonical()
        family_id = scenario.family_id

        scenario.delete()

        self.assertFalse(ScenarioFamily.objects.filter(pk=family_id).exists())

    def test_manual_translation_copy_preserves_family(self):
        scenario = self.create_canonical()
        phase = Phase.objects.create(
            name='Translation source phase',
            scenario=scenario,
            created_by=self.owner,
            updated_by=self.owner,
        )
        activity_type = ActivityType.objects.create(
            name='Translation source explanation',
            created_by=self.owner,
            updated_by=self.owner,
        )
        source_activity = Activity.objects.create(
            name='Translation source activity',
            text='Source',
            plain_text='Source',
            scenario=scenario,
            phase=phase,
            activity_type=activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('duplicate_scenario', args=[scenario.id]),
            {'variant': 'translation'},
        )

        clone = Scenario.objects.get(origin_scenario=scenario)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(clone.family_id, scenario.family_id)
        self.assertEqual(clone.variant_type, 'translation')
        self.assertEqual(
            clone.activities.get().lineage_key,
            source_activity.lineage_key,
        )
        self.assertIsNotNone(clone.current_version_id)
        self.assertEqual(ScenarioFamily.objects.count(), 1)

    def test_manual_copy_preserves_phase_less_start_activity_and_answers(self):
        scenario = self.create_canonical()
        activity_type = ActivityType.objects.create(
            name='Phase-less question',
            created_by=self.owner,
            updated_by=self.owner,
        )
        source_activity = Activity.objects.create(
            name='Phase-less start',
            text='Question',
            plain_text='Question',
            scenario=scenario,
            phase=None,
            activity_type=activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )
        Answer.objects.create(
            activity=source_activity,
            text='A. Answer',
            is_correct=True,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('duplicate_scenario', args=[scenario.id])
        )

        clone = Scenario.objects.get(origin_scenario=scenario)
        cloned_activity = clone.activities.get()
        self.assertEqual(response.status_code, 302)
        self.assertIsNone(cloned_activity.phase_id)
        self.assertEqual(clone.start_activity_id, cloned_activity.id)
        self.assertEqual(cloned_activity.answers.get().text, 'A. Answer')

    @patch('authoringtool.tasks.apply_proposals_to_cloned_scenario')
    @patch('authoringtool.tasks.assert_scenario_graph_integrity')
    def test_proposal_clone_preserves_family_and_subjects(
        self,
        graph_check,
        apply_proposals,
    ):
        scenario = self.create_canonical()
        activity_type = ActivityType.objects.create(
            name='Proposal phase-less explanation',
            created_by=self.owner,
            updated_by=self.owner,
        )
        source_activity = Activity.objects.create(
            name='Proposal phase-less start',
            text='Start',
            plain_text='Start',
            scenario=scenario,
            phase=None,
            activity_type=activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )

        clone_id = _build_personal_scenario(scenario.id, self.owner.id)

        clone = Scenario.objects.get(pk=clone_id)
        cloned_activity = clone.activities.get()
        self.assertEqual(clone.family_id, scenario.family_id)
        self.assertEqual(clone.variant_type, 'adaptation')
        self.assertTrue(clone.is_personal)
        self.assertIsNone(cloned_activity.phase_id)
        self.assertEqual(
            cloned_activity.lineage_key,
            source_activity.lineage_key,
        )
        self.assertEqual(clone.start_activity_id, cloned_activity.id)
        self.assertEqual(
            set(clone.subjects.values_list('id', flat=True)),
            {self.subject.id},
        )
        self.assertEqual(ScenarioFamily.objects.count(), 1)
        self.assertEqual(graph_check.call_count, 2)
        apply_proposals.assert_called_once()

    def test_scenario_page_reports_family_and_per_language_totals(self):
        canonical = self.create_canonical()
        translation = Scenario.objects.create(
            name='Family pendulum scenario Greek',
            language='Greek',
            created_by=self.owner,
            updated_by=self.owner,
            visibility_status='public',
            origin_scenario=canonical,
            family=canonical.family,
            variant_type='translation',
        )
        UserScenarioScore.objects.create(
            user=self.student_one,
            scenario=canonical,
        )
        UserScenarioScore.objects.create(
            user=self.student_one,
            scenario=translation,
        )
        UserScenarioScore.objects.create(
            user=self.student_two,
            scenario=translation,
        )
        self.client.force_login(self.owner)

        with patch(
            'authoringtool.views.generate_flowchart',
            return_value='graph TD',
        ):
            response = self.client.get(
                reverse('viewScenario', args=[canonical.id])
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['implementation_count'], 3)
        self.assertEqual(
            response.context['local_implementation_count'],
            1,
        )
        self.assertEqual(
            response.context['compatible_external_implementation_count'],
            2,
        )
        self.assertEqual(
            response.context['family_implementation_count'],
            3,
        )
        self.assertEqual(
            response.context['family_language_counts'],
            [
                {'language': 'English', 'implementation_count': 1},
                {'language': 'Greek', 'implementation_count': 2},
            ],
        )
        self.assertContains(
            response,
            'The reliability threshold uses',
        )
