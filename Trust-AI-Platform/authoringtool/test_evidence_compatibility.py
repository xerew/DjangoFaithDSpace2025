import csv
import tempfile
from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse

from authoringtool.evidence import (
    get_evidence_answers,
    get_evidence_context,
    get_evidence_implementation_count,
    get_evidence_signature,
    get_evidence_source_signature,
)
from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    ProposalGenerationRun,
    Scenario,
    ScenarioVersionCompatibility,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import (
    _compute_compatible_category_metrics_data,
    compute_category_metrics_per_phase_activity,
)
from authoringtool.utils import get_last_answers, get_scenario_evidence_cache_paths


class ScenarioEvidencePolicyTests(TestCase):
    def setUp(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user('policy_owner')
        self.owner.groups.add(teachers)
        self.current_student = User.objects.create_user('policy_current')
        self.legacy_student = User.objects.create_user('policy_legacy')
        self.other_student = User.objects.create_user('policy_other')
        activity_type = ActivityType.objects.create(
            name='Policy question',
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.scenario = Scenario.objects.create(
            name='Policy scenario',
            created_by=self.owner,
            updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Policy phase',
            scenario=self.scenario,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Policy activity',
            scenario=self.scenario,
            phase=phase,
            activity_type=activity_type,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.scenario.start_activity = self.activity
        self.scenario.save(update_fields=['start_activity'])
        self.scenario.ensure_current_version()
        UserScenarioScore.objects.create(
            user=self.current_student,
            scenario=self.scenario,
        )
        UserAnswer.objects.create(
            user=self.current_student,
            activity=self.activity,
            timing=10,
        )
        UserScenarioScore.objects.create(
            user=self.legacy_student,
            scenario=self.scenario,
            version_confidence='legacy_unknown',
        )
        UserAnswer.objects.create(
            user=self.legacy_student,
            activity=self.activity,
            timing=20,
            version_confidence='legacy_unknown',
        )
        self.other_scenario = Scenario.objects.create(
            name='Other family scenario',
            family=self.scenario.family,
            variant_type='translation',
            created_by=self.owner,
            updated_by=self.owner,
        )
        UserScenarioScore.objects.create(
            user=self.other_student,
            scenario=self.other_scenario,
        )

    def test_disabled_policy_uses_all_data_from_only_this_scenario(self):
        self.assertFalse(self.scenario.use_family_evidence_pooling)
        context = get_evidence_context(self.scenario, 'compatible')

        self.assertEqual(context['scope'], 'local')
        self.assertEqual(context['scenario_count'], 1)
        self.assertEqual(context['implementation_count'], 2)
        self.assertEqual(get_last_answers(self.scenario.id).count(), 2)

        self.client.force_login(self.owner)
        response = self.client.get(
            reverse('ai_metrics', args=[self.scenario.id]),
            {'scope': 'historical'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['evidence_scope'], 'local')
        self.assertEqual(response.context['implementation_count'], 2)
        self.assertTrue(response.context['proposal_generation_available'])
        self.assertContains(response, 'including its historical data')
        self.assertNotContains(response, 'Compatible family')

    def test_enabled_policy_separates_current_and_historical_evidence(self):
        self.scenario.use_family_evidence_pooling = True
        self.scenario.save(update_fields=['use_family_evidence_pooling'])

        self.assertEqual(
            get_evidence_implementation_count(self.scenario, 'local'),
            1,
        )
        self.assertEqual(
            get_evidence_implementation_count(self.scenario, 'historical'),
            1,
        )


class EvidenceCompatibilityMigrationTests(TransactionTestCase):
    migrate_from = ('authoringtool', '0062_scenario_evidence_versioning')
    migrate_to = ('authoringtool', '0063_evidence_compatibility_pooling')

    def setUp(self):
        super().setUp()
        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_from])
        apps = executor.loader.project_state([self.migrate_from]).apps

        UserModel = apps.get_model('auth', 'User')
        Family = apps.get_model('authoringtool', 'ScenarioFamily')
        ScenarioModel = apps.get_model('authoringtool', 'Scenario')
        Version = apps.get_model('authoringtool', 'ScenarioVersion')

        owner = UserModel.objects.create(
            username='compat_migration_owner',
            password='unused',
        )
        canonical = ScenarioModel.objects.create(
            name='Compatibility canonical',
            created_by_id=owner.id,
            updated_by_id=owner.id,
            variant_type='canonical',
        )
        family = Family.objects.create(
            title='Compatibility family',
            canonical_scenario_id=canonical.id,
            created_by_id=owner.id,
        )
        ScenarioModel.objects.filter(pk=canonical.id).update(
            family_id=family.id
        )
        adaptation = ScenarioModel.objects.create(
            name='Compatibility adaptation',
            created_by_id=owner.id,
            updated_by_id=owner.id,
            family_id=family.id,
            variant_type='adaptation',
        )
        canonical_version = Version.objects.create(
            scenario_id=canonical.id,
            version_number=1,
            structure_fingerprint='a' * 64,
            content_fingerprint='b' * 64,
            snapshot={},
            created_by_id=owner.id,
            is_current=True,
        )
        adaptation_version = Version.objects.create(
            scenario_id=adaptation.id,
            version_number=1,
            structure_fingerprint='a' * 64,
            content_fingerprint='c' * 64,
            snapshot={},
            created_by_id=owner.id,
            is_current=True,
        )
        ScenarioModel.objects.filter(pk=canonical.id).update(
            current_version_id=canonical_version.id
        )
        ScenarioModel.objects.filter(pk=adaptation.id).update(
            current_version_id=adaptation_version.id
        )
        self.version_ids = (
            canonical_version.id,
            adaptation_version.id,
        )

        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_to])
        self.apps = executor.loader.project_state([self.migrate_to]).apps

    def test_backfill_groups_structure_and_quarantines_adaptation(self):
        Compatibility = self.apps.get_model(
            'authoringtool',
            'ScenarioVersionCompatibility',
        )
        canonical = Compatibility.objects.get(
            scenario_version_id=self.version_ids[0]
        )
        adaptation = Compatibility.objects.get(
            scenario_version_id=self.version_ids[1]
        )

        self.assertEqual(canonical.cluster_id, adaptation.cluster_id)
        self.assertEqual(canonical.status, 'compatible')
        self.assertEqual(adaptation.status, 'needs_review')

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate([
            ('authoringtool', '0068_scenario_use_family_evidence_pooling')
        ])
        super().tearDown()


class CompatibleEvidencePoolingTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.owner = User.objects.create_user('compat_owner')
        self.viewer = User.objects.create_user('compat_viewer')
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner.groups.add(teachers)
        self.viewer.groups.add(teachers)
        self.student_en = User.objects.create_user('compat_student_en')
        self.student_el = User.objects.create_user('compat_student_el')
        department = SchoolDepartment.objects.create(name='Compatibility Lab')
        self.student_en.school_department = department
        self.student_el.school_department = department
        self.student_en.save(update_fields=['school_department'])
        self.student_el.save(update_fields=['school_department'])

        self.question_type = ActivityType.objects.create(
            name='Compatibility Question',
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.canonical = Scenario.objects.create(
            name='Pendulum English compatible',
            language='English',
            variant_type='canonical',
            use_family_evidence_pooling=True,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.family = self.canonical.family
        self.translation = Scenario.objects.create(
            name='Pendulum Greek compatible',
            language='Greek',
            variant_type='translation',
            family=self.family,
            origin_scenario=self.canonical,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.canonical_phase = Phase.objects.create(
            name='Investigate',
            scenario=self.canonical,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.translation_phase = Phase.objects.create(
            name='Διερεύνηση',
            scenario=self.translation,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.canonical_activity = Activity.objects.create(
            name='Period question',
            text='Does mass change the period?',
            plain_text='Does mass change the period?',
            scenario=self.canonical,
            phase=self.canonical_phase,
            activity_type=self.question_type,
            is_evaluatable=True,
            is_primary_ev=True,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.translation_activity = Activity.objects.create(
            name='Ερώτηση περιόδου',
            text='Αλλάζει η μάζα την περίοδο;',
            plain_text='Αλλάζει η μάζα την περίοδο;',
            scenario=self.translation,
            phase=self.translation_phase,
            activity_type=self.question_type,
            is_evaluatable=True,
            is_primary_ev=True,
            lineage_key=self.canonical_activity.lineage_key,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.canonical_answer = Answer.objects.create(
            activity=self.canonical_activity,
            text='A. No',
            is_correct=True,
            answer_weight=3,
            created_by=self.owner,
            updated_by=self.owner,
        )
        self.translation_answer = Answer.objects.create(
            activity=self.translation_activity,
            text='A. Όχι',
            is_correct=True,
            answer_weight=3,
            created_by=self.owner,
            updated_by=self.owner,
        )

    def _record_implementations(self):
        UserScenarioScore.objects.create(
            user=self.student_en,
            scenario=self.canonical,
        )
        UserAnswer.objects.create(
            user=self.student_en,
            activity=self.canonical_activity,
            answer=self.canonical_answer,
            timing=10,
        )
        UserScenarioScore.objects.create(
            user=self.student_el,
            scenario=self.translation,
        )
        UserAnswer.objects.create(
            user=self.student_el,
            activity=self.translation_activity,
            answer=self.translation_answer,
            timing=100,
        )

    def test_translation_versions_pool_by_structure_and_lineage(self):
        canonical_version = self.canonical.ensure_current_version()
        translation_version = self.translation.ensure_current_version()

        self.assertEqual(
            canonical_version.structure_fingerprint,
            translation_version.structure_fingerprint,
        )
        self.assertEqual(
            canonical_version.compatibility.cluster_id,
            translation_version.compatibility.cluster_id,
        )
        self.assertEqual(
            translation_version.compatibility.status,
            'compatible',
        )

        self._record_implementations()

        self.assertEqual(self.canonical.eligible_implementation_count(), 1)
        self.assertEqual(
            self.canonical.compatible_implementation_count(),
            2,
        )
        context = get_evidence_context(self.canonical, 'compatible')
        self.assertEqual(context['scenario_count'], 2)
        self.assertEqual(context['implementation_count'], 2)

    def test_adaptation_requires_review_before_pooling(self):
        adaptation = Scenario.objects.create(
            name='Pendulum teacher adaptation',
            language='English',
            variant_type='adaptation',
            family=self.family,
            origin_scenario=self.canonical,
            created_by=self.owner,
            updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Investigate',
            scenario=adaptation,
            created_by=self.owner,
            updated_by=self.owner,
        )
        Activity.objects.create(
            name='Period question',
            text=self.canonical_activity.text,
            plain_text=self.canonical_activity.plain_text,
            scenario=adaptation,
            phase=phase,
            activity_type=self.question_type,
            is_evaluatable=True,
            is_primary_ev=True,
            lineage_key=self.canonical_activity.lineage_key,
            created_by=self.owner,
            updated_by=self.owner,
        )
        adaptation_version = adaptation.ensure_current_version()

        self.assertEqual(
            adaptation_version.compatibility.status,
            'needs_review',
        )
        self.assertEqual(
            list(adaptation.compatible_current_versions()),
            [adaptation_version],
        )

    def test_admin_approval_allows_adaptation_pooling(self):
        self.canonical.ensure_current_version()
        translation_version = self.translation.ensure_current_version()
        translation_version.compatibility.status = 'needs_review'
        translation_version.compatibility.save(update_fields=['status'])

        self.assertEqual(
            self.canonical.compatible_current_versions().count(),
            1,
        )

        translation_version.compatibility.status = 'compatible'
        translation_version.compatibility.decision_source = 'admin'
        translation_version.compatibility.save(
            update_fields=['status', 'decision_source']
        )

        self.assertEqual(
            self.canonical.compatible_current_versions().count(),
            2,
        )

    def test_structure_change_moves_translation_to_a_new_pool(self):
        canonical_version = self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        Activity.objects.create(
            name='Additional translated activity',
            text='Extra structure',
            plain_text='Extra structure',
            scenario=self.translation,
            phase=self.translation_phase,
            activity_type=self.question_type,
            created_by=self.owner,
            updated_by=self.owner,
        )
        changed_version = self.translation.ensure_current_version()

        self.assertNotEqual(
            canonical_version.structure_fingerprint,
            changed_version.structure_fingerprint,
        )
        self.assertNotEqual(
            canonical_version.compatibility.cluster_id,
            changed_version.compatibility.cluster_id,
        )

    def test_correctness_pools_but_timing_stays_in_target_language(self):
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        self._record_implementations()

        rows = _compute_compatible_category_metrics_data(self.canonical)
        high = next(
            row
            for row in rows
            if row['Activity'] == self.canonical_activity.name
            and row['Category'] == 'High'
        )

        self.assertEqual(high['Total'], 2)
        self.assertEqual(high['Correct'], 2)
        self.assertEqual(high['Timing Total'], 1)
        self.assertEqual(high['Avg Time'], 10.0)
        self.assertEqual(high['Evidence Scope'], 'compatible')

    def test_quality_exclusion_changes_pool_signature_and_count(self):
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        self._record_implementations()
        before = get_evidence_context(self.canonical, 'compatible')
        translation_score = UserScenarioScore.objects.get(
            user=self.student_el,
            scenario=self.translation,
        )

        translation_score.data_quality_status = 'excluded'
        translation_score.save(update_fields=['data_quality_status'])
        after = get_evidence_context(self.canonical, 'compatible')

        self.assertNotEqual(before['signature'], after['signature'])
        self.assertEqual(
            before['source_signature'],
            after['source_signature'],
        )
        self.assertEqual(before['implementation_count'], 2)
        self.assertEqual(after['implementation_count'], 1)

    def test_new_answer_invalidates_data_cache_but_not_source_decisions(self):
        self.canonical.ensure_current_version()
        UserScenarioScore.objects.create(
            user=self.student_en,
            scenario=self.canonical,
        )
        source_before = get_evidence_source_signature(
            self.canonical,
            'compatible',
        )
        data_before = get_evidence_signature(
            self.canonical,
            'compatible',
        )

        UserAnswer.objects.create(
            user=self.student_en,
            activity=self.canonical_activity,
            answer=self.canonical_answer,
            timing=10,
        )

        self.assertEqual(
            source_before,
            get_evidence_source_signature(self.canonical, 'compatible'),
        )
        self.assertNotEqual(
            data_before,
            get_evidence_signature(self.canonical, 'compatible'),
        )

    def test_teacher_can_switch_between_compatible_and_local_metrics(self):
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        self._record_implementations()
        self.client.force_login(self.owner)

        compatible = self.client.get(
            reverse('ai_metrics', args=[self.canonical.id])
        )
        local = self.client.get(
            reverse('ai_metrics', args=[self.canonical.id]),
            {'scope': 'local'},
        )

        self.assertEqual(compatible.status_code, 200)
        self.assertEqual(compatible.context['evidence_scope'], 'compatible')
        self.assertEqual(compatible.context['implementation_count'], 2)
        self.assertContains(compatible, 'Compatible family')
        self.assertContains(compatible, 'Pendulum Greek compatible')

        self.assertEqual(local.status_code, 200)
        self.assertEqual(local.context['evidence_scope'], 'local')
        self.assertEqual(local.context['implementation_count'], 1)
        self.assertContains(local, 'This scenario only')

    def test_historical_scope_keeps_legacy_evidence_quarantined(self):
        current_version = self.canonical.ensure_current_version()
        legacy_score = UserScenarioScore.objects.create(
            user=self.student_en,
            scenario=self.canonical,
            version_confidence='legacy_unknown',
        )
        legacy_answer = UserAnswer.objects.create(
            user=self.student_en,
            activity=self.canonical_activity,
            answer=self.canonical_answer,
            timing=10,
            version_confidence='legacy_unknown',
        )

        context = get_evidence_context(self.canonical, 'historical')

        self.assertEqual(context['scope'], 'historical')
        self.assertTrue(context['is_legacy'])
        self.assertEqual(context['implementation_count'], 1)
        self.assertEqual(context['version_ids'], [])
        self.assertEqual(context['sources'][0]['version_id'], None)
        self.assertTrue(context['sources'][0]['is_legacy'])
        self.assertEqual(
            get_evidence_implementation_count(self.canonical, 'local'),
            0,
        )
        self.assertEqual(
            get_evidence_answers(self.canonical, 'historical').count(),
            1,
        )

        legacy_score.refresh_from_db()
        legacy_answer.refresh_from_db()
        self.assertIsNone(legacy_score.scenario_version_id)
        self.assertIsNone(legacy_answer.scenario_version_id)
        self.assertEqual(legacy_score.version_confidence, 'legacy_unknown')
        self.assertEqual(legacy_answer.version_confidence, 'legacy_unknown')
        self.assertEqual(self.canonical.current_version_id, current_version.id)

    def test_historical_metrics_are_visible_but_ai_generation_is_disabled(self):
        self.canonical.ensure_current_version()
        UserScenarioScore.objects.create(
            user=self.student_en,
            scenario=self.canonical,
            version_confidence='legacy_unknown',
        )
        UserAnswer.objects.create(
            user=self.student_en,
            activity=self.canonical_activity,
            answer=self.canonical_answer,
            timing=10,
            version_confidence='legacy_unknown',
        )
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('ai_metrics', args=[self.canonical.id]),
            {'scope': 'historical'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['evidence_scope'], 'historical')
        self.assertEqual(response.context['implementation_count'], 1)
        self.assertTrue(response.context['can_generate_proposals'])
        self.assertFalse(response.context['proposal_generation_available'])
        self.assertContains(response, 'Historical analytics')
        self.assertContains(response, 'Descriptive historical analytics only')
        self.assertContains(response, 'AI generation is disabled')
        self.assertNotContains(response, 'Generate LLM Context')
        self.assertNotContains(response, 'Force Rebuild')

    def test_historical_metrics_task_reads_only_legacy_answers(self):
        self.canonical.ensure_current_version()
        UserScenarioScore.objects.create(
            user=self.student_en,
            scenario=self.canonical,
            version_confidence='legacy_unknown',
        )
        UserAnswer.objects.create(
            user=self.student_en,
            activity=self.canonical_activity,
            answer=self.canonical_answer,
            timing=10,
            version_confidence='legacy_unknown',
        )
        with tempfile.TemporaryDirectory() as cache_root:
            with override_settings(AI_METRICS_CACHE_ROOT=cache_root):
                result = compute_category_metrics_per_phase_activity.run(
                    self.canonical.id,
                    evidence_scope='historical',
                )
                metrics_path = get_scenario_evidence_cache_paths(
                    self.canonical,
                    'historical',
                )['metrics']
                with open(metrics_path, newline='', encoding='utf-8') as file:
                    rows = list(csv.DictReader(file))

        self.assertEqual(result['evidence_scope'], 'historical')
        high_row = next(
            row
            for row in rows
            if row['Activity'] == self.canonical_activity.name
            and row['Category'] == 'High'
        )
        self.assertEqual(high_row['Total'], '1')
        self.assertEqual(high_row['Correct'], '1')
        self.assertEqual(high_row['Evidence Scope'], 'historical')
        self.assertEqual(high_row['Source Version IDs'], '')

    def test_scenario_threshold_uses_compatible_pool_but_keeps_local_total(self):
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        self._record_implementations()
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('viewScenario', args=[self.canonical.id])
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['implementation_count'], 2)
        self.assertEqual(response.context['local_implementation_count'], 1)
        self.assertEqual(
            response.context['compatible_external_implementation_count'],
            1,
        )

    def test_private_source_metadata_is_hidden_from_other_teachers(self):
        self.canonical.visibility_status = 'public'
        self.canonical.save(update_fields=['visibility_status'])
        self.translation.visibility_status = 'private'
        self.translation.save(update_fields=['visibility_status'])
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        self._record_implementations()
        self.client.force_login(self.viewer)

        response = self.client.get(
            reverse('viewScenario', args=[self.canonical.id])
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, self.translation.name)
        self.assertContains(response, 'Source details are hidden')
        self.assertEqual(
            response.context['evidence_context'][
                'restricted_implementation_count'
            ],
            1,
        )

    @patch('authoringtool.views.AsyncResult')
    def test_private_metric_task_results_are_not_exposed_by_task_id(
        self,
        async_result,
    ):
        self.translation.visibility_status = 'private'
        self.translation.save(update_fields=['visibility_status'])
        async_result.return_value.state = 'SUCCESS'
        async_result.return_value.result = {
            'scenario_id': self.translation.id,
            'evidence_scope': 'local',
        }
        self.client.force_login(self.viewer)

        for url_name in ('category_metrics_status', 'risk_flags_status'):
            with self.subTest(url_name=url_name):
                response = self.client.get(
                    reverse(url_name, args=['private-task-id'])
                )
                self.assertEqual(response.status_code, 403)

    def test_new_student_data_does_not_archive_existing_proposal_run(self):
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        context = get_evidence_context(self.canonical, 'compatible')
        run = ProposalGenerationRun.start_new(
            self.canonical,
            self.owner,
            evidence_scope='compatible',
            evidence_version_ids=context['version_ids'],
            evidence_summary=context,
        )
        UserScenarioScore.objects.create(
            user=self.student_en,
            scenario=self.canonical,
        )
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('proposal_list', args=[self.canonical.id])
        )

        run.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(run.is_current)

    def test_compatibility_change_archives_stale_proposal_run(self):
        self.canonical.ensure_current_version()
        translation_version = self.translation.ensure_current_version()
        context = get_evidence_context(self.canonical, 'compatible')
        run = ProposalGenerationRun.start_new(
            self.canonical,
            self.owner,
            evidence_scope='compatible',
            evidence_version_ids=context['version_ids'],
            evidence_summary=context,
        )
        translation_version.compatibility.status = 'excluded'
        translation_version.compatibility.save(update_fields=['status'])
        self.client.force_login(self.owner)

        response = self.client.get(
            reverse('proposal_list', args=[self.canonical.id])
        )

        run.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(run.is_current)

    def test_proposal_history_shows_stored_evidence_without_private_metadata(self):
        self.canonical.visibility_status = 'public'
        self.canonical.save(update_fields=['visibility_status'])
        self.translation.visibility_status = 'private'
        self.translation.save(update_fields=['visibility_status'])
        self.canonical.ensure_current_version()
        self.translation.ensure_current_version()
        self._record_implementations()
        context = get_evidence_context(self.canonical, 'compatible')
        run = ProposalGenerationRun.start_new(
            self.canonical,
            self.owner,
            evidence_scope='compatible',
            evidence_version_ids=context['version_ids'],
            evidence_summary=context,
        )
        run.is_current = False
        run.save(update_fields=['is_current'])
        self.client.force_login(self.viewer)

        history = self.client.get(
            reverse('proposal_history', args=[self.canonical.id])
        )
        detail = self.client.get(
            reverse(
                'proposal_history_run_detail',
                args=[self.canonical.id, run.id],
            )
        )

        self.assertEqual(history.status_code, 200)
        self.assertContains(history, '2 eligible implementations')
        self.assertNotContains(history, self.translation.name)
        self.assertEqual(detail.status_code, 200)
        self.assertContains(detail, 'Evidence used for this run')
        self.assertContains(detail, '1 restricted source')
        self.assertNotContains(detail, self.translation.name)
