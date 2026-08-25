import json
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import (
    Client,
    TestCase,
    TransactionTestCase,
    override_settings,
)
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Phase,
    ProposalGenerationRun,
    Scenario,
    ScenarioFamily,
    ScenarioFamilyCandidate,
    ScenarioFamilyMatchDecision,
    ScenarioSimilarityProfile,
    ScenarioVersionCompatibility,
    Subject,
    UserScenarioScore,
)
from authoringtool.scenario_matching import (
    apply_candidate_decision,
    build_scenario_similarity_profile,
    review_candidate_with_llm,
    scan_scenario_family_candidates,
)


class ScenarioFamilyDiscoveryMigrationTests(TransactionTestCase):
    migrate_from = ('authoringtool', '0063_evidence_compatibility_pooling')
    migrate_to = ('authoringtool', '0064_scenario_family_discovery')

    def setUp(self):
        super().setUp()
        executor = MigrationExecutor(connection)
        executor.migrate([self.migrate_from])
        apps = executor.loader.project_state([self.migrate_from]).apps

        UserModel = apps.get_model('auth', 'User')
        ScenarioModel = apps.get_model('authoringtool', 'Scenario')
        Family = apps.get_model('authoringtool', 'ScenarioFamily')
        Version = apps.get_model('authoringtool', 'ScenarioVersion')

        owner = UserModel.objects.create(
            username='matching_migration_owner',
            password='unused',
        )
        english = ScenarioModel.objects.create(
            name='Legacy Pendulum English',
            language='English',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        greek = ScenarioModel.objects.create(
            name='Legacy Pendulum Greek',
            language='Greek',
            created_by_id=owner.id,
            updated_by_id=owner.id,
        )
        english_family = Family.objects.create(
            title=english.name,
            canonical_scenario_id=english.id,
            created_by_id=owner.id,
        )
        greek_family = Family.objects.create(
            title=greek.name,
            canonical_scenario_id=greek.id,
            created_by_id=owner.id,
        )
        ScenarioModel.objects.filter(pk=english.id).update(
            family_id=english_family.id
        )
        ScenarioModel.objects.filter(pk=greek.id).update(
            family_id=greek_family.id
        )
        fingerprint = 'd' * 64
        english_version = Version.objects.create(
            scenario_id=english.id,
            version_number=1,
            structure_fingerprint=fingerprint,
            content_fingerprint='e' * 64,
            snapshot={},
            created_by_id=owner.id,
            is_current=True,
        )
        greek_version = Version.objects.create(
            scenario_id=greek.id,
            version_number=1,
            structure_fingerprint=fingerprint,
            content_fingerprint='f' * 64,
            snapshot={},
            created_by_id=owner.id,
            is_current=True,
        )
        ScenarioModel.objects.filter(pk=english.id).update(
            current_version_id=english_version.id
        )
        ScenarioModel.objects.filter(pk=greek.id).update(
            current_version_id=greek_version.id
        )
        self.scenario_ids = (english.id, greek.id)
        self.family_ids = (english_family.id, greek_family.id)

        executor = MigrationExecutor(connection)
        executor.migrate([
            ('authoringtool', '0067_family_review_dashboard_llm')
        ])
        self.apps = executor.loader.project_state([self.migrate_to]).apps

    def test_exact_structure_backfill_creates_review_only_candidate(self):
        Candidate = self.apps.get_model(
            'authoringtool',
            'ScenarioFamilyCandidate',
        )
        ScenarioModel = self.apps.get_model('authoringtool', 'Scenario')

        candidate = Candidate.objects.get(
            scenario_a_id=self.scenario_ids[0],
            scenario_b_id=self.scenario_ids[1],
        )

        self.assertEqual(candidate.decision, 'pending')
        self.assertEqual(candidate.suggested_relationship, 'translation')
        self.assertEqual(float(candidate.similarity_score), 0.88)
        self.assertEqual(
            list(
                ScenarioModel.objects
                .filter(id__in=self.scenario_ids)
                .order_by('id')
                .values_list('family_id', flat=True)
            ),
            list(self.family_ids),
        )

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate([
            ('authoringtool', '0067_family_review_dashboard_llm')
        ])
        super().tearDown()


class ScenarioFamilyDiscoveryTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_superuser(
            'matching_admin',
            'matching@example.com',
            'password',
        )
        self.teacher = User.objects.create_user('matching_teacher')
        self.subject = Subject.objects.create(
            name='Scenario Matching Physics',
            category='STEM',
        )
        self.activity_type = ActivityType.objects.create(
            name='Scenario Matching Question',
            created_by=self.admin,
            updated_by=self.admin,
        )

    def _create_scenario(
        self,
        name,
        language,
        *,
        text,
        origin=None,
        lineage_key=None,
        family=None,
        variant_type='canonical',
    ):
        scenario = Scenario.objects.create(
            name=name,
            language=language,
            description=text,
            learning_goals='Investigate the relationship between variables.',
            subject_domains='Physics',
            suggested_learning_time=20,
            origin_scenario=origin,
            family=family,
            variant_type=variant_type,
            created_by=self.admin,
            updated_by=self.admin,
        )
        scenario.subjects.add(self.subject)
        phase = Phase.objects.create(
            name='Investigation',
            description='Explore evidence.',
            scenario=scenario,
            created_by=self.admin,
            updated_by=self.admin,
        )
        activity_kwargs = {
            'name': f'{name} question',
            'text': text,
            'plain_text': text,
            'scenario': scenario,
            'phase': phase,
            'activity_type': self.activity_type,
            'is_evaluatable': True,
            'is_primary_ev': True,
            'created_by': self.admin,
            'updated_by': self.admin,
        }
        if lineage_key:
            activity_kwargs['lineage_key'] = lineage_key
        activity = Activity.objects.create(**activity_kwargs)
        version = scenario.ensure_current_version()
        return scenario, activity, version

    def _translation_pair(self):
        english, english_activity, _ = self._create_scenario(
            'Pendulum Investigation English',
            'English',
            text='Does changing the mass affect the pendulum period?',
        )
        greek, greek_activity, _ = self._create_scenario(
            'Διερεύνηση Εκκρεμούς',
            'Greek',
            text='Η αλλαγή της μάζας επηρεάζει την περίοδο του εκκρεμούς;',
        )
        return english, english_activity, greek, greek_activity

    def test_scan_suggests_translation_without_automatically_merging(self):
        english, _, greek, _ = self._translation_pair()
        original_families = {english.family_id, greek.family_id}

        result = scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=english,
            scenario_b=greek,
            is_current=True,
        )

        self.assertEqual(result['candidates'], 1)
        self.assertEqual(candidate.decision, 'pending')
        self.assertEqual(candidate.suggested_relationship, 'translation')
        self.assertGreaterEqual(
            candidate.component_scores['structure'],
            0.8,
        )
        english.refresh_from_db()
        greek.refresh_from_db()
        self.assertEqual(
            {english.family_id, greek.family_id},
            original_families,
        )

    def test_translation_decision_merges_logically_and_preserves_records(self):
        english, _, greek, _ = self._translation_pair()
        english_student = User.objects.create_user('matching_student_en')
        greek_student = User.objects.create_user('matching_student_el')
        UserScenarioScore.objects.create(
            user=english_student,
            scenario=english,
        )
        UserScenarioScore.objects.create(
            user=greek_student,
            scenario=greek,
        )
        run = ProposalGenerationRun.start_new(english, self.admin)
        source_family_id = greek.family_id
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=english,
            scenario_b=greek,
        )

        candidate, event = apply_candidate_decision(
            candidate,
            'translation',
            self.admin,
            notes='Confirmed as the official Greek translation.',
            target_family=english.family,
        )

        english.refresh_from_db()
        greek.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(english.family_id, greek.family_id)
        self.assertEqual(greek.variant_type, 'translation')
        self.assertFalse(ScenarioFamily.objects.filter(
            pk=source_family_id
        ).exists())
        self.assertEqual(
            set(UserScenarioScore.objects.values_list('scenario_id', flat=True)),
            {english.id, greek.id},
        )
        self.assertFalse(run.is_current)
        self.assertEqual(candidate.decision, 'translation')
        self.assertEqual(event.details['moved_scenario_ids'], [greek.id])
        self.assertEqual(
            ScenarioFamilyMatchDecision.objects.filter(
                candidate=candidate
            ).count(),
            1,
        )

    def test_adaptation_requires_compatibility_review_after_family_merge(self):
        canonical, canonical_activity, _ = self._create_scenario(
            'Pendulum Canonical Match',
            'English',
            text='Measure the period.',
        )
        adaptation, _, _ = self._create_scenario(
            'Teacher Pendulum Copy Match',
            'English',
            text='Measure the period.',
            origin=canonical,
            lineage_key=canonical_activity.lineage_key,
        )
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=canonical,
            scenario_b=adaptation,
        )

        apply_candidate_decision(
            candidate,
            'adaptation',
            self.admin,
            target_family=canonical.family,
        )

        canonical.refresh_from_db()
        adaptation.refresh_from_db()
        membership = ScenarioVersionCompatibility.objects.get(
            scenario_version=adaptation.current_version
        )
        self.assertEqual(canonical.family_id, adaptation.family_id)
        self.assertEqual(adaptation.variant_type, 'adaptation')
        self.assertEqual(membership.status, 'needs_review')
        self.assertEqual(
            list(
                canonical.compatible_current_versions()
                .values_list('scenario_id', flat=True)
            ),
            [canonical.id],
        )

    def test_related_topic_decision_never_changes_families(self):
        first, _, second, _ = self._translation_pair()
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=first,
            scenario_b=second,
        )
        family_ids = (first.family_id, second.family_id)

        apply_candidate_decision(
            candidate,
            'related_topic',
            self.admin,
            notes='Same subject, independently designed lesson.',
        )

        first.refresh_from_db()
        second.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(
            (first.family_id, second.family_id),
            family_ids,
        )
        self.assertEqual(candidate.decision, 'related_topic')

    def test_merging_a_family_moves_all_variants_and_can_not_be_silently_undone(self):
        target, _, _ = self._create_scenario(
            'Target Family Scenario',
            'English',
            text='Investigate force and motion.',
        )
        source, source_activity, _ = self._create_scenario(
            'Source Family Scenario',
            'English',
            text='Investigate force and motion.',
        )
        source_variant, _, _ = self._create_scenario(
            'Source Family Greek Variant',
            'Greek',
            text='Διερεύνηση δύναμης και κίνησης.',
            origin=source,
            lineage_key=source_activity.lineage_key,
            family=source.family,
            variant_type='translation',
        )
        source_family_id = source.family_id
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=target,
            scenario_b=source,
            is_current=True,
        )

        apply_candidate_decision(
            candidate,
            'adaptation',
            self.admin,
            target_family=target.family,
        )

        source.refresh_from_db()
        source_variant.refresh_from_db()
        self.assertEqual(source.family_id, target.family_id)
        self.assertEqual(source_variant.family_id, target.family_id)
        self.assertEqual(source.variant_type, 'adaptation')
        self.assertEqual(source_variant.variant_type, 'translation')
        self.assertFalse(
            ScenarioFamily.objects.filter(pk=source_family_id).exists()
        )
        with self.assertRaises(ValidationError):
            apply_candidate_decision(
                candidate,
                'unrelated',
                self.admin,
            )

    def test_changed_version_supersedes_the_old_candidate(self):
        english, _, greek, greek_activity = self._translation_pair()
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        old_candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=english,
            scenario_b=greek,
        )
        greek_activity.text += ' Additional teacher clarification.'
        greek_activity.plain_text = greek_activity.text
        greek_activity.save(update_fields=['text', 'plain_text'])
        greek.ensure_current_version()

        scan_scenario_family_candidates(
            force_profiles=True,
            include_embedding=False,
            min_score=0.45,
        )

        old_candidate.refresh_from_db()
        current = ScenarioFamilyCandidate.objects.get(
            scenario_a=english,
            scenario_b=greek,
            is_current=True,
        )
        self.assertFalse(old_candidate.is_current)
        self.assertNotEqual(old_candidate.id, current.id)
        self.assertNotEqual(
            old_candidate.scenario_b_version_id,
            current.scenario_b_version_id,
        )

    def test_profile_refresh_tracks_current_immutable_version(self):
        scenario, _, original_version = self._create_scenario(
            'Profile Refresh Scenario',
            'English',
            text='Original description.',
        )
        profile, _ = build_scenario_similarity_profile(
            scenario,
            include_embedding=False,
        )
        scenario.description = 'Changed pedagogical description.'
        scenario.save(update_fields=['description'])
        changed_version = scenario.ensure_current_version()
        profile.refresh_from_db()

        self.assertNotEqual(original_version.id, changed_version.id)
        self.assertTrue(profile.is_stale)

        refreshed, changed = build_scenario_similarity_profile(
            scenario,
            include_embedding=False,
        )
        self.assertTrue(changed)
        self.assertEqual(refreshed.scenario_version_id, changed_version.id)
        self.assertFalse(refreshed.is_stale)

    def test_admin_review_screen_and_decision_buttons(self):
        english, _, greek, _ = self._translation_pair()
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=english,
            scenario_b=greek,
        )
        self.client.force_login(self.admin)
        changelist_url = reverse(
            'admin:authoringtool_scenariofamilycandidate_changelist'
        )
        detail_url = reverse(
            'admin:authoringtool_scenariofamilycandidate_change',
            args=[candidate.id],
        )

        changelist = self.client.get(changelist_url)
        detail = self.client.get(detail_url)
        decision = self.client.post(detail_url, {
            'target_family': english.family_id,
            'review_notes': 'Keep related, but do not merge.',
            '_classify_related_topic': '1',
        })

        self.assertEqual(changelist.status_code, 200)
        self.assertContains(changelist, 'Scan scenarios')
        self.assertEqual(detail.status_code, 200)
        self.assertContains(detail, 'Side-by-side scenario comparison')
        self.assertContains(detail, 'Learning-flow structure')
        self.assertEqual(decision.status_code, 302)
        candidate.refresh_from_db()
        self.assertEqual(candidate.decision, 'related_topic')

    def test_family_review_dashboard_shows_variants_and_revisions(self):
        english, _, greek, _ = self._translation_pair()
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        self.client.force_login(self.admin)

        admin_index = self.client.get(reverse('admin:index'))
        response = self.client.get(reverse(
            'admin:authoringtool_scenariofamilyreviewproxy_changelist'
        ))

        self.assertEqual(admin_index.status_code, 200)
        self.assertContains(admin_index, 'Authoring Tool')
        self.assertContains(
            admin_index,
            'Scenario family review dashboard',
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Scenario family governance')
        self.assertContains(response, english.name)
        self.assertContains(response, greek.name)
        self.assertContains(response, 'Ask Ollama to review pending')
        self.assertContains(response, 'Open candidate inbox')
        self.assertContains(response, 'Associate scenarios manually')

    def test_manual_association_page_previews_without_changing_families(self):
        english, _, greek, _ = self._translation_pair()
        original_family_ids = (english.family_id, greek.family_id)
        self.client.force_login(self.admin)
        association_url = reverse(
            'admin:authoringtool_scenariofamilyreviewproxy_associate'
        )

        page = self.client.get(association_url)
        preview = self.client.post(association_url, {
            'target_scenario': english.id,
            'source_scenario': greek.id,
            'relationship': 'translation',
            'review_notes': 'Manual translation review.',
            'action': 'preview',
        })

        self.assertEqual(page.status_code, 200)
        self.assertContains(page, 'Revision or scenario association?')
        self.assertContains(page, 'Adaptation / revised copy')
        self.assertEqual(preview.status_code, 200)
        self.assertContains(
            preview,
            'Association preview: Official translation',
        )
        self.assertContains(
            preview,
            'Confirm Official translation',
        )
        self.assertTrue(preview.context['confirmation_token'])
        english.refresh_from_db()
        greek.refresh_from_db()
        self.assertEqual(
            (english.family_id, greek.family_id),
            original_family_ids,
        )
        self.assertFalse(ScenarioFamilyCandidate.objects.exists())

    def test_manual_translation_confirmation_merges_and_audits(self):
        english, _, greek, _ = self._translation_pair()
        source_family_id = greek.family_id
        self.client.force_login(self.admin)
        association_url = reverse(
            'admin:authoringtool_scenariofamilyreviewproxy_associate'
        )
        form_data = {
            'target_scenario': english.id,
            'source_scenario': greek.id,
            'relationship': 'translation',
            'review_notes': 'Confirmed manually by curriculum admin.',
        }
        preview = self.client.post(association_url, {
            **form_data,
            'action': 'preview',
        })

        confirmation = self.client.post(association_url, {
            **form_data,
            'action': 'confirm',
            'confirmation_token': preview.context[
                'confirmation_token'
            ],
        })

        self.assertEqual(confirmation.status_code, 302)
        english.refresh_from_db()
        greek.refresh_from_db()
        self.assertEqual(greek.family_id, english.family_id)
        self.assertEqual(greek.variant_type, 'translation')
        self.assertFalse(
            ScenarioFamily.objects.filter(pk=source_family_id).exists()
        )
        candidate = ScenarioFamilyCandidate.objects.get()
        self.assertEqual(candidate.decision, 'translation')
        self.assertEqual(candidate.detection_method, 'manual-admin-v1')
        event = ScenarioFamilyMatchDecision.objects.get(
            candidate=candidate
        )
        self.assertEqual(event.decided_by, self.admin)
        self.assertEqual(
            event.notes,
            'Confirmed manually by curriculum admin.',
        )

    def test_llm_review_is_structured_persisted_and_never_merges(self):
        english, _, greek, _ = self._translation_pair()
        original_family_ids = (english.family_id, greek.family_id)
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get(
            scenario_a=english,
            scenario_b=greek,
        )
        response = Mock()
        response.json.return_value = {
            'response': json.dumps({
                'relationship': 'translation',
                'confidence': 0.91,
                'reasoning': (
                    'The learning goals and branching structure are '
                    'materially equivalent across languages.'
                ),
                'evidence': [
                    'The immutable revisions have matching activity flows.',
                ],
                'warnings': [
                    'An administrator should verify the translated wording.',
                ],
            }),
        }

        with patch(
            'authoringtool.scenario_matching.requests.post',
            return_value=response,
        ) as request_mock:
            reviewed = review_candidate_with_llm(candidate.id)

        self.assertEqual(reviewed.llm_status, 'completed')
        self.assertEqual(
            reviewed.llm_suggested_relationship,
            'translation',
        )
        self.assertAlmostEqual(float(reviewed.llm_confidence), 0.91)
        self.assertIn('branching structure', reviewed.llm_reasoning)
        self.assertEqual(
            reviewed.llm_details['reviewed_scenario_a_version_id'],
            candidate.scenario_a_version_id,
        )
        request_payload = request_mock.call_args.kwargs['json']
        self.assertFalse(request_payload['stream'])
        self.assertIn('format', request_payload)
        english.refresh_from_db()
        greek.refresh_from_db()
        self.assertEqual(
            (english.family_id, greek.family_id),
            original_family_ids,
        )

    @patch(
        'authoringtool.tasks.'
        'review_scenario_family_candidates_with_llm_task.delay'
    )
    def test_admin_can_queue_pending_candidates_for_llm(self, delay_mock):
        delay_mock.return_value.id = 'family-review-task'
        self._translation_pair()
        scan_scenario_family_candidates(
            include_embedding=False,
            min_score=0.45,
        )
        candidate = ScenarioFamilyCandidate.objects.get()
        self.client.force_login(self.admin)

        response = self.client.post(reverse(
            'admin:authoringtool_scenariofamilycandidate_'
            'llm_review_pending'
        ))

        self.assertEqual(response.status_code, 302)
        candidate.refresh_from_db()
        self.assertEqual(candidate.llm_status, 'pending')
        delay_mock.assert_called_once_with([candidate.id])

    @override_settings(
        CELERY_TASK_ALWAYS_EAGER=True,
        CELERY_TASK_EAGER_PROPAGATES=True,
    )
    def test_admin_scan_endpoint_runs_the_background_discovery_task(self):
        english, _, greek, _ = self._translation_pair()
        self.client.force_login(self.admin)
        scan_url = reverse(
            'admin:authoringtool_scenariofamilycandidate_scan'
        )

        response = self.client.post(scan_url)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            ScenarioSimilarityProfile.objects.filter(
                scenario__in=[english, greek]
            ).count(),
            2,
        )
        self.assertTrue(
            ScenarioFamilyCandidate.objects.filter(
                scenario_a=english,
                scenario_b=greek,
                is_current=True,
            ).exists()
        )

    def test_non_staff_user_cannot_access_discovery_admin(self):
        self.client.force_login(self.teacher)
        response = self.client.get(
            reverse('admin:authoringtool_scenariofamilycandidate_changelist')
        )
        self.assertNotEqual(response.status_code, 200)
        dashboard = self.client.get(reverse(
            'admin:authoringtool_scenariofamilyreviewproxy_changelist'
        ))
        self.assertNotEqual(dashboard.status_code, 200)
        association = self.client.get(reverse(
            'admin:authoringtool_scenariofamilyreviewproxy_associate'
        ))
        self.assertNotEqual(association.status_code, 200)
