from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Phase,
    Scenario,
    ScenarioRevisionDraft,
    UserScenarioScore,
)


class RevisionProtectionBase(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('protect_owner', password='pass')
        self.owner.groups.add(teachers)
        self.student = User.objects.create_user('protect_student', password='pass')
        self.scenario = Scenario.objects.create(
            name='Protected pendulum', language='English',
            visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.first_phase = Phase.objects.create(
            name='Explore', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        self.second_phase = Phase.objects.create(
            name='Explain', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        activity_type = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        Activity.objects.create(
            name='Read', text='Read', scenario=self.scenario,
            phase=self.first_phase, activity_type=activity_type,
            created_by=self.owner, updated_by=self.owner,
        )
        self.scenario.refresh_from_db()
        self.scenario.ensure_current_version(created_by=self.owner)
        UserScenarioScore.objects.create(user=self.student, scenario=self.scenario)
        self.client.force_login(self.owner)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ProtectionSwitchedOffTests(RevisionProtectionBase):
    def test_edit_page_has_no_lock_and_save_is_enabled(self):
        response = self.client.get(
            reverse('updateScenario', args=[self.scenario.id]),
        )

        self.assertNotContains(response, 'Published evidence is protected')
        self.assertNotContains(response, 'Start Revision Draft')
        self.assertFalse(response.context['revision_protected'])

    def test_phase_can_be_edited_without_a_draft(self):
        # (Scenario details store an age range, which SQLite cannot hold;
        # a phase edit goes through the same revision guard.)
        self.client.post(
            reverse(
                'updatePhaseData',
                args=[self.scenario.id, self.first_phase.id],
            ),
            {'name': 'Explore renamed', 'description': 'Changed'},
        )

        self.first_phase.refresh_from_db()
        self.assertEqual(self.first_phase.name, 'Explore renamed')

    def test_phase_can_be_deleted_without_a_draft(self):
        self.client.get(
            reverse('deletePhase', args=[self.scenario.id, self.second_phase.id]),
        )

        self.assertFalse(Phase.objects.filter(pk=self.second_phase.pk).exists())

    def test_revision_draft_cannot_be_started(self):
        self.client.post(
            reverse('begin_scenario_revision', args=[self.scenario.id]),
        )

        self.assertFalse(ScenarioRevisionDraft.objects.exists())

    def test_saving_content_does_not_raise(self):
        self.assertIsNotNone(
            self.scenario.refresh_version_if_initialized(created_by=self.owner)
        )

    def test_scenario_page_offers_new_phase(self):
        response = self.client.get(
            reverse('viewScenario', args=[self.scenario.id]),
        )

        self.assertNotContains(
            response, 'Start a revision draft before changing this scenario',
        )

    def test_scenario_with_students_still_cannot_be_deleted(self):
        self.client.get(reverse('deleteScenario', args=[self.scenario.id]))

        self.assertTrue(Scenario.objects.filter(pk=self.scenario.pk).exists())


@override_settings(SCENARIO_REVISION_PROTECTION=True)
class ProtectionSwitchedOnTests(RevisionProtectionBase):
    def test_edit_page_shows_lock(self):
        response = self.client.get(
            reverse('updateScenario', args=[self.scenario.id]),
        )

        self.assertContains(response, 'Published evidence is protected')
        self.assertTrue(response.context['revision_protected'])

    def test_phase_cannot_be_deleted_without_a_draft(self):
        self.client.get(
            reverse('deletePhase', args=[self.scenario.id, self.second_phase.id]),
        )

        self.assertTrue(Phase.objects.filter(pk=self.second_phase.pk).exists())
