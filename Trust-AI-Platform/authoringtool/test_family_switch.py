from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.evidence import normalize_evidence_scope
from authoringtool.models import Scenario, ScenarioRevisionDraft


class FamilySwitchBase(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.admin = User.objects.create_superuser('switch_admin', password='pass')
        self.admin.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Pooled pendulum', language='English',
            visibility_status='public', use_family_evidence_pooling=True,
            created_by=self.admin, updated_by=self.admin,
        )
        self.client.force_login(self.admin)

    def admin_index(self):
        return self.client.get(reverse('admin:index'))


@override_settings(SCENARIO_FAMILY_FEATURES=False, SCENARIO_REVISION_PROTECTION=False)
class FamilyFeaturesOffTests(FamilySwitchBase):
    def test_pooling_is_off_even_when_the_scenario_asks_for_it(self):
        self.assertFalse(self.scenario.family_pooling_enabled)
        self.assertEqual(
            normalize_evidence_scope('compatible', self.scenario),
            'local',
        )

    def test_scenario_page_has_no_family_or_version_details(self):
        response = self.client.get(
            reverse('viewScenario', args=[self.scenario.id]),
        )

        self.assertNotContains(response, 'Compatible Evidence Pool')
        self.assertNotContains(response, 'Evidence Version')

    def test_admin_hides_family_and_version_screens(self):
        response = self.admin_index()

        for hidden in (
            'Scenario Families', 'Scenario family review dashboard',
            'Scenario Family Candidates', 'Scenario Family Match Decisions',
            'Evidence Compatibility Clusters',
            'Scenario Version Compatibilities',
            'Scenario Similarity Profiles', 'Activity Concepts',
            'Activity Matching', 'Scenario Versions', 'Activity Revisions',
            'Scenario Revision Drafts',
        ):
            with self.subTest(screen=hidden):
                self.assertNotContains(response, f'>{hidden}<')
        self.assertContains(response, '>Translation Matches<')
        self.assertContains(response, '>Scenarios<')

    def test_open_drafts_stay_reachable_in_admin(self):
        version = self.scenario.ensure_current_version(created_by=self.admin)
        ScenarioRevisionDraft.objects.create(
            scenario=self.scenario, base_version=version, created_by=self.admin,
        )

        response = self.admin_index()

        self.assertContains(response, '>Scenario Revision Drafts<')

    def test_scenario_admin_has_no_family_fields(self):
        changelist = self.client.get(
            reverse('admin:authoringtool_scenario_changelist'),
        )
        change = self.client.get(
            reverse('admin:authoringtool_scenario_change', args=[self.scenario.id]),
        )

        self.assertNotContains(changelist, 'scan_selected_for_family_matches')
        self.assertNotContains(changelist, 'column-family')
        self.assertNotContains(change, 'name="use_family_evidence_pooling"')
        self.assertNotContains(change, 'name="family"')


@override_settings(SCENARIO_FAMILY_FEATURES=True, SCENARIO_REVISION_PROTECTION=True)
class FamilyFeaturesOnTests(FamilySwitchBase):
    def test_pooling_follows_the_scenario_setting(self):
        self.assertTrue(self.scenario.family_pooling_enabled)

    def test_admin_shows_family_and_version_screens(self):
        response = self.admin_index()

        self.assertContains(response, '>Scenario Families<')
        self.assertContains(response, '>Scenario Versions<')

    def test_scenario_admin_has_family_fields(self):
        change = self.client.get(
            reverse('admin:authoringtool_scenario_change', args=[self.scenario.id]),
        )

        self.assertContains(change, 'name="use_family_evidence_pooling"')
