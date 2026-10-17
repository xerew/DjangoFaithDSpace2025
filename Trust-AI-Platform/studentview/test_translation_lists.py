"""Translations in the teachers' Student View list and the group scenario
picker, and the position of Plato's chat box."""
from django.test import override_settings
from django.urls import reverse

from authoringtool.models import Scenario
from authoringtool.test_translation_pooling import PoolingBase
from authoringtool.translation_matching import bundle_translations
from usergroups.models import UserGroup, UserGroupMembership


class BundleTranslationsTests(PoolingBase):
    def test_translations_are_bundled_with_english_first(self):
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        english = self.build('Pendulum', 'English')
        spanish = self.build('Péndulo', 'Español')
        other = self.build('Optics', 'English')
        self.confirm(english, greek)
        self.confirm(english, spanish)

        bundles = bundle_translations(
            [greek[0], other[0], english[0], spanish[0]],
        )

        self.assertEqual(
            [[s.id for s in bundle] for bundle in bundles],
            [[english[0].id, spanish[0].id, greek[0].id], [other[0].id]],
        )

    def test_without_english_the_oldest_comes_first(self):
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        romanian = self.build('Pendulul', 'Romanian')
        self.confirm(greek, romanian)

        bundles = bundle_translations([romanian[0], greek[0]])

        self.assertEqual([s.id for s in bundles[0]], [greek[0].id, romanian[0].id])


class StudentViewTranslationsTests(PoolingBase):
    def setUp(self):
        super().setUp()
        self.greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.english = self.build('Pendulum', 'English')
        self.confirm(self.english, self.greek)
        self.newer = self.build('Optics', 'English')

    def listed(self, user, **params):
        self.client.force_login(user)
        response = self.client.get(reverse('studentScenarios'), params)
        return response, [scenario.id for scenario in response.context['myScenarios']]

    def test_teacher_sees_one_card_per_group_translations_first(self):
        response, ids = self.listed(self.owner)

        self.assertEqual(ids, [self.english[0].id, self.newer[0].id])
        self.assertContains(response, 'Translations (1)')

    def test_translations_modal_starts_each_language(self):
        response, _ = self.listed(self.owner)

        self.assertContains(response, f'id="svTrModal{self.english[0].id}"')
        self.assertContains(
            response, reverse('studentView', args=[self.greek[0].id]),
        )
        self.assertContains(response, 'Ελληνικά')

    def test_language_filter_shows_the_matching_version(self):
        _, ids = self.listed(self.owner, language='Ελληνικά')

        self.assertEqual(ids, [self.greek[0].id])

    def test_students_list_is_unchanged(self):
        student = self.student_answers(self.english, True, 5)
        group = UserGroup.objects.create(
            name='Class A', created_by=self.owner, number_of_users=1,
        )
        group.assigned_scenarios.set([self.english[0], self.greek[0]])
        UserGroupMembership.objects.create(group=group, user=student)

        response, ids = self.listed(student)

        self.assertCountEqual(ids, [self.english[0].id, self.greek[0].id])
        self.assertNotContains(response, 'Translations (')


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class GroupScenarioPickerTests(PoolingBase):
    def setUp(self):
        super().setUp()
        self.greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.english = self.build('Pendulum', 'English')
        self.confirm(self.english, self.greek)
        self.optics = self.build('Optics', 'English')
        self.client.force_login(self.owner)

    def picker(self, response):
        html = response.content.decode()
        start = html.index('id="scenarioList"')
        return html[start:html.index('<script', start)]

    def test_translations_share_one_row_with_a_language_choice(self):
        picker = self.picker(self.client.get(reverse('create_group')))

        self.assertEqual(picker.count('class="scenario-item"'), 2)
        self.assertIn('class="form-select form-select-sm scenario-lang-select"', picker)
        self.assertIn(f'<option value="{self.english[0].id}"', picker)
        self.assertIn(f'<option value="{self.greek[0].id}"', picker)
        self.assertIn('Pendulum <span class="scenario-lang">(English)</span>', picker)

    def test_language_is_shown_for_single_scenarios(self):
        picker = self.picker(self.client.get(reverse('create_group')))

        self.assertIn('Optics <span class="scenario-lang">(English)</span>', picker)

    def test_language_choice_is_disabled_until_the_row_is_ticked(self):
        picker = self.picker(self.client.get(reverse('create_group')))

        select = picker[picker.index('<select'):picker.index('>', picker.index('<select'))]
        self.assertIn('disabled', select)
        self.assertIn('name="assigned_scenarios"', select)

    def test_edit_shows_the_assigned_language(self):
        group = UserGroup.objects.create(
            name='Class B', created_by=self.owner, number_of_users=0,
        )
        group.assigned_scenarios.set([self.greek[0]])

        picker = self.picker(self.client.get(reverse('edit_group', args=[group.id])))

        self.assertIn(f'<option value="{self.greek[0].id}" selected', picker)
        self.assertIn('Εκκρεμές <span class="scenario-lang">(Ελληνικά)</span>', picker)
        select = picker[picker.index('<select'):picker.index('>', picker.index('<select'))]
        self.assertNotIn('disabled', select)

    def test_two_assigned_languages_both_stay_assigned(self):
        group = UserGroup.objects.create(
            name='Class C', created_by=self.owner, number_of_users=0,
        )
        group.assigned_scenarios.set([self.english[0], self.greek[0]])

        picker = self.picker(self.client.get(reverse('edit_group', args=[group.id])))

        self.assertIn(f'<option value="{self.english[0].id}" selected', picker)
        self.assertIn(f'<option value="{self.greek[0].id}" selected', picker)


class PlatoChatBoxTests(PoolingBase):
    def test_chat_box_is_not_forced_out_of_sticky_positioning(self):
        scenario = self.build('Pendulum', 'English')[0]
        self.client.force_login(self.owner)

        response = self.client.get(reverse('studentView', args=[scenario.id]))

        self.assertContains(response, '<div class="chatbot-card">')
        self.assertNotContains(response, 'class="chatbot-card" style="position:relative;"')
