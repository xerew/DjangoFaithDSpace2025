"""Built-in reference values (subjects, types, groups, languages) get display labels."""
from unittest import mock

from django.contrib.auth.models import Group, User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import translation

from authoringtool.models import Scenario, Subject
from localization.catalog import marked_msgids
from localization.reference_labels import ui_label

KNOWN = [
    'Physics', 'Mathematics', 'Chemistry', 'Biology', 'Computer Science',
    'Environmental Science', 'Earth & Space Science', 'Engineering & Technology',
    'Statistics & Data Science',
    'STEM', 'Humanities', 'Social Sciences', 'Arts', 'Other',
    'Explanation', 'Experiment', 'Guidance', 'Question',
    'High', 'Moderate', 'Low',
]


class UiLabelTests(SimpleTestCase):
    def test_known_values_are_marked(self):
        marked = marked_msgids('el')
        for value in KNOWN:
            self.assertIn(value, marked)

    def test_language_names_are_native(self):
        for code in ('en', 'el', 'fr'):
            with translation.override(code):
                self.assertEqual(ui_label('Greek'), 'Ελληνικά')
                self.assertEqual(ui_label('Persian'), 'فارسی')

    def test_names_of_languages_not_offered_are_still_native(self):
        names = {'Basque': 'Euskara', 'Catalan': 'Català', 'Estonian': 'Eesti',
                 'Italian': 'Italiano', 'Turkish': 'Türkçe',
                 'Ukrainian': 'Українська', 'Persian': 'فارسی'}
        for code in ('en', 'de'):
            with translation.override(code):
                for table_name, native in names.items():
                    self.assertEqual(ui_label(table_name), native)

    def test_unknown_value_unchanged(self):
        self.assertEqual(ui_label('Robotics'), 'Robotics')
        self.assertIsNone(ui_label(None))
        self.assertEqual(ui_label(''), '')

    def test_english_is_identical_and_plain_str(self):
        with translation.override('en'):
            for value in KNOWN:
                label = ui_label(value)
                self.assertEqual(label, value)
                self.assertIs(type(label), str)


class ScenariosPageTests(TestCase):
    def test_scenarios_page_uses_ui_label(self):
        teacher = User.objects.create_user('ref_teacher', password='pass')
        teacher.groups.add(Group.objects.create(name='teachers'))
        self.client.force_login(teacher)
        subject, _ = Subject.objects.get_or_create(name='Physics', defaults={'category': 'STEM'})
        scenario = Scenario.objects.create(
            name='Ref scenario', created_by=teacher, updated_by=teacher, language='Greek')
        scenario.subjects.add(subject)
        with mock.patch('localization.reference_labels.ui_label',
                        return_value='PHYSICS-LABEL'):
            response = self.client.get(reverse('scenarios'))
        self.assertContains(response, 'PHYSICS-LABEL')


class VisibleLabelsKeepValuesTests(TestCase):
    def setUp(self):
        from organization.models import Organization  # noqa: F401
        self.teacher = User.objects.create_user('ref_teacher2', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))
        self.client.force_login(self.teacher)

    def test_language_filter_label_translated_value_english(self):
        Scenario.objects.create(name='Lang s', created_by=self.teacher,
                                updated_by=self.teacher, language='Greek')
        with mock.patch('localization.reference_labels.ui_label',
                        side_effect=lambda v: 'LBL-' + v if v else v):
            response = self.client.get(reverse('scenarios'))
        self.assertContains(response, 'LBL-Greek')
        self.assertContains(response, 'value="Greek"')

    def test_create_simulation_json_has_language_label(self):
        staff = User.objects.create_user('ref_staff', password='x', is_staff=True)
        self.client.force_login(staff)
        with mock.patch('localization.reference_labels.ui_label',
                        return_value='GR-LABEL'):
            r = self.client.post('/accounts/admin/simulations/create/', {
                'name': 'S', 'iframe_url': 'https://x.com', 'width': '800',
                'height': '600', 'allow_fullscreen': 'true', 'language': 'Greek'})
        data = r.json()
        self.assertEqual(data['language'], 'Greek')
        self.assertEqual(data['language_label'], 'GR-LABEL')
