"""Platform Administration: user names link to profiles, and a Languages tab
adds, renames and removes the languages scenarios and organizations use."""
from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import Language, Scenario
from organization.models import Organization


class AdminUserProfileLinkTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user('links_admin', password='pass', is_staff=True)
        teachers = Group.objects.create(name='teachers')
        partners = Group.objects.create(name='dspace_partners')
        self.teacher = User.objects.create_user('links_teacher', first_name='Tina', last_name='Teacher')
        self.teacher.groups.add(teachers)
        self.partner = User.objects.create_user('links_partner', first_name='Pat', last_name='Partner')
        self.partner.groups.add(partners)
        self.client.force_login(self.admin)

    def test_every_name_in_the_user_table_links_to_the_profile(self):
        response = self.client.get(reverse('admin_dashboard'))
        for user in (self.teacher, self.partner):
            with self.subTest(user=user.username):
                self.assertContains(response, f'href="{reverse("view_profile", args=[user.pk])}"')

    def test_admins_can_open_any_profile(self):
        response = self.client.get(reverse('view_profile', args=[self.partner.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Pat Partner')

    def test_teachers_still_cannot_open_non_teacher_profiles(self):
        self.client.force_login(self.teacher)
        response = self.client.get(reverse('view_profile', args=[self.partner.pk]))
        self.assertEqual(response.status_code, 404)


class AdminLanguageTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user('lang_admin', password='pass', is_staff=True)
        self.client.force_login(self.admin)
        self.klingon = Language.objects.create(name='Klingon')
        self.greek, _ = Language.objects.get_or_create(name='Greek')  # seeded by a migration

    def post(self, url_name, *args, **data):
        return self.client.post(reverse(url_name, args=args), data).json()

    def test_add_language(self):
        result = self.post('admin_create_language', name='  Vulcan ')
        self.assertTrue(result['success'])
        self.assertTrue(Language.objects.filter(name='Vulcan').exists())

    def test_add_refuses_empty_and_duplicate_names(self):
        self.assertFalse(self.post('admin_create_language', name='  ')['success'])
        result = self.post('admin_create_language', name='klingon')
        self.assertFalse(result['success'])
        self.assertIn('already exists', result['error'])

    def test_rename_updates_scenarios_and_organizations(self):
        owner = User.objects.create_user('lang_owner')
        scenario = Scenario.objects.create(
            name='Q', language='Klingon', created_by=owner, updated_by=owner,
        )
        org = Organization.objects.create(name='Org', short_name='org', language='Klingon')
        result = self.post('admin_rename_language', self.klingon.id, name='tlhIngan Hol')
        self.assertTrue(result['success'])
        scenario.refresh_from_db()
        org.refresh_from_db()
        self.assertEqual(scenario.language, 'tlhIngan Hol')
        self.assertEqual(org.language, 'tlhIngan Hol')

    def test_rename_refused_for_languages_with_built_in_texts(self):
        result = self.post('admin_rename_language', self.greek.id, name='Ellinika')
        self.assertFalse(result['success'])
        self.greek.refresh_from_db()
        self.assertEqual(self.greek.name, 'Greek')

    def test_remove_refused_while_in_use(self):
        owner = User.objects.create_user('lang_owner2')
        Scenario.objects.create(name='Q', language='Klingon', created_by=owner, updated_by=owner)
        result = self.post('admin_delete_language', self.klingon.id)
        self.assertFalse(result['success'])
        self.assertIn('1 scenario', result['error'])
        self.assertTrue(Language.objects.filter(pk=self.klingon.pk).exists())

    def test_remove_unused_language(self):
        self.assertTrue(self.post('admin_delete_language', self.klingon.id)['success'])
        self.assertFalse(Language.objects.filter(pk=self.klingon.pk).exists())

    def test_only_admins_can_change_languages(self):
        teacher = User.objects.create_user('lang_teacher')
        teacher.groups.add(Group.objects.create(name='teachers'))
        self.client.force_login(teacher)
        response = self.client.post(reverse('admin_create_language'), {'name': 'Vulcan'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(Language.objects.filter(name='Vulcan').exists())

    def test_dashboard_lists_languages_with_usage(self):
        owner = User.objects.create_user('lang_owner3')
        Scenario.objects.create(name='Q', language='Klingon', created_by=owner, updated_by=owner)
        response = self.client.get(reverse('admin_dashboard'))
        languages = {row['language'].name: row for row in response.context['languages']}
        self.assertEqual(languages['Klingon']['scenario_count'], 1)
        self.assertTrue(languages['Greek']['built_in'])
        self.assertContains(response, 'id="tab-languages"')
