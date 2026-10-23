from django.conf import settings
from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse

from accounts.models import UserProfile
from localization.languages import code_for_table_name

COOKIE = settings.LANGUAGE_COOKIE_NAME


class LanguageChoiceTests(TestCase):
    def make_user(self, language=None):
        user = User.objects.create_user('lang_user', password='pw12345!')
        user.groups.add(Group.objects.get_or_create(name='teachers')[0])
        if language is not None:
            UserProfile.objects.create(user=user, language=language)
        return user

    def login(self, user):
        self.assertTrue(self.client.login(username=user.username, password='pw12345!'))

    def test_browser_language_used_on_first_visit(self):
        response = self.client.get('/accounts/login/', HTTP_ACCEPT_LANGUAGE='el')
        self.assertEqual(response.context['LANGUAGE_CODE'], 'el')
        self.assertIn('lang="el"', response.content.decode())

    def test_unsupported_browser_language_falls_back_to_english(self):
        response = self.client.get('/accounts/login/', HTTP_ACCEPT_LANGUAGE='ja')
        self.assertEqual(response.context['LANGUAGE_CODE'], 'en')

    def test_switch_sets_cookie_and_profile(self):
        user = self.make_user()
        self.login(user)
        response = self.client.post(reverse('switch_language'),
                                    {'language': 'de', 'next': '/accounts/profile/'})
        self.assertRedirects(response, '/accounts/profile/', fetch_redirect_response=False)
        self.assertEqual(response.cookies[COOKIE].value, 'de')
        self.assertEqual(UserProfile.objects.get(user=user).language, 'de')

    def test_switch_rejects_unknown_code(self):
        response = self.client.post(reverse('switch_language'), {'language': 'fa'})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(COOKIE, response.cookies)

    def test_switch_ignores_foreign_next(self):
        response = self.client.post(reverse('switch_language'),
                                    {'language': 'fr', 'next': 'https://evil.test/'})
        self.assertRedirects(response, '/', fetch_redirect_response=False)

    def test_switch_requires_post(self):
        self.assertEqual(self.client.get(reverse('switch_language')).status_code, 405)

    def test_saved_language_applies_after_login(self):
        user = self.make_user('it')
        self.client.cookies[COOKIE] = 'en'
        self.login(user)
        response = self.client.get(reverse('scenarios'))
        self.assertEqual(response.context['LANGUAGE_CODE'], 'it')
        self.assertEqual(response.cookies[COOKIE].value, 'it')

    def test_first_login_saves_browser_language(self):
        user = self.make_user('')
        self.client.cookies[COOKIE] = 'fr'
        self.login(user)
        self.client.get(reverse('scenarios'))
        self.assertEqual(UserProfile.objects.get(user=user).language, 'fr')

    def test_unknown_saved_language_falls_back_to_english(self):
        user = self.make_user('fa')
        self.login(user)
        response = self.client.get(reverse('scenarios'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['LANGUAGE_CODE'], 'en')

    def test_switcher_lists_all_languages_with_flags(self):
        user = self.make_user('en')
        self.login(user)
        html = self.client.get(reverse('scenarios')).content.decode()
        self.assertIn('flag-icons/4x3/es-pv.svg', html)
        self.assertIn('Ελληνικά', html)
        self.assertIn(reverse('switch_language'), html)
        self.assertEqual(html.count('name="language" value="'), 13)
        self.assertNotIn('value="fa"', html)

    def test_code_for_table_name(self):
        for name in ('Greek', ' greek ', 'Ελληνικά', 'el'):
            self.assertEqual(code_for_table_name(name), 'el')
        self.assertIsNone(code_for_table_name('Persian'))
        self.assertIsNone(code_for_table_name('Klingon'))
        self.assertIsNone(code_for_table_name(None))
