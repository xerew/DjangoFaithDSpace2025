"""Keeping corrections current: database failures must not cost a request or an email."""
from unittest.mock import PropertyMock, patch

from django.conf import settings
from django.contrib.auth.models import User
from django.core.cache import cache
from django.db import OperationalError, connection
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import translation
from django.utils.translation import gettext

from accounts.models import UserProfile
from localization.languages import recipient_language
from localization.models import TranslationOverride
from localization.overrides import bump_overrides_version, overrides_version, reset_overrides

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
CATALOG_CHECK_AGAIN = 'Νέος έλεγχος'


def failing_query(*args):
    """A real failed statement: on PostgreSQL it breaks the surrounding transaction."""
    with connection.cursor() as cursor:
        cursor.execute('SELECT * FROM localization_no_such_table')


@override_settings(CACHES=LOCMEM)
class OverrideVersionTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)

    def test_every_version_is_new(self):
        bump_overrides_version()
        first = overrides_version()
        bump_overrides_version()
        second = overrides_version()
        self.assertIsInstance(first, str)
        self.assertNotEqual(first, second)
        cache.clear()
        self.assertIsInstance(overrides_version(), str)


@override_settings(CACHES=LOCMEM)
class DatabaseFailureTests(TestCase):
    def setUp(self):
        cache.clear()
        reset_overrides()
        self.user = User.objects.create_user('db_failure_user')
        UserProfile.objects.create(user=self.user, language='el')

    def tearDown(self):
        reset_overrides()
        cache.clear()

    def test_request_survives_when_overrides_cannot_load(self):
        TranslationOverride.objects.create(language='el', msgid='Check again', msgstr='ΔΙΟΡΘΩΣΗ')
        self.client.force_login(self.user)
        with patch('localization.overrides._wanted', side_effect=OperationalError('down')):
            with self.assertLogs('localization.overrides', level='ERROR'):
                response = self.client.get(reverse('studentScenarios'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['LANGUAGE_CODE'], 'el')
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), CATALOG_CHECK_AGAIN)

    def test_request_survives_when_the_profile_cannot_load(self):
        self.client.force_login(self.user)
        self.client.cookies[settings.LANGUAGE_COOKIE_NAME] = 'de'
        with patch.object(UserProfile.objects, 'filter', side_effect=OperationalError('down')):
            with self.assertLogs('localization.middleware', level='ERROR'):
                response = self.client.get(reverse('studentScenarios'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['LANGUAGE_CODE'], 'de')
        self.assertNotIn(settings.LANGUAGE_COOKIE_NAME, response.cookies)

    def test_recipient_language_is_english_when_the_profile_cannot_load(self):
        with patch.object(User, 'profile', PropertyMock(side_effect=OperationalError('down'))):
            with self.assertLogs('localization.languages', level='ERROR'):
                self.assertEqual(recipient_language(self.user), 'en')

    def test_recipient_language_keeps_the_code_when_overrides_cannot_load(self):
        TranslationOverride.objects.create(language='el', msgid='Check again', msgstr='ΔΙΟΡΘΩΣΗ')
        with patch('localization.overrides._wanted', side_effect=OperationalError('down')):
            with self.assertLogs('localization.overrides', level='ERROR'):
                self.assertEqual(recipient_language(self.user), 'el')
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), CATALOG_CHECK_AGAIN)

    def test_failed_override_query_leaves_the_transaction_usable(self):
        # TestCase runs inside a transaction, as an atomic call site would.
        with patch('localization.overrides._wanted', side_effect=failing_query):
            with self.assertLogs('localization.overrides', level='ERROR'):
                self.assertEqual(recipient_language(self.user), 'el')
        self.assertTrue(User.objects.filter(pk=self.user.pk).exists())

    def test_failed_profile_query_leaves_the_transaction_usable(self):
        with patch.object(User, 'profile', PropertyMock(side_effect=failing_query)):
            with self.assertLogs('localization.languages', level='ERROR'):
                self.assertEqual(recipient_language(self.user), 'en')
        self.assertTrue(User.objects.filter(pk=self.user.pk).exists())

    def test_a_later_call_retries_and_applies_the_overrides(self):
        TranslationOverride.objects.create(language='el', msgid='Check again', msgstr='ΔΙΟΡΘΩΣΗ')
        with patch('localization.overrides._wanted', side_effect=OperationalError('down')):
            with self.assertLogs('localization.overrides', level='ERROR'):
                recipient_language(self.user)
        recipient_language(self.user)
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), 'ΔΙΟΡΘΩΣΗ')
