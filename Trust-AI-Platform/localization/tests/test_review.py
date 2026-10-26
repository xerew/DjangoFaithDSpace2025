"""Reviewer corrections: assigned reviewers fix translations without a deploy."""
from unittest.mock import patch

import requests
from django.contrib.auth.models import Group, User
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import translation
from django.utils.translation import gettext, ngettext, trans_real

from accounts.models import UserProfile
from authoringtool.machine_translation.health import check_ollama
from authoringtool.models import Scenario
from localization.models import TranslationOverride, TranslationOverrideLog, TranslationReviewer
from localization.catalog import (
    EXTRA_DOUBLE_QUOTE, STRAY_PERCENT, UNBALANCED_BRACKETS, UNEXPECTED_LINE_BREAK,
)
from localization.languages import recipient_language
from localization.overrides import (
    apply_overrides, bump_overrides_version, ensure_overrides, reset_overrides,
)

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
GREEK_CHECK_AGAIN = 'Έλεγξε ξανά'
CATALOG_CHECK_AGAIN = 'Νέος έλεγχος'  # what the Greek catalog says without an override


@override_settings(CACHES=LOCMEM)
class ReviewTestBase(TestCase):
    def setUp(self):
        cache.clear()
        reset_overrides()
        self.teachers, _ = Group.objects.get_or_create(name='teachers')
        self.reviewer = self.make_teacher('el_reviewer')
        TranslationReviewer.objects.create(user=self.reviewer, language='el')
        self.teacher = self.make_teacher('plain_teacher')
        self.staff = User.objects.create_user('review_staff', is_staff=True)

    def tearDown(self):
        reset_overrides()
        cache.clear()

    def make_teacher(self, username, language='en'):
        user = User.objects.create_user(username)
        user.groups.add(self.teachers)
        UserProfile.objects.create(user=user, language=language)
        return user

    def save(self, code, **data):
        return self.client.post(reverse('save_translation', args=[code]), data)


class ReviewPermissionTests(ReviewTestBase):
    def test_only_assigned_reviewer_can_edit_their_language(self):
        el_page = reverse('review_translations', args=['el'])
        de_page = reverse('review_translations', args=['de'])

        self.client.force_login(self.reviewer)
        self.assertEqual(self.client.get(el_page).status_code, 200)
        self.assertEqual(self.client.get(de_page).status_code, 403)
        self.assertEqual(self.save('de', msgid='Check again', msgstr='Erneut prüfen').status_code, 403)
        self.assertEqual(self.client.get(reverse('save_translation', args=['el'])).status_code, 405)

        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(el_page).status_code, 403)
        self.assertEqual(self.save('el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN).status_code, 403)
        self.assertFalse(TranslationOverride.objects.exists())

        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(el_page).status_code, 200)
        self.assertEqual(self.client.get(de_page).status_code, 200)

    def test_english_cannot_be_reviewed(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(reverse('review_translations', args=['en'])).status_code, 404)

    def test_translations_link_only_for_reviewers_and_staff(self):
        link = f'href="{reverse("review_index")}"'
        for user, visible in ((self.reviewer, True), (self.staff, True), (self.teacher, False)):
            with self.subTest(user=user.username):
                self.client.force_login(user)
                response = self.client.get(reverse('studentScenarios'))
                if visible:
                    self.assertContains(response, link)
                else:
                    self.assertNotContains(response, link)

    def test_index_sends_a_single_language_reviewer_to_their_language(self):
        self.client.force_login(self.reviewer)
        response = self.client.get(reverse('review_index'))
        self.assertRedirects(response, reverse('review_translations', args=['el']),
                             fetch_redirect_response=False)
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(reverse('review_index')).status_code, 403)


class ReviewPageTests(ReviewTestBase):
    def test_page_lists_source_current_translation_and_edit_box(self):
        TranslationOverride.objects.create(language='el', msgid='Check again',
                                           msgstr=GREEK_CHECK_AGAIN, updated_by=self.reviewer)
        self.client.force_login(self.reviewer)
        response = self.client.get(reverse('review_translations', args=['el']), {'q': 'check again'})
        self.assertContains(response, 'Check again')
        self.assertContains(response, GREEK_CHECK_AGAIN)
        self.assertContains(response, '<textarea')
        self.assertEqual(response.context['page_obj'].paginator.per_page, 100)

    def test_needs_review_filter_and_paging(self):
        self.client.force_login(self.reviewer)
        url = reverse('review_translations', args=['el'])
        everything = self.client.get(url).context['page_obj'].paginator.count
        needs_review = self.client.get(url, {'filter': 'needs_review'}).context['page_obj'].paginator.count
        self.assertGreater(everything, 100)
        self.assertLessEqual(needs_review, everything)
        self.assertEqual(len(self.client.get(url).context['page_obj'].object_list), 100)


class OverrideTests(ReviewTestBase):
    def metrics_page(self):
        scenario = Scenario.objects.create(name='Override scenario', visibility_status='private',
                                           created_by=self.reviewer, updated_by=self.reviewer)
        with patch('authoringtool.machine_translation.health.requests.get',
                   side_effect=requests.ConnectionError('Connection refused')):
            check_ollama()  # the banner with "Check again" shows while the AI server is offline
        return reverse('ai_metrics', args=[scenario.id])

    def test_override_applies_immediately(self):
        url = self.metrics_page()
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN)
        self.assertEqual(response.status_code, 200, response.content)

        UserProfile.objects.filter(user=self.reviewer).update(language='el')
        self.assertContains(self.client.get(url), GREEK_CHECK_AGAIN)

        UserProfile.objects.filter(user=self.reviewer).update(language='en')
        english = self.client.get(url)
        self.assertContains(english, 'Check again')
        self.assertNotContains(english, GREEK_CHECK_AGAIN)

    def test_blank_override_reverts_to_the_catalog(self):
        self.client.force_login(self.reviewer)
        self.save('el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN)
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), GREEK_CHECK_AGAIN)
        self.assertEqual(self.save('el', msgid='Check again', msgstr='  ').status_code, 200)
        self.assertFalse(TranslationOverride.objects.exists())
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), CATALOG_CHECK_AGAIN)

    def test_removed_override_restores_the_original_value(self):
        with translation.override('el'):
            original = gettext('Save')
        TranslationOverride.objects.create(language='el', msgid='Save', msgstr='Φύλαξη')
        apply_overrides('el')
        with translation.override('el'):
            self.assertEqual(gettext('Save'), 'Φύλαξη')
        with translation.override('en'):
            self.assertEqual(gettext('Save'), 'Save')
        TranslationOverride.objects.all().delete()
        apply_overrides('el')
        with translation.override('el'):
            self.assertEqual(gettext('Save'), original)

    def test_plural_forms_are_overridden_separately(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='%(n)d item', msgstr_0='%(n)d στοιχείο',
                             msgstr_1='%(n)d στοιχεία')
        self.assertEqual(response.status_code, 200, response.content)
        with translation.override('el'):
            self.assertEqual(ngettext('%(n)d item', '%(n)d items', 1), '%(n)d στοιχείο')
            self.assertEqual(ngettext('%(n)d item', '%(n)d items', 3), '%(n)d στοιχεία')

    def test_other_processes_reload_when_the_version_changes(self):
        ensure_overrides('el')
        # Written by another process: this one keeps the version it applied ...
        TranslationOverride.objects.create(language='el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN)
        ensure_overrides('el')
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), CATALOG_CHECK_AGAIN)
        # ... until the shared version moves on.
        bump_overrides_version()
        ensure_overrides('el')
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), GREEK_CHECK_AGAIN)

    def test_rebuilt_translation_objects_get_the_overrides_again(self):
        TranslationOverride.objects.create(language='el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN)
        ensure_overrides('el')
        trans_real._translations = {}
        ensure_overrides('el')
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), GREEK_CHECK_AGAIN)

    def test_emails_pick_up_overrides_through_recipient_language(self):
        TranslationOverride.objects.create(language='el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN)
        bump_overrides_version()
        recipient = self.make_teacher('greek_recipient', language='el')
        self.assertEqual(recipient_language(recipient), 'el')
        with translation.override('el'):
            self.assertEqual(gettext('Check again'), GREEK_CHECK_AGAIN)

    def test_override_cannot_drop_placeholder(self):
        self.client.force_login(self.reviewer)
        msgid = 'A language called "%(name)s" already exists.'
        response = self.save('el', msgid=msgid, msgstr='Υπάρχει ήδη γλώσσα με αυτό το όνομα.')
        self.assertEqual(response.status_code, 400)
        self.assertIn('missing %(name)s', response.json()['problems'])
        response = self.save('el', msgid='%(n)d item', msgstr_0='ένα στοιχείο',
                             msgstr_1='%(n)d στοιχεία')
        self.assertEqual(response.status_code, 400)
        self.assertFalse(TranslationOverride.objects.exists())

    def test_override_cannot_add_html(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='Check again',
                             msgstr='Έλεγξε <script>alert(1)</script>ξανά')
        self.assertEqual(response.status_code, 400)
        self.assertTrue(any('script' in p for p in response.json()['problems']))
        self.assertFalse(TranslationOverride.objects.exists())

    def assertRefused(self, response, problem):
        self.assertEqual(response.status_code, 400, response.content)
        result = response.json()
        self.assertIn(problem, result['problems'])
        self.assertEqual(len(result['messages']), len(result['problems']))
        self.assertTrue(all(result['messages']))
        self.assertFalse(TranslationOverride.objects.exists())

    def test_override_cannot_add_an_unclosed_tag(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='Check again', msgstr='X <img src=x onerror=alert(1) ')
        self.assertRefused(response, UNBALANCED_BRACKETS)

    def test_override_cannot_add_attributes_to_a_source_tag(self):
        self.client.force_login(self.reviewer)
        msgid = 'The <strong>Trust AI Lab platform</strong> consists of the following components:'
        response = self.save('el', msgid=msgid, msgstr=(
            'Η <strong onmouseover=alert(1)>πλατφόρμα Trust AI Lab</strong> αποτελείται από:'))
        self.assertRefused(response, 'unexpected <strong onmouseover=alert(1)>')

    def test_override_cannot_break_out_of_an_attribute(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='Search source or translation…',
                             msgstr='S" autofocus onfocus="alert(1)')
        self.assertRefused(response, EXTRA_DOUBLE_QUOTE)

    def test_override_cannot_add_a_line_break(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='Check again', msgstr='Έλεγξε\nξανά')
        self.assertRefused(response, UNEXPECTED_LINE_BREAK)

    def test_stray_percent_is_refused_only_in_formatted_texts(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='A language called "%(name)s" already exists.',
                             msgstr='Η γλώσσα «%(name)s» υπάρχει ήδη 100%.')
        self.assertRefused(response, STRAY_PERCENT)
        response = self.save('el', msgid='Check again', msgstr='Έλεγξε ξανά 100%')
        self.assertEqual(response.status_code, 200, response.content)

    def test_typographic_quotes_and_matching_tags_are_accepted(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='A language called "%(name)s" already exists.',
                             msgstr='Υπάρχει ήδη μια γλώσσα με το όνομα «%(name)s».')
        self.assertEqual(response.status_code, 200, response.content)
        msgid = 'The <strong>Trust AI Lab platform</strong> consists of the following components:'
        response = self.save('el', msgid=msgid, msgstr=(
            'Η <strong>πλατφόρμα Trust AI Lab</strong> αποτελείται από τα ακόλουθα μέρη:'))
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(TranslationOverride.objects.count(), 2)

    def test_unknown_msgid_is_refused(self):
        self.client.force_login(self.reviewer)
        response = self.save('el', msgid='Not a platform text', msgstr='Τίποτα')
        self.assertEqual(response.status_code, 400)
        self.assertFalse(TranslationOverride.objects.exists())

    def test_override_is_logged(self):
        self.client.force_login(self.reviewer)
        self.save('el', msgid='Check again', msgstr='Ξαναέλεγξε')
        self.save('el', msgid='Check again', msgstr=GREEK_CHECK_AGAIN)
        logs = list(TranslationOverrideLog.objects.order_by('at', 'id'))
        self.assertEqual(len(logs), 2)
        self.assertEqual((logs[0].old, logs[0].new), (CATALOG_CHECK_AGAIN, 'Ξαναέλεγξε'))
        self.assertEqual((logs[1].old, logs[1].new), ('Ξαναέλεγξε', GREEK_CHECK_AGAIN))
        override = TranslationOverride.objects.get()
        for log in logs:
            self.assertEqual(log.user, self.reviewer)
            self.assertEqual(log.language, 'el')
            self.assertEqual(log.msgid, 'Check again')
            self.assertEqual(log.override, override)
        self.assertEqual(override.updated_by, self.reviewer)


class AdminReviewerTests(ReviewTestBase):
    def post(self, url_name, *args, **data):
        return self.client.post(reverse(url_name, args=args), data)

    def test_admin_assigns_reviewer_on_languages_tab(self):
        self.client.force_login(self.staff)
        dashboard = self.client.get(reverse('admin_dashboard'))
        self.assertContains(dashboard, 'Translation reviewers')
        self.assertContains(dashboard, reverse('admin_add_translation_reviewer'))

        result = self.post('admin_add_translation_reviewer', user_id=self.teacher.id, language='de').json()
        self.assertTrue(result['success'], result)
        assignment = TranslationReviewer.objects.get(user=self.teacher, language='de')
        self.assertContains(self.client.get(reverse('admin_dashboard')),
                            f'id="reviewer-{assignment.id}"')

        outsider = User.objects.create_user('no_role_user')
        self.assertFalse(self.post('admin_add_translation_reviewer',
                                   user_id=outsider.id, language='de').json()['success'])
        self.assertFalse(self.post('admin_add_translation_reviewer',
                                   user_id=self.teacher.id, language='fa').json()['success'])
        self.assertFalse(self.post('admin_add_translation_reviewer',
                                   user_id='abc', language='de').json()['success'])
        self.assertFalse(self.post('admin_add_translation_reviewer',
                                   user_id=self.teacher.id, language='en').json()['success'])
        duplicate = self.post('admin_add_translation_reviewer', user_id=self.teacher.id, language='de').json()
        self.assertFalse(duplicate['success'])

        result = self.post('admin_remove_translation_reviewer', assignment.id).json()
        self.assertTrue(result['success'])
        self.assertFalse(TranslationReviewer.objects.filter(id=assignment.id).exists())

    def test_only_staff_manage_reviewers(self):
        self.client.force_login(self.reviewer)
        response = self.post('admin_add_translation_reviewer', user_id=self.teacher.id, language='de')
        self.assertEqual(response.status_code, 403)
        assignment = TranslationReviewer.objects.get(user=self.reviewer)
        response = self.post('admin_remove_translation_reviewer', assignment.id)
        self.assertEqual(response.status_code, 403)
        self.assertTrue(TranslationReviewer.objects.filter(id=assignment.id).exists())
