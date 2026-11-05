from types import SimpleNamespace

from django.contrib.auth.models import User
from django.core import mail
from django.test import TestCase, override_settings

from accounts.models import UserProfile
from accounts.outbox import drain_outbox
from accounts.test_email_outbox import make_campaign, queue
from authoringtool.machine_translation.collaborators import _send_review_email
from localization.languages import recipient_language

BACKEND = 'localization.tests.utils.LanguageRecordingBackend'


def make_user(name, language=None):
    user = User.objects.create_user(name, email=f'{name}@example.com', password='pw12345!')
    if language is not None:
        UserProfile.objects.create(user=user, language=language)
    return user


@override_settings(EMAIL_BACKEND=BACKEND)
class EmailLanguageTests(TestCase):
    def test_recipient_language_fallbacks(self):
        self.assertEqual(recipient_language(None), 'en')
        self.assertEqual(recipient_language(make_user('noprofile')), 'en')
        self.assertEqual(recipient_language(make_user('bad', 'xx')), 'en')
        self.assertEqual(recipient_language(make_user('greek', 'el')), 'el')

    def test_password_reset_email_in_users_language(self):
        make_user('greek', 'el')
        self.client.post('/accounts/password_reset/', {'email': 'greek@example.com'},
                         HTTP_ACCEPT_LANGUAGE='en')
        self.assertEqual([m.recorded_language for m in mail.outbox], ['el'])

    def test_campaign_frame_in_recipients_language(self):
        users = [make_user('a_de', 'de'), make_user('b_none')]
        queue(make_campaign(), users)
        drain_outbox()
        self.assertEqual([m.recorded_language for m in mail.outbox], ['de', 'en'])
        self.assertIn('<html lang="de">', mail.outbox[0].alternatives[0][0])
        self.assertIn('<html lang="en">', mail.outbox[1].alternatives[0][0])

    def test_reviewer_email_in_reviewers_language(self):
        reviewer = make_user('rev', 'fr')
        admin = make_user('adm')
        scenario = SimpleNamespace(id=1, name='Case')
        job = SimpleNamespace(target_language='French')
        _send_review_email(scenario, job, reviewer, admin)
        self.assertEqual([m.recorded_language for m in mail.outbox], ['fr'])
