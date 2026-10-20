"""Tests for the rate-limited email campaign outbox."""

import smtplib
from contextlib import contextmanager
from datetime import timedelta
from unittest import mock

from django.conf import settings
from django.contrib.auth.models import User
from django.core import mail
from django.core.mail.backends.locmem import EmailBackend
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from accounts.models import BulkEmailCampaign, BulkEmailDelivery, EmailOutboxState
from accounts.outbox import claim, drain_outbox, pause, remaining_allowance
from accounts.outbox_rules import (
    INTERRUPTED_REASON,
    RATE_LIMIT_REASON,
    REBUILT_REASON,
    is_rate_limit,
    reconstruct_deliveries,
)

# Papaki's real reply when the hourly sending cap is reached.
PAPAKI_TEXT = (
    '5.7.0 Your message could not be sent. The limit on the number of allowed '
    'outgoing messages was exceeded. Try again later.'
)
PAPAKI = f"(554, b'{PAPAKI_TEXT}')"
PAPAKI_BYTES = PAPAKI_TEXT.encode()


class OutboxRulesTests(SimpleTestCase):
    def test_papaki_reply_is_rate_limit(self):
        self.assertTrue(is_rate_limit(PAPAKI))
        self.assertTrue(is_rate_limit('452 Rate LIMIT Exceeded'))
        self.assertFalse(is_rate_limit('550 5.1.1 <x@y.eu>: user unknown'))

    def test_reconstruct_matches_real_campaign(self):
        ids = list(range(1, 441))
        summary = '\n'.join(f'User {i}: {PAPAKI}' for i in range(301, 321))

        rows = reconstruct_deliveries(ids, 300, summary, in_progress=False)

        self.assertEqual([i for i in ids if rows[i][0] == 'sent'], list(range(1, 301)))
        self.assertEqual(rows[301], ('failed', RATE_LIMIT_REASON))
        self.assertEqual(rows[400], ('failed', REBUILT_REASON))

    def test_reconstruct_named_non_limit_error_keeps_line(self):
        line = 'User 2: inactive account or missing email'

        rows = reconstruct_deliveries([1, 2, 3, 4], 3, line, in_progress=False)

        self.assertEqual(rows[2], ('failed', line))
        self.assertEqual([rows[i][0] for i in (1, 3, 4)], ['sent', 'sent', 'sent'])

    def test_reconstruct_in_progress_fails_rest_as_uncertain(self):
        # The old task saved its count every 25 emails, so some of these may
        # have received it: never queue them automatically.
        rows = reconstruct_deliveries([1, 2, 3], 1, '', in_progress=True)

        self.assertEqual(
            [rows[i] for i in (1, 2, 3)],
            [('sent', ''), ('failed', INTERRUPTED_REASON), ('failed', INTERRUPTED_REASON)],
        )

    def test_size_and_quota_errors_are_not_rate_limits(self):
        self.assertFalse(is_rate_limit('552 5.3.4 Message size exceeds fixed limit'))
        self.assertFalse(is_rate_limit('552 5.2.2 Mailbox size limit exceeded'))
        self.assertFalse(is_rate_limit('452 4.2.2 Quota limit exceeded for user'))
        self.assertTrue(is_rate_limit(PAPAKI))


def make_campaign(**fields):
    defaults = {
        'target_type': BulkEmailCampaign.TARGET_ALL_TEACHERS,
        'subject': 'Platform news',
        'body_html': '<p>Hello</p>',
        'site_url': 'https://platform.test',
    }
    defaults.update(fields)
    return BulkEmailCampaign.objects.create(**defaults)


def make_users(count, prefix='teacher'):
    return [
        User.objects.create_user(f'{prefix}{i}', email=f'{prefix}{i}@example.com')
        for i in range(count)
    ]


def queue(campaign, users):
    """Create one waiting delivery per user, as sending a campaign does."""
    campaign.recipients.add(*users)
    campaign.recipient_count = campaign.recipients.count()
    campaign.save(update_fields=['recipient_count'])
    return [
        BulkEmailDelivery.objects.create(
            campaign=campaign,
            user=user,
            email=user.email,
            recipient_name=user.get_full_name() or user.username,
        )
        for user in users
    ]


class CampaignCountsTests(TestCase):
    def setUp(self):
        self.campaign = make_campaign()
        self.rows = queue(self.campaign, make_users(3))

    def set_statuses(self, statuses):
        for row, status in zip(self.rows, statuses):
            BulkEmailDelivery.objects.filter(pk=row.pk).update(status=status)
        self.campaign.refresh_counts()

    def test_status_follows_rows(self):
        self.set_statuses(['waiting', 'waiting', 'waiting'])
        self.assertEqual(self.campaign.status, BulkEmailCampaign.STATUS_QUEUED)

        self.set_statuses(['sent', 'waiting', 'sending'])
        self.assertEqual(
            (self.campaign.status, self.campaign.sent_count, self.campaign.waiting_count),
            (BulkEmailCampaign.STATUS_SENDING, 1, 2),
        )
        self.assertIsNotNone(self.campaign.started_at)
        self.assertIsNone(self.campaign.completed_at)

        self.set_statuses(['sent', 'sent', 'sent'])
        self.assertEqual(self.campaign.status, BulkEmailCampaign.STATUS_COMPLETED)
        self.assertIsNotNone(self.campaign.completed_at)

        self.set_statuses(['sent', 'failed', 'sent'])
        self.assertEqual(
            (self.campaign.status, self.campaign.failed_count),
            (BulkEmailCampaign.STATUS_PARTIAL, 1),
        )

        self.set_statuses(['failed', 'failed', 'failed'])
        self.assertEqual(self.campaign.status, BulkEmailCampaign.STATUS_FAILED)

        self.campaign.refresh_from_db()
        self.assertEqual(self.campaign.failed_count, 3)

    def test_one_row_per_user_per_campaign(self):
        user = self.rows[0].user
        with self.assertRaises(IntegrityError), transaction.atomic():
            BulkEmailDelivery.objects.create(campaign=self.campaign, user=user, email=user.email)

        # Rows whose user was deleted do not collide with each other.
        BulkEmailDelivery.objects.create(campaign=self.campaign, user=None, email='a@example.com')
        BulkEmailDelivery.objects.create(campaign=self.campaign, user=None, email='b@example.com')


class OutboxStateTests(TestCase):
    def test_outbox_state_singleton(self):
        now = timezone.now()
        self.assertEqual(EmailOutboxState.load().pk, EmailOutboxState.load().pk)
        self.assertEqual(EmailOutboxState.objects.count(), 1)

        state = EmailOutboxState.load()
        state.paused_until = now + timedelta(minutes=5)
        state.save()

        self.assertTrue(EmailOutboxState.load().is_paused(now))
        self.assertFalse(EmailOutboxState.load().is_paused(now + timedelta(minutes=6)))


@contextmanager
def failing_send(rules):
    """Make the locmem backend raise for chosen recipients.

    `rules` maps an email address to an exception, or to a list of exceptions
    raised on successive attempts (None in the list means "send normally").
    """
    real_send = EmailBackend.send_messages
    remaining = {
        email: list(value) if isinstance(value, list) else [value]
        for email, value in rules.items()
    }

    def send_messages(backend, messages):
        email = messages[0].to[0]
        if remaining.get(email):
            exc = remaining[email].pop(0)
            if exc is not None:
                raise exc
        return real_send(backend, messages)

    with mock.patch.object(EmailBackend, 'send_messages', send_messages):
        yield


def statuses(rows):
    return [BulkEmailDelivery.objects.get(pk=row.pk).status for row in rows]


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    DEFAULT_FROM_EMAIL='Trust AI Lab <noreply@trust-ai.test>',
    EMAIL_CAMPAIGN_HOURLY_LIMIT=200,
)
class DrainOutboxTests(TestCase):
    def setUp(self):
        self.now = timezone.now()
        self.campaign = make_campaign()

    def paused(self, at=None):
        return EmailOutboxState.load().is_paused(at or self.now)

    def test_sends_waiting_rows_and_updates_campaign(self):
        rows = queue(self.campaign, make_users(3))

        result = drain_outbox(self.now)

        self.assertEqual(result['sent'], 3)
        self.assertEqual(sorted(m.to[0] for m in mail.outbox), sorted(r.email for r in rows))
        self.assertEqual(statuses(rows), ['sent', 'sent', 'sent'])
        self.assertTrue(all(BulkEmailDelivery.objects.get(pk=r.pk).sent_at for r in rows))
        self.campaign.refresh_from_db()
        self.assertEqual(
            (self.campaign.status, self.campaign.sent_count),
            (BulkEmailCampaign.STATUS_COMPLETED, 3),
        )

    def test_message_matches_campaign_template(self):
        queue(self.campaign, make_users(1))

        drain_outbox(self.now)

        message = mail.outbox[0]
        self.assertEqual(message.subject, 'Platform news')
        self.assertEqual(message.from_email, 'Trust AI Lab <noreply@trust-ai.test>')
        self.assertIn('Hello teacher0', message.body)
        self.assertIn('Platform Announcement', message.alternatives[0][0])

    def test_respects_hourly_allowance(self):
        sent_rows = queue(make_campaign(), make_users(150, prefix='earlier'))
        BulkEmailDelivery.objects.filter(pk__in=[r.pk for r in sent_rows]).update(
            status='sent', sent_at=self.now - timedelta(minutes=30),
        )
        queue(self.campaign, make_users(100))

        result = drain_outbox(self.now)

        self.assertEqual(result['sent'], 50)
        self.assertEqual(self.campaign.deliveries.filter(status='waiting').count(), 50)

    def test_allowance_never_negative(self):
        rows = queue(make_campaign(), make_users(250, prefix='earlier'))
        BulkEmailDelivery.objects.filter(pk__in=[r.pk for r in rows]).update(
            status='sent', sent_at=self.now - timedelta(minutes=5),
        )
        queue(self.campaign, make_users(2))

        self.assertEqual(remaining_allowance(self.now), 0)
        self.assertEqual(drain_outbox(self.now)['sent'], 0)
        self.assertEqual(len(mail.outbox), 0)

    @override_settings(EMAIL_CAMPAIGN_HOURLY_LIMIT=2)
    def test_older_campaign_goes_first(self):
        older_rows = queue(self.campaign, make_users(2, prefix='older'))
        newer = make_campaign(subject='Later news')
        newer_rows = queue(newer, make_users(2, prefix='newer'))

        drain_outbox(self.now)

        self.assertEqual(statuses(older_rows), ['sent', 'sent'])
        self.assertEqual(statuses(newer_rows), ['waiting', 'waiting'])

    def test_papaki_limit_pauses_and_keeps_waiting(self):
        rows = queue(self.campaign, make_users(4))
        refusal = smtplib.SMTPDataError(554, PAPAKI_BYTES)

        with failing_send({rows[1].email: refusal}):
            result = drain_outbox(self.now)

        self.assertEqual(result['sent'], 1)
        self.assertTrue(result['paused'])
        self.assertEqual(statuses(rows), ['sent', 'waiting', 'waiting', 'waiting'])
        refused = BulkEmailDelivery.objects.get(pk=rows[1].pk)
        self.assertIn('limit', refused.error)
        self.assertEqual(refused.attempts, 0)
        self.assertTrue(self.paused())

        self.assertEqual(drain_outbox(self.now + timedelta(minutes=10))['sent'], 0)
        self.assertEqual(drain_outbox(self.now + timedelta(minutes=61))['sent'], 3)
        self.assertEqual(statuses(rows), ['sent'] * 4)

    def test_papaki_limit_at_recipient_stage_also_pauses(self):
        rows = queue(self.campaign, make_users(2))
        refusal = smtplib.SMTPRecipientsRefused({rows[0].email: (554, PAPAKI_BYTES)})

        with failing_send({rows[0].email: refusal}):
            drain_outbox(self.now)

        self.assertEqual(statuses(rows), ['waiting', 'waiting'])
        self.assertTrue(self.paused())

    def test_permanent_5xx_fails_only_that_row(self):
        rows = queue(self.campaign, make_users(3))
        refusal = smtplib.SMTPRecipientsRefused({rows[0].email: (550, b'5.1.1 user unknown')})

        with failing_send({rows[0].email: refusal}):
            result = drain_outbox(self.now)

        self.assertEqual(statuses(rows), ['failed', 'sent', 'sent'])
        self.assertIn('user unknown', BulkEmailDelivery.objects.get(pk=rows[0].pk).error)
        self.assertFalse(result['paused'])
        self.assertFalse(self.paused())

    def test_temporary_4xx_skips_then_fails_after_three(self):
        rows = queue(self.campaign, make_users(3))
        refusal = smtplib.SMTPRecipientsRefused({rows[0].email: (451, b'4.4.0 try again later')})

        with failing_send({rows[0].email: [refusal, refusal, refusal]}):
            drain_outbox(self.now)
            self.assertEqual(statuses(rows), ['waiting', 'sent', 'sent'])
            drain_outbox(self.now + timedelta(minutes=61))
            self.assertEqual(statuses(rows)[0], 'waiting')
            drain_outbox(self.now + timedelta(minutes=122))

        self.assertEqual(statuses(rows)[0], 'failed')
        self.assertIn('try again later', BulkEmailDelivery.objects.get(pk=rows[0].pk).error)
        self.assertFalse(self.paused(self.now + timedelta(minutes=122)))

    def test_disconnect_mid_send_is_uncertain_and_pauses(self):
        rows = queue(self.campaign, make_users(3))

        with failing_send({rows[1].email: smtplib.SMTPServerDisconnected('Connection lost')}):
            drain_outbox(self.now)

        self.assertEqual(statuses(rows), ['sent', 'failed', 'waiting'])
        self.assertEqual(BulkEmailDelivery.objects.get(pk=rows[1].pk).error, INTERRUPTED_REASON)
        self.assertTrue(self.paused())

    def test_cannot_connect_pauses_without_touching_rows(self):
        rows = queue(self.campaign, make_users(2))
        connection = mock.Mock()
        connection.open.side_effect = OSError('Connection refused')

        with mock.patch('accounts.outbox.get_connection', return_value=connection):
            result = drain_outbox(self.now)

        self.assertTrue(result['paused'])
        self.assertIn('Connection refused', EmailOutboxState.load().pause_reason)
        self.assertEqual(statuses(rows), ['waiting', 'waiting'])
        self.assertEqual([BulkEmailDelivery.objects.get(pk=r.pk).attempts for r in rows], [0, 0])

    def test_claim_is_exclusive(self):
        first, second = queue(self.campaign, make_users(2))

        self.assertTrue(claim(first.pk, self.now))
        self.assertFalse(claim(first.pk, self.now))

        drain_outbox(self.now)
        self.assertEqual([m.to[0] for m in mail.outbox], [second.email])

    def test_stuck_sending_becomes_uncertain_failure(self):
        row = queue(self.campaign, make_users(1))[0]
        BulkEmailDelivery.objects.filter(pk=row.pk).update(
            status='sending', claimed_at=self.now - timedelta(minutes=31),
        )

        drain_outbox(self.now)

        row.refresh_from_db()
        self.assertEqual((row.status, row.error), ('failed', INTERRUPTED_REASON))
        self.assertEqual(len(mail.outbox), 0)
        self.campaign.refresh_from_db()
        self.assertEqual(self.campaign.status, BulkEmailCampaign.STATUS_FAILED)

    def test_recent_sending_row_is_left_alone(self):
        row = queue(self.campaign, make_users(1))[0]
        BulkEmailDelivery.objects.filter(pk=row.pk).update(
            status='sending', claimed_at=self.now - timedelta(minutes=5),
        )

        drain_outbox(self.now)

        self.assertEqual(statuses([row]), ['sending'])
        self.assertEqual(len(mail.outbox), 0)

    def test_deactivated_and_deleted_users_fail_with_reason(self):
        inactive, deleted, active = make_users(3)
        rows = queue(self.campaign, [inactive, deleted, active])
        inactive.is_active = False
        inactive.save()
        deleted.delete()

        drain_outbox(self.now)

        self.assertEqual(statuses(rows), ['failed', 'failed', 'sent'])
        self.assertIn('deactivated', BulkEmailDelivery.objects.get(pk=rows[0].pk).error)
        self.assertIn('deleted', BulkEmailDelivery.objects.get(pk=rows[1].pk).error)
        self.assertEqual([m.to[0] for m in mail.outbox], [active.email])

    def test_sends_to_queued_address(self):
        user = make_users(1)[0]
        queue(self.campaign, [user])
        user.email = 'changed@example.com'
        user.save()

        drain_outbox(self.now)

        self.assertEqual(mail.outbox[0].to, ['teacher0@example.com'])

    def test_provider_wide_4xx_streak_pauses_and_keeps_rows_waiting(self):
        rows = queue(self.campaign, make_users(5))
        rejected = {
            row.email: smtplib.SMTPRecipientsRefused({row.email: (450, b'4.7.1 Client host rejected')})
            for row in rows
        }

        with failing_send(rejected):
            result = drain_outbox(self.now)

        self.assertTrue(result['paused'])
        self.assertTrue(self.paused())
        self.assertEqual(statuses(rows), ['waiting'] * 5)
        self.assertEqual([BulkEmailDelivery.objects.get(pk=r.pk).attempts for r in rows], [0] * 5)

    @override_settings(EMAIL_CAMPAIGN_HOURLY_LIMIT=3)
    def test_failed_attempts_count_against_hourly_allowance(self):
        rows = queue(self.campaign, make_users(5))
        unknown = {
            row.email: smtplib.SMTPRecipientsRefused({row.email: (550, b'5.1.1 user unknown')})
            for row in rows
        }

        with failing_send(unknown):
            drain_outbox(self.now)
            drain_outbox(self.now + timedelta(minutes=5))

        self.assertEqual(statuses(rows), ['failed', 'failed', 'failed', 'waiting', 'waiting'])

    def test_overlapping_run_does_nothing(self):
        rows = queue(self.campaign, make_users(2))

        with mock.patch('accounts.outbox.cache.add', return_value=False):
            result = drain_outbox(self.now)

        self.assertEqual(result['sent'], 0)
        self.assertEqual(statuses(rows), ['waiting', 'waiting'])
        self.assertEqual(len(mail.outbox), 0)

    def test_beat_entry_drops_backed_up_runs(self):
        entry = settings.CELERY_BEAT_SCHEDULE['drain-email-outbox']
        self.assertEqual(entry['options'], {'expires': 240})

    def test_beat_schedule_runs_drain_task(self):
        from celery import current_app
        import accounts.tasks  # noqa: F401  (registers the task)

        entry = settings.CELERY_BEAT_SCHEDULE['drain-email-outbox']
        self.assertEqual(entry['task'], 'accounts.tasks.drain_email_outbox')
        self.assertEqual(entry['schedule'], 300.0)
        self.assertIn('accounts.tasks.drain_email_outbox', current_app.tasks)


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    DEFAULT_FROM_EMAIL='Trust AI Lab <noreply@trust-ai.test>',
    SITE_URL='https://platform.test',
)
class QueueOnSendTests(TestCase):
    def setUp(self):
        from django.contrib.auth.models import Group

        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.staff = User.objects.create_user(
            'outbox_staff', password='pass', email='staff@example.com', is_staff=True,
        )
        self.teachers = make_users(2)
        for teacher in self.teachers:
            teacher.groups.add(teachers)
        self.client.login(username='outbox_staff', password='pass')

    def send(self):
        from django.urls import reverse

        return self.client.post(reverse('admin_send_bulk_email'), {
            'target_type': 'all_teachers',
            'subject': 'Platform news',
            'body_html': '<p>Hello</p>',
            'confirmed': 'true',
        })

    def test_send_creates_waiting_rows_and_sends_nothing(self):
        response = self.send()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], BulkEmailCampaign.STATUS_QUEUED)
        self.assertEqual(len(mail.outbox), 0)
        campaign = BulkEmailCampaign.objects.get()
        self.assertEqual(campaign.status, BulkEmailCampaign.STATUS_QUEUED)
        rows = list(campaign.deliveries.order_by('email'))
        self.assertEqual(
            [(r.email, r.status, r.recipient_name) for r in rows],
            [
                ('teacher0@example.com', 'waiting', 'teacher0'),
                ('teacher1@example.com', 'waiting', 'teacher1'),
            ],
        )

    def test_password_reset_sends_immediately_while_campaign_waits(self):
        from django.urls import reverse

        self.send()
        pause('Sending paused for the test', timezone.now())
        # Django skips password resets for accounts without a usable password.
        self.teachers[0].set_password('old-pass')
        self.teachers[0].save()

        response = self.client.post(reverse('password_reset'), {'email': 'teacher0@example.com'})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['teacher0@example.com'])
        self.assertEqual(
            BulkEmailDelivery.objects.filter(status='waiting').count(), 2,
        )


def backfill_migration():
    import importlib

    return importlib.import_module('accounts.migrations.0007_backfill_bulk_email_deliveries')


class BackfillMigrationTests(TestCase):
    def run_backfill(self):
        from django.apps import apps

        backfill_migration().backfill(apps, None)

    def test_rebuilds_partial_campaign(self):
        finished = timezone.now() - timedelta(days=1)
        users = make_users(440)
        summary = '\n'.join(f'User {u.id}: {PAPAKI}' for u in users[300:320])
        campaign = make_campaign(
            status=BulkEmailCampaign.STATUS_PARTIAL,
            recipient_count=440, sent_count=300, failed_count=140,
            completed_at=finished, error_summary=summary,
        )
        campaign.recipients.set(users)

        self.run_backfill()

        rows = list(campaign.deliveries.order_by('user_id'))
        self.assertEqual(len(rows), 440)
        self.assertTrue(all(r.reconstructed for r in rows))
        self.assertEqual({r.status for r in rows[:300]}, {'sent'})
        self.assertEqual({r.sent_at for r in rows[:300]}, {finished})
        self.assertEqual({r.status for r in rows[300:]}, {'failed'})
        self.assertEqual(rows[300].error, RATE_LIMIT_REASON)
        self.assertEqual(rows[439].error, REBUILT_REASON)
        self.assertEqual(rows[0].email, users[0].email)
        campaign.refresh_from_db()
        self.assertEqual(
            (campaign.status, campaign.sent_count, campaign.failed_count),
            (BulkEmailCampaign.STATUS_PARTIAL, 300, 140),
        )

    def test_rebuilds_in_progress_campaign_without_queueing_anyone(self):
        users = make_users(3)
        campaign = make_campaign(
            status=BulkEmailCampaign.STATUS_SENDING, recipient_count=3, sent_count=1,
        )
        campaign.recipients.set(users)

        self.run_backfill()

        self.assertEqual(
            list(campaign.deliveries.order_by('user_id').values_list('status', 'error')),
            [('sent', ''), ('failed', INTERRUPTED_REASON), ('failed', INTERRUPTED_REASON)],
        )

    def test_backfill_skips_campaigns_that_already_have_rows(self):
        users = make_users(2)
        campaign = make_campaign(
            status=BulkEmailCampaign.STATUS_COMPLETED, recipient_count=2, sent_count=2,
        )
        campaign.recipients.set(users)

        self.run_backfill()
        self.run_backfill()

        self.assertEqual(campaign.deliveries.count(), 2)


class CampaignDetailsTests(TestCase):
    def setUp(self):
        from django.urls import reverse

        self.staff = User.objects.create_user(
            'details_staff', password='pass', email='staff@example.com', is_staff=True,
        )
        self.regular = User.objects.create_user('details_regular', password='pass')
        self.client.login(username='details_staff', password='pass')
        self.campaign = make_campaign()
        self.rows = queue(self.campaign, make_users(3))
        self.details_url = reverse('admin_bulk_email_campaign', args=[self.campaign.id])
        self.resend_url = reverse('admin_bulk_email_resend', args=[self.campaign.id])

    def set_row(self, row, **fields):
        BulkEmailDelivery.objects.filter(pk=row.pk).update(**fields)

    def test_details_lists_recipients_with_status_and_reason(self):
        self.set_row(self.rows[0], status='sent', sent_at=timezone.now())
        self.set_row(self.rows[1], status='failed', error='550 user unknown')

        response = self.client.get(self.details_url)

        self.assertEqual(response.status_code, 200)
        for row in self.rows:
            self.assertContains(response, row.email)
        self.assertContains(response, '550 user unknown')
        self.assertEqual(response.context['counts'], {'all': 3, 'waiting': 1, 'sent': 1, 'failed': 1})

    def test_details_filter_failed(self):
        self.set_row(self.rows[1], status='failed', error='550 user unknown')

        response = self.client.get(self.details_url, {'status': 'failed'})

        self.assertContains(response, self.rows[1].email)
        self.assertNotContains(response, self.rows[0].email)

    def test_details_shows_reconstructed_note(self):
        self.set_row(self.rows[0], reconstructed=True)

        response = self.client.get(self.details_url)

        self.assertContains(response, 'reconstructed')

    def test_details_pages_and_keeps_filter(self):
        failed = queue(self.campaign, make_users(250, prefix='bulk'))
        BulkEmailDelivery.objects.filter(pk__in=[r.pk for r in failed]).update(status='failed')

        response = self.client.get(self.details_url, {'status': 'failed', 'page': 3})

        self.assertEqual(len(response.context['page_obj'].object_list), 50)
        self.assertContains(response, '?status=failed&amp;page=2')

    def test_details_shows_pause(self):
        pause('Connection refused', timezone.now())

        response = self.client.get(self.details_url)

        self.assertContains(response, 'Sending paused until')
        self.assertContains(response, 'Connection refused')

    def test_resend_failed_requeues_only_failed_rows_with_users(self):
        self.set_row(self.rows[0], status='failed', error='550 user unknown', attempts=1)
        self.set_row(self.rows[1], status='sent')
        orphan = BulkEmailDelivery.objects.create(
            campaign=self.campaign, user=None, email='gone@example.com', status='failed',
        )

        response = self.client.post(self.resend_url, {'mode': 'failed'})

        self.assertRedirects(response, f'{self.details_url}?resent=1')
        row = BulkEmailDelivery.objects.get(pk=self.rows[0].pk)
        self.assertEqual((row.status, row.error, row.attempts), ('waiting', '', 1))
        self.assertEqual(statuses([self.rows[1], orphan]), ['sent', 'failed'])
        self.campaign.refresh_from_db()
        self.assertEqual(self.campaign.failed_count, 1)

    def test_resend_selected_can_requeue_sent_rows(self):
        self.set_row(self.rows[0], status='sent')

        self.client.post(self.resend_url, {'mode': 'selected', 'delivery_ids': [self.rows[0].pk]})

        self.assertEqual(statuses([self.rows[0]]), ['waiting'])

    def test_resend_ignores_other_campaigns_rows(self):
        other = queue(make_campaign(subject='Other'), make_users(1, prefix='other'))[0]
        self.set_row(other, status='failed')

        response = self.client.post(
            self.resend_url, {'mode': 'selected', 'delivery_ids': [other.pk]},
        )

        self.assertRedirects(response, f'{self.details_url}?resent=0')
        self.assertEqual(statuses([other]), ['failed'])

    def test_resend_twice_is_harmless(self):
        self.set_row(self.rows[0], status='failed')

        self.client.post(self.resend_url, {'mode': 'failed'})
        response = self.client.post(self.resend_url, {'mode': 'failed'})

        self.assertRedirects(response, f'{self.details_url}?resent=0')
        self.assertEqual(statuses(self.rows), ['waiting', 'waiting', 'waiting'])

    def test_resent_banner(self):
        response = self.client.get(self.details_url, {'resent': 4})

        self.assertContains(response, 'Queued 4 emails again.')

    def test_details_and_resend_are_staff_only(self):
        self.client.logout()
        self.client.login(username='details_regular', password='pass')

        self.assertEqual(self.client.get(self.details_url).status_code, 403)
        self.assertEqual(self.client.post(self.resend_url, {'mode': 'failed'}).status_code, 403)

        self.client.login(username='details_staff', password='pass')
        self.assertEqual(self.client.get(self.resend_url).status_code, 405)

    def test_email_tab_shows_waiting_count_details_link_and_pause(self):
        from django.urls import reverse

        response = self.client.get(reverse('admin_dashboard'))
        self.assertContains(response, self.details_url)
        self.assertContains(response, '3 waiting')
        self.assertNotContains(response, 'Sending paused until')

        pause('Connection refused', timezone.now())
        response = self.client.get(reverse('admin_dashboard'))
        self.assertContains(response, 'Sending paused until')


class FakeSMTP:
    """Stands in for smtplib.SMTP under Django's real SMTP backend.

    Like smtplib, a 421 reply closes the connection before raising, so the
    next sendmail fails with SMTPServerDisconnected.
    """

    refuse = {}
    delivered = []

    def __init__(self, host, port, **kwargs):
        self.sock = object()

    def starttls(self, **kwargs):
        pass

    def login(self, user, password):
        pass

    def sendmail(self, from_addr, to_addrs, msg, *args, **kwargs):
        if self.sock is None:
            raise smtplib.SMTPServerDisconnected('please run connect() first')
        to = to_addrs[0]
        if to in self.refuse:
            code, text = self.refuse[to]
            if code == 421:
                self.sock = None
            raise smtplib.SMTPRecipientsRefused({to: (code, text)})
        FakeSMTP.delivered.append(to)
        return {}

    def quit(self):
        self.sock = None

    def close(self):
        self.sock = None


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.smtp.EmailBackend',
    EMAIL_HOST='smtp.test', EMAIL_PORT=587, EMAIL_USE_TLS=False,
    EMAIL_HOST_USER='', EMAIL_HOST_PASSWORD='',
    DEFAULT_FROM_EMAIL='Trust AI Lab <noreply@trust-ai.test>',
)
class RealSmtpBackendTests(TestCase):
    def setUp(self):
        FakeSMTP.refuse = {}
        FakeSMTP.delivered = []
        patcher = mock.patch('django.core.mail.backends.smtp.smtplib.SMTP', FakeSMTP)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.now = timezone.now()
        self.rows = queue(make_campaign(), make_users(3))

    def test_421_pauses_and_next_row_is_not_marked_uncertain(self):
        FakeSMTP.refuse = {self.rows[0].email: (421, b'4.7.0 Too many messages, try later')}

        result = drain_outbox(self.now)

        self.assertTrue(result['paused'])
        self.assertEqual(statuses(self.rows), ['waiting', 'waiting', 'waiting'])
        self.assertEqual([BulkEmailDelivery.objects.get(pk=r.pk).attempts for r in self.rows], [0, 0, 0])
        self.assertEqual(FakeSMTP.delivered, [])

    def test_sends_through_real_backend(self):
        drain_outbox(self.now)

        self.assertEqual(FakeSMTP.delivered, [r.email for r in self.rows])
        self.assertEqual(statuses(self.rows), ['sent', 'sent', 'sent'])
