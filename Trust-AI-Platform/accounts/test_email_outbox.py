"""Tests for the rate-limited email campaign outbox."""

from datetime import timedelta

from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from accounts.models import BulkEmailCampaign, BulkEmailDelivery, EmailOutboxState
from accounts.outbox_rules import (
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

    def test_reconstruct_in_progress_leaves_rest_waiting(self):
        rows = reconstruct_deliveries([1, 2, 3], 1, '', in_progress=True)

        self.assertEqual([rows[i][0] for i in (1, 2, 3)], ['sent', 'waiting', 'waiting'])


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
