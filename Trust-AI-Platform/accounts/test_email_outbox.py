"""Tests for the rate-limited email campaign outbox."""

from django.test import SimpleTestCase

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
