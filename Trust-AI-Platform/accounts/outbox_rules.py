"""Rules shared by the email outbox and its migrations.

No model imports here: the backfill migration uses these functions too.
"""

import re

INTERRUPTED_REASON = 'Sending was interrupted; this person may or may not have received it.'
RATE_LIMIT_REASON = 'Not sent: hourly sending limit reached'
REBUILT_REASON = "Not sent (rebuilt from the campaign's totals)"

_NAMED_ERROR_RE = re.compile(r'^User (\d+): (.*)$', re.MULTILINE)
_NOT_RATE_LIMIT_WORDS = ('size', 'quota', 'mailbox', '5.3.4', '5.2.2', '4.2.2')


def is_rate_limit(text):
    """True for a provider refusal caused by a sending limit, whatever its code.

    Papaki answers "554 5.7.0 ... limit ... was exceeded", a code that
    normally means permanent failure, so only the text can be trusted.
    Message-size and mailbox-quota refusals also say "limit exceeded" but
    concern one message or recipient, so they are not rate limits.
    """
    text = (text or '').lower()
    if any(word in text for word in _NOT_RATE_LIMIT_WORDS):
        return False
    return 'limit' in text and 'exceed' in text


def reconstruct_deliveries(recipient_ids, sent_count, error_summary, in_progress):
    """Best guess at each recipient's outcome for a campaign sent before the outbox.

    The old task sent to recipients in id order and logged up to 20 errors as
    "User <id>: <error>". Named users failed; of the rest, the first
    `sent_count` were sent, and the others failed. For a campaign still in
    progress the old task saved its count only every 25 emails, so some of the
    rest may have received it: they fail as uncertain and are never queued
    automatically. Returns {user_id: (status, error)}.
    """
    named = {}
    for user_id, message in _NAMED_ERROR_RE.findall(error_summary or ''):
        line = f'User {user_id}: {message}'
        named[int(user_id)] = RATE_LIMIT_REASON if is_rate_limit(line) else line

    rows = {}
    sent = 0
    for user_id in recipient_ids:
        if user_id in named:
            rows[user_id] = ('failed', named[user_id])
        elif sent < sent_count:
            rows[user_id] = ('sent', '')
            sent += 1
        elif in_progress:
            rows[user_id] = ('failed', INTERRUPTED_REASON)
        else:
            rows[user_id] = ('failed', REBUILT_REASON)
    return rows
