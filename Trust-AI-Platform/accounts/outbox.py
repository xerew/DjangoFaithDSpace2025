"""Rate-limited sending of campaign email from the delivery outbox.

The mail provider (Papaki) caps outgoing mail per hour. Campaign email is sent
from per-recipient rows at most EMAIL_CAMPAIGN_HOURLY_LIMIT per rolling hour,
leaving room for password resets, which are sent immediately elsewhere.
A provider refusal that affects every message pauses sending for an hour
instead of failing recipients; nothing is ever resent automatically.
"""

import smtplib
import socket
from datetime import timedelta
from html import unescape

from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives, get_connection
from django.db.models import F, Q
from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.html import strip_tags

from .models import BulkEmailCampaign, BulkEmailDelivery, EmailOutboxState
from .outbox_rules import INTERRUPTED_REASON, is_rate_limit

PAUSE_FOR = timedelta(minutes=60)
STUCK_AFTER = timedelta(minutes=30)
MAX_TEMPORARY_ATTEMPTS = 3
# This many temporary refusals in a row means the server, not the recipients.
PROVIDER_WIDE_STREAK = 3
LOCK_KEY = 'accounts:drain-email-outbox'
LOCK_TIMEOUT = 25 * 60

# Outcomes of one send attempt.
_SENT = 'sent'
_FAILED = 'failed'
_TEMPORARY = 'temporary'
_PAUSE = 'pause'
_INTERRUPTED = 'interrupted'


def plain_text_message(campaign, user):
    greeting_name = user.first_name or user.get_username()
    content = unescape(strip_tags(campaign.body_html)).strip()
    return (
        f'Hello {greeting_name},\n\n'
        f'{campaign.subject}\n\n'
        f'{content}\n\n'
        f'Trust AI Lab\n{campaign.site_url}'
    )


def build_message(delivery, connection):
    campaign = delivery.campaign
    message = EmailMultiAlternatives(
        subject=campaign.subject,
        body=plain_text_message(campaign, delivery.user),
        from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', None),
        to=[delivery.email],
        connection=connection,
    )
    message.attach_alternative(render_to_string('email/admin_bulk_email.html', {
        'campaign': campaign,
        'recipient': delivery.user,
        'site_url': campaign.site_url,
    }), 'text/html')
    return message


def claim(delivery_id, now):
    """Move one waiting row to sending; False if another run already has it."""
    return BulkEmailDelivery.objects.filter(
        pk=delivery_id,
        status=BulkEmailDelivery.STATUS_WAITING,
    ).update(
        status=BulkEmailDelivery.STATUS_SENDING,
        claimed_at=now,
        attempts=F('attempts') + 1,
    ) == 1


def remaining_allowance(now):
    """Attempts left this rolling hour. Refused attempts count too, so a
    server that refuses everything is not retried more than the limit."""
    limit = getattr(settings, 'EMAIL_CAMPAIGN_HOURLY_LIMIT', 200)
    since = now - timedelta(hours=1)
    attempted_last_hour = BulkEmailDelivery.objects.filter(
        Q(sent_at__gte=since) | Q(claimed_at__gte=since),
    ).count()
    return max(0, limit - attempted_last_hour)


def release_stuck(now):
    """Fail rows left in sending by a crashed run; they may have been delivered."""
    stuck = BulkEmailDelivery.objects.filter(
        status=BulkEmailDelivery.STATUS_SENDING,
        claimed_at__lt=now - STUCK_AFTER,
    )
    campaign_ids = set(stuck.values_list('campaign_id', flat=True))
    count = stuck.update(status=BulkEmailDelivery.STATUS_FAILED, error=INTERRUPTED_REASON)
    _refresh_campaigns(campaign_ids)
    return count


def pause(reason, now):
    state = EmailOutboxState.load()
    state.paused_until = now + PAUSE_FOR
    state.pause_reason = reason
    state.save()


def _classify(exc):
    """Map a send exception to an outcome, following the spec's failure rules."""
    # Papaki reports its hourly cap with a 554, so the text is checked first.
    if is_rate_limit(str(exc)):
        return _PAUSE
    if isinstance(exc, (smtplib.SMTPSenderRefused, smtplib.SMTPAuthenticationError)):
        return _PAUSE
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        codes = [code for code, _ in exc.recipients.values()]
        code = codes[0] if codes else 550
    elif isinstance(exc, smtplib.SMTPResponseException):
        code = exc.smtp_code
    elif isinstance(exc, (smtplib.SMTPServerDisconnected, socket.timeout, OSError)):
        return _INTERRUPTED
    else:
        return _FAILED
    if code == 421:
        # "Service not available": smtplib has already closed the connection.
        return _PAUSE
    return _TEMPORARY if 400 <= code < 500 else _FAILED


def _connection_lost(connection):
    """True when the SMTP socket is already closed, so nothing can be sent."""
    smtp = getattr(connection, 'connection', None)
    return smtp is not None and getattr(smtp, 'sock', True) is None


def _send_one(delivery, connection, now):
    """Send one claimed row and record its outcome. Returns (outcome, exception)."""
    update = BulkEmailDelivery.objects.filter(pk=delivery.pk).update
    user = delivery.user
    if user is None:
        update(status=BulkEmailDelivery.STATUS_FAILED, error='Recipient account was deleted.')
        return _FAILED, None
    if not user.is_active:
        update(status=BulkEmailDelivery.STATUS_FAILED, error='Recipient account is deactivated.')
        return _FAILED, None

    try:
        build_message(delivery, connection).send()
    except Exception as exc:  # classified below; other recipients continue
        outcome = _classify(exc)
        if outcome == _PAUSE:
            # Not this recipient's fault: back in the queue, attempt not counted.
            update(
                status=BulkEmailDelivery.STATUS_WAITING,
                error=str(exc),
                attempts=F('attempts') - 1,
            )
        elif outcome == _INTERRUPTED:
            update(status=BulkEmailDelivery.STATUS_FAILED, error=INTERRUPTED_REASON)
        elif outcome == _TEMPORARY:
            # A provider-wide streak may still put it back (see drain_outbox).
            give_up = delivery.attempts >= MAX_TEMPORARY_ATTEMPTS
            update(
                status=BulkEmailDelivery.STATUS_FAILED if give_up else BulkEmailDelivery.STATUS_WAITING,
                error=str(exc),
            )
        else:
            update(status=BulkEmailDelivery.STATUS_FAILED, error=str(exc))
            outcome = _FAILED
        return outcome, exc

    update(status=BulkEmailDelivery.STATUS_SENT, sent_at=now, error='')
    return _SENT, None


def _refresh_campaigns(campaign_ids):
    for campaign in BulkEmailCampaign.objects.filter(pk__in=campaign_ids):
        campaign.refresh_counts()


def drain_outbox(now=None):
    """Send waiting campaign email within the hourly allowance.

    Only one run sends at a time; an overlapping run returns at once.
    """
    now = now or timezone.now()
    result = {'sent': 0, 'failed': 0, 'deferred': 0, 'paused': False, 'reason': ''}
    if not cache.add(LOCK_KEY, '1', LOCK_TIMEOUT):
        result['reason'] = 'Another run is sending.'
        return result
    try:
        _drain(now, result)
    finally:
        cache.delete(LOCK_KEY)
    return result


def _pause_run(result, reason, now):
    pause(reason, now)
    result.update(paused=True, reason=reason)


def _drain(now, result):
    release_stuck(now)
    state = EmailOutboxState.load()
    if state.is_paused(now):
        result.update(paused=True, reason=state.pause_reason)
        return

    allowance = remaining_allowance(now)
    if not allowance:
        return
    waiting_ids = list(
        BulkEmailDelivery.objects.filter(status=BulkEmailDelivery.STATUS_WAITING)
        .order_by('campaign__created_at', 'campaign_id', 'id')
        .values_list('id', flat=True)
    )
    if not waiting_ids:
        return

    connection = get_connection()
    try:
        connection.open()
    except Exception as exc:
        _pause_run(result, f'Could not connect to the mail server: {exc}', now)
        return

    touched = set()
    attempted = 0
    streak = []  # consecutive temporary refusals in this run
    try:
        for delivery_id in waiting_ids:
            if attempted >= allowance:
                break
            if _connection_lost(connection):
                _pause_run(result, 'The mail server closed the connection.', now)
                break
            if not claim(delivery_id, now):
                continue
            attempted += 1
            delivery = BulkEmailDelivery.objects.select_related('campaign', 'user').get(pk=delivery_id)
            touched.add(delivery.campaign_id)

            outcome, exc = _send_one(delivery, connection, now)
            if outcome == _TEMPORARY:
                streak.append(delivery_id)
                result['deferred'] += 1
                if len(streak) >= PROVIDER_WIDE_STREAK:
                    # The server is refusing everyone: put the streak back
                    # untouched, without using up anyone's attempts.
                    BulkEmailDelivery.objects.filter(pk__in=streak).update(
                        status=BulkEmailDelivery.STATUS_WAITING,
                        attempts=F('attempts') - 1,
                    )
                    _pause_run(result, f'The mail server refused {len(streak)} messages in a row: {exc}', now)
                    break
                continue
            streak = []
            if outcome == _SENT:
                result['sent'] += 1
            elif outcome == _FAILED:
                result['failed'] += 1
            else:
                if outcome == _INTERRUPTED:
                    result['failed'] += 1
                _pause_run(result, f'The mail server refused or dropped the connection: {exc}', now)
                break
    finally:
        try:
            connection.close()
        except Exception:
            pass
        _refresh_campaigns(touched)
