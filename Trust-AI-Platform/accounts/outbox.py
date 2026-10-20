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
from django.core.mail import EmailMultiAlternatives, get_connection
from django.db.models import F
from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.html import strip_tags

from .models import BulkEmailCampaign, BulkEmailDelivery, EmailOutboxState
from .outbox_rules import INTERRUPTED_REASON, is_rate_limit

PAUSE_FOR = timedelta(minutes=60)
STUCK_AFTER = timedelta(minutes=30)
MAX_TEMPORARY_ATTEMPTS = 3

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
    limit = getattr(settings, 'EMAIL_CAMPAIGN_HOURLY_LIMIT', 200)
    sent_last_hour = BulkEmailDelivery.objects.filter(
        status=BulkEmailDelivery.STATUS_SENT,
        sent_at__gte=now - timedelta(hours=1),
    ).count()
    return max(0, limit - sent_last_hour)


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
    return _TEMPORARY if 400 <= code < 500 else _FAILED


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
        elif outcome == _TEMPORARY and delivery.attempts < MAX_TEMPORARY_ATTEMPTS:
            update(status=BulkEmailDelivery.STATUS_WAITING, error=str(exc))
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
    """Send waiting campaign email within the hourly allowance."""
    now = now or timezone.now()
    result = {'sent': 0, 'failed': 0, 'deferred': 0, 'paused': False, 'reason': ''}

    release_stuck(now)
    state = EmailOutboxState.load()
    if state.is_paused(now):
        result.update(paused=True, reason=state.pause_reason)
        return result

    allowance = remaining_allowance(now)
    if not allowance:
        return result
    waiting_ids = list(
        BulkEmailDelivery.objects.filter(status=BulkEmailDelivery.STATUS_WAITING)
        .order_by('campaign__created_at', 'campaign_id', 'id')
        .values_list('id', flat=True)
    )
    if not waiting_ids:
        return result

    connection = get_connection()
    try:
        connection.open()
    except Exception as exc:
        reason = f'Could not connect to the mail server: {exc}'
        pause(reason, now)
        result.update(paused=True, reason=reason)
        return result

    touched = set()
    try:
        for delivery_id in waiting_ids:
            if result['sent'] >= allowance:
                break
            if not claim(delivery_id, now):
                continue
            delivery = BulkEmailDelivery.objects.select_related('campaign', 'user').get(pk=delivery_id)
            touched.add(delivery.campaign_id)

            outcome, exc = _send_one(delivery, connection, now)
            if outcome == _SENT:
                result['sent'] += 1
            elif outcome == _FAILED:
                result['failed'] += 1
            elif outcome == _TEMPORARY:
                result['deferred'] += 1
            else:
                if outcome == _INTERRUPTED:
                    result['failed'] += 1
                reason = f'The mail server refused or dropped the connection: {exc}'
                pause(reason, now)
                result.update(paused=True, reason=reason)
                break
    finally:
        try:
            connection.close()
        except Exception:
            pass
        _refresh_campaigns(touched)
    return result
