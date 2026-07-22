"""Asynchronous tasks owned by the accounts application."""

from html import unescape

from celery import shared_task
from django.conf import settings
from django.core.mail import EmailMultiAlternatives, get_connection
from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.html import strip_tags

from .models import BulkEmailCampaign


def _plain_text_message(campaign, user):
    greeting_name = user.first_name or user.get_username()
    content = unescape(strip_tags(campaign.body_html)).strip()
    return (
        f'Hello {greeting_name},\n\n'
        f'{campaign.subject}\n\n'
        f'{content}\n\n'
        f'Trust AI Lab\n{campaign.site_url}'
    )


@shared_task
def send_bulk_email_campaign(campaign_id):
    """Send a campaign once, without automatic retries that could duplicate mail."""
    campaign = BulkEmailCampaign.objects.get(pk=campaign_id)
    campaign.status = BulkEmailCampaign.STATUS_SENDING
    campaign.started_at = timezone.now()
    campaign.error_summary = ''
    campaign.save(update_fields=['status', 'started_at', 'error_summary'])

    recipients = list(campaign.recipients.all().order_by('id'))
    sent_count = 0
    failed_count = 0
    errors = []
    from_email = getattr(
        settings,
        'DEFAULT_FROM_EMAIL',
        getattr(settings, 'EMAIL_HOST_USER', None),
    )
    connection = get_connection()

    try:
        connection.open()
        for index, user in enumerate(recipients, start=1):
            email = (user.email or '').strip()
            if not user.is_active or not email:
                failed_count += 1
                errors.append(f'User {user.id}: inactive account or missing email')
                continue

            html_message = render_to_string('email/admin_bulk_email.html', {
                'campaign': campaign,
                'recipient': user,
                'site_url': campaign.site_url,
            })
            message = EmailMultiAlternatives(
                subject=campaign.subject,
                body=_plain_text_message(campaign, user),
                from_email=from_email,
                to=[email],
                connection=connection,
            )
            message.attach_alternative(html_message, 'text/html')
            try:
                if message.send() == 1:
                    sent_count += 1
                else:
                    failed_count += 1
                    errors.append(f'User {user.id}: mail backend returned no delivery')
            except Exception as exc:  # keep processing other recipients
                failed_count += 1
                errors.append(f'User {user.id}: {exc}')

            if index % 25 == 0:
                BulkEmailCampaign.objects.filter(pk=campaign.pk).update(
                    sent_count=sent_count,
                    failed_count=failed_count,
                )
    except Exception as exc:
        unsent = max(0, len(recipients) - sent_count - failed_count)
        failed_count += unsent
        errors.append(f'Mail connection: {exc}')
    finally:
        connection.close()

    if sent_count == len(recipients) and not failed_count:
        status = BulkEmailCampaign.STATUS_COMPLETED
    elif sent_count:
        status = BulkEmailCampaign.STATUS_PARTIAL
    else:
        status = BulkEmailCampaign.STATUS_FAILED

    campaign.status = status
    campaign.sent_count = sent_count
    campaign.failed_count = failed_count
    campaign.error_summary = '\n'.join(errors[:20])
    campaign.completed_at = timezone.now()
    campaign.save(update_fields=[
        'status',
        'sent_count',
        'failed_count',
        'error_summary',
        'completed_at',
    ])
    return {
        'campaign_id': campaign.id,
        'sent': sent_count,
        'failed': failed_count,
        'status': status,
    }
