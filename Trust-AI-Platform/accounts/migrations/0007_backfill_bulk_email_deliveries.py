"""Rebuild outbox rows for campaigns sent before the outbox existed.

The old sender kept only totals, so each recipient's outcome is a best guess
from the campaign's counts and logged errors (see reconstruct_deliveries).
Rows are marked `reconstructed` so the admin can tell.
"""

from django.db import migrations

from accounts.outbox_rules import reconstruct_deliveries


def backfill(apps, schema_editor):
    BulkEmailCampaign = apps.get_model('accounts', 'BulkEmailCampaign')
    BulkEmailDelivery = apps.get_model('accounts', 'BulkEmailDelivery')

    for campaign in BulkEmailCampaign.objects.filter(deliveries__isnull=True).distinct():
        users = list(campaign.recipients.order_by('id'))
        outcomes = reconstruct_deliveries(
            [user.id for user in users],
            campaign.sent_count,
            campaign.error_summary,
            in_progress=campaign.status in ('queued', 'sending'),
        )
        rows = []
        for user in users:
            status, error = outcomes[user.id]
            full_name = f'{user.first_name} {user.last_name}'.strip()
            rows.append(BulkEmailDelivery(
                campaign=campaign,
                user=user,
                email=(user.email or '').strip(),
                recipient_name=full_name or user.username,
                status=status,
                error=error,
                sent_at=campaign.completed_at if status == 'sent' else None,
                reconstructed=True,
            ))
        BulkEmailDelivery.objects.bulk_create(rows)


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0006_bulkemaildelivery_emailoutboxstate'),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
