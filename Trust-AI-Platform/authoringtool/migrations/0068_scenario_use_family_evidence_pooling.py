from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0067_family_review_dashboard_llm'),
    ]

    operations = [
        migrations.AddField(
            model_name='scenario',
            name='use_family_evidence_pooling',
            field=models.BooleanField(
                'Use family and historical evidence logic',
                default=False,
                help_text=(
                    'When enabled, Metrics & AI uses compatible current '
                    'versions from this scenario family and separates legacy '
                    'evidence into Historical Analytics. When disabled, all '
                    'current and historical data belonging to this scenario '
                    'is used, as in the original analytics behaviour.'
                ),
            ),
        ),
    ]
