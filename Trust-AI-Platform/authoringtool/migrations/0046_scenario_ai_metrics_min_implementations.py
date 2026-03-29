from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0045_language_model_and_normalize'),
    ]

    operations = [
        migrations.AddField(
            model_name='scenario',
            name='ai_metrics_min_implementations',
            field=models.PositiveIntegerField(
                default=300,
                help_text='Minimum number of student implementations required before the Scenario Metrics & AI button is shown.'
            ),
        ),
    ]
