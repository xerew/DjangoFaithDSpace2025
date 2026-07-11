from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0054_simulation_language_choices'),
    ]

    operations = [
        migrations.AddField(
            model_name='userproposalreview',
            name='rejection_reasons',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
