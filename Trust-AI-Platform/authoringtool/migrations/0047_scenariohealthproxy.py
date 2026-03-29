from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0046_scenario_ai_metrics_min_implementations'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScenarioHealthProxy',
            fields=[],
            options={
                'verbose_name': 'Scenario Health Check',
                'verbose_name_plural': 'Scenario Health Check',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('authoringtool.scenario',),
        ),
    ]
