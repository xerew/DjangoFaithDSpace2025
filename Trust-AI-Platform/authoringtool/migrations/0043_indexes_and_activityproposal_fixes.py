from django.db import migrations, models


class Migration(migrations.Migration):
    """
    INDEX-1  – composite index on UserScenarioScore(user, scenario)
    INDEX-2  – index on ActivityFlag(activity) for FK traversal
    INDEX-3  – db_index=True on ActivityProposal.scenario FK
    PERF-2   – remove null=True from ActivityProposal.flag M2M (no-op in DB,
                silently ignored by Django; recorded here for completeness)
    """

    dependencies = [
        ('authoringtool', '0042_useranswer_user_activity_idx'),
    ]

    operations = [
        # INDEX-1: composite index on UserScenarioScore(user, scenario)
        migrations.AddIndex(
            model_name='userscenarioscore',
            index=models.Index(fields=['user', 'scenario'], name='uss_user_scenario_idx'),
        ),

        # INDEX-2: index on ActivityFlag(activity)
        migrations.AddIndex(
            model_name='activityflag',
            index=models.Index(fields=['activity'], name='actflag_activity_idx'),
        ),

        # INDEX-3: db_index=True on ActivityProposal.scenario
        # Django auto-creates an index for FKs, but AlterField makes it explicit.
        migrations.AlterField(
            model_name='activityproposal',
            name='scenario',
            field=models.ForeignKey(
                db_index=True,
                on_delete=models.deletion.CASCADE,
                related_name='proposals',
                to='authoringtool.scenario',
            ),
        ),
    ]
