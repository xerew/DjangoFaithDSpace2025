import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def backfill_activity_lineage_and_mark_legacy_evidence(apps, schema_editor):
    Activity = apps.get_model('authoringtool', 'Activity')
    Scenario = apps.get_model('authoringtool', 'Scenario')
    UserAnswer = apps.get_model('authoringtool', 'UserAnswer')
    UserScenarioScore = apps.get_model(
        'authoringtool',
        'UserScenarioScore',
    )

    scenarios = {
        scenario.id: scenario
        for scenario in Scenario.objects.all().order_by('id')
    }
    activities_by_scenario = {
        scenario_id: list(
            Activity.objects
            .filter(scenario_id=scenario_id)
            .order_by(
                'phase__created_on',
                'phase_id',
                'created_on',
                'id',
            )
        )
        for scenario_id in scenarios
    }
    processed = set()

    def assign_independent_lineage(scenario_id):
        for activity in activities_by_scenario[scenario_id]:
            Activity.objects.filter(pk=activity.pk).update(
                lineage_key=uuid.uuid4()
            )
            activity.lineage_key = (
                Activity.objects
                .values_list('lineage_key', flat=True)
                .get(pk=activity.pk)
            )
        processed.add(scenario_id)

    for scenario_id, scenario in scenarios.items():
        if (
            not scenario.origin_scenario_id
            or scenario.origin_scenario_id not in scenarios
        ):
            assign_independent_lineage(scenario_id)

    remaining = set(scenarios) - processed
    while remaining:
        progressed = False
        for scenario_id in sorted(remaining):
            scenario = scenarios[scenario_id]
            parent_id = scenario.origin_scenario_id
            if parent_id not in processed:
                continue

            child_activities = activities_by_scenario[scenario_id]
            parent_activities = activities_by_scenario[parent_id]
            child_signature = [
                (
                    activity.activity_type_id,
                    activity.is_evaluatable,
                    activity.is_primary_ev,
                )
                for activity in child_activities
            ]
            parent_signature = [
                (
                    activity.activity_type_id,
                    activity.is_evaluatable,
                    activity.is_primary_ev,
                )
                for activity in parent_activities
            ]
            structures_match = child_signature == parent_signature

            for index, activity in enumerate(child_activities):
                lineage_key = (
                    parent_activities[index].lineage_key
                    if structures_match
                    else uuid.uuid4()
                )
                Activity.objects.filter(pk=activity.pk).update(
                    lineage_key=lineage_key
                )
                activity.lineage_key = lineage_key

            processed.add(scenario_id)
            remaining.remove(scenario_id)
            progressed = True
            break

        if not progressed:
            # Malformed origin cycles must never imply evidence equivalence.
            fallback_id = min(remaining)
            assign_independent_lineage(fallback_id)
            remaining.remove(fallback_id)

    # Historical rows cannot prove which mutable scenario structure they used.
    UserScenarioScore.objects.all().update(
        version_confidence='legacy_unknown'
    )
    UserAnswer.objects.all().update(version_confidence='legacy_unknown')


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0061_scenario_family'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='activity',
            name='lineage_key',
            field=models.UUIDField(
                blank=True,
                db_index=True,
                editable=False,
                help_text=(
                    'Stable activity identity shared by translations and '
                    'unchanged copies.'
                ),
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='useranswer',
            name='version_confidence',
            field=models.CharField(
                choices=[
                    ('exact', 'Exact scenario version'),
                    ('legacy_unknown', 'Legacy version unknown'),
                ],
                default='exact',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='userscenarioscore',
            name='data_quality_status',
            field=models.CharField(
                choices=[
                    ('unreviewed', 'Unreviewed'),
                    ('clean', 'Clean'),
                    ('suspect', 'Suspect'),
                    ('excluded', 'Excluded'),
                ],
                default='unreviewed',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='userscenarioscore',
            name='version_confidence',
            field=models.CharField(
                choices=[
                    ('exact', 'Exact scenario version'),
                    ('legacy_unknown', 'Legacy version unknown'),
                ],
                default='exact',
                max_length=20,
            ),
        ),
        migrations.CreateModel(
            name='ScenarioVersion',
            fields=[
                (
                    'id',
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name='ID',
                    ),
                ),
                ('version_number', models.PositiveIntegerField()),
                (
                    'structure_fingerprint',
                    models.CharField(db_index=True, max_length=64),
                ),
                (
                    'content_fingerprint',
                    models.CharField(db_index=True, max_length=64),
                ),
                ('snapshot', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                (
                    'change_summary',
                    models.CharField(blank=True, max_length=255),
                ),
                ('is_current', models.BooleanField(default=True)),
                (
                    'created_by',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='created_scenario_versions',
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    'previous_version',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='next_versions',
                        to='authoringtool.scenarioversion',
                    ),
                ),
                (
                    'scenario',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='versions',
                        to='authoringtool.scenario',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Scenario Version',
                'verbose_name_plural': 'Scenario Versions',
                'ordering': ['scenario', '-version_number'],
            },
        ),
        migrations.AddField(
            model_name='proposalgenerationrun',
            name='scenario_version',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='proposal_generation_runs',
                to='authoringtool.scenarioversion',
            ),
        ),
        migrations.AddField(
            model_name='scenario',
            name='current_version',
            field=models.ForeignKey(
                blank=True,
                help_text=(
                    'Current immutable evidence definition for this scenario.'
                ),
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='current_for_scenarios',
                to='authoringtool.scenarioversion',
            ),
        ),
        migrations.AddField(
            model_name='useranswer',
            name='scenario_version',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='user_answers',
                to='authoringtool.scenarioversion',
            ),
        ),
        migrations.AddField(
            model_name='userscenarioscore',
            name='scenario_version',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='implementation_scores',
                to='authoringtool.scenarioversion',
            ),
        ),
        migrations.RunPython(
            backfill_activity_lineage_and_mark_legacy_evidence,
            migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name='activity',
            name='lineage_key',
            field=models.UUIDField(
                db_index=True,
                default=uuid.uuid4,
                editable=False,
                help_text=(
                    'Stable activity identity shared by translations and '
                    'unchanged copies.'
                ),
            ),
        ),
        migrations.AddIndex(
            model_name='useranswer',
            index=models.Index(
                fields=['scenario_version', 'version_confidence'],
                name='useranswer_version_idx',
            ),
        ),
        migrations.AddIndex(
            model_name='userscenarioscore',
            index=models.Index(
                fields=[
                    'scenario',
                    'scenario_version',
                    'version_confidence',
                ],
                name='uss_version_evidence_idx',
            ),
        ),
        migrations.AddConstraint(
            model_name='scenarioversion',
            constraint=models.UniqueConstraint(
                fields=('scenario', 'version_number'),
                name='unique_scenario_version_number',
            ),
        ),
        migrations.AddConstraint(
            model_name='scenarioversion',
            constraint=models.UniqueConstraint(
                condition=models.Q(('is_current', True)),
                fields=('scenario',),
                name='unique_current_scenario_version',
            ),
        ),
    ]
