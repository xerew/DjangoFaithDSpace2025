import hashlib
import json

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def _fingerprint(payload):
    return hashlib.sha256(
        json.dumps(
            payload or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()


def backfill_activity_and_implementation_lineage(apps, schema_editor):
    Activity = apps.get_model('authoringtool', 'Activity')
    ActivityConcept = apps.get_model('authoringtool', 'ActivityConcept')
    ActivityRevision = apps.get_model('authoringtool', 'ActivityRevision')
    ScenarioImplementation = apps.get_model(
        'authoringtool',
        'ScenarioImplementation',
    )
    ScenarioVersion = apps.get_model('authoringtool', 'ScenarioVersion')
    UserAnswer = apps.get_model('authoringtool', 'UserAnswer')
    UserScenarioScore = apps.get_model(
        'authoringtool',
        'UserScenarioScore',
    )

    concepts = {}
    activities = (
        Activity.objects
        .filter(scenario__family__isnull=False)
        .select_related('scenario')
        .order_by(
            'scenario__family_id',
            'scenario_id',
            'phase__created_on',
            'created_on',
            'id',
        )
    )
    for activity in activities.iterator():
        key = (activity.scenario.family_id, str(activity.lineage_key))
        concept = concepts.get(key)
        if concept is None:
            concept = ActivityConcept.objects.create(
                family_id=activity.scenario.family_id,
                title=activity.name,
                created_by_id=activity.created_by_id,
            )
            concepts[key] = concept
        Activity.objects.filter(pk=activity.pk).update(concept_id=concept.id)

    live_activities = {
        (activity.scenario_id, str(activity.lineage_key)): activity
        for activity in (
            Activity.objects
            .filter(concept__isnull=False)
            .only(
                'id',
                'scenario_id',
                'concept_id',
                'lineage_key',
            )
        )
    }
    for version in ScenarioVersion.objects.all().iterator():
        structure_by_lineage = {}
        content_by_lineage = {}
        snapshot = version.snapshot or {}
        for phase in snapshot.get('structure', {}).get('phases', []):
            for activity in phase.get('activities', []):
                lineage_key = activity.get('lineage_key')
                if lineage_key:
                    structure_by_lineage[lineage_key] = activity
        for phase in snapshot.get('content', {}).get('phases', []):
            for activity in phase.get('activities', []):
                lineage_key = activity.get('lineage_key')
                if lineage_key:
                    content_by_lineage[lineage_key] = activity

        for lineage_key, structure in structure_by_lineage.items():
            activity = live_activities.get(
                (version.scenario_id, lineage_key)
            )
            if not activity:
                continue
            content = content_by_lineage.get(lineage_key, {})
            ActivityRevision.objects.get_or_create(
                scenario_version_id=version.id,
                lineage_key=lineage_key,
                defaults={
                    'activity_id': activity.id,
                    'concept_id': activity.concept_id,
                    'revision_number': version.version_number,
                    'structure_fingerprint': _fingerprint(structure),
                    'content_fingerprint': _fingerprint(content),
                    'snapshot': {
                        'schema': 1,
                        'structure': structure,
                        'content': content,
                    },
                },
            )

    implementation_by_score_key = {}
    for score in UserScenarioScore.objects.all().order_by('id').iterator():
        confidence = score.version_confidence
        if confidence == 'exact' and not score.scenario_version_id:
            confidence = 'legacy_unknown'
        status = (
            'legacy'
            if confidence == 'legacy_unknown'
            else 'completed'
        )
        implementation = ScenarioImplementation.objects.create(
            user_id=score.user_id,
            scenario_id=score.scenario_id,
            scenario_version_id=score.scenario_version_id,
            status=status,
            version_confidence=confidence,
            data_quality_status=score.data_quality_status,
            last_activity_id=score.last_activity_id,
        )
        UserScenarioScore.objects.filter(pk=score.pk).update(
            implementation_id=implementation.id,
            version_confidence=confidence,
        )
        key = (
            score.user_id,
            score.scenario_id,
            score.scenario_version_id,
            confidence,
        )
        implementation_by_score_key.setdefault(key, implementation)

    revision_by_version_lineage = {
        (revision.scenario_version_id, str(revision.lineage_key)): revision.id
        for revision in ActivityRevision.objects.all().only(
            'id',
            'scenario_version_id',
            'lineage_key',
        )
    }
    for answer in (
        UserAnswer.objects
        .select_related('activity')
        .all()
        .order_by('id')
        .iterator()
    ):
        confidence = answer.version_confidence
        if confidence == 'exact' and not answer.scenario_version_id:
            confidence = 'legacy_unknown'
        key = (
            answer.user_id,
            answer.activity.scenario_id,
            answer.scenario_version_id,
            confidence,
        )
        implementation = implementation_by_score_key.get(key)
        if implementation is None:
            implementation = ScenarioImplementation.objects.create(
                user_id=answer.user_id,
                scenario_id=answer.activity.scenario_id,
                scenario_version_id=answer.scenario_version_id,
                status=(
                    'legacy'
                    if confidence == 'legacy_unknown'
                    else 'completed'
                ),
                version_confidence=confidence,
                data_quality_status='unreviewed',
            )
            implementation_by_score_key[key] = implementation
        activity_revision_id = revision_by_version_lineage.get(
            (
                answer.scenario_version_id,
                str(answer.activity.lineage_key),
            )
        )
        UserAnswer.objects.filter(pk=answer.pk).update(
            implementation_id=implementation.id,
            activity_revision_id=activity_revision_id,
            version_confidence=confidence,
        )


class Migration(migrations.Migration):
    # PostgreSQL must commit the FK-heavy backfill before the following index
    # and constraint DDL; otherwise it reports pending trigger events.
    atomic = False

    dependencies = [
        ('authoringtool', '0064_scenario_family_discovery'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ActivityMatchingProxy',
            fields=[],
            options={
                'verbose_name': 'Activity Matching',
                'verbose_name_plural': 'Activity Matching',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('authoringtool.activity',),
        ),
        migrations.CreateModel(
            name='ActivityConcept',
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
                ('title', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                (
                    'created_by',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='created_activity_concepts',
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    'family',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='activity_concepts',
                        to='authoringtool.scenariofamily',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Activity Concept',
                'verbose_name_plural': 'Activity Concepts',
                'ordering': ['family', 'title', 'id'],
            },
        ),
        migrations.AddField(
            model_name='activity',
            name='concept',
            field=models.ForeignKey(
                blank=True,
                help_text=(
                    'Language-independent family concept used to compare '
                    'equivalent activities across variants.'
                ),
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='activities',
                to='authoringtool.activityconcept',
            ),
        ),
        migrations.CreateModel(
            name='ActivityRevision',
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
                ('lineage_key', models.UUIDField(db_index=True)),
                ('revision_number', models.PositiveIntegerField()),
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
                    'activity',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='revisions',
                        to='authoringtool.activity',
                    ),
                ),
                (
                    'concept',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name='revisions',
                        to='authoringtool.activityconcept',
                    ),
                ),
                (
                    'scenario_version',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='activity_revisions',
                        to='authoringtool.scenarioversion',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Activity Revision',
                'verbose_name_plural': 'Activity Revisions',
                'ordering': [
                    'scenario_version__scenario',
                    'scenario_version__version_number',
                    'revision_number',
                ],
            },
        ),
        migrations.AddField(
            model_name='useranswer',
            name='activity_revision',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name='user_answers',
                to='authoringtool.activityrevision',
            ),
        ),
        migrations.CreateModel(
            name='ScenarioImplementation',
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
                (
                    'status',
                    models.CharField(
                        choices=[
                            ('active', 'Active'),
                            ('completed', 'Completed'),
                            ('abandoned', 'Abandoned'),
                            ('legacy', 'Legacy imported attempt'),
                        ],
                        default='active',
                        max_length=20,
                    ),
                ),
                (
                    'version_confidence',
                    models.CharField(
                        choices=[
                            ('exact', 'Exact scenario version'),
                            (
                                'legacy_unknown',
                                'Legacy version unknown',
                            ),
                        ],
                        default='exact',
                        max_length=20,
                    ),
                ),
                (
                    'data_quality_status',
                    models.CharField(
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
                ('started_at', models.DateTimeField(auto_now_add=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                (
                    'last_activity',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='implementation_progress',
                        to='authoringtool.activity',
                    ),
                ),
                (
                    'scenario',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='implementations',
                        to='authoringtool.scenario',
                    ),
                ),
                (
                    'scenario_version',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name='implementations',
                        to='authoringtool.scenarioversion',
                    ),
                ),
                (
                    'user',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='scenario_implementations',
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                'verbose_name': 'Scenario Implementation',
                'verbose_name_plural': 'Scenario Implementations',
                'ordering': ['-started_at', '-id'],
            },
        ),
        migrations.AddField(
            model_name='useranswer',
            name='implementation',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='answers',
                to='authoringtool.scenarioimplementation',
            ),
        ),
        migrations.AddField(
            model_name='userscenarioscore',
            name='implementation',
            field=models.OneToOneField(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='score',
                to='authoringtool.scenarioimplementation',
            ),
        ),
        migrations.RunPython(
            backfill_activity_and_implementation_lineage,
            migrations.RunPython.noop,
            atomic=True,
        ),
        migrations.AddIndex(
            model_name='useranswer',
            index=models.Index(
                fields=['implementation', 'activity_revision'],
                name='useranswer_impl_actrev_idx',
            ),
        ),
        migrations.AddIndex(
            model_name='activityrevision',
            index=models.Index(
                fields=['concept', 'scenario_version'],
                name='actrev_concept_version_idx',
            ),
        ),
        migrations.AddConstraint(
            model_name='activityrevision',
            constraint=models.UniqueConstraint(
                fields=('scenario_version', 'lineage_key'),
                name='unique_activity_revision_per_scenario_version',
            ),
        ),
        migrations.AddIndex(
            model_name='scenarioimplementation',
            index=models.Index(
                fields=['scenario', 'scenario_version', 'status'],
                name='impl_scenario_version_idx',
            ),
        ),
        migrations.AddIndex(
            model_name='scenarioimplementation',
            index=models.Index(
                fields=['user', 'scenario', 'status'],
                name='impl_user_scenario_idx',
            ),
        ),
        migrations.AddIndex(
            model_name='scenarioimplementation',
            index=models.Index(
                fields=['data_quality_status', 'version_confidence'],
                name='impl_quality_conf_idx',
            ),
        ),
        migrations.AddConstraint(
            model_name='scenarioimplementation',
            constraint=models.UniqueConstraint(
                condition=models.Q(('status', 'active')),
                fields=('user', 'scenario'),
                name='unique_active_implementation_per_user_scenario',
            ),
        ),
        migrations.AddConstraint(
            model_name='scenarioimplementation',
            constraint=models.CheckConstraint(
                check=models.Q(
                    ('version_confidence', 'legacy_unknown'),
                    ('scenario_version__isnull', False),
                    _connector='OR',
                ),
                name='exact_implementation_requires_version',
            ),
        ),
    ]
