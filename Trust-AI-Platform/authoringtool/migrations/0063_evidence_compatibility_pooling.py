import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def backfill_version_compatibility(apps, schema_editor):
    Cluster = apps.get_model(
        'authoringtool',
        'EvidenceCompatibilityCluster',
    )
    Compatibility = apps.get_model(
        'authoringtool',
        'ScenarioVersionCompatibility',
    )
    ScenarioVersion = apps.get_model(
        'authoringtool',
        'ScenarioVersion',
    )

    clusters = {}
    versions = (
        ScenarioVersion.objects
        .select_related('scenario')
        .exclude(scenario__family_id__isnull=True)
        .order_by('scenario__family_id', 'structure_fingerprint', 'id')
    )
    for version in versions.iterator():
        family_id = version.scenario.family_id
        fingerprint = version.structure_fingerprint
        key = (family_id, fingerprint)
        cluster = clusters.get(key)
        if cluster is None:
            cluster, _ = Cluster.objects.get_or_create(
                family_id=family_id,
                cluster_key=f'auto:{fingerprint}',
                defaults={
                    'name': f'Structure {fingerprint[:12]}',
                    'structure_fingerprint': fingerprint,
                    'is_automatic': True,
                    'created_by_id': version.created_by_id,
                },
            )
            clusters[key] = cluster

        needs_review = version.scenario.variant_type == 'adaptation'
        Compatibility.objects.get_or_create(
            scenario_version_id=version.id,
            defaults={
                'cluster_id': cluster.id,
                'status': (
                    'needs_review' if needs_review else 'compatible'
                ),
                'decision_source': 'automatic',
                'reason': (
                    'Teacher adaptations require review before their '
                    'evidence is pooled with another scenario.'
                    if needs_review
                    else (
                        'Automatically matched by family and structural '
                        'fingerprint.'
                    )
                ),
            },
        )


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0062_scenario_evidence_versioning'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='EvidenceCompatibilityCluster',
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
                ('name', models.CharField(max_length=255)),
                ('cluster_key', models.CharField(max_length=100)),
                (
                    'structure_fingerprint',
                    models.CharField(
                        blank=True,
                        db_index=True,
                        max_length=64,
                    ),
                ),
                ('is_automatic', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                (
                    'created_by',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='created_evidence_clusters',
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    'family',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='evidence_clusters',
                        to='authoringtool.scenariofamily',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Evidence Compatibility Cluster',
                'verbose_name_plural': 'Evidence Compatibility Clusters',
                'ordering': ['family', 'name', 'id'],
            },
        ),
        migrations.CreateModel(
            name='ScenarioVersionCompatibility',
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
                            ('compatible', 'Compatible'),
                            ('needs_review', 'Needs review'),
                            (
                                'excluded',
                                'Excluded from family evidence',
                            ),
                        ],
                        default='compatible',
                        max_length=20,
                    ),
                ),
                (
                    'decision_source',
                    models.CharField(
                        choices=[
                            (
                                'automatic',
                                'Automatic structural match',
                            ),
                            ('admin', 'Administrator decision'),
                        ],
                        default='automatic',
                        max_length=20,
                    ),
                ),
                ('reason', models.CharField(blank=True, max_length=500)),
                ('reviewed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                (
                    'cluster',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='memberships',
                        to='authoringtool.evidencecompatibilitycluster',
                    ),
                ),
                (
                    'reviewed_by',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='reviewed_scenario_compatibilities',
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    'scenario_version',
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='compatibility',
                        to='authoringtool.scenarioversion',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Scenario Version Compatibility',
                'verbose_name_plural': (
                    'Scenario Version Compatibilities'
                ),
                'ordering': [
                    'cluster__family',
                    'cluster',
                    'scenario_version__scenario',
                    '-scenario_version__version_number',
                ],
            },
        ),
        migrations.AddField(
            model_name='activityflag',
            name='evidence_scope',
            field=models.CharField(
                choices=[
                    ('local', 'This scenario only'),
                    ('compatible', 'Compatible family evidence'),
                ],
                default='local',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='activityflag',
            name='evidence_signature',
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name='activityflag',
            name='evidence_sources',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='proposalgenerationrun',
            name='evidence_scope',
            field=models.CharField(
                choices=[
                    ('local', 'This scenario only'),
                    ('compatible', 'Compatible family evidence'),
                ],
                default='local',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='proposalgenerationrun',
            name='evidence_summary',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='proposalgenerationrun',
            name='evidence_version_ids',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddConstraint(
            model_name='evidencecompatibilitycluster',
            constraint=models.UniqueConstraint(
                fields=('family', 'cluster_key'),
                name='unique_family_evidence_cluster_key',
            ),
        ),
        migrations.RunPython(
            backfill_version_compatibility,
            migrations.RunPython.noop,
        ),
    ]
