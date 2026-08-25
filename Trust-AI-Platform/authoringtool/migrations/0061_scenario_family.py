import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def _normalized_language(value):
    return (value or '').strip().casefold()


def backfill_scenario_families(apps, schema_editor):
    Scenario = apps.get_model('authoringtool', 'Scenario')
    ScenarioFamily = apps.get_model('authoringtool', 'ScenarioFamily')

    scenarios = {
        scenario.id: scenario
        for scenario in Scenario.objects.all().order_by('id')
    }
    family_by_scenario = {}

    def create_family(root):
        family = ScenarioFamily.objects.create(
            title=root.name,
            description=root.description or '',
            created_by_id=root.created_by_id,
        )
        Scenario.objects.filter(pk=root.id).update(
            family_id=family.id,
            variant_type='canonical',
        )
        ScenarioFamily.objects.filter(pk=family.id).update(
            canonical_scenario_id=root.id,
        )
        family.subjects.set(root.subjects.all())
        family_by_scenario[root.id] = family.id

    roots = [
        scenario
        for scenario in scenarios.values()
        if (
            not scenario.origin_scenario_id
            or scenario.origin_scenario_id not in scenarios
        )
    ]
    for root in roots:
        create_family(root)

    remaining = {
        scenario_id
        for scenario_id in scenarios
        if scenario_id not in family_by_scenario
    }
    while remaining:
        progressed = False
        for scenario_id in sorted(remaining):
            scenario = scenarios[scenario_id]
            parent_id = scenario.origin_scenario_id
            family_id = family_by_scenario.get(parent_id)
            if not family_id:
                continue

            parent = scenarios[parent_id]
            parent_language = _normalized_language(parent.language)
            scenario_language = _normalized_language(scenario.language)
            is_translation = (
                bool(parent_language)
                and bool(scenario_language)
                and parent_language != scenario_language
            )
            Scenario.objects.filter(pk=scenario_id).update(
                family_id=family_id,
                variant_type=(
                    'translation' if is_translation else 'adaptation'
                ),
            )
            family_by_scenario[scenario_id] = family_id
            remaining.remove(scenario_id)
            progressed = True
            break

        if not progressed:
            # Malformed legacy origin cycles are kept safe and independent.
            fallback_id = min(remaining)
            create_family(scenarios[fallback_id])
            remaining.remove(fallback_id)


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0060_scenario_start_graph_bandit_controls'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScenarioFamily',
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
                ('created_on', models.DateTimeField(auto_now_add=True)),
                ('updated_on', models.DateTimeField(auto_now=True)),
                (
                    'canonical_scenario',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='canonical_for_families',
                        to='authoringtool.scenario',
                    ),
                ),
                (
                    'created_by',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='created_scenario_families',
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    'subjects',
                    models.ManyToManyField(
                        blank=True,
                        related_name='scenario_families',
                        to='authoringtool.subject',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Scenario Family',
                'verbose_name_plural': 'Scenario Families',
                'ordering': ['title', 'id'],
            },
        ),
        migrations.AddField(
            model_name='scenario',
            name='family',
            field=models.ForeignKey(
                blank=True,
                help_text=(
                    'Shared lesson identity across translations and teacher '
                    'adaptations.'
                ),
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='scenarios',
                to='authoringtool.scenariofamily',
            ),
        ),
        migrations.AddField(
            model_name='scenario',
            name='variant_type',
            field=models.CharField(
                choices=[
                    ('canonical', 'Canonical'),
                    ('translation', 'Official translation'),
                    ('adaptation', 'Teacher adaptation'),
                ],
                default='canonical',
                help_text=(
                    'How this scenario relates to the rest of its family.'
                ),
                max_length=20,
            ),
        ),
        migrations.RunPython(
            backfill_scenario_families,
            migrations.RunPython.noop,
        ),
    ]
