from django.db import migrations


LEGACY_THRESHOLD = 300
CURRENT_THRESHOLD = 200


def normalize_ai_metrics_threshold(apps, schema_editor):
    Scenario = apps.get_model("authoringtool", "Scenario")
    Scenario.objects.filter(
        ai_metrics_min_implementations=LEGACY_THRESHOLD
    ).update(ai_metrics_min_implementations=CURRENT_THRESHOLD)


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0003_bulkemailcampaign"),
        ("authoringtool", "0058_backfill_proposal_generation_runs"),
    ]

    operations = [
        migrations.RunPython(
            normalize_ai_metrics_threshold,
            migrations.RunPython.noop,
        ),
    ]
