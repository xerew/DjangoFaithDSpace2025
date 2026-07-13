from django.db import migrations


def backfill_generation_runs(apps, schema_editor):
    ActivityProposal = apps.get_model('authoringtool', 'ActivityProposal')
    ProposalGenerationRun = apps.get_model('authoringtool', 'ProposalGenerationRun')

    scenario_ids = ActivityProposal.objects.filter(
        generation_run__isnull=True
    ).values_list('scenario_id', flat=True).distinct()

    for scenario_id in scenario_ids:
        proposals = ActivityProposal.objects.filter(
            scenario_id=scenario_id, generation_run__isnull=True
        ).order_by('created_at')
        first_proposal = proposals.first()
        if first_proposal is None:
            continue
        run = ProposalGenerationRun.objects.create(
            scenario_id=scenario_id,
            created_by_id=first_proposal.scenario.created_by_id,
            is_current=True,
        )
        # created_at has auto_now_add=True, which silently ignores any
        # value passed to create() — override it afterwards via
        # .update(), which bypasses the auto_now_add pre-save behavior,
        # so the backfilled run's timestamp matches the oldest proposal
        # it covers instead of "now".
        ProposalGenerationRun.objects.filter(pk=run.pk).update(created_at=first_proposal.created_at)
        proposals.update(generation_run_id=run.pk)


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0057_proposalgenerationrun_and_more'),
    ]

    operations = [
        migrations.RunPython(backfill_generation_runs, noop_reverse),
    ]
