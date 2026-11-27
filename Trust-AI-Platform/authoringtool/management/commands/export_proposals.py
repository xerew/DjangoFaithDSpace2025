"""Write proposal runs to a JSON file, for import_proposals on the server.

Run it on a local copy of the platform after generating proposals there:

    python manage.py export_proposals --output /data/proposals.json

By default every scenario's current run is written; runs that are already on
the server (they came with the dump) are skipped there on import.
"""
import json

from django.core.management.base import BaseCommand, CommandError
from django.utils.dateparse import parse_datetime

from authoringtool.models import ProposalGenerationRun
from authoringtool.proposal_transfer import export_runs


class Command(BaseCommand):
    help = 'Export current proposal runs to a JSON file for import_proposals.'

    def add_arguments(self, parser):
        parser.add_argument('--output', required=True, help='File to write.')
        parser.add_argument(
            '--since',
            help='Only runs created at or after this time (e.g. 2026-11-28T09:00).',
        )
        parser.add_argument(
            '--scenario', type=int, nargs='+', help='Only these scenario ids.',
        )

    def handle(self, *args, **options):
        runs = ProposalGenerationRun.objects.filter(is_current=True).select_related('scenario')
        if options['since']:
            since = parse_datetime(options['since'])
            if since is None:
                raise CommandError(f"--since is not a date and time: {options['since']}")
            runs = runs.filter(created_at__gte=since)
        if options['scenario']:
            runs = runs.filter(scenario_id__in=options['scenario'])
        runs = list(runs.order_by('scenario_id'))
        data = export_runs(runs)
        with open(options['output'], 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
        proposals = sum(len(run['proposals']) for run in data['runs'])
        self.stdout.write(
            f"Wrote {len(runs)} run{'s' if len(runs) != 1 else ''} "
            f"({proposals} proposal{'s' if proposals != 1 else ''}) to {options['output']}."
        )
        for run in data['runs']:
            self.stdout.write(
                f"  scenario {run['scenario_id']} \"{run['scenario_name']}\": "
                f"{len(run['proposals'])} proposals, created {run['created_at']}"
            )
