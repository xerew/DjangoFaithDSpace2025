"""Import proposal runs written by export_proposals on a local copy.

    python manage.py import_proposals /data/proposals.json --dry-run
    python manage.py import_proposals /data/proposals.json

Each imported run becomes its scenario's current run (the old one moves to
Proposal History), as after a finished Generate. Skipped, with the reason:
runs already here, scenarios not here, scenarios with newer proposals here
(unless --force) or with a proposal job running, and proposals whose
activity was deleted or renamed since the dump.
"""
import json

from django.core.management.base import BaseCommand, CommandError

from authoringtool.proposal_transfer import import_runs


class Command(BaseCommand):
    help = 'Import proposal runs exported from a local copy of the platform.'

    def add_arguments(self, parser):
        parser.add_argument('path', help='The JSON file from export_proposals.')
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Report what would be imported without changing anything.',
        )
        parser.add_argument(
            '--force', action='store_true',
            help='Also replace proposals that are newer on this server.',
        )

    def handle(self, *args, **options):
        try:
            with open(options['path'], encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, ValueError) as error:
            raise CommandError(f'Could not read {options["path"]}: {error}') from error
        try:
            report = import_runs(data, dry_run=options['dry_run'], force=options['force'])
        except ValueError as error:
            raise CommandError(str(error)) from error
        for line in report.lines:
            self.stdout.write(line)
