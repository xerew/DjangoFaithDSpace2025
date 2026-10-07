from django.core.management.base import BaseCommand

from authoringtool.translation_matching import scan_translation_matches


class Command(BaseCommand):
    help = (
        "Score possible 1:1 translations between scenarios and store them for "
        "review in Admin > Translation matches. Review decisions are kept."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--scenario",
            type=int,
            action="append",
            dest="scenario_ids",
            help="Only compare these scenarios (repeat the option for more).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the candidates and their confidence; write nothing.",
        )

    def handle(self, *args, **options):
        summary = scan_translation_matches(
            scenario_ids=options["scenario_ids"],
            dry_run=options["dry_run"],
        )
        if options["dry_run"]:
            candidates = sorted(
                summary["candidates"],
                key=lambda row: -row["confidence"],
            )
            self.stdout.write(f"{len(candidates)} candidate(s), nothing written:")
            for row in candidates:
                self.stdout.write(
                    f"{row['confidence']:>3}%  {row['scenario_a']} <-> "
                    f"{row['scenario_b']}  {'; '.join(row['differences'][:3])}"
                )
            return
        self.stdout.write(self.style.SUCCESS(
            f"{summary['matches']} candidate(s): {summary['created']} new, "
            f"{summary['changed_since_review']} changed since review, "
            f"{summary['removed']} removed."
        ))
