from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Q

from authoringtool.models import ActivityProposal, ActivityFlag


class Command(BaseCommand):
    help = (
        "Relink ActivityProposal.flag (M2M) using ActivityProposal.activity + categories_in_risk "
        "by matching ActivityFlag.activity + ActivityFlag.category."
    )

    def add_arguments(self, parser):
        parser.add_argument("--scenario", type=int, required=True, help="Scenario ID")
        parser.add_argument(
            "--only-new",
            action="store_true",
            help="Relink only proposals with status='new' (default: relink all statuses).",
        )
        parser.add_argument(
            "--strict-scenario-phase",
            action="store_true",
            help="Also require ActivityFlag.scenario == proposal.scenario AND ActivityFlag.phase == proposal.phase (safer).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would change but do not write to DB.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        scenario_id = options["scenario"]
        only_new = options["only_new"]
        strict = options["strict_scenario_phase"]
        dry_run = options["dry_run"]

        qs = ActivityProposal.objects.filter(scenario_id=scenario_id).select_related("activity", "scenario", "phase")
        if only_new:
            qs = qs.filter(status="new")

        total = qs.count()
        self.stdout.write(f"Found {total} proposals for scenario={scenario_id} (only_new={only_new})")

        updated = 0
        skipped_no_categories = 0
        skipped_no_flags_found = 0

        for p in qs:
            cats = list(p.categories_in_risk.values_list("name", flat=True))  # High/Moderate/Low
            if not cats:
                skipped_no_categories += 1
                continue

            flag_qs = ActivityFlag.objects.filter(
                activity=p.activity,
                category__in=cats,
                is_at_risk=True,
            )

            if strict:
                # ActivityFlag has scenario/phase fields in your model :contentReference[oaicite:1]{index=1}
                flag_qs = flag_qs.filter(
                    Q(scenario__isnull=True) | Q(scenario=p.scenario),
                    Q(phase__isnull=True) | Q(phase=p.phase),
                )

            flags = list(flag_qs.distinct())

            if not flags:
                skipped_no_flags_found += 1
                self.stdout.write(
                    self.style.WARNING(
                        f"[WARN] Proposal {p.id}: no flags found for activity={p.activity_id} cats={cats} strict={strict}"
                    )
                )
                continue

            before = p.flag.count()
            after = len(flags)

            if dry_run:
                self.stdout.write(
                    f"[DRY] Proposal {p.id}: flags {before} -> {after} | activity={p.activity_id} cats={cats}"
                )
            else:
                p.flag.set(flags)
                updated += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Proposal {p.id}: flags {before} -> {after} | activity={p.activity_id} cats={cats}"
                    )
                )

        if dry_run:
            self.stdout.write(self.style.NOTICE("Dry-run complete. No DB changes were made."))
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Updated={updated}, skipped(no_categories)={skipped_no_categories}, skipped(no_flags_found)={skipped_no_flags_found}"
            )
        )
