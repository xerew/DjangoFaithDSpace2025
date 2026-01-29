from django.core.management.base import BaseCommand
from django.db import transaction

from authoringtool.models import QValue, ActivityProposal, update_q_value
from authoringtool.models import UserProposalReview  # βάλε το σωστό import path


class Command(BaseCommand):
    help = "Rebuild QValues from UserProposalReview accepted/rejected history (requires proposals to have flags linked)."

    def add_arguments(self, parser):
        parser.add_argument("--scenario", type=int, required=True)
        parser.add_argument("--reset", action="store_true", help="Delete all QValue rows before replay (recommended).")
        parser.add_argument("--dry-run", action="store_true")

    @transaction.atomic
    def handle(self, *args, **opts):
        scenario_id = opts["scenario"]
        reset = opts["reset"]
        dry = opts["dry_run"]

        reviews = (
            UserProposalReview.objects
            .filter(proposal__scenario_id=scenario_id, status__in=["accepted", "rejected"])
            .select_related("proposal")
            .prefetch_related("proposal__flag")
            .order_by("reviewed_at", "id")  # chronological replay
        )

        total = reviews.count()
        self.stdout.write(f"Found {total} user reviews for scenario={scenario_id}")

        if dry:
            with_flags = 0
            for r in reviews:
                if r.proposal.flag.exists():
                    with_flags += 1
            self.stdout.write(f"[DRY] Reviews whose proposal currently has flags: {with_flags}/{total}")
            return

        if reset:
            deleted, _ = QValue.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Deleted QValue rows: {deleted}"))

        applied_updates = 0
        skipped_no_flags = 0

        for r in reviews:
            p = r.proposal
            reward = 1 if r.status == "accepted" else -1

            flags = list(p.flag.all())
            if not flags:
                skipped_no_flags += 1
                continue

            for f in flags:
                # Uses your existing EMA update function :contentReference[oaicite:3]{index=3}
                update_q_value(
                    flag_type=f.flag_type,
                    category=f.category,
                    action=p.proposal_type,
                    reward=reward
                )
                applied_updates += 1

        self.stdout.write(self.style.SUCCESS(
            f"Done. Updates={applied_updates}, skipped(no flags)={skipped_no_flags}"
        ))

        # --- PRINT USERS WITH AT LEAST ONE REVIEW ---
        users_with_reviews = (
            reviews
            .values_list("user__username", flat=True)
            .distinct()
        )

        self.stdout.write(
            self.style.NOTICE(
                "Users with at least one accepted/rejected proposal review:"
            )
        )

        for username in users_with_reviews:
            self.stdout.write(f" - {username}")

        self.stdout.write("")  # κενή γραμμή για καθαρό output
