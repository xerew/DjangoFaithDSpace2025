from collections import defaultdict

from django.db import migrations, models


REJECTION_NUDGES = {
    "wrong_action_create": "create",
    "wrong_action_revise": "revise",
    "wrong_action_skip": "skip",
    "not_relevant": "skip",
    "already_covered": "skip",
}


def backfill_reward_counts(apps, schema_editor):
    QValue = apps.get_model("authoringtool", "QValue")
    UserProposalReview = apps.get_model("authoringtool", "UserProposalReview")

    counts = defaultdict(int)
    reviews = (
        UserProposalReview.objects
        .filter(status__in=["accepted", "rejected"])
        .select_related("proposal")
        .prefetch_related("proposal__flag")
    )

    for review in reviews.iterator(chunk_size=500):
        proposal = review.proposal
        flags = list(proposal.flag.all())
        for flag in flags:
            counts[(flag.flag_type, flag.category, proposal.proposal_type)] += 1

        if review.status == "rejected":
            for reason in review.rejection_reasons or []:
                alternative_action = REJECTION_NUDGES.get(reason)
                if alternative_action:
                    for flag in flags:
                        counts[(flag.flag_type, flag.category, alternative_action)] += 1

    for (flag_type, category, action), count in counts.items():
        QValue.objects.update_or_create(
            flag_type=flag_type,
            category=category,
            action=action,
            defaults={"reward_count": count},
        )


def clear_reward_counts(apps, schema_editor):
    apps.get_model("authoringtool", "QValue").objects.update(reward_count=0)


class Migration(migrations.Migration):

    dependencies = [
        ("authoringtool", "0058_backfill_proposal_generation_runs"),
    ]

    operations = [
        migrations.AddField(
            model_name="qvalue",
            name="reward_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.RunPython(backfill_reward_counts, clear_reward_counts),
    ]
