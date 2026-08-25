from collections import defaultdict

from django.db import migrations, models
import django.db.models.deletion


REJECTION_NUDGES = {
    "wrong_action_create": ("create", 0.5),
    "wrong_action_revise": ("revise", 0.5),
    "wrong_action_skip": ("skip", 0.5),
    "not_relevant": ("skip", 0.3),
    "already_covered": ("skip", 0.3),
}


def backfill_start_and_bandit_counts(apps, schema_editor):
    Scenario = apps.get_model("authoringtool", "Scenario")
    Activity = apps.get_model("authoringtool", "Activity")
    QValue = apps.get_model("authoringtool", "QValue")
    UserProposalReview = apps.get_model("authoringtool", "UserProposalReview")
    BanditPolicyConfiguration = apps.get_model(
        "authoringtool",
        "BanditPolicyConfiguration",
    )

    for scenario in Scenario.objects.filter(start_activity__isnull=True).iterator(
        chunk_size=500
    ):
        start_id = (
            Activity.objects.filter(scenario_id=scenario.id)
            .order_by("id")
            .values_list("id", flat=True)
            .first()
        )
        if start_id:
            Scenario.objects.filter(pk=scenario.id).update(
                start_activity_id=start_id
            )

    counters = defaultdict(
        lambda: {
            "positive_reward_count": 0,
            "negative_reward_count": 0,
            "reward_sum": 0.0,
        }
    )
    reviews = (
        UserProposalReview.objects.filter(status__in=["accepted", "rejected"])
        .select_related("proposal")
        .prefetch_related("proposal__flag")
    )
    for review in reviews.iterator(chunk_size=500):
        reasons = list(review.rejection_reasons or [])
        if "structural_invalid" in reasons:
            UserProposalReview.objects.filter(pk=review.pk).update(
                feedback_type="structural"
            )
            continue

        proposal = review.proposal
        flags = list(proposal.flag.all())
        reward = 1.0 if review.status == "accepted" else -1.0
        for flag in flags:
            key = (flag.flag_type, flag.category, proposal.proposal_type)
            if reward > 0:
                counters[key]["positive_reward_count"] += 1
            else:
                counters[key]["negative_reward_count"] += 1
            counters[key]["reward_sum"] += reward

        if review.status == "rejected":
            for reason in reasons:
                nudge = REJECTION_NUDGES.get(reason)
                if not nudge:
                    continue
                action, reward_value = nudge
                for flag in flags:
                    key = (flag.flag_type, flag.category, action)
                    counters[key]["positive_reward_count"] += 1
                    counters[key]["reward_sum"] += reward_value

    QValue.objects.update(
        reward_count=0,
        positive_reward_count=0,
        negative_reward_count=0,
        reward_sum=0.0,
    )
    for (flag_type, category, action), values in counters.items():
        reward_count = (
            values["positive_reward_count"]
            + values["negative_reward_count"]
        )
        QValue.objects.update_or_create(
            flag_type=flag_type,
            category=category,
            action=action,
            defaults={
                **values,
                "reward_count": reward_count,
            },
        )

    BanditPolicyConfiguration.objects.get_or_create(
        name="Default policy",
        defaults={
            "is_active": True,
            "policy": "thompson",
            "minimum_context_rewards": 200,
            "create_weight": 0.50,
            "skip_weight": 0.30,
            "revise_weight": 0.20,
        },
    )


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("authoringtool", "0059_qvalue_reward_count"),
    ]

    operations = [
        migrations.AddField(
            model_name="scenario",
            name="start_activity",
            field=models.ForeignKey(
                blank=True,
                help_text=(
                    "Explicit entry point for this scenario. It must belong "
                    "to the same scenario."
                ),
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="starting_scenarios",
                to="authoringtool.activity",
            ),
        ),
        migrations.AddField(
            model_name="qvalue",
            name="negative_reward_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="qvalue",
            name="positive_reward_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="qvalue",
            name="reward_sum",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="userproposalreview",
            name="feedback_type",
            field=models.CharField(
                choices=[
                    ("pedagogical", "Pedagogical"),
                    ("structural", "Structural / malformed"),
                ],
                default="pedagogical",
                help_text=(
                    "Structural feedback is tracked separately and never "
                    "updates the learning bandit."
                ),
                max_length=16,
            ),
        ),
        migrations.CreateModel(
            name="BanditPolicyConfiguration",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "name",
                    models.CharField(
                        default="Default policy",
                        max_length=100,
                        unique=True,
                    ),
                ),
                ("is_active", models.BooleanField(default=True)),
                (
                    "policy",
                    models.CharField(
                        choices=[
                            ("thompson", "Thompson Sampling"),
                            ("ucb", "Upper Confidence Bound (UCB)"),
                        ],
                        default="thompson",
                        max_length=20,
                    ),
                ),
                (
                    "minimum_context_rewards",
                    models.PositiveIntegerField(
                        default=200,
                        help_text=(
                            "Use weighted cold-start exploration until this "
                            "many rewards exist for the flag type/category "
                            "context."
                        ),
                    ),
                ),
                ("create_weight", models.FloatField(default=0.5)),
                ("skip_weight", models.FloatField(default=0.3)),
                ("revise_weight", models.FloatField(default=0.2)),
                ("thompson_prior_alpha", models.FloatField(default=1.0)),
                ("thompson_prior_beta", models.FloatField(default=1.0)),
                (
                    "ucb_exploration_strength",
                    models.FloatField(default=1.41421356237),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Bandit Policy Configuration",
                "verbose_name_plural": "Bandit Policy Configuration",
            },
        ),
        migrations.CreateModel(
            name="ProposalStructuralFailure",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("selected_action", models.CharField(blank=True, max_length=16)),
                (
                    "stage",
                    models.CharField(
                        choices=[
                            ("generation", "LLM generation"),
                            ("translation", "Translation"),
                            ("acceptance", "Teacher acceptance"),
                            ("teacher_review", "Teacher review"),
                            ("application", "Proposal application"),
                            ("graph_integrity", "Scenario graph integrity"),
                        ],
                        max_length=24,
                    ),
                ),
                ("errors", models.JSONField(default=list)),
                ("raw_output", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("resolved", models.BooleanField(default=False)),
                ("resolved_at", models.DateTimeField(blank=True, null=True)),
                (
                    "activity",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="proposal_structural_failures",
                        to="authoringtool.activity",
                    ),
                ),
                (
                    "generation_run",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="structural_failures",
                        to="authoringtool.proposalgenerationrun",
                    ),
                ),
                (
                    "proposal",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="structural_failures",
                        to="authoringtool.activityproposal",
                    ),
                ),
                (
                    "scenario",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="proposal_structural_failures",
                        to="authoringtool.scenario",
                    ),
                ),
            ],
            options={
                "verbose_name": "Proposal Structural Failure",
                "verbose_name_plural": "Proposal Structural Failures",
                "ordering": ["-created_at"],
            },
        ),
        migrations.RunPython(backfill_start_and_bandit_counts, noop_reverse),
    ]
