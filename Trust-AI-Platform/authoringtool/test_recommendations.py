import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.contrib.auth.models import Group, User
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityFlag,
    ActivityProposal,
    ActivityType,
    Answer,
    BanditPolicyConfiguration,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    ProposalGenerationRun,
    ProposalStructuralFailure,
    QValue,
    Scenario,
    UserProposalReview,
    update_q_value,
)
from authoringtool.graph_validation import validate_scenario_graph
from authoringtool.evidence import get_evidence_context
from authoringtool.tasks import (
    ProposalValidationError,
    _get_exploration_rate,
    _build_personal_scenario,
    apply_user_proposals_to_new_scenario,
    build_proposal_json_schema,
    create_activity_from_json_action,
    format_proposal_answer_text,
    get_best_q_action,
    merge_proposal_edits,
    proposal_requires_insert_after,
    request_validated_proposal,
    validate_proposal_data,
)


def valid_question_proposal(action="create"):
    return {
        "action": action,
        "activity_name": "Check the pendulum model",
        "activity_type": "Question",
        "content": "Which statement best matches the measurements?",
        "answers": [
            {"text": "A. Period is independent of mass", "is_correct": True, "weight": 3},
            {"text": "B. Period doubles with mass", "is_correct": False, "weight": 2},
            {"text": "C. Period equals mass", "is_correct": False, "weight": 1},
        ],
        "insert_location": "after",
        "explanation": "This checks the misconception directly. The distractors reflect likely errors.",
    }


class ProposalValidationTests(SimpleTestCase):
    def test_valid_question_is_normalized_and_accepted(self):
        proposal = valid_question_proposal()
        proposal["activity_type"] = "question"
        proposal["insert_location"] = "after flagged activity"

        validated = validate_proposal_data(proposal, expected_action="create")

        self.assertEqual(validated["activity_type"], "Question")
        self.assertEqual(validated["insert_location"], "after")
        self.assertEqual(len(validated["answers"]), 3)

    def test_answer_labels_are_canonicalized_by_position(self):
        proposal = valid_question_proposal()
        proposal["answers"][0]["text"] = "C. First answer"
        proposal["answers"][1]["text"] = "B) Second answer"
        proposal["answers"][2]["text"] = "Third answer"

        validated = validate_proposal_data(proposal)

        self.assertEqual(
            [answer["text"] for answer in validated["answers"]],
            ["A. First answer", "B. Second answer", "C. Third answer"],
        )

    def test_duplicate_answer_bodies_are_rejected_despite_different_labels(self):
        proposal = valid_question_proposal()
        proposal["answers"][0]["text"] = "A. Same answer"
        proposal["answers"][1]["text"] = "B. Same answer"

        with self.assertRaisesRegex(
            ProposalValidationError,
            "answer text must be unique",
        ):
            validate_proposal_data(proposal)

    def test_answer_formatter_uses_requested_position(self):
        self.assertEqual(
            format_proposal_answer_text("D) Example answer", 0),
            "A. Example answer",
        )

    def test_question_without_answers_is_rejected(self):
        proposal = valid_question_proposal()
        proposal["answers"] = []

        with self.assertRaisesRegex(
            ProposalValidationError,
            "Question activities must contain 2 to 4 answers",
        ):
            validate_proposal_data(proposal, expected_action="create")

    def test_question_requires_exactly_one_weight_three_correct_answer(self):
        proposal = valid_question_proposal()
        proposal["answers"][1]["is_correct"] = True
        proposal["answers"][1]["weight"] = 3

        with self.assertRaisesRegex(
            ProposalValidationError,
            "exactly one correct answer",
        ):
            validate_proposal_data(proposal)

    def test_policy_selected_action_cannot_be_overridden(self):
        proposal = valid_question_proposal(action="revise")

        with self.assertRaisesRegex(
            ProposalValidationError,
            "action must remain create",
        ):
            validate_proposal_data(proposal, expected_action="create")

    def test_non_question_cannot_contain_answers(self):
        proposal = valid_question_proposal()
        proposal["activity_type"] = "Explanation"

        with self.assertRaisesRegex(
            ProposalValidationError,
            "must not contain answers",
        ):
            validate_proposal_data(proposal)

    def test_question_revision_can_require_existing_answer_count(self):
        proposal = valid_question_proposal(action="revise")
        proposal["answers"] = proposal["answers"][:2]

        with self.assertRaisesRegex(
            ProposalValidationError,
            "keep exactly 3 answers",
        ):
            validate_proposal_data(
                proposal,
                expected_action="revise",
                expected_activity_type="Question",
                expected_answer_count=3,
            )

    def test_merge_teacher_edits_preserves_answer_metadata(self):
        base = valid_question_proposal(action="revise")
        edited = {
            "content": "A clearer question stem",
            "answers": [
                {"text": "A. Revised correct answer"},
                {"text": "B. Revised distractor"},
                {"text": "C. Another revised distractor"},
            ],
        }

        merged = merge_proposal_edits(base, edited)

        self.assertEqual(merged["content"], "A clearer question stem")
        self.assertTrue(merged["answers"][0]["is_correct"])
        self.assertEqual(merged["answers"][0]["weight"], 3)
        self.assertFalse(merged["answers"][1]["is_correct"])

    def test_schema_locks_action_and_question_answer_count(self):
        schema = build_proposal_json_schema(
            "revise",
            expected_activity_type="Question",
            expected_answer_count=3,
        )

        self.assertEqual(schema["properties"]["action"]["enum"], ["revise"])
        self.assertEqual(schema["properties"]["activity_type"]["enum"], ["Question"])
        self.assertEqual(schema["properties"]["answers"]["minItems"], 3)
        self.assertEqual(schema["properties"]["answers"]["maxItems"], 3)

    def test_entry_activity_create_schema_only_allows_after(self):
        schema = build_proposal_json_schema(
            "create",
            require_insert_after=True,
        )

        self.assertEqual(
            schema["properties"]["insert_location"]["enum"],
            ["after"],
        )

    def test_entry_activity_create_before_is_rejected(self):
        proposal = valid_question_proposal()
        proposal["insert_location"] = "before"

        with self.assertRaisesRegex(
            ProposalValidationError,
            "first activity must use insert_location after",
        ):
            validate_proposal_data(
                proposal,
                expected_action="create",
                require_insert_after=True,
            )


class BanditPolicyTests(TestCase):
    def test_create_has_first_50_percent_of_weighted_exploration(self):
        QValue.objects.create(
            flag_type="Systemic failure",
            category="Low",
            action="skip",
            q_value=1.0,
            reward_count=199,
        )

        with patch("authoringtool.tasks.random.random", return_value=0.49):
            action = get_best_q_action([])

        self.assertEqual(action, "create")

    def test_skip_has_next_30_percent_of_weighted_exploration(self):
        with patch("authoringtool.tasks.random.random", return_value=0.65):
            action = get_best_q_action([])

        self.assertEqual(action, "skip")

    def test_revise_has_final_20_percent_of_weighted_exploration(self):
        with patch("authoringtool.tasks.random.random", return_value=0.90):
            action = get_best_q_action([])

        self.assertEqual(action, "revise")

    def test_exploration_decays_linearly_between_200_and_500_rewards(self):
        self.assertEqual(_get_exploration_rate(200), 1.0)
        self.assertAlmostEqual(_get_exploration_rate(350), 0.525)
        self.assertEqual(_get_exploration_rate(500), 0.05)

    def test_q_values_control_action_at_500_rewards(self):
        flag = SimpleNamespace(flag_type="Systemic failure", category="Low")
        configuration = BanditPolicyConfiguration.get_active()
        configuration.policy = "ucb"
        configuration.minimum_context_rewards = 1
        configuration.save()
        QValue.objects.create(
            flag_type=flag.flag_type,
            category=flag.category,
            action="create",
            q_value=0.1,
            reward_count=500,
            positive_reward_count=275,
            negative_reward_count=225,
            reward_sum=50,
        )
        QValue.objects.create(
            flag_type=flag.flag_type,
            category=flag.category,
            action="revise",
            q_value=0.9,
            reward_count=100,
            positive_reward_count=95,
            negative_reward_count=5,
            reward_sum=90,
        )
        QValue.objects.create(
            flag_type=flag.flag_type,
            category=flag.category,
            action="skip",
            q_value=-0.2,
            reward_count=100,
            positive_reward_count=40,
            negative_reward_count=60,
            reward_sum=-20,
        )

        action = get_best_q_action([flag])

        self.assertEqual(action, "revise")

    def test_q_value_update_increments_reward_count(self):
        update_q_value("Systemic failure", "Low", "create", reward=1)
        update_q_value("Systemic failure", "Low", "create", reward=-1)

        q_value = QValue.objects.get(
            flag_type="Systemic failure",
            category="Low",
            action="create",
        )
        self.assertEqual(q_value.reward_count, 2)
        self.assertEqual(q_value.positive_reward_count, 1)
        self.assertEqual(q_value.negative_reward_count, 1)
        self.assertEqual(q_value.reward_sum, 0)

    def test_thompson_sampling_is_available_for_mature_context(self):
        configuration = BanditPolicyConfiguration.get_active()
        configuration.policy = "thompson"
        configuration.minimum_context_rewards = 1
        configuration.save()
        flag = SimpleNamespace(flag_type="Systemic failure", category="Low")
        for action in ("create", "revise", "skip"):
            QValue.objects.create(
                flag_type=flag.flag_type,
                category=flag.category,
                action=action,
                reward_count=10,
                positive_reward_count=5,
                negative_reward_count=5,
            )

        with patch(
            "authoringtool.tasks.random.betavariate",
            side_effect=[0.9, 0.2, 0.1],
        ):
            action = get_best_q_action([flag])

        self.assertEqual(action, "create")


class StructuredProposalRequestTests(SimpleTestCase):
    @patch("authoringtool.tasks.requests.post")
    def test_invalid_question_is_retried_with_validation_feedback(self, post):
        invalid = valid_question_proposal()
        invalid["answers"] = []
        first_response = Mock()
        first_response.raise_for_status.return_value = None
        first_response.json.return_value = {"response": json.dumps(invalid)}

        second_response = Mock()
        second_response.raise_for_status.return_value = None
        second_response.json.return_value = {
            "response": json.dumps(valid_question_proposal())
        }
        post.side_effect = [first_response, second_response]

        raw, structured = request_validated_proposal(
            "Return the proposal as JSON.",
            required_action="create",
        )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(structured["action"], "create")
        self.assertEqual(len(structured["answers"]), 3)
        self.assertEqual(json.loads(raw)["activity_type"], "Question")
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["format"]["properties"]["action"]["enum"], ["create"])
        self.assertFalse(first_payload["think"])
        self.assertIn("Question activities must contain 2 to 4 answers", second_payload["prompt"])

    @patch("authoringtool.tasks.requests.post")
    def test_thinking_field_is_used_as_compatibility_fallback(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "response": "",
            "thinking": json.dumps(valid_question_proposal()),
            "done": True,
            "done_reason": "stop",
        }
        post.return_value = response

        raw, structured = request_validated_proposal(
            "Return the proposal as JSON.",
            required_action="create",
        )

        self.assertEqual(structured["action"], "create")
        self.assertEqual(json.loads(raw)["activity_type"], "Question")

    @patch("authoringtool.tasks.requests.post")
    def test_entry_activity_request_constrains_insert_location_to_after(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "response": json.dumps(valid_question_proposal()),
        }
        post.return_value = response

        request_validated_proposal(
            "Return the proposal as JSON.",
            required_action="create",
            require_insert_after=True,
        )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload["format"]["properties"]["insert_location"]["enum"],
            ["after"],
        )


class CreateActivityFromProposalTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("proposal_creator", password="pass")
        self.scenario = Scenario.objects.create(
            name="Structured proposal scenario",
            created_by=self.user,
            updated_by=self.user,
        )
        self.phase = Phase.objects.create(
            name="Phase 1",
            scenario=self.scenario,
            created_by=self.user,
            updated_by=self.user,
        )
        ActivityType.objects.create(
            name="Question",
            created_by=self.user,
            updated_by=self.user,
        )

    def test_question_without_answers_is_not_created(self):
        invalid = valid_question_proposal()
        invalid["answers"] = []

        with self.assertRaises(ProposalValidationError):
            create_activity_from_json_action(
                invalid,
                self.phase,
                self.scenario,
                self.user,
            )

        self.assertFalse(Activity.objects.filter(name=invalid["activity_name"]).exists())

    def test_valid_question_and_answers_are_created_atomically(self):
        proposal = valid_question_proposal()

        activity = create_activity_from_json_action(
            proposal,
            self.phase,
            self.scenario,
            self.user,
        )

        self.assertEqual(activity.answers.count(), 3)
        self.assertEqual(activity.answers.filter(is_correct=True).count(), 1)
        self.assertEqual(
            list(activity.answers.order_by("id").values_list("text", flat=True)),
            [
                "A. Period is independent of mass",
                "B. Period doubles with mass",
                "C. Period equals mass",
            ],
        )
        self.assertEqual(
            Answer.objects.get(activity=activity, is_correct=True).answer_weight,
            3,
        )


class AcceptProposalStructureTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user("proposal_reviewer", password="pass")
        teachers, _ = Group.objects.get_or_create(name="teachers")
        self.user.groups.add(teachers)
        self.client.login(username="proposal_reviewer", password="pass")
        self.scenario = Scenario.objects.create(
            name="Proposal review scenario",
            created_by=self.user,
            updated_by=self.user,
        )
        self.phase = Phase.objects.create(
            name="Phase 1",
            scenario=self.scenario,
            created_by=self.user,
            updated_by=self.user,
        )
        question_type = ActivityType.objects.create(
            name="Question",
            created_by=self.user,
            updated_by=self.user,
        )
        self.activity = Activity.objects.create(
            name="Existing question",
            text="Original stem",
            plain_text="Original stem",
            scenario=self.scenario,
            phase=self.phase,
            activity_type=question_type,
            created_by=self.user,
            updated_by=self.user,
        )
        for answer in valid_question_proposal()["answers"]:
            Answer.objects.create(
                activity=self.activity,
                text=answer["text"],
                is_correct=answer["is_correct"],
                answer_weight=answer["weight"],
                created_by=self.user,
                updated_by=self.user,
            )

    def create_proposal(self, data, proposal_type="revise"):
        raw = json.dumps(data)
        return ActivityProposal.objects.create(
            scenario=self.scenario,
            phase=self.phase,
            activity=self.activity,
            proposal_type=proposal_type,
            suggested_action=raw,
            translated_action=raw,
            json_action=raw,
            json_translated_action=raw,
        )

    def test_accept_blocks_question_proposal_without_answers(self):
        invalid = valid_question_proposal(action="revise")
        invalid["answers"] = []
        proposal = self.create_proposal(invalid)

        response = self.client.post(
            reverse("accept_proposal", args=[self.scenario.id, proposal.id])
        )

        self.assertEqual(response.status_code, 302)
        review = UserProposalReview.objects.get(proposal=proposal, user=self.user)
        self.assertEqual(review.status, "new")

    def test_accept_allows_structurally_valid_question_proposal(self):
        proposal = self.create_proposal(valid_question_proposal(action="revise"))

        response = self.client.post(
            reverse("accept_proposal", args=[self.scenario.id, proposal.id])
        )

        self.assertEqual(response.status_code, 302)
        review = UserProposalReview.objects.get(proposal=proposal, user=self.user)
        self.assertEqual(review.status, "accepted")

    def test_accept_blocks_create_before_scenario_entry_activity(self):
        invalid = valid_question_proposal()
        invalid["insert_location"] = "before"
        proposal = self.create_proposal(invalid, proposal_type="create")

        response = self.client.post(
            reverse("accept_proposal", args=[self.scenario.id, proposal.id])
        )

        self.assertEqual(response.status_code, 302)
        review = UserProposalReview.objects.get(proposal=proposal, user=self.user)
        self.assertEqual(review.status, "new")

    def test_only_first_activity_requires_create_after(self):
        self.assertTrue(proposal_requires_insert_after(self.activity, "create"))

        later_activity = Activity.objects.create(
            name="Later question",
            text="Later stem",
            plain_text="Later stem",
            scenario=self.scenario,
            phase=self.phase,
            activity_type=self.activity.activity_type,
            created_by=self.user,
            updated_by=self.user,
        )

        self.assertFalse(proposal_requires_insert_after(later_activity, "create"))
        self.assertFalse(proposal_requires_insert_after(self.activity, "revise"))

    def test_undo_accept_resets_review_to_pending(self):
        proposal = self.create_proposal(
            valid_question_proposal(action="revise")
        )
        self.client.post(
            reverse("accept_proposal", args=[self.scenario.id, proposal.id])
        )

        response = self.client.post(
            reverse(
                "reset_proposal_review",
                args=[self.scenario.id, proposal.id],
            )
        )

        self.assertEqual(response.status_code, 302)
        review = UserProposalReview.objects.get(
            proposal=proposal,
            user=self.user,
        )
        self.assertEqual(review.status, "new")

    def test_undo_reject_resets_metadata_but_preserves_teacher_edits(self):
        proposal = self.create_proposal(
            valid_question_proposal(action="revise")
        )
        review = UserProposalReview.objects.create(
            proposal=proposal,
            user=self.user,
            teacher_edited_json={"content": "Teacher revision"},
        )
        self.client.post(
            reverse("reject_proposal", args=[self.scenario.id, proposal.id]),
            {"rejection_reasons": ["structural_invalid"]},
        )

        self.client.post(
            reverse(
                "reset_proposal_review",
                args=[self.scenario.id, proposal.id],
            )
        )

        review.refresh_from_db()
        self.assertEqual(review.status, "new")
        self.assertEqual(review.rejection_reasons, [])
        self.assertEqual(review.feedback_type, "pedagogical")
        self.assertEqual(
            review.teacher_edited_json,
            {"content": "Teacher revision"},
        )

    def test_reset_review_requires_post(self):
        proposal = self.create_proposal(
            valid_question_proposal(action="revise")
        )
        UserProposalReview.objects.create(
            proposal=proposal,
            user=self.user,
            status="accepted",
        )

        response = self.client.get(
            reverse(
                "reset_proposal_review",
                args=[self.scenario.id, proposal.id],
            )
        )

        self.assertEqual(response.status_code, 405)


class ScenarioStartActivityTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("start_owner", password="pass")
        self.scenario = Scenario.objects.create(
            name="Explicit start scenario",
            created_by=self.user,
            updated_by=self.user,
        )
        self.phase = Phase.objects.create(
            name="Phase",
            scenario=self.scenario,
            created_by=self.user,
            updated_by=self.user,
        )
        self.activity_type = ActivityType.objects.create(
            name="Explanation",
            created_by=self.user,
            updated_by=self.user,
        )

    def create_activity(self, name):
        return Activity.objects.create(
            name=name,
            text=name,
            plain_text=name,
            scenario=self.scenario,
            phase=self.phase,
            activity_type=self.activity_type,
            created_by=self.user,
            updated_by=self.user,
        )

    def test_first_created_activity_initializes_explicit_start(self):
        first = self.create_activity("First")
        self.create_activity("Second")

        self.scenario.refresh_from_db()

        self.assertEqual(self.scenario.start_activity, first)

    def test_explicit_start_controls_entry_proposal_rule(self):
        first = self.create_activity("First")
        second = self.create_activity("Second")
        self.scenario.start_activity = second
        self.scenario.save(update_fields=["start_activity"])

        self.assertFalse(proposal_requires_insert_after(first, "create"))
        self.assertTrue(proposal_requires_insert_after(second, "create"))

    def test_deleting_start_selects_a_remaining_entry(self):
        first = self.create_activity("First")
        second = self.create_activity("Second")
        self.scenario.refresh_from_db()

        first.delete()
        self.scenario.refresh_from_db()

        self.assertEqual(self.scenario.start_activity, second)


class ScenarioGraphIntegrityTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("graph_owner", password="pass")
        self.scenario = Scenario.objects.create(
            name="Graph scenario",
            created_by=self.user,
            updated_by=self.user,
        )
        self.phase = Phase.objects.create(
            name="Phase",
            scenario=self.scenario,
            created_by=self.user,
            updated_by=self.user,
        )
        self.explanation_type = ActivityType.objects.create(
            name="Explanation",
            created_by=self.user,
            updated_by=self.user,
        )
        self.question_type = ActivityType.objects.create(
            name="Question",
            created_by=self.user,
            updated_by=self.user,
        )

    def activity(self, name, activity_type=None, is_evaluatable=False):
        return Activity.objects.create(
            name=name,
            text=name,
            plain_text=name,
            scenario=self.scenario,
            phase=self.phase,
            activity_type=activity_type or self.explanation_type,
            is_evaluatable=is_evaluatable,
            created_by=self.user,
            updated_by=self.user,
        )

    def issue_codes(self):
        self.scenario.refresh_from_db()
        return {
            issue["code"]
            for issue in validate_scenario_graph(self.scenario)
        }

    def test_valid_linear_graph_has_no_issues(self):
        first = self.activity("First")
        second = self.activity("Second")
        NextQuestionLogic.objects.create(
            activity=first,
            next_activity=second,
        )

        self.assertEqual(self.issue_codes(), set())

    def test_unreachable_activity_is_reported(self):
        self.activity("Reachable start")
        self.activity("Orphan")

        self.assertIn("unreachable_activity", self.issue_codes())

    def test_missing_answer_route_is_reported(self):
        question = self.activity("Question", self.question_type)
        target = self.activity("Target")
        answers = [
            Answer.objects.create(
                activity=question,
                text=f"{label}. Answer",
                is_correct=(label == "A"),
                answer_weight=3 if label == "A" else 1,
                created_by=self.user,
                updated_by=self.user,
            )
            for label in ("A", "B")
        ]
        NextQuestionLogic.objects.create(
            activity=question,
            answer=answers[0],
            next_activity=target,
        )

        self.assertIn("missing_answer_routes", self.issue_codes())

    def test_incomplete_branching_is_reported(self):
        evaluation = self.activity(
            "Evaluation",
            self.question_type,
            is_evaluatable=True,
        )
        target = self.activity("Target")
        EvQuestionBranching.objects.create(
            activity=evaluation,
            next_question_on_high=target,
        )

        self.assertIn("incomplete_branching", self.issue_codes())

    def test_cycle_is_reported(self):
        first = self.activity("First")
        second = self.activity("Second")
        NextQuestionLogic.objects.create(
            activity=first,
            next_activity=second,
        )
        NextQuestionLogic.objects.create(
            activity=second,
            next_activity=first,
        )

        self.assertIn("activity_cycle", self.issue_codes())

    def test_personal_clone_preserves_explicit_start(self):
        first = self.activity("First")
        second = self.activity("Second")
        NextQuestionLogic.objects.create(
            activity=first,
            next_activity=second,
        )

        cloned_id = _build_personal_scenario(self.scenario.id, self.user.id)
        clone = Scenario.objects.get(pk=cloned_id)

        self.assertEqual(clone.start_activity.name, "First")

    def test_invalid_graph_rolls_back_clone_and_records_failure(self):
        self.activity("Start")
        self.activity("Unreachable")

        result = apply_user_proposals_to_new_scenario.run(
            self.scenario.id,
            self.user.id,
        )

        self.assertEqual(result, "scenario error")
        self.assertFalse(
            Scenario.objects.filter(origin_scenario=self.scenario).exists()
        )
        failure = ProposalStructuralFailure.objects.get(
            scenario=self.scenario,
            stage="graph_integrity",
        )
        self.assertTrue(
            any(
                issue["code"] == "unreachable_activity"
                for issue in failure.errors
            )
        )


class StructuralFeedbackTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("structure_owner", password="pass")
        self.scenario = Scenario.objects.create(
            name="Structural feedback scenario",
            created_by=self.user,
            updated_by=self.user,
            use_family_evidence_pooling=True,
        )
        self.phase = Phase.objects.create(
            name="Phase",
            scenario=self.scenario,
            created_by=self.user,
            updated_by=self.user,
        )
        activity_type = ActivityType.objects.create(
            name="Explanation",
            created_by=self.user,
            updated_by=self.user,
        )
        self.activity = Activity.objects.create(
            name="Activity",
            text="Content",
            plain_text="Content",
            scenario=self.scenario,
            phase=self.phase,
            activity_type=activity_type,
            created_by=self.user,
            updated_by=self.user,
        )
        evidence_context = get_evidence_context(
            self.scenario,
            'compatible',
        )
        self.run = ProposalGenerationRun.start_new(
            self.scenario,
            self.user,
            evidence_scope='compatible',
            evidence_version_ids=evidence_context['version_ids'],
            evidence_summary=evidence_context,
        )
        raw = json.dumps({
            "action": "revise",
            "activity_name": "Activity",
            "activity_type": "Explanation",
            "content": "Content",
            "answers": [],
            "insert_location": "after",
            "explanation": "Explanation",
        })
        self.proposal = ActivityProposal.objects.create(
            scenario=self.scenario,
            generation_run=self.run,
            phase=self.phase,
            activity=self.activity,
            proposal_type="revise",
            suggested_action=raw,
            translated_action=raw,
            json_action=raw,
            json_translated_action=raw,
        )
        self.flag = ActivityFlag.objects.create(
            activity=self.activity,
            scenario=self.scenario,
            phase=self.phase,
            category="Low",
            flag_type="Systemic failure",
            flag_reason="Test",
        )
        self.proposal.flag.add(self.flag)

    def test_structural_rejection_does_not_update_bandit(self):
        review = UserProposalReview.objects.create(
            proposal=self.proposal,
            user=self.user,
        )

        with self.captureOnCommitCallbacks(execute=True):
            review.reject(reasons=["structural_invalid"])

        review.refresh_from_db()
        self.assertEqual(review.feedback_type, "structural")
        self.assertFalse(QValue.objects.exists())

    def test_reset_to_pending_removes_previous_bandit_reward(self):
        review = UserProposalReview.objects.create(
            proposal=self.proposal,
            user=self.user,
        )
        with self.captureOnCommitCallbacks(execute=True):
            review.accept()
        q_value = QValue.objects.get(
            flag_type=self.flag.flag_type,
            category=self.flag.category,
            action=self.proposal.proposal_type,
        )
        self.assertEqual(q_value.reward_count, 1)
        self.assertEqual(q_value.reward_sum, 1)

        with self.captureOnCommitCallbacks(execute=True):
            review.reset_to_pending()

        q_value.refresh_from_db()
        self.assertEqual(review.status, "new")
        self.assertEqual(q_value.reward_count, 0)
        self.assertEqual(q_value.reward_sum, 0)
        self.assertEqual(q_value.q_value, 0)

    def test_acceptance_validation_failure_is_recorded_separately(self):
        client = Client()
        teachers, _ = Group.objects.get_or_create(name="teachers")
        self.user.groups.add(teachers)
        client.login(username="structure_owner", password="pass")
        invalid = json.loads(self.proposal.json_action)
        invalid["content"] = ""
        self.proposal.json_action = json.dumps(invalid)
        self.proposal.json_translated_action = self.proposal.json_action
        self.proposal.save(
            update_fields=["json_action", "json_translated_action"]
        )

        client.post(
            reverse(
                "accept_proposal",
                args=[self.scenario.id, self.proposal.id],
            )
        )

        failure = ProposalStructuralFailure.objects.get(
            proposal=self.proposal,
            stage="acceptance",
        )
        self.assertIn("content must be a non-empty string", failure.errors)

    def test_teacher_view_shows_create_insertion_preview(self):
        teachers, _ = Group.objects.get_or_create(name="teachers")
        self.user.groups.add(teachers)
        data = {
            "action": "create",
            "activity_name": "New support activity",
            "activity_type": "Explanation",
            "content": "Support content",
            "answers": [],
            "insert_location": "after",
            "explanation": "Support is useful.",
        }
        raw = json.dumps(data)
        self.proposal.proposal_type = "create"
        self.proposal.suggested_action = raw
        self.proposal.translated_action = raw
        self.proposal.json_action = raw
        self.proposal.json_translated_action = raw
        self.proposal.save()
        client = Client()
        client.login(username="structure_owner", password="pass")

        response = client.get(
            reverse("proposal_list", args=[self.scenario.id])
        )

        self.assertContains(response, "Insertion Preview")
        self.assertContains(response, "New support activity")
        self.assertContains(response, "End")
