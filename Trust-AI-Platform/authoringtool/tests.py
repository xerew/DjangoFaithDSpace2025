import csv
import io

from django.contrib.auth.models import User
from django.test import TestCase

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    Scenario,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import compute_student_performance_metrics


class PerActivityCSVColumnTest(TestCase):
    """
    Verify that compute_student_performance_metrics with include_activity_detail=True
    produces a CSV whose header contains the expected per-activity column names and
    whose header length matches every data row.
    """

    def setUp(self):
        # --- Admin / default user required by SET_DEFAULT FKs ---
        self.admin = User.objects.create_user(username="admin", password="admin")

        # --- School department so the user passes the empty-group_ids filter ---
        dept = SchoolDepartment.objects.create(name="Test Dept")

        # --- Student user ---
        self.student = User.objects.create_user(
            username="student1", password="pass"
        )
        self.student.school_department = dept
        self.student.save()

        # --- Scenario ---
        self.scenario = Scenario.objects.create(
            name="Test Scenario",
            suggested_learning_time=60,
            created_by=self.admin,
            updated_by=self.admin,
        )

        # --- Phase ---
        self.phase = Phase.objects.create(
            name="Phase 1",
            scenario=self.scenario,
            created_by=self.admin,
            updated_by=self.admin,
        )

        # --- Activity types ---
        at_read = ActivityType.objects.create(
            name="Read Text",
            created_by=self.admin,
            updated_by=self.admin,
        )
        at_quiz = ActivityType.objects.create(
            name="Quiz",
            created_by=self.admin,
            updated_by=self.admin,
        )

        # --- Activities ---
        # Non-evaluatable reading activity
        self.act_read = Activity.objects.create(
            name="Read Text",
            text="Read this passage.",
            is_evaluatable=False,
            is_primary_ev=False,
            phase=self.phase,
            scenario=self.scenario,
            activity_type=at_read,
            created_by=self.admin,
            updated_by=self.admin,
        )
        # Evaluatable quiz activity (primary evaluatable)
        self.act_quiz = Activity.objects.create(
            name="Quiz",
            text="Answer this question.",
            is_evaluatable=True,
            is_primary_ev=True,
            phase=self.phase,
            scenario=self.scenario,
            activity_type=at_quiz,
            created_by=self.admin,
            updated_by=self.admin,
        )

        # --- Answers for the quiz activity ---
        self.answer_correct = Answer.objects.create(
            activity=self.act_quiz,
            text="Correct answer",
            is_correct=True,
            answer_weight=10,
            created_by=self.admin,
            updated_by=self.admin,
        )
        Answer.objects.create(
            activity=self.act_quiz,
            text="Wrong answer",
            is_correct=False,
            answer_weight=0,
            created_by=self.admin,
            updated_by=self.admin,
        )

        # --- UserScenarioScore: required for empty group_ids lookup ---
        UserScenarioScore.objects.create(
            user=self.student,
            scenario=self.scenario,
            user_score=10,
        )

        # --- UserAnswers ---
        UserAnswer.objects.create(
            user=self.student,
            activity=self.act_read,
            answer=None,
            timing=30,
        )
        UserAnswer.objects.create(
            user=self.student,
            activity=self.act_quiz,
            answer=self.answer_correct,
            timing=45,
        )

    def test_per_activity_header_names_and_column_count(self):
        result = compute_student_performance_metrics(
            scenario_id=self.scenario.id,
            group_ids=[],
            start_date=None,
            end_date=None,
            include_activity_detail=True,
        )

        self.assertNotIn("error", result, msg=f"Task returned error: {result}")
        self.assertIn("csv_content", result)

        reader = csv.reader(io.StringIO(result["csv_content"]))
        rows = list(reader)
        self.assertGreaterEqual(len(rows), 2, "Expected at least a header and one data row")

        header = rows[0]
        data_row = rows[1]

        # --- Assert per-activity column names ---
        # Activities are sorted by id; act_read was created first so it comes first.
        expected_cols = [
            "Phase 1 > Read Text Type",
            "Phase 1 > Read Text Time (s)",
            "Phase 1 > Read Text Score",
            "Phase 1 > Quiz Type",
            "Phase 1 > Quiz Time (s)",
            "Phase 1 > Quiz Score",
        ]
        for col in expected_cols:
            self.assertIn(
                col,
                header,
                msg=f"Expected column '{col}' not found in header: {header}",
            )

        # --- Assert header and data row lengths match ---
        self.assertEqual(
            len(header),
            len(data_row),
            msg=(
                f"Header has {len(header)} columns but data row has {len(data_row)} columns.\n"
                f"Header: {header}\nData: {data_row}"
            ),
        )

        # --- Assert Quiz Score cell value ---
        quiz_score_idx = header.index("Phase 1 > Quiz Score")
        self.assertEqual(data_row[quiz_score_idx], "10", "Quiz score should equal the correct answer weight")
