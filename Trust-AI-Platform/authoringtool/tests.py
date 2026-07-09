import csv
import io
import json
import openpyxl

from django.contrib.auth.models import User, Group
from django.test import TestCase, Client
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    AnswerFeedback,
    NextQuestionLogic,
    Phase,
    Scenario,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import compute_student_performance_metrics


def make_xlsx(
    scenario_name='Test Scenario',
    phases=None,
    activities=None,
    answers=None,
    routing=None,
    evaluation=None,
    missing_sheet=None,
):
    """Build a minimal valid .xlsx in memory. Each argument is a list of tuples (row values)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'README'
    ws['A1'] = 'Instructions'

    sheets = {
        'Scenario': (
            ['Name', 'Description', 'Visibility'],
            [[scenario_name, '', 'private']],
        ),
        'Phases': (
            ['Phase Name', 'Description', 'Video URL'],
            phases or [['Phase 1', '', '']],
        ),
        'Activities': (
            ['Activity Name', 'Phase Name', 'Text', 'Activity Type', 'Helper',
             'Is Evaluatable', 'Is Primary Evaluation', 'Must Wait', 'Score Limit',
             'Simulation Name', 'Remote Lab Name', 'VR Lab Name', 'Image URL', 'Video URL'],
            activities or [['Act 1', 'Phase 1', 'Hello world', '', '', 'No', 'No', 'No', '', '', '', '', '', '']],
        ),
        'Answers': (
            ['Activity Name', 'Answer Text', 'Is Correct', 'Answer Weight',
             'Image URL', 'Video URL', 'Feedback Text', 'Feedback Image URL', 'Feedback Video URL'],
            answers or [],
        ),
        'Next Activity': (
            ['Source Activity Name', 'Answer Text', 'Next Activity Name'],
            routing or [],
        ),
        'Evaluation': (
            ['Primary Activity Name', 'Grouped Activities', 'High Branch Activity',
             'High Branch Feedback', 'Mid Branch Activity', 'Mid Branch Feedback',
             'Low Branch Activity', 'Low Branch Feedback'],
            evaluation or [],
        ),
    }

    for name, (header, rows) in sheets.items():
        if name == missing_sheet:
            continue
        ws2 = wb.create_sheet(name)
        ws2.append(header)
        for row in rows:
            ws2.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


class ImporterValidationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('teacher', password='pass')

    def _run(self, **kwargs):
        from authoringtool.importer import ScenarioImporter
        buf = make_xlsx(**kwargs)
        importer = ScenarioImporter(buf, self.user)
        importer._parse()
        if not importer.errors:
            importer._validate()
        return importer.errors

    def test_valid_file_no_errors(self):
        errors = self._run()
        self.assertEqual(errors, [])

    def test_missing_sheet_reported(self):
        errors = self._run(missing_sheet='Activities')
        messages = [e['message'] for e in errors]
        self.assertTrue(any('Activities' in m for m in messages))

    def test_missing_scenario_name(self):
        errors = self._run(scenario_name='')
        messages = [e['message'] for e in errors]
        self.assertTrue(any('Name' in m or 'required' in m.lower() for m in messages))

    def test_duplicate_scenario_name(self):
        # Scenario.objects.create() fails on SQLite because Scenario.age_of_students is an
        # IntegerRangeField (postgres-only).  Mock the DB lookup instead.
        from unittest.mock import patch
        from authoringtool.importer import ScenarioImporter
        buf = make_xlsx(scenario_name='Test Scenario')
        importer = ScenarioImporter(buf, self.user)
        importer._parse()
        with patch('authoringtool.importer.Scenario.objects') as mock_qs:
            mock_qs.filter.return_value.exists.return_value = True
            importer._load_db_lookups()
            importer._validate_scenario()
        messages = [e['message'] for e in importer.errors]
        self.assertTrue(any('already exists' in m for m in messages))

    def test_invalid_visibility(self):
        # build with bad visibility by patching scenario_data after parse
        from authoringtool.importer import ScenarioImporter
        buf = make_xlsx()
        importer = ScenarioImporter(buf, self.user)
        importer._parse()
        importer.scenario_data['Visibility'] = 'invalid'
        importer._load_db_lookups()
        importer._validate_scenario()
        self.assertTrue(any('Visibility' in e['column'] for e in importer.errors))

    def test_activity_references_unknown_phase(self):
        errors = self._run(
            activities=[['Act 1', 'NonexistentPhase', 'Hello', '', '', 'No', 'No', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('Phase' in e['message'] for e in errors))

    def test_duplicate_activity_names(self):
        errors = self._run(
            activities=[
                ['Act 1', 'Phase 1', 'Hello', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
                ['Act 1', 'Phase 1', 'Dupe', '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
            ],
        )
        self.assertTrue(any('Duplicate' in e['message'] for e in errors))

    def test_primary_ev_without_evaluatable(self):
        errors = self._run(
            activities=[['Act 1', 'Phase 1', 'Hello', '', '', 'No', 'Yes', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('Is Evaluatable' in e['message'] or 'Is Primary' in e['message'] for e in errors))

    def test_evaluatable_without_evaluation_row(self):
        errors = self._run(
            activities=[['Quiz', 'Phase 1', 'Q?', '', '', 'Yes', 'Yes', 'No', '', '', '', '', '', '']],
            answers=[['Quiz', 'Option A', 'Yes', '1', '', '', '', '', '']],
            evaluation=[],  # no evaluation row
        )
        self.assertTrue(any('Evaluation' in e['message'] or 'evaluatable' in e['message'].lower() for e in errors))

    def test_answer_references_unknown_activity(self):
        errors = self._run(
            answers=[['NonexistentAct', 'Option A', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('NonexistentAct' in e['message'] for e in errors))

    def test_routing_references_unknown_next_activity(self):
        errors = self._run(
            routing=[['Act 1', '', 'GhostActivity']],
        )
        self.assertTrue(any('GhostActivity' in e['message'] for e in errors))

    def test_routing_answer_not_found(self):
        errors = self._run(
            answers=[['Act 1', 'Option A', 'Yes', '1', '', '', '', '', '']],
            routing=[['Act 1', 'Wrong Answer Text', '']],
        )
        self.assertTrue(any('Wrong Answer Text' in e['message'] for e in errors))

    def test_invalid_boolean_field(self):
        errors = self._run(
            activities=[['Act 1', 'Phase 1', 'Hello', '', '', 'maybe', 'No', 'No', '', '', '', '', '', '']],
        )
        self.assertTrue(any('Yes' in e['message'] or 'No' in e['message'] for e in errors))


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
