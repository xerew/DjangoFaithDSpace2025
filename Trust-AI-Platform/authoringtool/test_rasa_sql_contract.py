"""Rasa writes student answers with raw SQL (RasaFaith/actions/actions.py).

These tests replay those exact INSERT statements against the real schema, so
a model change that adds a NOT NULL column without a database default (which
would make every chat answer fail to save) is caught here.
"""
from django.contrib.auth.models import User
from django.db import connection
from django.test import TestCase
from django.utils import timezone

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    Scenario,
    UserAnswer,
    UserScenarioScore,
)


class RasaRawInsertContractTests(TestCase):
    def setUp(self):
        owner = User.objects.create_user('sql_owner')
        self.student = User.objects.create_user('sql_student')
        self.scenario = Scenario.objects.create(name='SQL', created_by=owner, updated_by=owner)
        phase = Phase.objects.create(name='P', scenario=self.scenario, created_by=owner, updated_by=owner)
        question = ActivityType.objects.create(name='Question', created_by=owner, updated_by=owner)
        self.activity = Activity.objects.create(
            name='Q', text='q', phase=phase, scenario=self.scenario,
            activity_type=question, created_by=owner, updated_by=owner,
        )
        self.answer = Answer.objects.create(
            activity=self.activity, text='A', is_correct=True,
            created_by=owner, updated_by=owner,
        )

    def test_handle_answer_insert(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO authoringtool_useranswer "
                "(user_id, activity_id, answer_id, timing, created_on, "
                "scenario_version_id, version_confidence, implementation_id, "
                "activity_revision_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, "
                "(SELECT id FROM authoringtool_activityrevision "
                " WHERE scenario_version_id = %s AND activity_id = %s "
                " ORDER BY id DESC LIMIT 1))",
                (self.student.id, self.activity.id, self.answer.id, None,
                 timezone.now(), None, 'exact', None, None, self.activity.id),
            )
        saved = UserAnswer.objects.get(user=self.student)
        self.assertIsNone(saved.timing)
        self.assertFalse(saved.detached_from)

    def test_confirm_read_insert(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO authoringtool_useranswer "
                "(user_id, activity_id, timing, created_on, "
                "scenario_version_id, version_confidence, implementation_id, "
                "activity_revision_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, "
                "(SELECT id FROM authoringtool_activityrevision "
                " WHERE scenario_version_id = %s AND activity_id = %s "
                " ORDER BY id DESC LIMIT 1))",
                (self.student.id, self.activity.id, 12, timezone.now(),
                 None, 'exact', None, None, self.activity.id),
            )
        self.assertEqual(UserAnswer.objects.get(user=self.student).timing, 12)

    def test_score_inserts(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO authoringtool_userscenarioscore "
                "(user_id, last_activity_id, scenario_id, user_score, "
                "scenario_version_id, version_confidence, "
                "data_quality_status, implementation_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'unreviewed', %s)",
                (self.student.id, self.activity.id, self.scenario.id, 3, None, 'exact', None),
            )
        self.assertTrue(UserScenarioScore.objects.filter(user=self.student).exists())
