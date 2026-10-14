"""Teacher deletes never destroy student evidence.

Scenarios with student data are archived; phases, activities and answer
options can be removed after a warning, but the students' answers are kept,
detached, with a snapshot of what was removed.
"""
from datetime import timedelta

from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from authoringtool.models import (
    Activity,
    ActivityProposal,
    ActivityType,
    Answer,
    Phase,
    RemoteLabSession,
    Scenario,
    UserAnswer,
    UserProposalReview,
    UserScenarioScore,
)
from usergroups.models import UserGroup, UserGroupMembership


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class RetentionBase(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('ret_owner', password='pass')
        self.other_teacher = User.objects.create_user('ret_other', password='pass')
        for teacher in (self.owner, self.other_teacher):
            teacher.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Retention', visibility_status='public',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Explore', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        question = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Pendulum question', text='Which is faster?', phase=self.phase,
            scenario=self.scenario, activity_type=question,
            created_by=self.owner, updated_by=self.owner,
        )
        self.right = Answer.objects.create(
            activity=self.activity, text='The short one', is_correct=True,
            answer_weight=3, created_by=self.owner, updated_by=self.owner,
        )
        self.wrong = Answer.objects.create(
            activity=self.activity, text='The heavy one', is_correct=False,
            answer_weight=0, created_by=self.owner, updated_by=self.owner,
        )
        self.group = UserGroup.objects.create(
            name='Retention class', prefix='ret', number_of_users=2, created_by=self.owner,
        )
        self.group.assigned_scenarios.add(self.scenario)
        self.students = []
        for i, answer in enumerate((self.right, self.wrong)):
            student = User.objects.create_user(f'ret{i + 1}', password='pass')
            UserGroupMembership.objects.create(group=self.group, user=student)
            UserAnswer.objects.create(user=student, activity=self.activity, answer=answer, timing=40)
            UserScenarioScore.objects.create(user=student, scenario=self.scenario, user_score=3)
            self.students.append(student)
        self.client.force_login(self.owner)


class ScenarioDeleteTests(RetentionBase):
    def test_get_shows_confirmation_and_changes_nothing(self):
        response = self.client.get(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '2 students')
        self.scenario.refresh_from_db()
        self.assertIsNone(self.scenario.archived_at)

    def test_scenario_with_student_data_is_archived_not_deleted(self):
        self.client.post(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        self.scenario.refresh_from_db()
        self.assertIsNotNone(self.scenario.archived_at)
        self.assertEqual(UserAnswer.objects.filter(activity=self.activity).count(), 2)

    def test_scenario_without_student_data_is_deleted(self):
        empty = Scenario.objects.create(
            name='Empty', created_by=self.owner, updated_by=self.owner,
        )
        self.client.post(reverse('deleteScenario', kwargs={'id': empty.id}))
        self.assertFalse(Scenario.objects.filter(pk=empty.pk).exists())

    def test_archived_scenario_disappears_from_lists_and_access(self):
        self.client.post(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        listed = self.client.get(reverse('scenarios'))
        self.assertNotContains(listed, 'Retention</')
        dashboard = self.client.get(reverse('index'))
        self.assertNotIn(self.scenario, list(dashboard.context['scenarios']))
        self.client.force_login(self.other_teacher)
        self.assertEqual(
            self.client.get(reverse('viewScenario', args=[self.scenario.id])).status_code, 403,
        )
        self.client.force_login(self.students[0])
        self.assertEqual(
            self.client.get(reverse('studentView', args=[self.scenario.id])).status_code, 404,
        )
        student_list = self.client.get(reverse('studentScenarios'))
        self.assertNotContains(student_list, 'Retention</')

    def test_owner_restores_archived_scenario(self):
        self.client.post(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        archived = self.client.get(reverse('archived_scenarios'))
        self.assertIn(self.scenario, list(archived.context['scenarios']))
        self.client.post(reverse('restore_scenario', args=[self.scenario.id]))
        self.scenario.refresh_from_db()
        self.assertIsNone(self.scenario.archived_at)

    def test_other_teacher_cannot_delete_or_restore(self):
        self.client.force_login(self.other_teacher)
        self.client.post(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        self.scenario.refresh_from_db()
        self.assertIsNone(self.scenario.archived_at)
        self.scenario.archived_at = self.scenario.updated_on
        self.scenario.save(update_fields=['archived_at'])
        response = self.client.post(reverse('restore_scenario', args=[self.scenario.id]))
        self.assertEqual(response.status_code, 403)


class ActivityDeleteTests(RetentionBase):
    def setUp(self):
        super().setUp()
        # A scenario must keep at least one phase.
        Phase.objects.create(
            name='Explain', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )

    def url(self, name):
        if name == 'deleteActivity':
            return reverse(name, args=[self.scenario.id, self.phase.id, self.activity.id])
        if name == 'deletePhase':
            return reverse(name, args=[self.scenario.id, self.phase.id])
        return reverse(name, args=[self.scenario.id, self.phase.id, self.activity.id, self.wrong.id])

    def test_get_warns_how_many_students_answered(self):
        for name in ('deleteActivity', 'deletePhase', 'deleteAnswer'):
            with self.subTest(view=name):
                response = self.client.get(self.url(name))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'student')
        self.assertTrue(Activity.objects.filter(pk=self.activity.pk).exists())

    def test_confirmed_activity_delete_keeps_answers_detached(self):
        self.client.post(self.url('deleteActivity'))
        self.assertFalse(Activity.objects.filter(pk=self.activity.pk).exists())
        kept = UserAnswer.objects.filter(user__in=self.students).order_by('user__username')
        self.assertEqual(kept.count(), 2)
        first = kept[0]
        self.assertIsNone(first.activity_id)
        self.assertEqual(first.detached_from['activity']['name'], 'Pendulum question')
        self.assertEqual(first.detached_from['activity']['scenario_id'], self.scenario.id)
        self.assertEqual(first.detached_from['answer']['text'], 'The short one')
        self.assertTrue(first.detached_from['answer']['is_correct'])
        self.assertEqual(first.detached_from['detached_by'], self.owner.id)

    def test_confirmed_phase_delete_keeps_answers_detached(self):
        self.client.post(self.url('deletePhase'))
        self.assertFalse(Phase.objects.filter(pk=self.phase.pk).exists())
        kept = UserAnswer.objects.filter(user__in=self.students)
        self.assertEqual(kept.count(), 2)
        self.assertTrue(all(a.detached_from['activity']['phase'] == 'Explore' for a in kept))

    def test_confirmed_answer_option_delete_keeps_what_the_student_chose(self):
        self.client.post(self.url('deleteAnswer'))
        self.assertFalse(Answer.objects.filter(pk=self.wrong.pk).exists())
        chose_wrong = UserAnswer.objects.get(user=self.students[1])
        self.assertIsNone(chose_wrong.answer_id)
        self.assertEqual(chose_wrong.activity_id, self.activity.id)
        self.assertEqual(chose_wrong.detached_from['answer']['text'], 'The heavy one')

    def test_removing_an_option_in_the_answer_editor_keeps_the_choice(self):
        self.client.post(
            reverse('updateAnswers', args=[self.scenario.id, self.phase.id, self.activity.id]),
            {
                'answers[0][id]': self.right.id, 'answers[0][text]': self.right.text,
                'answers[0][is_correct]': 'on', 'answers[0][answer_weight]': 3,
            },
        )
        self.assertFalse(Answer.objects.filter(pk=self.wrong.pk).exists())
        chose_wrong = UserAnswer.objects.get(user=self.students[1])
        self.assertEqual(chose_wrong.detached_from['answer']['text'], 'The heavy one')

    def test_get_never_deletes(self):
        self.client.get(self.url('deleteActivity'))
        self.client.get(self.url('deletePhase'))
        self.client.get(self.url('deleteAnswer'))
        self.assertTrue(Activity.objects.filter(pk=self.activity.pk).exists())
        self.assertTrue(Answer.objects.filter(pk=self.wrong.pk).exists())


class TeacherResearchDataRetentionTests(RetentionBase):
    """AI proposals, teacher reviews and lab sessions survive content removal."""

    def setUp(self):
        super().setUp()
        Phase.objects.create(
            name='Explain', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        self.proposal = ActivityProposal.objects.create(
            scenario=self.scenario, phase=self.phase, activity=self.activity,
            proposal_type='revise', suggested_action='{}', translated_action='{}',
            json_action='{}', json_translated_action='{}',
        )
        self.review = UserProposalReview.objects.create(
            proposal=self.proposal, user=self.owner, status='accepted',
        )
        self.lab = RemoteLabSession.objects.create(
            activity=self.activity, phase=self.phase, scenario=self.scenario,
            user=self.students[0], start=timezone.now(), end=timezone.now(),
            lab_name='Pendulum', pre_duration=timedelta(seconds=5),
            exec_duration=timedelta(seconds=30), mass='light', angle=20, iteration=1,
        )

    def test_activity_delete_keeps_proposals_reviews_and_lab_sessions(self):
        self.client.post(reverse(
            'deleteActivity', args=[self.scenario.id, self.phase.id, self.activity.id],
        ))
        self.proposal.refresh_from_db()
        self.assertIsNone(self.proposal.activity_id)
        self.assertEqual(self.proposal.detached_from['activity']['name'], 'Pendulum question')
        self.assertTrue(UserProposalReview.objects.filter(pk=self.review.pk).exists())
        self.lab.refresh_from_db()
        self.assertIsNone(self.lab.activity_id)
        self.assertEqual(self.lab.detached_from['activity']['name'], 'Pendulum question')

    def test_phase_delete_keeps_proposals(self):
        self.client.post(reverse('deletePhase', args=[self.scenario.id, self.phase.id]))
        self.proposal.refresh_from_db()
        self.assertIsNone(self.proposal.phase_id)
        self.assertEqual(self.proposal.detached_from['activity']['phase'], 'Explore')

    def test_detached_proposals_leave_the_review_list(self):
        self.client.post(reverse(
            'deleteActivity', args=[self.scenario.id, self.phase.id, self.activity.id],
        ))
        response = self.client.get(reverse('proposal_list', args=[self.scenario.id]))
        self.assertEqual(response.status_code, 200)

    def test_scenario_with_proposals_but_no_students_is_archived(self):
        quiet = Scenario.objects.create(
            name='Pooled only', created_by=self.owner, updated_by=self.owner,
        )
        quiet_phase = Phase.objects.create(
            name='P', scenario=quiet, created_by=self.owner, updated_by=self.owner,
        )
        quiet_activity = Activity.objects.create(
            name='Q', text='q', phase=quiet_phase, scenario=quiet,
            activity_type=self.activity.activity_type,
            created_by=self.owner, updated_by=self.owner,
        )
        ActivityProposal.objects.create(
            scenario=quiet, phase=quiet_phase, activity=quiet_activity,
            proposal_type='skip', suggested_action='{}', translated_action='{}',
            json_action='{}', json_translated_action='{}',
        )
        self.client.post(reverse('deleteScenario', kwargs={'id': quiet.id}))
        quiet.refresh_from_db()
        self.assertIsNotNone(quiet.archived_at)
