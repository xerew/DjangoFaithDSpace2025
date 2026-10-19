"""Applying a "skip" proposal reroutes everything that led to the skipped
activity, so the personalised scenario's flow stays complete."""
from django.contrib.auth.models import User
from django.test import TestCase

from authoringtool.graph_validation import (
    ScenarioGraphValidationError,
    validate_scenario_graph,
)
from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    Scenario,
)
from authoringtool.tasks import skip_activity_in_flow


class SkipActivityInFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('skip_owner')
        self.scenario = Scenario.objects.create(
            name='Skip flow', created_by=self.user, updated_by=self.user,
        )
        self.phase = Phase.objects.create(
            name='Phase', scenario=self.scenario,
            created_by=self.user, updated_by=self.user,
        )
        self.types = {
            name: ActivityType.objects.create(
                name=name, created_by=self.user, updated_by=self.user,
            )
            for name in ('Question', 'Explanation')
        }

    def activity(self, name, kind='Explanation', answers=0, evaluated=False):
        activity = Activity.objects.create(
            name=name, text=name, plain_text=name, scenario=self.scenario,
            phase=self.phase, activity_type=self.types[kind],
            is_evaluatable=evaluated, is_primary_ev=evaluated,
            created_by=self.user, updated_by=self.user,
        )
        for number in range(answers):
            Answer.objects.create(
                activity=activity, text=f'{name} answer {number}',
                is_correct=number == 0, answer_weight=number + 1,
            )
        return activity

    def route(self, activity, target, answer=None):
        NextQuestionLogic.objects.create(
            activity=activity, answer=answer, next_activity=target,
        )

    def branch(self, activity, high, mid, low):
        EvQuestionBranching.objects.create(
            activity=activity, next_question_on_high=high,
            next_question_on_mid=mid, next_question_on_low=low,
        )

    def start_at(self, activity):
        Scenario.objects.filter(pk=self.scenario.pk).update(start_activity=activity)
        self.scenario.refresh_from_db()

    def assertFlowValid(self):
        self.assertEqual(validate_scenario_graph(self.scenario), [])

    def test_branch_into_a_question_without_branching_goes_to_its_next_activity(self):
        # The Pendulum case: Activity 7 High -> Activity 8 (no branching) -> Activity 9.
        seven = self.activity('Activity 7', 'Question', answers=2, evaluated=True)
        eight = self.activity('Activity 8', 'Question', answers=2)
        nine = self.activity('Activity 9')
        self.start_at(seven)
        self.branch(seven, high=eight, mid=nine, low=nine)
        for answer in eight.answers.all():
            self.route(eight, nine, answer)

        skip_activity_in_flow(self.scenario, eight)

        seven.branching.refresh_from_db()
        self.assertEqual(seven.branching.next_question_on_high, nine)
        self.assertFlowValid()

    def test_answer_routes_into_the_skipped_activity_are_kept(self):
        start = self.activity('Start', 'Question', answers=2)
        skipped = self.activity('Skipped')
        after = self.activity('After')
        other = self.activity('Other')
        self.start_at(start)
        first, second = start.answers.order_by('id')
        self.route(start, skipped, first)
        self.route(start, other, second)
        self.route(skipped, after)
        self.route(other, after)

        skip_activity_in_flow(self.scenario, skipped)

        self.assertEqual(
            NextQuestionLogic.objects.get(activity=start, answer=first).next_activity, after,
        )
        self.assertEqual(
            NextQuestionLogic.objects.get(activity=start, answer=second).next_activity, other,
        )
        self.assertFlowValid()

    def test_same_level_branch_is_followed_when_the_skipped_activity_has_one(self):
        start = self.activity('Start', 'Question', answers=2, evaluated=True)
        skipped = self.activity('Skipped', 'Question', answers=2, evaluated=True)
        high, mid, low = self.activity('High'), self.activity('Mid'), self.activity('Low')
        self.start_at(start)
        self.branch(start, high=skipped, mid=skipped, low=skipped)
        self.branch(skipped, high=high, mid=mid, low=low)

        skip_activity_in_flow(self.scenario, skipped)

        start.branching.refresh_from_db()
        self.assertEqual(start.branching.next_question_on_high, high)
        self.assertEqual(start.branching.next_question_on_mid, mid)
        self.assertEqual(start.branching.next_question_on_low, low)
        self.assertFlowValid()

    def test_plain_route_into_a_branching_activity_takes_its_main_successor(self):
        start = self.activity('Start')
        skipped = self.activity('Skipped', 'Question', answers=2, evaluated=True)
        common, rare = self.activity('Common'), self.activity('Rare')
        self.start_at(start)
        self.route(start, skipped)
        self.branch(skipped, high=rare, mid=common, low=common)
        # Rare is still reachable another way, so the flow stays valid.
        self.route(common, rare)

        skip_activity_in_flow(self.scenario, skipped)

        self.assertEqual(NextQuestionLogic.objects.get(activity=start).next_activity, common)
        self.assertFlowValid()

    def test_skipping_the_start_moves_the_start_to_its_successor(self):
        first = self.activity('First')
        second = self.activity('Second')
        self.start_at(first)
        self.route(first, second)

        skip_activity_in_flow(self.scenario, first)

        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.start_activity, second)
        self.assertFalse(Activity.objects.filter(pk=first.pk).exists())

    def test_ambiguous_start_is_refused(self):
        first = self.activity('First', 'Question', answers=2)
        a, b = self.activity('A'), self.activity('B')
        self.start_at(first)
        one, two = first.answers.order_by('id')
        self.route(first, a, one)
        self.route(first, b, two)

        with self.assertRaises(ScenarioGraphValidationError):
            skip_activity_in_flow(self.scenario, first)


from unittest.mock import MagicMock, patch

from django.contrib.auth.models import Group
from django.test import override_settings
from django.urls import reverse

from authoringtool.models import ProposalStructuralFailure


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class PersonalScenarioOutcomeTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user('outcome_teacher', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))
        self.scenario = Scenario.objects.create(
            name='Pendulum', created_by=self.teacher, updated_by=self.teacher,
        )
        self.client.force_login(self.teacher)
        self.list_url = reverse('proposal_list', args=[self.scenario.id])

    def create(self):
        task = MagicMock(id='task-1')
        with patch('authoringtool.views.apply_user_proposals_to_new_scenario.delay', return_value=task):
            self.client.post(reverse('create_personal_scenario', args=[self.scenario.id]))

    def visit_list(self, ready, result):
        outcome = MagicMock(ready=MagicMock(return_value=ready), result=result)
        with patch('authoringtool.views.AsyncResult', return_value=outcome):
            return self.client.get(self.list_url, follow=True)

    def test_failure_is_reported_with_its_reason(self):
        self.create()
        ProposalStructuralFailure.objects.create(
            scenario=self.scenario, stage='graph_integrity',
            errors=[{'code': 'incomplete_branching',
                     'message': "Activity 'Activity 7' has incomplete branches: High is missing."}],
        )

        response = self.visit_list(ready=True, result='scenario error')

        self.assertContains(response, 'could not be created')
        self.assertContains(response, 'High is missing')

    def test_success_links_to_the_new_scenario(self):
        copy = Scenario.objects.create(
            name='Pendulum Copy Made by outcome_teacher',
            created_by=self.teacher, updated_by=self.teacher,
        )
        self.create()

        response = self.visit_list(ready=True, result=copy.id)

        self.assertContains(response, 'is ready')
        self.assertContains(response, reverse('viewScenario', args=[copy.id]))

    def test_outcome_is_reported_once(self):
        self.create()
        self.visit_list(ready=True, result='scenario error')

        response = self.visit_list(ready=True, result='scenario error')

        self.assertNotContains(response, 'could not be created')

    def test_unfinished_task_says_nothing_yet(self):
        self.create()

        response = self.visit_list(ready=False, result=None)

        self.assertNotContains(response, 'could not be created')
        self.assertNotContains(response, 'is ready')
