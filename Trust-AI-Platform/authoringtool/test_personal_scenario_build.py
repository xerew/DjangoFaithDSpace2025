"""Building a personalised scenario from accepted proposals keeps every
path students could take, and is not blocked by problems the original
scenario already had."""
import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from authoringtool.graph_validation import validate_scenario_graph
from authoringtool.models import (
    Activity,
    ActivityProposal,
    ActivityType,
    Answer,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    ProposalStructuralFailure,
    Scenario,
    UserProposalReview,
)
from authoringtool.tasks import (
    _build_personal_scenario,
    apply_proposals_to_cloned_scenario,
    apply_user_proposals_to_new_scenario,
)


class PersonalScenarioBuildBase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('build_owner')
        self.scenario = Scenario.objects.create(
            name='Pendulum', created_by=self.user, updated_by=self.user,
        )
        self.phase = Phase.objects.create(
            name='Phase', scenario=self.scenario,
            created_by=self.user, updated_by=self.user,
        )
        self.types = {
            name: ActivityType.objects.create(name=name, created_by=self.user, updated_by=self.user)
            for name in ('Question', 'Explanation', 'Experiment')
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
                is_correct=number == 0, answer_weight=3 if number == 0 else 1,
            )
        return activity

    def route(self, activity, target):
        NextQuestionLogic.objects.create(activity=activity, next_activity=target)

    def start_at(self, activity):
        Scenario.objects.filter(pk=self.scenario.pk).update(start_activity=activity)
        self.scenario.refresh_from_db()


class CreateAfterBranchingActivityTests(PersonalScenarioBuildBase):
    """The Pendulum case: Activity 6 sends High, Moderate and Low students
    to three different experiments."""

    def setUp(self):
        super().setUp()
        self.six = self.activity('Activity 6', 'Question', answers=3, evaluated=True)
        self.high = self.activity('Experiment 2')
        self.mid = self.activity('Experiment after activity 6 - 1')
        self.low = self.activity('Experiment after activity 6 - 2')
        self.end = self.activity('End')
        self.start_at(self.six)
        EvQuestionBranching.objects.create(
            activity=self.six, next_question_on_high=self.high,
            next_question_on_mid=self.mid, next_question_on_low=self.low,
        )
        for target in (self.high, self.mid, self.low):
            self.route(target, self.end)

    def apply(self, target_category, kind='Question'):
        answers = [
            {'text': 'A. Longer', 'is_correct': True, 'weight': 3},
            {'text': 'B. Shorter', 'is_correct': False, 'weight': 1},
        ] if kind == 'Question' else []
        payload = json.dumps({
            'action': 'create', 'activity_name': 'Period Analysis', 'activity_type': kind,
            'content': 'Look at your data.', 'answers': answers,
            'insert_location': 'after', 'explanation': 'Helps.',
            'target_category': target_category,
        })
        proposal = ActivityProposal.objects.create(
            scenario=self.scenario, phase=self.phase, activity=self.six,
            proposal_type='create', json_action=payload, json_translated_action=payload,
        )
        review = UserProposalReview.objects.create(proposal=proposal, user=self.user, status='accepted')
        mapping = {activity.id: activity for activity in self.scenario.activities.all()}
        with patch('authoringtool.tasks.get_accepted_reviews_for_personal_scenario', return_value=[review]), \
                patch('authoringtool.tasks.refactor_all_proposals_to_json'):
            apply_proposals_to_cloned_scenario(self.scenario, self.scenario, mapping, self.user)
        self.six.branching.refresh_from_db()

    def assertLeadsTo(self, activity, target):
        targets = {logic.next_activity_id for logic in activity.next_logic.all()}
        self.assertEqual(targets, {target.id})

    def test_every_level_keeps_its_own_path(self):
        self.apply(['All'])

        branching = self.six.branching
        self.assertEqual(branching.next_question_on_high.name, 'Period Analysis (High)')
        self.assertEqual(branching.next_question_on_mid.name, 'Period Analysis (Moderate)')
        self.assertEqual(branching.next_question_on_low.name, 'Period Analysis (Low)')
        self.assertLeadsTo(branching.next_question_on_high, self.high)
        self.assertLeadsTo(branching.next_question_on_mid, self.mid)
        self.assertLeadsTo(branching.next_question_on_low, self.low)
        self.assertEqual(validate_scenario_graph(self.scenario), [])

    def test_levels_sharing_a_path_share_one_activity(self):
        EvQuestionBranching.objects.filter(activity=self.six).update(next_question_on_low=self.mid)
        Activity.objects.filter(pk=self.low.pk).delete()

        self.apply(['All'], kind='Explanation')

        branching = self.six.branching
        self.assertEqual(branching.next_question_on_mid_id, branching.next_question_on_low_id)
        self.assertEqual(branching.next_question_on_mid.name, 'Period Analysis (Moderate, Low)')
        self.assertLeadsTo(branching.next_question_on_mid, self.mid)
        self.assertLeadsTo(branching.next_question_on_high, self.high)
        self.assertEqual(validate_scenario_graph(self.scenario), [])

    def test_a_single_level_keeps_its_name(self):
        self.apply(['Low'])

        branching = self.six.branching
        self.assertEqual(branching.next_question_on_low.name, 'Period Analysis')
        self.assertEqual(branching.next_question_on_mid, self.mid)
        self.assertEqual(validate_scenario_graph(self.scenario), [])


class OriginalScenarioProblemsTests(PersonalScenarioBuildBase):
    def test_problems_already_in_the_original_do_not_block_the_copy(self):
        # The Greek Pendulum case: a leftover activity nothing leads to.
        first, second = self.activity('First'), self.activity('Second')
        self.activity('Βοήθεια Δασκάλου')
        self.start_at(first)
        self.route(first, second)

        clone = Scenario.objects.get(pk=_build_personal_scenario(self.scenario.id, self.user.id))

        self.assertEqual(clone.activities.count(), 3)

    def test_problems_made_by_the_proposals_still_block_the_copy(self):
        first, second = self.activity('First'), self.activity('Second')
        self.start_at(first)
        self.route(first, second)

        def strand_an_activity(original_scenario, new_scenario, **kwargs):
            Activity.objects.create(
                name='Stranded', text='x', plain_text='x', scenario=new_scenario,
                activity_type=self.types['Explanation'],
                created_by=self.user, updated_by=self.user,
            )

        with patch('authoringtool.tasks.apply_proposals_to_cloned_scenario', side_effect=strand_an_activity):
            result = apply_user_proposals_to_new_scenario.run(self.scenario.id, self.user.id)

        self.assertEqual(result, 'scenario error')
        self.assertFalse(Scenario.objects.filter(origin_scenario=self.scenario).exists())
        failure = ProposalStructuralFailure.objects.get(scenario=self.scenario, stage='graph_integrity')
        self.assertEqual(
            [issue['message'] for issue in failure.errors],
            ["Activity 'Stranded' is unreachable from the scenario start."],
        )
