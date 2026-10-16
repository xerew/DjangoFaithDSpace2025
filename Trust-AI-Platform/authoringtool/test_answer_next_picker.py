"""The answers page picks each answer's next activity with the same
searchable picker as the activity page's "Next Activity"."""
from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    NextQuestionLogic,
    Phase,
    Scenario,
)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class AnswerNextActivityPickerTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('picker_owner', password='pass')
        self.owner.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Picker', created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Explore', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        self.later_phase = Phase.objects.create(
            name='Explain', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        explanation = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        question = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Which pendulum?', text='Which?', phase=self.phase,
            scenario=self.scenario, activity_type=question,
            created_by=self.owner, updated_by=self.owner,
        )
        self.target = Activity.objects.create(
            name='Swing explanation', text='It swings.', phase=self.later_phase,
            scenario=self.scenario, activity_type=explanation,
            created_by=self.owner, updated_by=self.owner,
        )
        self.answer = Answer.objects.create(
            activity=self.activity, text='The long one', is_correct=True,
            created_by=self.owner, updated_by=self.owner,
        )
        self.other_answer = Answer.objects.create(
            activity=self.activity, text='The short one', is_correct=False,
            created_by=self.owner, updated_by=self.owner,
        )
        NextQuestionLogic.objects.create(
            activity=self.activity, answer=self.answer, next_activity=self.target,
        )
        self.url = reverse(
            'updateAnswers',
            args=[self.scenario.id, self.phase.id, self.activity.id],
        )
        self.client.force_login(self.owner)

    def page(self):
        return self.client.get(self.url).content.decode()

    def test_each_answer_has_a_searchable_picker_instead_of_a_select(self):
        html = self.page()

        self.assertNotIn('<select name="next_activity_0"', html)
        # Two answers, plus the template used for added answers.
        self.assertEqual(html.count('class="np-value"'), 3)
        self.assertEqual(html.count('class="np-search"'), 3)

    def test_current_route_is_preselected(self):
        html = self.page()

        self.assertIn(
            f'name="next_activity_0" class="np-value" value="{self.target.id}"',
            html,
        )
        self.assertIn('name="next_activity_1" class="np-value" value=""', html)
        self.assertIn('<span class="np-label">Swing explanation</span>', html)
        self.assertIn('Select Next Activity (Optional)…', html)

    def test_activities_are_grouped_by_phase_without_the_activity_itself(self):
        Activity.objects.create(
            name='Hint', text='Look again.', phase=self.phase,
            scenario=self.scenario, activity_type=self.target.activity_type,
            created_by=self.owner, updated_by=self.owner,
        )

        html = self.page()

        self.assertIn('Explore (Current)', html)
        self.assertIn('Explain', html)
        self.assertNotIn(f'data-val="{self.activity.id}"', html)
        self.assertIn('data-val="create_new"', html)

    def test_new_answers_get_the_picker_from_a_template(self):
        html = self.page()

        self.assertIn('<template id="nextPickerTemplate">', html)
        self.assertIn('name="__NAME__"', html)

    def test_route_back_to_the_same_activity_is_kept(self):
        NextQuestionLogic.objects.filter(answer=self.answer).update(
            next_activity=self.activity,
        )

        html = self.page()

        self.assertIn(
            f'name="next_activity_0" class="np-value" value="{self.activity.id}"',
            html,
        )
        self.assertIn('<span class="np-label">Which pendulum?</span>', html)

    def test_saving_a_picked_activity_routes_the_answer(self):
        self.client.post(self.url, {
            'answers[0][id]': self.answer.id,
            'answers[0][text]': 'The long one',
            'answers[0][is_correct]': 'on',
            'answers[0][answer_weight]': 1,
            'next_activity_0': '',
            'answers[1][id]': self.other_answer.id,
            'answers[1][text]': 'The short one',
            'answers[1][answer_weight]': 1,
            'next_activity_1': self.target.id,
        })

        self.assertFalse(
            NextQuestionLogic.objects.filter(answer=self.answer).exists()
        )
        self.assertEqual(
            NextQuestionLogic.objects.get(answer=self.other_answer).next_activity,
            self.target,
        )
