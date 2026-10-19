"""What a revise proposal changes: the current activity next to the proposal."""
import json

from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityProposal,
    ActivityType,
    Answer,
    Phase,
    ProposalGenerationRun,
    Scenario,
    UserProposalReview,
)
from authoringtool.proposal_diff import revision_changes, word_diff


class WordDiffTests(TestCase):
    def test_unchanged_text_is_one_same_part(self):
        self.assertEqual(word_diff('The period stays', 'The period stays'), [('same', 'The period stays')])

    def test_added_and_removed_words_are_marked(self):
        parts = word_diff('The period doubles', 'The period stays the same')

        self.assertIn(('same', 'The period'), parts)
        self.assertIn(('del', 'doubles'), parts)
        self.assertIn(('add', 'stays the same'), parts)


class ActivityFixture:
    def setUp(self):
        user = User.objects.create_user('diff_owner')
        scenario = Scenario.objects.create(name='Diff', created_by=user, updated_by=user)
        phase = Phase.objects.create(name='Phase', scenario=scenario, created_by=user, updated_by=user)
        question = ActivityType.objects.create(name='Question', created_by=user, updated_by=user)
        self.activity = Activity.objects.create(
            name='Activity 6', text='<p>Which is <b>true</b>?</p>', plain_text='Which is true?',
            scenario=scenario, phase=phase, activity_type=question,
            created_by=user, updated_by=user,
        )
        Answer.objects.create(activity=self.activity, text='Longer is slower', is_correct=True, answer_weight=3)
        Answer.objects.create(activity=self.activity, text='Mass matters', is_correct=False, answer_weight=1)


class RevisionChangesTests(ActivityFixture, TestCase):
    def rows(self, proposed):
        return {row['label']: row for row in revision_changes(self.activity, proposed)}

    def test_rows_show_current_and_proposed_with_change_flags(self):
        rows = self.rows({
            'activity_name': 'Activity 6',
            'content': 'Based on your data, which is true?',
            'answers': [
                {'text': 'Longer is slower', 'is_correct': True},
                {'text': 'Mass does not matter', 'is_correct': False},
            ],
        })

        self.assertFalse(rows['Name']['changed'])
        self.assertTrue(rows['Content']['changed'])
        self.assertEqual(rows['Content']['current'], 'Which is true?')
        # "Which" -> "which" is a real change too.
        self.assertEqual(rows['Content']['diff'], [
            ('del', 'Which'), ('add', 'Based on your data, which'), ('same', 'is true?'),
        ])
        answers = rows['Answers']
        self.assertTrue(answers['changed'])
        self.assertFalse(answers['answers'][0]['changed'])
        self.assertTrue(answers['answers'][1]['changed'])
        self.assertEqual(answers['answers'][1]['current'], 'Mass matters')

    def test_a_change_of_correct_answer_counts_as_a_change(self):
        rows = self.rows({
            'activity_name': 'Activity 6', 'content': 'Which is true?',
            'answers': [
                {'text': 'Longer is slower', 'is_correct': False},
                {'text': 'Mass matters', 'is_correct': True},
            ],
        })

        self.assertFalse(rows['Content']['changed'])
        self.assertTrue(rows['Answers']['answers'][0]['changed'])
        self.assertTrue(rows['Answers']['changed'])

    def test_activity_without_answers_has_no_answers_row(self):
        self.activity.answers.all().delete()

        rows = self.rows({'activity_name': 'Activity 6', 'content': 'Which is true?', 'answers': []})

        self.assertNotIn('Answers', rows)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ProposalListRevisionChangesTests(ActivityFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.teacher = User.objects.create_user('diff_teacher', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))
        scenario = self.activity.scenario
        Scenario.objects.filter(pk=scenario.pk).update(created_by=self.teacher)
        self.run = ProposalGenerationRun.objects.create(scenario=scenario, is_current=True)
        self.client.force_login(self.teacher)
        self.url = reverse('proposal_list', args=[scenario.id])

    def proposal(self, kind, **data):
        payload = json.dumps({
            'action': kind, 'activity_name': 'Activity 6', 'activity_type': 'Question',
            'content': 'Which is true?', 'answers': [
                {'text': 'Longer is slower', 'is_correct': True, 'weight': 3},
                {'text': 'Mass matters', 'is_correct': False, 'weight': 1},
            ], 'insert_location': 'after', 'explanation': 'x', **data,
        })
        return ActivityProposal.objects.create(
            scenario=self.activity.scenario, generation_run=self.run, phase=self.activity.phase,
            activity=self.activity, proposal_type=kind,
            json_action=payload, json_translated_action=payload,
        )

    def test_revise_proposal_shows_what_changes(self):
        self.proposal('revise', content='Based on your data, which is true?')

        response = self.client.get(self.url)

        self.assertContains(response, 'What changes')
        self.assertContains(response, 'Which is true?')
        self.assertContains(response, '<ins class="diff-add">Based on your data, which</ins>')
        self.assertContains(response, '<del class="diff-del">Which</del>')

    def test_teacher_edits_are_compared_too(self):
        proposal = self.proposal('revise')
        UserProposalReview.objects.create(
            proposal=proposal, user=self.teacher,
            teacher_edited_json={'content': 'Edited by the teacher'},
        )

        response = self.client.get(self.url)

        self.assertContains(response, 'Edited by the teacher')

    def test_create_proposal_has_no_comparison(self):
        self.proposal('create', activity_name='New one')

        response = self.client.get(self.url)

        self.assertNotContains(response, 'What changes')

    def test_proposed_text_is_escaped(self):
        self.proposal('revise', content='<script>alert(1)</script>')

        response = self.client.get(self.url)

        self.assertNotContains(response, '<script>alert(1)</script>')
