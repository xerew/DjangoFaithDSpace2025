import html

from django.test import TestCase
from django.urls import reverse

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.machine_translation.editing import original_of
from authoringtool.models import Activity, Scenario, ScenarioTranslationJob
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher


class TextOnlyTemplateTests(TestCase):
    def setUp(self):
        self.owner = make_teacher('mt_tpl_owner')
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.owner, 'Greek', 'English')
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(status='done')
        self.target = self.job.target_scenario
        self.ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        Activity.objects.filter(pk=self.ask.pk).update(
            name='Διπλασιασμός', text='<p>Τι συμβαίνει;</p>',
        )
        self.client.force_login(self.owner)

    def get(self, name, *args):
        response = self.client.get(reverse(name, args=args))
        self.assertEqual(response.status_code, 200)
        return html.unescape(response.content.decode())

    def test_original_of_maps_rows(self):
        self.job.refresh_from_db()
        original = original_of(self.job, self.ask)
        self.assertEqual(original.name, 'Doubling resistance')
        self.assertEqual(original_of(self.job, self.target), self.source)
        answer = self.ask.answers.order_by('id').first()
        self.assertEqual(original_of(self.job, answer).text, 'It halves')
        self.assertEqual(original_of(self.job, self.ask.phase).name, 'Explore')

    def test_view_scenario_hides_structure_controls(self):
        content = self.get('viewScenario', self.target.id)
        self.assertNotIn(reverse('createPhase', args=[self.target.id]), content)
        self.assertNotIn(reverse('move_phase', args=[self.target.id, self.ask.phase_id, 'up']), content)
        normal = self.get('viewScenario', self.source.id)
        self.assertIn(reverse('createPhase', args=[self.source.id]), normal)

    def test_view_phase_hides_add_and_delete_activity(self):
        content = self.get('phase', self.target.id, self.ask.phase_id)
        self.assertNotIn(reverse('createActivity', args=[self.target.id, self.ask.phase_id]), content)
        self.assertNotIn(
            reverse('deleteActivity', args=[self.target.id, self.ask.phase_id, self.ask.id]), content,
        )

    def test_view_activity_hides_answer_and_start_controls(self):
        content = self.get('activity', self.target.id, self.ask.phase_id, self.ask.id)
        self.assertNotIn(reverse('set_start_activity', args=[self.target.id, self.ask.id]), content)
        self.assertNotIn(
            reverse('createAnswers', args=[self.target.id, self.ask.phase_id, self.ask.id]), content,
        )
        self.assertIn(
            reverse('updateAnswers', args=[self.target.id, self.ask.phase_id, self.ask.id]), content,
        )

    def test_update_activity_shows_original_text(self):
        content = self.get('updateActivity', self.target.id, self.ask.phase_id, self.ask.id)
        self.assertIn('Original: Doubling resistance', content)
        self.assertIn('What happens to the current if the resistance doubles', content)

    def test_update_activity_hides_type_and_routing_inputs(self):
        content = self.get('updateActivity', self.target.id, self.ask.phase_id, self.ask.id)
        for name in ('activity_type', 'next_activity_id', 'is_evaluatable', 'is_primary_ev', 'selected_activities'):
            self.assertNotIn(f'name="{name}"', content)
        self.assertIn('name="experiment_type"', content)
        normal = Activity.objects.get(scenario=self.source, name='Doubling resistance')
        normal_content = self.get('updateActivity', self.source.id, normal.phase_id, normal.id)
        self.assertIn('name="activity_type"', normal_content)

    def test_update_answers_hides_correctness_and_shows_original(self):
        content = self.get('updateAnswers', self.target.id, self.ask.phase_id, self.ask.id)
        self.assertNotIn('answers[0][is_correct]"', content)
        self.assertNotIn('answers[0][answer_weight]"', content)
        self.assertIn('Original: It halves', content)

    def test_update_phase_and_scenario_show_original(self):
        self.assertIn('Original: Explore', self.get('updatePhase', self.target.id, self.ask.phase_id))
        self.assertIn("Original: Ohm's Law", self.get('updateScenario', self.target.id))

    def test_original_missing_when_source_deleted(self):
        Scenario.objects.filter(pk=self.source.pk).delete()
        content = self.get('updateActivity', self.target.id, self.ask.phase_id, self.ask.id)
        self.assertNotIn('Original:', content)
        self.assertNotIn('name="activity_type"', content)
