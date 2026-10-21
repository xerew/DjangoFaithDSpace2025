from django.test import TestCase
from django.urls import reverse

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    ScenarioTranslationJob,
    Simulation,
)
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher
from authoringtool.translation_matching import build_flow_profile


class TextOnlyEnforcementTests(TestCase):
    def setUp(self):
        self.owner = make_teacher('mt_edit_owner')
        self.admin = make_teacher('mt_edit_admin', admin=True)
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.owner, 'Greek', 'English')
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(status='done')
        self.target = self.job.target_scenario
        self.ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        self.explain = Activity.objects.get(scenario=self.target, name='Why it halves')
        self.phase = self.ask.phase
        self.client.force_login(self.owner)

    def structural_requests(self, scenario, phase, activity, answer):
        s, p, a = scenario.id, phase.id, activity.id
        return [
            ('createPhaseData', [s], {'name': 'New'}),
            ('deletePhase', [s, p], {}),
            ('createActivityData', [s, p], {'activity_name': 'New', 'activity_text': 'x'}),
            ('deleteActivity', [s, p, a], {}),
            ('move_activity', [s, p, a, 'down'], {}),
            ('move_phase', [s, p, 'down'], {}),
            ('set_start_activity', [s, a], {}),
            ('deleteAnswer', [s, p, a, answer.id], {}),
            ('createCriterionData', [s, p, a], {}),
            ('updateCriterionData', [s, p, a], {}),
            ('createAnswers', [s, p, a], {}),
        ]

    def assert_structure_untouched(self):
        self.assertEqual(Phase.objects.filter(scenario=self.target).count(), 2)
        self.assertEqual(Activity.objects.filter(scenario=self.target).count(), 3)
        self.assertEqual(Answer.objects.filter(activity__scenario=self.target).count(), 3)
        self.assertEqual(
            build_flow_profile(self.target)['signature'],
            build_flow_profile(self.source)['signature'],
        )

    def test_delete_activity_refused(self):
        answer = self.ask.answers.first()
        for name, args, data in self.structural_requests(self.target, self.phase, self.ask, answer):
            with self.subTest(view=name):
                response = self.client.post(reverse(name, args=args), data)
                self.assertRedirects(
                    response, reverse('viewScenario', args=[self.target.id]),
                    fetch_redirect_response=False,
                )
        self.assert_structure_untouched()

    def test_admin_also_refused_structural_change(self):
        self.client.force_login(self.admin)
        self.client.post(reverse('deleteActivity', args=[self.target.id, self.phase.id, self.ask.id]))
        self.assertTrue(Activity.objects.filter(pk=self.ask.pk).exists())

    def test_activity_update_saves_text_only(self):
        explanation = ActivityType.objects.get(name='Explanation')
        response = self.client.post(
            reverse('updateActivityData', args=[self.target.id, self.phase.id, self.ask.id]),
            {
                'activity_name': 'Διπλασιασμός',
                'activity_text': '<p>Τι συμβαίνει;</p>',
                'helping_quote': 'Σκέψου',
                'is_evaluatable': 'on',
                'activity_type': explanation.id,
                'next_activity_id': 'create_new',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.ask.refresh_from_db()
        self.assertEqual(self.ask.name, 'Διπλασιασμός')
        self.assertEqual(self.ask.plain_text, 'Τι συμβαίνει;')
        self.assertEqual(self.ask.helper, 'Σκέψου')
        self.assertFalse(self.ask.is_evaluatable)
        self.assertEqual(self.ask.activity_type.name, 'Question')
        self.assert_structure_untouched()

    def test_activity_update_allows_simulation_swap(self):
        experiment, _ = ActivityType.objects.get_or_create(
            name='Experiment', defaults={'created_by': self.owner, 'updated_by': self.owner},
        )
        sim_en = Simulation.objects.create(name='EN', iframe_url='https://example.com/en')
        sim_el = Simulation.objects.create(name='EL', iframe_url='https://example.com/el', language='Greek')
        Activity.objects.filter(pk=self.explain.pk).update(activity_type=experiment, simulation=sim_en)
        self.client.post(
            reverse('updateActivityData', args=[self.target.id, self.phase.id, self.explain.id]),
            {
                'activity_name': 'Γιατί',
                'activity_text': '<p>Υποδιπλασιάζεται.</p>',
                'experiment_type': 'simulation',
                'simulation': sim_el.id,
                'activity_type': experiment.id,
            },
        )
        self.explain.refresh_from_db()
        self.assertEqual(self.explain.simulation, sim_el)
        self.assertEqual(self.explain.activity_type, experiment)

    def test_answers_update_saves_text_only(self):
        first, second = self.ask.answers.order_by('id')
        response = self.client.post(
            reverse('updateAnswers', args=[self.target.id, self.phase.id, self.ask.id]),
            {
                'answers[0][id]': first.id,
                'answers[0][text]': 'Υποδιπλασιάζεται',
                'answers[0][answer_weight]': 3,
                # is_correct omitted: would make it wrong in the normal editor
                'answers[2][text]': 'A brand new answer',
                'next_activity_0': '',
            },
        )
        self.assertEqual(response.status_code, 302)
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual(first.text, 'Υποδιπλασιάζεται')
        self.assertTrue(first.is_correct)
        self.assertEqual(first.answer_weight, 2)
        self.assertEqual(second.text, 'It doubles')
        self.assert_structure_untouched()

    def test_pending_translation_cannot_be_opened(self):
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(status='waiting')
        for name, args in [
            ('viewScenario', [self.target.id]),
            ('updateScenario', [self.target.id]),
            ('phase', [self.target.id, self.phase.id]),
            ('updateActivity', [self.target.id, self.phase.id, self.ask.id]),
        ]:
            with self.subTest(view=name):
                response = self.client.get(reverse(name, args=args))
                self.assertRedirects(response, reverse('scenarios'), fetch_redirect_response=False)

    def test_normal_scenarios_unaffected(self):
        explain = Activity.objects.get(scenario=self.source, name='Why it halves')
        self.client.force_login(self.owner)
        self.client.post(reverse('deleteActivity', args=[self.source.id, explain.phase_id, explain.id]))
        self.assertFalse(Activity.objects.filter(pk=explain.pk).exists())
