from django.test import TestCase

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.models import Activity, Answer, Phase
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher
from authoringtool.translation_matching import build_flow_profile


class CopyForTranslationTests(TestCase):
    def setUp(self):
        self.owner = make_teacher('mt_copy_owner')
        self.requester = make_teacher('mt_copy_requester')
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.requester, 'Greek', 'English')
        self.copy = self.job.target_scenario

    def test_copy_is_private_translation_owned_by_requester(self):
        self.assertEqual(self.copy.visibility_status, 'private')
        self.assertEqual(self.copy.variant_type, 'translation')
        self.assertEqual(self.copy.origin_scenario, self.source)
        self.assertEqual(self.copy.created_by, self.requester)
        self.assertEqual(self.copy.language, 'Greek')
        self.assertEqual(self.copy.family_id, self.source.family_id)
        self.assertEqual(self.job.status, 'waiting')

    def test_copy_matches_structure(self):
        self.assertEqual(
            build_flow_profile(self.copy)['signature'],
            build_flow_profile(self.source)['signature'],
        )

    def test_copy_includes_answer_feedback(self):
        copied = Answer.objects.get(activity__scenario=self.copy, text='It doubles')
        self.assertEqual([f.text for f in copied.feedbacks.all()], ['Not quite.'])

    def test_copy_maps_every_row(self):
        activities = self.job.id_map['activities']
        self.assertEqual(len(activities), 3)
        for new_id, old_id in activities.items():
            self.assertEqual(
                Activity.objects.get(pk=int(new_id)).lineage_key,
                Activity.objects.get(pk=old_id).lineage_key,
            )
        self.assertEqual(len(self.job.id_map['phases']), 2)
        for new_id, old_id in self.job.id_map['phases'].items():
            self.assertEqual(
                Phase.objects.get(pk=int(new_id)).name,
                Phase.objects.get(pk=old_id).name,
            )
        self.assertEqual(len(self.job.id_map['answers']), 3)
        for new_id, old_id in self.job.id_map['answers'].items():
            self.assertEqual(
                Answer.objects.get(pk=int(new_id)).text,
                Answer.objects.get(pk=old_id).text,
            )

    def test_units_total_counts_header_phases_activities(self):
        self.assertEqual(self.job.units_total, 6)

    def test_second_copy_gets_numbered_name(self):
        second = copy_for_translation(self.source, self.requester, 'Greek', 'English')
        self.assertEqual(self.copy.name, "Ohm's Law (Greek)")
        self.assertEqual(second.target_scenario.name, "Ohm's Law (Greek) (2)")

    def test_copy_does_not_carry_counts_or_llm_context(self):
        copied = Activity.objects.get(scenario=self.copy, name='Doubling resistance')
        self.assertEqual(copied.correct_count, 0)
        self.assertEqual(copied.incorrect_count, 0)
        self.assertFalse(copied.llm_context)

    def test_start_activity_and_branching_are_remapped(self):
        start = self.copy.start_activity
        self.assertEqual(start.scenario_id, self.copy.id)
        check = Activity.objects.get(scenario=self.copy, name='Final check')
        self.assertEqual(check.branching.next_question_on_low.scenario_id, self.copy.id)
        self.assertEqual(check.branching.next_question_on_low_feedback, 'Try again.')
        bunch = check.question_bunches.get()
        self.assertEqual(
            set(Activity.objects.filter(id__in=bunch.activity_ids).values_list('scenario_id', flat=True)),
            {self.copy.id},
        )
