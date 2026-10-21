from unittest.mock import patch

from django.test import TestCase

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Simulation,
    TranslationMatch,
)
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher
from authoringtool.translation_matching import (
    build_flow_profile,
    scan_translation_matches,
)

CANDIDATES = 'authoringtool.translation_matching.find_translation_candidates'


def confirmed_pair(first, second, origin):
    a, b = sorted([first, second], key=lambda s: s.id)
    profile_a, profile_b = build_flow_profile(a), build_flow_profile(b)
    return TranslationMatch.objects.create(
        scenario_a=a,
        scenario_b=b,
        status='confirmed',
        origin=origin,
        exact_match=True,
        confidence=100,
        activity_map=[[1, 2]],
        flow_signature_a=profile_a['signature'],
        flow_signature_b=profile_b['signature'],
        structure_signature_a=profile_a['structure_signature'],
        structure_signature_b=profile_b['structure_signature'],
    )


class StructureSignatureTests(TestCase):
    def setUp(self):
        self.owner = make_teacher('mt_scan_owner')
        self.source = build_source_scenario(self.owner)
        self.target = copy_for_translation(self.source, self.owner, 'Greek', 'English').target_scenario
        self.sim_en = Simulation.objects.create(name='Ohm EN', iframe_url='https://example.com/en', language='English')
        self.sim_el = Simulation.objects.create(name='Ohm EL', iframe_url='https://example.com/el', language='Greek')
        explain = Activity.objects.get(scenario=self.target, name='Why it halves')
        explain.simulation = self.sim_en
        explain.save()

    def swap_simulation(self):
        explain = Activity.objects.get(scenario=self.target, name='Why it halves')
        explain.simulation = self.sim_el
        explain.save()

    def test_structure_signature_ignores_simulation(self):
        before = build_flow_profile(self.target)
        self.swap_simulation()
        after = build_flow_profile(self.target)
        self.assertNotEqual(before['signature'], after['signature'])
        self.assertEqual(before['structure_signature'], after['structure_signature'])

    def test_scan_keeps_mt_pair_confirmed_when_not_a_candidate(self):
        match = confirmed_pair(self.source, self.target, 'machine_translation')
        with patch(CANDIDATES, return_value=[]):
            scan_translation_matches()
        match.refresh_from_db()
        self.assertEqual(match.status, 'confirmed')
        self.assertEqual(match.activity_map, [[1, 2]])

    def test_scan_keeps_mt_pair_after_media_swap(self):
        match = confirmed_pair(self.source, self.target, 'machine_translation')
        self.swap_simulation()
        scan_translation_matches()
        match.refresh_from_db()
        self.assertEqual(match.status, 'confirmed')
        self.assertEqual(match.activity_map, [[1, 2]])

    def test_scan_marks_mt_pair_changed_after_structure_change(self):
        match = confirmed_pair(self.source, self.target, 'machine_translation')
        check = Activity.objects.get(scenario=self.source, name='Final check')
        Answer.objects.create(activity=check, text='I = V/R', created_by=self.owner, updated_by=self.owner)
        with patch(CANDIDATES, return_value=[]):
            scan_translation_matches()
        match.refresh_from_db()
        self.assertEqual(match.status, 'changed')

    def test_scan_behaviour_unchanged_for_scan_pairs(self):
        match = confirmed_pair(self.source, self.target, 'scan')
        # The fixture is smaller than the scan's minimum; lower it so the
        # pair is eligible, as a real scanned pair would be.
        with patch(CANDIDATES, return_value=[]), \
                patch('authoringtool.translation_matching.MIN_ACTIVITIES', 1):
            scan_translation_matches()
        match.refresh_from_db()
        self.assertEqual(match.status, 'changed')
