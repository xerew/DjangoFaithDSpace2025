import json
import os
import shutil
import tempfile
import unittest

from simulated_students import analyze_walkthrough as aw
from simulated_students.tests.scenario_fixture import H, L, M, P1, P2, Q1, Q2, scenario

HIGH = [(P1, None), (Q1, 21), (Q2, 41), (H, None)]
MID_DIRECT = [(P1, None), (Q1, 21), (Q2, 42), (M, None)]
MID = [(P1, None), (Q1, 22), (P2, None), (Q2, 41), (M, None)]
LOW = [(P1, None), (Q1, 22), (P2, None), (Q2, 42), (L, None)]


def row(persona, steps, end_reason='end', model='m', scenario_id=9):
    options = {21: (True, 1), 22: (False, 0), 41: (True, 1), 42: (False, 0)}
    return {
        'scenario_id': scenario_id, 'model': model, 'persona': persona, 'end_reason': end_reason,
        'steps': [
            {'activity_id': a, 'kind': 'page'} if o is None else
            {'activity_id': a, 'kind': 'question', 'choice_option_id': o,
             'correct': options[o][0], 'weight': options[o][1]}
            for a, o in steps
        ],
        'total_score': sum(options[o][1] for _, o in steps if o),
    }


class SummariseTests(unittest.TestCase):
    def test_simulated_runs_are_summarised_like_real_ones(self):
        rows = [row('high', HIGH)] * 3 + [row('mid', MID)] * 2 + [row('low', LOW[:2], 'invalid_reply')]
        summary = aw.summarise(aw.nodes_of(scenario()), [aw.run_of(r) for r in rows])
        self.assertEqual((summary['students'], summary['completed']), (6, 5))
        self.assertEqual(summary['transitions'][str(Q1)], {str(Q2): 3, str(P2): 2, 'STOP': 1})
        self.assertEqual(summary['branching'][str(Q2)], {'high': 3, 'mid': 2})
        self.assertEqual(summary['score_histogram'], {'2': 3, '1': 2})
        self.assertAlmostEqual(summary['items'][str(Q1)]['p_correct'], 3 / 6)
        self.assertEqual(summary['paths'][0], {'path': [P1, Q1, Q2, H], 'n': 3})


class CompareTests(unittest.TestCase):
    def test_a_simulated_class_like_the_real_one_scores_perfectly(self):
        data = scenario()
        # The real class: 9 high, 1 + 5 mid, 3 low (and 2 who stopped).
        rows = ([row('high', HIGH)] * 9 + [row('mid', MID_DIRECT)] + [row('mid', MID)] * 5
                + [row('low', LOW)] * 3)
        result = aw.compare(data, rows)
        self.assertAlmostEqual(result['choice_point_jsd'], 0)
        self.assertAlmostEqual(result['band_jsd'], 0)
        self.assertAlmostEqual(result['score_wasserstein'], 0)
        # Rare real paths are not exported, so compare the share on frequent paths.
        self.assertAlmostEqual(result['frequent_path_share_real'], 14 / 18)
        self.assertAlmostEqual(result['frequent_path_share_sim'], 14 / 18)
        self.assertEqual(result['path_coverage_real'], 1.0)
        self.assertEqual(result['completion_real'], 18 / 20)
        self.assertEqual(result['completion_sim'], 1.0)
        self.assertTrue(result['persona_ordered'])
        self.assertEqual(result['invalid_rate'], 0)

    def test_a_class_that_always_answers_right_is_far_from_the_real_one(self):
        result = aw.compare(scenario(), [row('high', HIGH)] * 10)
        self.assertAlmostEqual(result['band_jsd'], aw.metrics.jsd({'high': 9 / 18, 'mid': 6 / 18, 'low': 3 / 18},
                                                                 {'high': 1.0}))
        self.assertGreater(result['score_wasserstein'], 0)
        self.assertEqual(result['path_coverage_real'], 9 / 14)
        self.assertEqual(result['mean_score_sim'], 2)
        self.assertIsNone(result['persona_ordered'])  # only one persona

    def test_personas_in_the_wrong_order_are_flagged(self):
        result = aw.compare(scenario(), [row('high', LOW), row('mid', MID), row('low', HIGH)])
        self.assertFalse(result['persona_ordered'])
        self.assertEqual(result['persona_mean_score'], {'low': 2, 'mid': 1, 'high': 0})


class ReportTests(unittest.TestCase):
    def test_report_files_per_model(self):
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        rows = [row('high', HIGH, model='a')] * 3 + [row('low', LOW, model='b')] * 3
        aw.write_report([scenario()], rows, out)
        with open(os.path.join(out, 'summary.json'), encoding='utf-8') as handle:
            summary = json.load(handle)
        self.assertEqual(sorted(summary['models']), ['a', 'b'])
        self.assertEqual(summary['models']['a']['scenarios'], 1)
        with open(os.path.join(out, 'report.md'), encoding='utf-8') as handle:
            report = handle.read()
        self.assertIn('Pendulum', report)
        self.assertTrue(os.path.exists(os.path.join(out, 'scenarios.csv')))


if __name__ == '__main__':
    unittest.main()
