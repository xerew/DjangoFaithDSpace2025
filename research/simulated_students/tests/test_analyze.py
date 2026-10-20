import csv
import os
import shutil
import tempfile
import unittest

from simulated_students import analyze


def item(item_id, p_real, shares, language='English', images=(), by_ability=None):
    return {
        'item_id': item_id, 'language': language, 'images': list(images),
        'options': [
            {'id': 1, 'is_correct': True, 'image': None},
            {'id': 2, 'is_correct': False, 'image': None},
            {'id': 3, 'is_correct': False, 'image': None},
        ],
        'real': {'n': 50, 'p_correct': p_real, 'option_shares': shares,
                 'by_ability': by_ability or {}},
    }


def answers(item_id, persona, picks, model='m', context='none'):
    return [
        {'item_id': item_id, 'model': model, 'context': context, 'persona': persona,
         'choice_option_id': pick, 'correct': pick == 1 if pick else None,
         'error': None if pick else 'unusable reply'}
        for pick in picks
    ]


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            item(1, 0.9, {'1': 0.9, '2': 0.1}, by_ability={
                'low': {'n': 10, 'p_correct': 0.8}, 'high': {'n': 10, 'p_correct': 1.0}}),
            item(2, 0.5, {'1': 0.5, '3': 0.5}, language='Greek'),
            item(3, 0.2, {'1': 0.2, '2': 0.6, '3': 0.2}, images=['images/3_1.png']),
        ]
        self.rows = (
            answers(1, 'low', [1, 1, 2]) + answers(1, 'high', [1, 1, 1])
            + answers(2, 'low', [1, 3, 3]) + answers(2, 'high', [1, 1, 3])
            + answers(3, 'low', [2, 2, 2]) + answers(3, 'high', [1, 2, None])
        )

    def test_item_level_simulated_difficulty(self):
        table = analyze.item_table(self.items, self.rows, model='m', context='none')
        by_id = {row['item_id']: row for row in table}
        self.assertAlmostEqual(by_id[1]['p_sim'], 5 / 6)
        self.assertAlmostEqual(by_id[3]['p_sim'], 1 / 5)  # the unusable reply is left out
        self.assertEqual(by_id[3]['invalid'], 1)
        self.assertAlmostEqual(by_id[1]['p_low'], 2 / 3)
        self.assertAlmostEqual(by_id[1]['p_high'], 1.0)
        self.assertEqual(by_id[3]['real_top_distractor'], '2')
        self.assertEqual(by_id[3]['sim_top_distractor'], '2')

    def test_summary_compares_with_real_students(self):
        table = analyze.item_table(self.items, self.rows, model='m', context='none')
        summary = analyze.summarize(table, rounds=200)
        self.assertEqual(summary['items'], 3)
        self.assertAlmostEqual(summary['spearman'], 1.0)
        self.assertEqual(summary['distractor_agreement'], 1.0)
        self.assertEqual(summary['persona_monotonic_share'], 1.0)

    def test_report_files(self):
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        analyze.write_report(self.items, self.rows, out, rounds=100)
        with open(os.path.join(out, 'report.md'), encoding='utf-8') as handle:
            report = handle.read()
        self.assertIn('| m | none | all |', report)
        self.assertIn('Greek', report)
        self.assertIn('with images', report)
        with open(os.path.join(out, 'items.csv'), encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)


if __name__ == '__main__':
    unittest.main()
