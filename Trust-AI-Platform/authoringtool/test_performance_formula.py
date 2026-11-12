from django.test import SimpleTestCase

from authoringtool.performance_formula import (
    categorize,
    phase_score,
    phase_weights,
    scored_activity_ids,
    weighted_percentage,
)


class FormulaUnitTests(SimpleTestCase):
    def test_categorize_thresholds(self):
        self.assertEqual(categorize(83.3), 'High')
        self.assertEqual(categorize(83.29), 'Moderate')
        self.assertEqual(categorize(49.7), 'Moderate')
        self.assertEqual(categorize(49.69), 'Low')

    def test_phase_weights(self):
        self.assertEqual(phase_weights(3), [0.2, 0.2, 0.45, 0.15])
        self.assertEqual(phase_weights(4), [0.2, 0.2, 0.45, 0.15])
        self.assertEqual(phase_weights(5), [0.2, 0.2, 0.3, 0.15, 0.15])
        self.assertEqual(phase_weights(7), [0.2, 0.2, 0.3, 0.15, 0.15])

    def test_scored_activity_ids_deduplicates_bunches(self):
        bunches = {1: [1, 2], 3: [2, 3, 4]}
        self.assertEqual(scored_activity_ids([1, 3, 5], bunches), [1, 2, 3, 4, 5])

    def test_scored_activity_ids_known_filter(self):
        bunches = {1: [1, 99]}
        self.assertEqual(scored_activity_ids([1], bunches, known_ids={1}), [1])

    def test_phase_score_missing_answers_count_zero(self):
        self.assertEqual(phase_score([1, 2, 3], {1: 2, 3: 1}, {1: 2, 2: 2, 3: 2}), (3, 6))

    def test_weighted_percentage_rescales_skipped_phase(self):
        # Phase 2 has no scored activity: weights 0.2 and 0.45 remain.
        result = weighted_percentage([(1, 1), (0, 0), (0, 2)], [0.2, 0.2, 0.45, 0.15])
        self.assertAlmostEqual(result, 0.2 / 0.65 * 100)

    def test_weighted_percentage_beyond_weights_is_ignored(self):
        scores = [(1, 1)] * 5 + [(0, 1)]
        self.assertAlmostEqual(weighted_percentage(scores, phase_weights(6)), 100)

    def test_weighted_percentage_none_when_nothing_counts(self):
        self.assertIsNone(weighted_percentage([(0, 0)], [0.2]))
