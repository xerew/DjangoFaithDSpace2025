import unittest

from simulated_students import metrics


class MetricsTests(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        # ties get average ranks
        self.assertAlmostEqual(metrics.spearman([1, 1, 2], [1, 1, 2]), 1.0)

    def test_pearson_and_mae(self):
        self.assertAlmostEqual(metrics.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(metrics.mae([0.5, 0.7], [0.4, 0.9]), 0.15)

    def test_bootstrap_interval_contains_the_estimate(self):
        x = list(range(40))
        y = [v + (3 if v % 2 else -3) for v in x]
        low, high = metrics.bootstrap_ci(x, y, metrics.spearman, rounds=300, seed=1)
        estimate = metrics.spearman(x, y)
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)
        self.assertLess(high - low, 0.2)

    def test_jensen_shannon(self):
        self.assertAlmostEqual(metrics.jsd({'a': 1.0}, {'a': 1.0}), 0.0)
        self.assertAlmostEqual(metrics.jsd({'a': 1.0}, {'b': 1.0}), 1.0)
        self.assertGreater(metrics.jsd({'a': 0.5, 'b': 0.5}, {'a': 0.9, 'b': 0.1}), 0)

    def test_top_distractor(self):
        real = {'1': 0.6, '2': 0.3, '3': 0.1}
        self.assertEqual(metrics.top_distractor(real, correct_id='1'), '2')
        self.assertIsNone(metrics.top_distractor({'1': 1.0}, correct_id='1'))


if __name__ == '__main__':
    unittest.main()
