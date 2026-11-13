from unittest import mock

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from authoringtool.performance_formula import categorize
from categorization import training
from categorization.features import as_model_frame, input_columns, model_categories

LANGUAGES = {1: 'English', 2: 'Greek', 3: 'Spanish', 4: 'Unspecified'}


def synthetic_final(runs=200, students=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(runs):
        group = i % 4 + 1
        p1 = float(rng.uniform(0, 100))
        rows.append({
            'implementation_id': i + 1, 'user_id': i % students + 1, 'username': f'u{i % students}',
            'scenario_id': group, 'scenario_name': f'Scenario {group}', 'language': LANGUAGES[group],
            'group_key': group, 'is_current_version': bool(i % 3), 'label': categorize(p1),
            'final_percentage': p1,
            'p1_pct': p1, 'p2_pct': float(rng.uniform(0, 100)), 'retries': float(rng.integers(0, 4)),
            'scenario': str(group),
        })
    return pd.DataFrame(rows)


def synthetic_checkpoints(final):
    table = final.copy()
    table['checkpoint'] = 1.0
    table['pct_so_far'] = table['p1_pct']
    table['baseline_label'] = [categorize(v) for v in table['p1_pct']]
    return table


class TrainingTests(SimpleTestCase):
    def setUp(self):
        self.final = synthetic_final()
        self.checkpoints = synthetic_checkpoints(self.final)
        self.columns = input_columns(self.final)
        self.categories = model_categories(self.final, self.checkpoints)

    def test_no_student_in_train_and_test(self):
        trained_on = []
        real_fit = training.fit_booster

        def recording_fit(X, y, groups):
            trained_on.append(set(groups))
            return real_fit(X, y, groups)

        with mock.patch.object(training, 'fit_booster', side_effect=recording_fit):
            oof = training.grouped_oof(self.final, self.columns, self.categories, lambda text: None)
        self.assertEqual(len(trained_on), training.FOLDS)
        for fold, users in enumerate(trained_on):
            tested = set(self.final.loc[oof['fold'] == fold, 'user_id'])
            self.assertTrue(tested)
            self.assertFalse(tested & users, fold)
        self.assertEqual(len(oof), len(self.final))

    def test_unseen_scenario_test_keeps_its_students_out_of_training(self):
        # 50 students over 200 runs and 4 scenarios: each student has runs
        # in two scenarios.
        table = synthetic_final(students=50)
        trained_on = []
        real_fit = training.fit_booster

        def recording_fit(X, y, groups):
            trained_on.append(set(groups))
            return real_fit(X, y, groups)

        with mock.patch.object(training, 'fit_booster', side_effect=recording_fit):
            results = training.leave_one_group_out(table, self.columns, self.categories, lambda text: None)
        self.assertEqual(len(results), len(trained_on))
        for entry, users in zip(results, trained_on):
            tested = set(table.loc[table.group_key == entry['group_key'], 'user_id'])
            self.assertTrue(users)
            self.assertFalse(tested & users, entry['group_key'])

    def test_fold_missing_a_category_still_gives_three_columns(self):
        subset = self.final[self.final.label != 'High']
        X = as_model_frame(subset, self.columns, self.categories)
        booster = training.fit_booster(
            X, subset.label.map(training.LABEL_INDEX).to_numpy(), subset.user_id.to_numpy(),
        )
        probabilities = training.predict_proba(booster, X)
        self.assertEqual(probabilities.shape, (len(subset), 3))

    def test_unseen_scenario_and_blank_language_predict(self):
        results = training.leave_one_group_out(self.final, self.columns, self.categories, lambda text: None)
        held_out = {entry['group_key']: entry for entry in results}
        self.assertIn(4, held_out)
        self.assertEqual(held_out[4]['runs'], 50)
        self.assertEqual(held_out[4]['scenario_names'], 'Scenario 4')
        self.assertGreater(held_out[4]['accuracy'], 0.5)

    def test_same_data_same_results(self):
        first, _ = training.train_all(self.final, self.checkpoints, lambda text: None)
        second, _ = training.train_all(self.final, self.checkpoints, lambda text: None)
        first.pop('seconds')
        second.pop('seconds')
        self.assertEqual(first, second)

    def test_results_follow_the_schema(self):
        results, boosters = training.train_all(self.final, self.checkpoints, lambda text: None)
        self.assertEqual(set(boosters), {'final', 'early'})
        self.assertEqual(results['final']['overall']['runs'], 200)
        self.assertEqual(set(results['final']['by_version']), {'current', 'older'})
        prediction = results['final']['predictions'][0]
        self.assertEqual(set(prediction), {
            'implementation_id', 'user_id', 'username', 'scenario_id', 'scenario_name', 'language',
            'group_key', 'is_current_version', 'pisa', 'model', 'confidence', 'final_percentage',
        })
        checkpoint = results['early']['checkpoints'][0]
        self.assertEqual(checkpoint['checkpoint'], 1)
        self.assertEqual(checkpoint['baseline']['accuracy'], 1.0)
        self.assertIn('1', results['early']['importances'])
        self.assertEqual(set(results['seconds']), {'total', 'loso'})

    def test_too_little_data_raises(self):
        small = self.final.head(20)
        with self.assertRaises(training.NotEnoughData) as caught:
            training.train_all(small, synthetic_checkpoints(small), lambda text: None)
        self.assertEqual(
            str(caught.exception),
            'Not enough runs to train: 20 found, at least 50 needed, '
            'with at least two categories.',
        )

    def test_single_category_raises(self):
        single = self.final.copy()
        single['label'] = 'High'
        with self.assertRaises(training.NotEnoughData):
            training.train_all(single, synthetic_checkpoints(single), lambda text: None)

    def test_metrics_confusion_order(self):
        result = training.metrics(['High', 'Low'], ['High', 'Moderate'])
        self.assertEqual(result['confusion'][2][2], 1)
        self.assertEqual(result['confusion'][0][1], 1)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['runs'], 2)

    def test_metrics_of_nothing(self):
        result = training.metrics([], [])
        self.assertEqual(result['runs'], 0)
        self.assertIsNone(result['accuracy'])

    def test_importances_sorted_and_capped(self):
        X = as_model_frame(self.final, self.columns, self.categories)
        booster = training.fit_booster(
            X, self.final.label.map(training.LABEL_INDEX).to_numpy(), self.final.user_id.to_numpy(),
        )
        ranking = training.importances(booster, X, self.columns)
        self.assertLessEqual(len(ranking), 15)
        self.assertEqual(ranking[0]['input'], 'p1_pct')
        values = [entry['mean_abs_shap'] for entry in ranking]
        self.assertEqual(values, sorted(values, reverse=True))
