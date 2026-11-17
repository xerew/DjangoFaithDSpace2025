"""Train and evaluate the two categorization models. No Django here.

Every reported number comes from students the model did not train on:
5-fold cross-validation grouped by student, plus leave-one-scenario-out
(a scenario and its confirmed translations held out together).
"""

import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from authoringtool.performance_formula import CATEGORIES

from .features import as_model_frame, input_columns, model_categories

SEED = 42
FOLDS = 5
LOSO_MIN_RUNS = 30
MIN_LABELLED_RUNS = 50
EARLY_STOPPING_ROUNDS = 30
NUM_BOOST_ROUND = 400
ROUNDS_WITHOUT_VALIDATION = 200
PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eta': 0.08,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'seed': SEED,
}
LABEL_INDEX = {label: index for index, label in enumerate(CATEGORIES)}


class NotEnoughData(Exception):
    """Shown on the dashboard as it is."""


def _matrix(X, y=None, weight=None):
    return xgb.DMatrix(X, label=y, weight=weight, enable_categorical=True)


def fit_booster(X, y, groups):
    """Fit with balanced class weights; num_class stays 3 even if a class is missing."""
    weights = compute_sample_weight('balanced', y)
    if len(set(groups)) >= 10:
        split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        train_index, valid_index = next(split.split(X, y, groups))
        return xgb.train(
            PARAMS,
            _matrix(X.iloc[train_index], y[train_index], weights[train_index]),
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(_matrix(X.iloc[valid_index], y[valid_index], weights[valid_index]), 'valid')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
    return xgb.train(PARAMS, _matrix(X, y, weights), num_boost_round=ROUNDS_WITHOUT_VALIDATION)


def _iterations(booster):
    best = booster.attr('best_iteration')
    return (0, int(best) + 1) if best is not None else (0, 0)


def predict_proba(booster, X):
    """Probabilities, shape (n, 3), columns in CATEGORIES order."""
    return booster.predict(_matrix(X), iteration_range=_iterations(booster))


def metrics(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    per_class_empty = {label: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
                       for label in CATEGORIES}
    if not y_true:
        return {'runs': 0, 'accuracy': None, 'macro_f1': None,
                'per_class': per_class_empty, 'confusion': [[0] * 3 for _ in range(3)]}
    # Macro F1 over the categories that occur, so a filter showing one
    # scenario without High students isn't penalised for an empty class.
    present = [label for label in CATEGORIES if label in y_true or label in y_pred]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CATEGORIES), zero_division=0,
    )
    return {
        'runs': len(y_true),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0)),
        'per_class': {
            label: {
                'precision': float(precision[i]), 'recall': float(recall[i]),
                'f1': float(f1[i]), 'support': int(support[i]),
            }
            for i, label in enumerate(CATEGORIES)
        },
        'confusion': confusion_matrix(y_true, y_pred, labels=list(CATEGORIES)).tolist(),
    }


def _targets(table):
    return table['label'].map(LABEL_INDEX).to_numpy()


def grouped_oof(table, columns, categories, progress, name='Final model'):
    """Out-of-fold predictions; no student is ever in its own fold's training data."""
    X = as_model_frame(table, columns, categories)
    y = _targets(table)
    groups = table['user_id'].to_numpy()
    folds = min(FOLDS, len(set(groups)))
    probabilities = np.zeros((len(table), 3))
    fold_of_row = np.zeros(len(table), dtype=int)
    for fold, (train_index, test_index) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        progress(f'{name}: cross-validation {fold + 1} / {folds}')
        booster = fit_booster(X.iloc[train_index], y[train_index], groups[train_index])
        probabilities[test_index] = predict_proba(booster, X.iloc[test_index])
        fold_of_row[test_index] = fold
    out = table[[]].copy()
    out['pred'] = [CATEGORIES[i] for i in probabilities.argmax(axis=1)]
    out['confidence'] = probabilities.max(axis=1)
    for i, label in enumerate(CATEGORIES):
        out[f'prob_{label}'] = probabilities[:, i]
    out['fold'] = fold_of_row
    return out


def leave_one_group_out(table, columns, categories, progress, name='Final model'):
    """Accuracy on each scenario group with enough runs, trained on the others."""
    run_counts = table.groupby('group_key')['implementation_id'].nunique()
    eligible = sorted(key for key, count in run_counts.items() if count >= LOSO_MIN_RUNS)
    X = as_model_frame(table, columns, categories)
    y = _targets(table)
    groups = table['user_id'].to_numpy()
    results = []
    for index, key in enumerate(eligible, start=1):
        progress(f'{name}: unseen scenarios {index} / {len(eligible)}')
        held = (table['group_key'] == key).to_numpy()
        # Students of the held-out scenario are left out of training
        # entirely, including their runs on other scenarios.
        train = ~held & ~table['user_id'].isin(set(table.loc[held, 'user_id'])).to_numpy()
        if len(set(y[train])) < 2:
            continue
        booster = fit_booster(X[train], y[train], groups[train])
        predicted = [CATEGORIES[i] for i in predict_proba(booster, X[held]).argmax(axis=1)]
        part = table[held]
        scores = metrics(part['label'], predicted)
        entry = {
            'group_key': int(key),
            'scenario_names': ', '.join(sorted(set(part['scenario_name']))),
            'runs': int(run_counts[key]),
            'accuracy': scores['accuracy'],
            'macro_f1': scores['macro_f1'],
        }
        if 'checkpoint' in table:
            by_checkpoint = {}
            frame = part.assign(model=predicted)
            for k, rows in frame.groupby('checkpoint'):
                by_checkpoint[str(int(k))] = {
                    'runs': int(len(rows)),
                    'accuracy': metrics(rows['label'], rows['model'])['accuracy'],
                    'baseline_accuracy': metrics(rows['label'], rows['baseline_label'])['accuracy'],
                }
            entry['by_checkpoint'] = by_checkpoint
        results.append(entry)
    return results


def importances(booster, X, columns, top=15):
    """Mean absolute SHAP value per input, from XGBoost's own contributions."""
    contributions = booster.predict(
        _matrix(X), pred_contribs=True, iteration_range=_iterations(booster),
    )
    # (rows, classes, inputs + bias): average over rows and classes, drop bias.
    values = np.abs(contributions).mean(axis=(0, 1))[:-1]
    ranking = sorted(zip(columns, values), key=lambda pair: (-pair[1], pair[0]))
    return [{'input': name, 'mean_abs_shap': round(float(value), 6)} for name, value in ranking[:top]]


def _fit_all(table, columns, categories):
    X = as_model_frame(table, columns, categories)
    return fit_booster(X, _targets(table), table['user_id'].to_numpy()), X


def _final_predictions(table, oof):
    rows = []
    for (_, run), (_, pred) in zip(table.iterrows(), oof.iterrows()):
        rows.append({
            'implementation_id': int(run.implementation_id),
            'user_id': int(run.user_id),
            'username': run.username,
            'scenario_id': int(run.scenario_id),
            'scenario_name': run.scenario_name,
            'language': run.language,
            'group_key': int(run.group_key),
            'is_current_version': bool(run.is_current_version),
            'pisa': run.label,
            'model': pred.pred,
            'confidence': round(float(pred.confidence), 4),
            'final_percentage': round(float(run.final_percentage), 2),
        })
    return rows


def _early_predictions(table, oof):
    rows = []
    for (_, run), (_, pred) in zip(table.iterrows(), oof.iterrows()):
        rows.append({
            'implementation_id': int(run.implementation_id),
            'scenario_id': int(run.scenario_id),
            'scenario_name': run.scenario_name,
            'language': run.language,
            'group_key': int(run.group_key),
            'is_current_version': bool(run.is_current_version),
            'checkpoint': int(run.checkpoint),
            'truth': run.label,
            'model': pred.pred,
            'baseline': run.baseline_label,
        })
    return rows


def train_all(final, checkpoints, progress):
    """Evaluate and fit both models. Returns (results, {'final', 'early'} boosters)."""
    started = time.monotonic()
    if len(final) < MIN_LABELLED_RUNS or final['label'].nunique() < 2:
        raise NotEnoughData(
            f'Not enough runs to train: {len(final)} found, at least '
            f'{MIN_LABELLED_RUNS} needed, with at least two categories.'
        )
    categories = model_categories(final, checkpoints)
    loso_seconds = 0.0

    columns = input_columns(final)
    oof = grouped_oof(final, columns, categories, progress)
    current = (final['is_current_version'] == True).to_numpy()  # noqa: E712
    loso_start = time.monotonic()
    final_loso = leave_one_group_out(final, columns, categories, progress)
    loso_seconds += time.monotonic() - loso_start
    progress('Final model: fitting on all runs')
    final_booster, final_X = _fit_all(final, columns, categories)
    results = {'final': {
        'overall': metrics(final['label'], oof['pred']),
        'by_version': {
            'current': metrics(final['label'][current], oof['pred'][current]),
            'older': metrics(final['label'][~current], oof['pred'][~current]),
        },
        'importances': importances(final_booster, final_X, columns),
        'loso': final_loso,
        'predictions': _final_predictions(final, oof),
    }}

    boosters = {'final': final_booster, 'early': None}
    early = {'checkpoints': [], 'importances': {}, 'loso': [], 'predictions': []}
    if len(checkpoints) and checkpoints['label'].nunique() >= 2:
        early_columns = input_columns(checkpoints)
        early_oof = grouped_oof(checkpoints, early_columns, categories, progress, name='Early model')
        frame = checkpoints.assign(model=early_oof['pred'].to_numpy())
        for k, rows in frame.groupby('checkpoint'):
            early['checkpoints'].append({
                'checkpoint': int(k),
                'xgb': metrics(rows['label'], rows['model']),
                'baseline': metrics(rows['label'], rows['baseline_label']),
            })
        loso_start = time.monotonic()
        early['loso'] = leave_one_group_out(
            checkpoints, early_columns, categories, progress, name='Early model',
        )
        loso_seconds += time.monotonic() - loso_start
        progress('Early model: fitting on all runs')
        early_booster, early_X = _fit_all(checkpoints, early_columns, categories)
        for k in sorted(checkpoints['checkpoint'].unique()):
            mask = (checkpoints['checkpoint'] == k).to_numpy()
            early['importances'][str(int(k))] = importances(early_booster, early_X[mask], early_columns)
        early['predictions'] = _early_predictions(checkpoints, early_oof)
        boosters['early'] = early_booster
    results['early'] = early
    results['seconds'] = {
        'total': round(time.monotonic() - started, 1),
        'loso': round(loso_seconds, 1),
    }
    return results, boosters
