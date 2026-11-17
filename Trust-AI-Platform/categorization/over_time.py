"""Over time: the models as if they had been running since the first student.

Runs are ordered by the date of their first answer and grouped by month;
a month with fewer than MIN_PERIOD_RUNS runs is joined with the next one
(the last small months with the period before). For each period, both
models are trained only on the runs that started before it, including the
time norms, and tested on the period's runs. A student who comes back in a
later period has their earlier runs in the training data, as they would
in real use.
"""

from collections import defaultdict

from django.utils import timezone

from authoringtool.performance_formula import CATEGORIES

from .features import (
    build_norms,
    as_model_frame,
    checkpoint_table,
    final_table,
    input_columns,
    model_categories,
)
from .training import LABEL_INDEX, fit_booster, metrics, predict_proba

MIN_PERIOD_RUNS = 30
MIN_TRAIN_RUNS = 50


def started_on(run):
    return min(event.created_on for event in run.answers)


def periods(runs, min_runs=MIN_PERIOD_RUNS):
    """[(first (year, month), last (year, month), runs)] in date order."""
    by_month = defaultdict(list)
    for run in runs:
        when = started_on(run)
        # Answers are stored in UTC; a month starts at midnight on the platform (Athens).
        if timezone.is_aware(when):
            when = timezone.localtime(when)
        by_month[(when.year, when.month)].append(run)
    found = []
    current = None
    for month in sorted(by_month):
        if current is None:
            current = [month, month, []]
        current[1] = month
        current[2].extend(by_month[month])
        if len(current[2]) >= min_runs:
            found.append(tuple(current))
            current = None
    if current:
        if found:
            first, _, part = found.pop()
            found.append((first, current[1], part + current[2]))
        else:
            found.append(tuple(current))
    return found


def _short(scores):
    return {key: scores[key] for key in ('runs', 'accuracy', 'macro_f1')}


def _predict(train, test, categories):
    """Fit on `train`, return the predicted labels of `test`."""
    columns = input_columns(train)
    test = test.copy()
    for column in columns:
        if column not in test:
            test[column] = float('nan')
    booster = fit_booster(
        as_model_frame(train, columns, categories),
        train['label'].map(LABEL_INDEX).to_numpy(),
        train['user_id'].to_numpy(),
    )
    probabilities = predict_proba(booster, as_model_frame(test, columns, categories))
    return [CATEGORIES[i] for i in probabilities.argmax(axis=1)]


def _evaluate_period(past, current):
    norms = build_norms(past)
    train_final, test_final = final_table(past, norms), final_table(current, norms)
    train_early, test_early = checkpoint_table(past, norms), checkpoint_table(current, norms)
    categories = model_categories(train_final, test_final, train_early, test_early)
    result = {'final': {'xgb': _short(metrics(
        test_final['label'], _predict(train_final, test_final, categories),
    ))}}
    checkpoints = {}
    if len(train_early) and train_early['label'].nunique() >= 2 and len(test_early):
        frame = test_early.assign(model=_predict(train_early, test_early, categories))
        for k, rows in frame.groupby('checkpoint'):
            checkpoints[str(int(k))] = {
                'xgb': _short(metrics(rows['label'], rows['model'])),
                'baseline': _short(metrics(rows['label'], rows['baseline_label'])),
            }
    result['checkpoints'] = checkpoints
    return result


def evaluate(runs, progress=lambda text: None):
    labelled = [run for run in runs if run.label is not None and run.answers]
    found = periods(labelled)
    out = []
    past = []
    for index, (first, last, current) in enumerate(found, start=1):
        progress(f'Over time: period {index} / {len(found)}')
        entry = {
            'start': f'{first[0]}-{first[1]:02d}',
            'end': f'{last[0]}-{last[1]:02d}',
            'trained_on': len(past),
            'students_trained': len({run.user_id for run in past}),
            'tested': len(current),
        }
        if len(past) >= MIN_TRAIN_RUNS and len({run.label for run in past}) >= 2:
            entry.update(_evaluate_period(past, current))
        out.append(entry)
        past = past + current
    return {'min_period_runs': MIN_PERIOD_RUNS, 'min_train_runs': MIN_TRAIN_RUNS, 'periods': out}
