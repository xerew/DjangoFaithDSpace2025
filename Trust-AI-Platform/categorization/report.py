"""Dashboard numbers from a training run's stored out-of-fold predictions.

Filters narrow the stored predictions and the measures are recomputed from
them; nothing is retrained.
"""

from collections import defaultdict
from datetime import date

from django.utils.formats import date_format

from authoringtool.performance_formula import HIGH_THRESHOLD, MODERATE_THRESHOLD

from .training import metrics


def filter_rows(rows, scenario='', language='', version=''):
    selected = []
    for row in rows:
        if scenario and str(row['scenario_id']) != str(scenario):
            continue
        if language and row['language'] != language:
            continue
        if version == 'current' and not row['is_current_version']:
            continue
        if version == 'older' and row['is_current_version']:
            continue
        selected.append(row)
    return selected


def _group_names(rows):
    names = defaultdict(set)
    for row in rows:
        names[row['group_key']].add(row['scenario_name'])
    return {key: ', '.join(sorted(values)) for key, values in names.items()}


def final_view(results, filters):
    final = results['final']
    rows = filter_rows(final['predictions'], **filters)
    disagreements = []
    for row in rows:
        if row['pisa'] == row['model']:
            continue
        pct = row['final_percentage']
        disagreements.append({
            **row,
            'threshold_distance': round(
                min(abs(pct - HIGH_THRESHOLD), abs(pct - MODERATE_THRESHOLD)), 2,
            ),
        })
    disagreements.sort(key=lambda row: (-row['confidence'], row['implementation_id']))

    by_group = defaultdict(list)
    for row in rows:
        by_group[row['group_key']].append(row)
    loso = {entry['group_key']: entry['accuracy'] for entry in final['loso']}
    names = _group_names(rows)
    per_scenario = [
        {
            'group_key': key,
            'names': names[key],
            'runs': len(group),
            'agreement': metrics([r['pisa'] for r in group], [r['model'] for r in group])['accuracy'],
            'loso': loso.get(key),
        }
        for key, group in sorted(by_group.items(), key=lambda item: names[item[0]].casefold())
    ]
    overall = metrics([r['pisa'] for r in rows], [r['model'] for r in rows])
    by_version = {}
    for name, flag in (('current', True), ('older', False)):
        part = [r for r in rows if r['is_current_version'] == flag]
        by_version[name] = metrics([r['pisa'] for r in part], [r['model'] for r in part])
    return {
        'overall': overall,
        'confusion': overall['confusion'],
        'by_version': by_version,
        'disagreements': disagreements,
        'per_scenario': per_scenario,
        'importances': final['importances'],
    }


def early_view(results, filters, checkpoint=None):
    early = results['early']
    rows = filter_rows(early['predictions'], **filters)
    by_checkpoint = defaultdict(list)
    for row in rows:
        by_checkpoint[row['checkpoint']].append(row)
    table = []
    for k in sorted(by_checkpoint):
        group = by_checkpoint[k]
        truth = [r['truth'] for r in group]
        table.append({
            'checkpoint': k,
            'xgb': metrics(truth, [r['model'] for r in group]),
            'baseline': metrics(truth, [r['baseline'] for r in group]),
        })
    checkpoints = [entry['checkpoint'] for entry in table]
    picked = checkpoint if checkpoint in checkpoints else (checkpoints[0] if checkpoints else None)
    picked_row = next((entry for entry in table if entry['checkpoint'] == picked), None)
    loso = []
    for entry in early['loso']:
        at_k = entry.get('by_checkpoint', {}).get(str(picked))
        if at_k:
            loso.append({
                'group_key': entry['group_key'],
                'scenario_names': entry['scenario_names'],
                **at_k,
            })
    return {
        'rows': table,
        'checkpoints': checkpoints,
        'picked': picked,
        'confusion': picked_row['xgb']['confusion'] if picked_row else None,
        'importances': early['importances'].get(str(picked), []) if picked is not None else [],
        'loso': loso,
        'chart': {
            'checkpoints': checkpoints,
            'xgb_accuracy': [entry['xgb']['accuracy'] for entry in table],
            'baseline_accuracy': [entry['baseline']['accuracy'] for entry in table],
            'xgb_f1': [entry['xgb']['macro_f1'] for entry in table],
            'baseline_f1': [entry['baseline']['macro_f1'] for entry in table],
        },
    }


def filter_options(results):
    rows = results.get('final', {}).get('predictions', [])
    scenarios = sorted(
        {(row['scenario_id'], row['scenario_name']) for row in rows},
        key=lambda pair: pair[1].casefold(),
    )
    languages = sorted({row['language'] for row in rows}, key=str.casefold)
    return {'scenarios': scenarios, 'languages': languages}


def _period_label(start, end):
    def month(text):
        year, number = map(int, text.split('-'))
        return date_format(date(year, number, 1), 'M Y')
    return month(start) if start == end else f'{month(start)} – {month(end)}'


def over_time_view(results, prediction=None):
    """One row per period for the picked prediction ('1', '2', ... or 'final')."""
    data = results.get('over_time')
    if not data:
        return None
    if 'error' in data:
        return {'error': data['error']}
    periods = data['periods']
    checkpoints = sorted({k for p in periods for k in p.get('checkpoints', {})}, key=int)
    options = checkpoints + ['final']
    picked = prediction if prediction in options else options[0]
    rows = []
    for period in periods:
        if picked == 'final':
            xgb, baseline = period.get('final', {}).get('xgb'), None
        else:
            entry = period.get('checkpoints', {}).get(picked, {})
            xgb, baseline = entry.get('xgb'), entry.get('baseline')
        rows.append({
            'label': _period_label(period['start'], period['end']),
            'trained_on': period['trained_on'],
            'students_trained': period['students_trained'],
            'tested': xgb['runs'] if xgb else period['tested'],
            'xgb': xgb,
            'baseline': baseline,
            'evaluated': xgb is not None,
        })

    def series(kind, key):
        return [row[kind][key] if row[kind] else None for row in rows]

    return {
        'options': options,
        'picked': picked,
        'rows': rows,
        'min_period_runs': data.get('min_period_runs'),
        'min_train_runs': data.get('min_train_runs'),
        'chart': {
            'labels': [row['label'] for row in rows],
            'xgb_accuracy': series('xgb', 'accuracy'),
            'baseline_accuracy': series('baseline', 'accuracy'),
            'xgb_f1': series('xgb', 'macro_f1'),
            'baseline_f1': series('baseline', 'macro_f1'),
            'trained_on': [row['trained_on'] for row in rows],
            'tested': [row['tested'] for row in rows],
        },
    }
