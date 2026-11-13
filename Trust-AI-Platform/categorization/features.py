"""Model inputs that mean the same thing in every scenario.

No activity ids, phase names or raw points: phases are described by their
position (1st, 2nd, ...), times relative to other students on the same
activity. A checkpoint row (after phase k) uses only answers and help from
phases 1..k, so the early model never sees the future.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import pandas as pd

from authoringtool.performance_formula import (
    CATEGORIES,
    categorize,
    phase_weights,
    weighted_percentage,
)

from .dataset import run_until

MAX_POSITIONS = 6
FAST_ANSWER_RATIO = 0.1
FAST_ANSWER_RULE = (
    'An answer is very fast when its time is under 10% of the median time '
    'on that activity.'
)
META_COLUMNS = [
    'implementation_id', 'user_id', 'username', 'scenario_id', 'scenario_name',
    'language', 'group_key', 'is_current_version', 'label', 'final_percentage',
]
CATEGORICAL = ['scenario', 'language']
NAN = float('nan')


@dataclass
class Norms:
    """Label-free statistics over all runs."""
    time_median: dict
    visit_share: dict


def build_norms(runs):
    times = defaultdict(list)
    visits = defaultdict(int)
    runs_per_scenario = defaultdict(int)
    for run in runs:
        runs_per_scenario[run.scenario_id] += 1
        seen = set()
        for event in run.answers:
            if event.activity_id is None:
                continue
            seen.add(event.activity_id)
            if event.is_last and event.active_seconds:
                times[event.activity_id].append(event.active_seconds)
        for activity_id in seen:
            visits[(run.scenario_id, activity_id)] += 1
    return Norms(
        time_median={aid: median(values) for aid, values in times.items()},
        visit_share={
            key: count / runs_per_scenario[key[0]] for key, count in visits.items()
        },
    )


def _mean(values):
    return sum(values) / len(values) if values else NAN


def run_inputs(run, norms, upto=None):
    """Inputs for a whole run (upto=None) or for phases 1..upto only."""
    limit = run.phase_count if upto is None else upto
    events = [
        e for e in run.answers
        if e.activity_id is not None and e.phase_position is not None and e.phase_position <= limit
    ]
    last = [e for e in events if e.is_last]
    weights = phase_weights(run.phase_count)
    inputs = {}
    for position in range(1, MAX_POSITIONS + 1):
        reached = position <= limit and position <= run.phase_count
        pct = share = category = ratio = NAN
        if reached:
            total, maximum = run.phase_scores[position - 1]
            if maximum > 0:
                pct = total / maximum * 100
                category = float(CATEGORIES.index(categorize(pct)))
            answered, scored = run.scored_by_position.get(position, (0, 0))
            if scored:
                share = answered / scored
            ratios = [
                e.active_seconds / norms.time_median[e.activity_id]
                for e in last
                if e.phase_position == position and e.active_seconds
                and norms.time_median.get(e.activity_id)
            ]
            ratio = median(ratios) if ratios else NAN
        # Phase weights are scenario structure, known from the start.
        weight = (
            (weights[position - 1] if position - 1 < len(weights) else 0.0)
            if position <= run.phase_count else NAN
        )
        inputs[f'p{position}_pct'] = pct
        inputs[f'p{position}_answered_share'] = share
        inputs[f'p{position}_category'] = category
        inputs[f'p{position}_weight'] = weight
        inputs[f'p{position}_time_ratio'] = ratio

    timed = [
        e for e in last if e.active_seconds and norms.time_median.get(e.activity_id)
    ]
    fast = [
        e for e in timed
        if e.active_seconds < FAST_ANSWER_RATIO * norms.time_median[e.activity_id]
    ]
    help_positions = [
        value for position, value in run.help_by_position.items() if position <= limit
    ]
    answered_ids = {e.activity_id for e in events}
    inputs.update({
        'phase_count': float(run.phase_count),
        'fast_share': len(fast) / len(timed) if timed else NAN,
        'retries': float(len(events) - len(answered_ids)),
        'help_requests': float(sum(v['requests'] for v in help_positions)),
        'help_seconds': float(sum(v['seconds'] for v in help_positions)),
        'path_commonness': _mean([
            norms.visit_share.get((run.scenario_id, aid), 0.0) for aid in answered_ids
        ]),
        'is_current_version': float(run.is_current_version),
        'scenario': str(run.scenario_id),
        'language': run.language or 'Unspecified',
    })
    return inputs


def _meta(run):
    return {
        'implementation_id': run.implementation_id,
        'user_id': run.user_id,
        'username': run.username,
        'scenario_id': run.scenario_id,
        'scenario_name': run.scenario_name,
        'language': run.language or 'Unspecified',
        'group_key': run.group_key,
        'is_current_version': run.is_current_version,
        'label': run.label,
        'final_percentage': run.final_percentage,
    }


def _frame(rows):
    return pd.DataFrame(rows).reset_index(drop=True)


def final_table(runs, norms):
    """One row per labelled run, inputs from the whole run."""
    rows = []
    for run in runs:
        if run.label is None:
            continue
        row = _meta(run)
        row.update(run_inputs(run, norms, upto=None))
        rows.append(row)
    return _frame(rows)


def checkpoint_table(runs, norms):
    """One row per labelled run and checkpoint k (1..phases-1) it moved past.

    "After phase k" is the moment the student first answered past phase k.
    A student who stopped inside phase k never reached that moment, so the
    run has no row for k: their final state is not a picture of a student
    who just finished phase k.
    """
    rows = []
    for run in runs:
        if run.label is None:
            continue
        weights = phase_weights(run.phase_count)
        reached = max((e.phase_position for e in run.answers if e.phase_position), default=0)
        for k in range(1, min(run.phase_count, reached)):
            # Only what the student had done before first moving past phase k.
            at_k = run_until(run, k)
            if not any(e.phase_position == k for e in at_k.answers):
                continue
            so_far = weighted_percentage(at_k.phase_scores[:k], weights)
            if so_far is None:
                continue
            row = _meta(run)
            row.update(run_inputs(at_k, norms, upto=k))
            row['checkpoint'] = float(k)
            row['pct_so_far'] = so_far
            row['baseline_label'] = categorize(so_far)
            rows.append(row)
    return _frame(rows)


def input_columns(table):
    """Every model input: all columns but the meta ones and the baseline.

    `language` and `is_current_version` are both meta columns and inputs.
    """
    meta = set(META_COLUMNS) - {'language', 'is_current_version'}
    return [c for c in table.columns if c not in meta and c != 'baseline_label']


def model_categories(*tables):
    """Fixed category lists for the categorical inputs, from all tables."""
    categories = {}
    for column in CATEGORICAL:
        values = set()
        for table in tables:
            if column in table:
                values.update(table[column].astype(str))
        categories[column] = sorted(values)
    return categories


def as_model_frame(table, columns, categories):
    frame = table[columns].copy()
    for column in columns:
        if column in CATEGORICAL:
            frame[column] = pd.Categorical(frame[column].astype(str), categories=categories[column])
        else:
            frame[column] = frame[column].astype(float)
    return frame
