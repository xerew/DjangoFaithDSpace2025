"""Compare simulated walk-throughs of whole scenarios with real students.

For each model and scenario: do simulated students take the same routes at
the choice points, land in the same evaluation bands (high/mid/low), get
similar total scores, follow the paths most real students follow, find the
same activities hard, and do the personas order themselves low < mid < high?
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from . import metrics
from .walkthrough import RouteError, load_scenarios, nodes_of, route

PERSONAS = ('low', 'mid', 'high')
MIN_REAL_ITEM = 10   # students per activity before its difficulty is compared
MIN_SIM_ITEM = 3     # simulated students per activity
FAILED = ('invalid_reply', 'error')


def run_of(row):
    return [(step['activity_id'], step.get('choice_option_id')) for step in row['steps']]


def point_biserial(pairs):
    """Correlation between getting an item right (0/1) and the rest score."""
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if not sxx or not syy:
        return None
    return sxy / (sxx * syy) ** 0.5


def summarise(nodes, runs, min_path_students=1):
    """The same aggregates export_scenario_bank computes for real students
    (keep the two in step), here for simulated runs. Simulated students
    always follow the current routes, so no run is left out."""
    transitions = defaultdict(Counter)
    bands = defaultdict(Counter)
    scores, lengths, paths = Counter(), Counter(), Counter()
    first_answers = defaultdict(list)
    completed = 0
    weights = {o['id']: o['weight'] for n in nodes.values() for o in n['options']}
    for run in runs:
        finished = False
        for index, (activity_id, option_id) in enumerate(run):
            node = nodes.get(activity_id)
            try:
                following, band = route(nodes, node, option_id, run[:index + 1]) if node else (None, None)
            except RouteError:
                following, band = 'error', None
            if band:
                bands[activity_id][band] += 1
            if index + 1 < len(run):
                actual = run[index + 1][0]
                transitions[activity_id][str(actual)] += 1
            elif following is None:
                finished = True
                transitions[activity_id]['END'] += 1
            else:
                transitions[activity_id]['STOP'] += 1
        if finished:
            completed += 1
            scores[str(sum(weights.get(option, 0) for _, option in run))] += 1
            lengths[str(len(run))] += 1
            paths[tuple(activity for activity, _ in run)] += 1

        right = {}
        for activity_id, option_id in run:
            node = nodes.get(activity_id)
            if node and node['kind'] == 'question' and activity_id not in right:
                has_correct = any(o['is_correct'] for o in node['options'])
                correct = next((o['is_correct'] for o in node['options'] if o['id'] == option_id), False)
                right[activity_id] = (option_id, correct if has_correct else None)
        for activity_id, (option_id, correct) in right.items():
            others = [ok for other, (_, ok) in right.items() if other != activity_id and ok is not None]
            rest = sum(others) / len(others) if others else None
            first_answers[activity_id].append((option_id, correct, rest))

    items = {}
    for activity_id, answers in first_answers.items():
        n = len(answers)
        graded = [correct for _, correct, _ in answers if correct is not None]
        items[str(activity_id)] = {
            'n': n,
            'p_correct': sum(graded) / len(graded) if graded else None,
            'option_shares': {str(option): count / n
                              for option, count in Counter(o for o, _, _ in answers).items()},
            'point_biserial': point_biserial(
                [(int(correct), rest) for _, correct, rest in answers
                 if correct is not None and rest is not None]
            ),
        }

    return {
        'students': len(runs),
        'completed': completed,
        'transitions': {str(key): dict(value) for key, value in transitions.items()},
        'branching': {str(key): dict(value) for key, value in bands.items()},
        'items': items,
        'score_histogram': dict(scores),
        'path_length_histogram': dict(lengths),
        'paths': [
            {'path': list(path), 'n': n}
            for path, n in sorted(paths.items(), key=lambda pair: (-pair[1], pair[0]))
            if n >= min_path_students
        ],
    }


# ── comparison ────────────────────────────────────────────────────────────
def _shares(counts):
    counts = {k: v for k, v in (counts or {}).items() if k != 'STOP'}
    total = sum(counts.values())
    return ({k: v / total for k, v in counts.items()}, total) if total else (None, 0)


def _weighted_jsd(real, sim, keys):
    total = weight = 0.0
    for key in keys:
        p, n = _shares(real.get(key))
        q, _ = _shares(sim.get(key))
        if p and q:
            total += n * metrics.jsd(p, q)
            weight += n
    return total / weight if weight else None


def _mean_of_histogram(histogram):
    n = sum(histogram.values())
    return sum(float(k) * v for k, v in histogram.items()) / n if n else None


def _paired_spearman(real_items, sim_items, field):
    pairs = [
        (real[field], sim_items[key][field])
        for key, real in real_items.items()
        if key in sim_items and real['n'] >= MIN_REAL_ITEM and sim_items[key]['n'] >= MIN_SIM_ITEM
        and real.get(field) is not None and sim_items[key].get(field) is not None
    ]
    if len(pairs) < 3:
        return None, len(pairs)
    return metrics.spearman(*zip(*pairs)), len(pairs)


def choice_points(nodes):
    """Questions where students can go different ways."""
    points = []
    for activity_id, node in nodes.items():
        if node['kind'] != 'question':
            continue
        if node['evaluation'] or len({o['next'] for o in node['options']}) > 1:
            points.append(str(activity_id))
    return points


def compare(scenario, rows):
    """Metrics for one scenario and one model; rows are its simulated students."""
    real = scenario['real']
    nodes = nodes_of(scenario)
    sim = summarise(nodes, [run_of(row) for row in rows])

    real_paths = {tuple(p['path']): p['n'] for p in real['paths']}
    sim_paths = {tuple(p['path']): p['n'] for p in sim['paths']}
    real_frequent = sum(real_paths.values())

    persona_scores = defaultdict(list)
    for row in rows:
        if row['end_reason'] == 'end':
            persona_scores[row['persona']].append(row['total_score'])
    persona_mean = {p: sum(s) / len(s) for p, s in persona_scores.items()}
    ordered = [persona_mean[p] for p in PERSONAS if p in persona_mean]
    difficulty, difficulty_items = _paired_spearman(real['items'], sim['items'], 'p_correct')
    discrimination, _ = _paired_spearman(real['items'], sim['items'], 'point_biserial')

    return {
        'scenario_id': scenario['scenario_id'],
        'scenario': scenario['scenario'],
        'language': scenario.get('language', ''),
        'students_real': real['runs_used'],
        'students_sim': len(rows),
        'completion_real': real['completed'] / real['runs_used'] if real['runs_used'] else None,
        'completion_sim': sim['completed'] / sim['students'] if sim['students'] else None,
        'choice_point_jsd': _weighted_jsd(real['transitions'], sim['transitions'], choice_points(nodes)),
        'band_jsd': _weighted_jsd(real['branching'], sim['branching'], real['branching']),
        'mean_score_real': _mean_of_histogram(real['score_histogram']),
        'mean_score_sim': _mean_of_histogram(sim['score_histogram']),
        'score_wasserstein': metrics.wasserstein(real['score_histogram'], sim['score_histogram']),
        'mean_length_real': _mean_of_histogram(real['path_length_histogram']),
        'mean_length_sim': _mean_of_histogram(sim['path_length_histogram']),
        'frequent_path_share_real': real_frequent / real['completed'] if real['completed'] else None,
        'frequent_path_share_sim': (sum(n for path, n in sim_paths.items() if path in real_paths)
                                    / sim['completed']) if sim['completed'] else None,
        'path_coverage_real': (sum(n for path, n in real_paths.items() if path in sim_paths)
                               / real_frequent) if real_frequent else None,
        'difficulty_spearman': difficulty,
        'difficulty_items': difficulty_items,
        'discrimination_spearman': discrimination,
        'persona_mean_score': persona_mean,
        'persona_ordered': all(a < b for a, b in zip(ordered, ordered[1:])) if len(ordered) > 1 else None,
        'invalid_rate': sum(row['end_reason'] in FAILED for row in rows) / len(rows) if rows else None,
        'route_error_rate': sum(row['end_reason'] == 'route_error' for row in rows) / len(rows) if rows else None,
        'real_follows_routes': real.get('follows_routes'),
    }


# ── report ────────────────────────────────────────────────────────────────
COLUMNS = [
    ('students_real', 'Real students (current routes)'), ('students_sim', 'Simulated'),
    ('completion_real', 'Completed real'), ('completion_sim', 'Completed sim'),
    ('choice_point_jsd', 'Route JSD'), ('band_jsd', 'Band JSD'),
    ('mean_score_real', 'Score real'), ('mean_score_sim', 'Score sim'),
    ('score_wasserstein', 'Score W1'),
    ('frequent_path_share_real', 'Frequent paths real'), ('frequent_path_share_sim', 'Frequent paths sim'),
    ('path_coverage_real', 'Real paths covered'),
    ('difficulty_spearman', 'Difficulty ρ'), ('discrimination_spearman', 'Discrimination ρ'),
    ('persona_ordered', 'Personas ordered'), ('invalid_rate', 'Invalid'),
    ('real_follows_routes', 'Real follow routes'),
]


def _fmt(value):
    if value is None:
        return '–'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def write_report(scenarios, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row['model'], row['scenario_id'])].append(row)
    models = sorted({row['model'] for row in rows})

    summary, table = {'models': {}}, []
    lines = ['# Simulated walk-throughs vs real students', '',
             'Route JSD: divergence of the next activity chosen at choice points (0 = same). '
             'Band JSD: divergence of high/mid/low evaluation outcomes. '
             'Score W1: how far total scores are apart, in score points. '
             'Frequent paths: share of completed runs on a path that at least the export threshold '
             'of real students took. Real follow routes: share of real runs consistent with the '
             'current routes (lower means the scenario changed while students used it); only those runs '
             'are compared.', '']
    for model in models:
        results = [compare(s, by_key[(model, s['scenario_id'])])
                   for s in scenarios if by_key.get((model, s['scenario_id']))]
        means = {key: _mean(r[key] for r in results) for key, _ in COLUMNS}
        summary['models'][model] = {'scenarios': len(results), 'mean': means, 'per_scenario': results}
        table += [dict(model=model, **{k: v for k, v in r.items() if k != 'persona_mean_score'},
                       **{f'persona_{p}': r['persona_mean_score'].get(p) for p in PERSONAS})
                  for r in results]

        lines += [f'## {model}', '',
                  '| Scenario | Language | ' + ' | '.join(label for _, label in COLUMNS) + ' |',
                  '|---|---|' + '---|' * len(COLUMNS)]
        for r in results:
            lines.append(f'| {r["scenario"]} | {r["language"]} | '
                         + ' | '.join(_fmt(r[key]) for key, _ in COLUMNS) + ' |')
        lines.append('| **Mean** | | ' + ' | '.join(_fmt(means[key]) for key, _ in COLUMNS) + ' |')
        lines.append('')

    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'scenarios.csv'), 'w', encoding='utf-8', newline='') as handle:
        if table:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenarios', required=True, help='scenario_bank.jsonl from export_scenario_bank')
    parser.add_argument('--results', action='append', required=True, help='walkthrough results JSONL')
    parser.add_argument('--out', required=True, help='folder for report.md, summary.json, scenarios.csv')
    args = parser.parse_args(argv)
    rows = []
    for path in args.results:
        with open(path, encoding='utf-8') as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    write_report(load_scenarios(args.scenarios), rows, args.out)
    print(f'Report written to {args.out}')


if __name__ == '__main__':
    main()
