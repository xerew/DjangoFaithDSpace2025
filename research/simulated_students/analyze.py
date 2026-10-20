"""Compare simulated students with real ones and write a report.

For each model and context: does the simulated class find the same
activities hard (rank correlation of difficulty), choose the same wrong
options, and separate by ability like real low/mid/high students? Results are
also broken down by language and by whether the activity has images.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

from . import metrics

PERSONAS = ('low', 'mid', 'high')


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def item_table(items, rows, model, context):
    """One row per activity: real vs simulated difficulty and option choices."""
    by_item = defaultdict(list)
    for row in rows:
        if row['model'] == model and row['context'] == context:
            by_item[row['item_id']].append(row)
    table = []
    for item in items:
        answers = by_item.get(item['item_id'])
        if not answers:
            continue
        valid = [a for a in answers if a['choice_option_id'] is not None]
        if not valid:
            continue
        correct_id = next(o['id'] for o in item['options'] if o['is_correct'])
        sim_counts = defaultdict(int)
        for answer in valid:
            sim_counts[str(answer['choice_option_id'])] += 1
        sim_shares = {k: v / len(valid) for k, v in sim_counts.items()}
        real = item['real']
        record = {
            'model': model, 'context': context, 'item_id': item['item_id'],
            'language': item.get('language') or '', 'has_images': bool(
                item.get('images') or any(o.get('image') for o in item['options'])),
            'n_real': real['n'], 'p_real': real['p_correct'],
            'n_sim': len(valid), 'invalid': len(answers) - len(valid),
            'p_sim': _mean([1.0 if a['correct'] else 0.0 for a in valid]),
            'jsd': metrics.jsd(real['option_shares'], sim_shares),
            'real_top_distractor': metrics.top_distractor(real['option_shares'], correct_id),
            'sim_top_distractor': metrics.top_distractor(sim_shares, correct_id),
            'n_wrong_options': len(item['options']) - 1,
        }
        for persona in PERSONAS:
            mine = [a for a in valid if a['persona'] == persona]
            record[f'p_{persona}'] = _mean([1.0 if a['correct'] else 0.0 for a in mine]) if mine else None
            group = real.get('by_ability', {}).get(persona)
            record[f'real_{persona}'] = group['p_correct'] if group else None
        table.append(record)
    return table


def summarize(table, rounds=2000):
    real = [row['p_real'] for row in table]
    sim = [row['p_sim'] for row in table]
    low_ci, high_ci = (metrics.bootstrap_ci(real, sim, metrics.spearman, rounds=rounds)
                       if len(table) >= 3 else (float('nan'), float('nan')))
    both = [row for row in table if row['real_top_distractor'] and row['sim_top_distractor']]
    persona_rows = [
        [row[f'p_{p}'] for p in PERSONAS if row[f'p_{p}'] is not None]
        for row in table
    ]
    persona_rows = [values for values in persona_rows if len(values) >= 2]
    pairs = [
        (row[f'p_{p}'], row[f'real_{p}']) for row in table for p in PERSONAS
        if row[f'p_{p}'] is not None and row[f'real_{p}'] is not None
    ]
    total = sum(row['n_sim'] + row['invalid'] for row in table)
    return {
        'items': len(table),
        'spearman': metrics.spearman(real, sim) if len(table) >= 2 else float('nan'),
        'spearman_ci': (low_ci, high_ci),
        'pearson': metrics.pearson(real, sim) if len(table) >= 2 else float('nan'),
        'mae': metrics.mae(real, sim) if table else float('nan'),
        'mean_jsd': _mean([row['jsd'] for row in table]),
        'distractor_items': len(both),
        'distractor_agreement': (_mean([1.0 if r['real_top_distractor'] == r['sim_top_distractor'] else 0.0
                                        for r in both]) if both else None),
        'distractor_chance': _mean([1 / r['n_wrong_options'] for r in both]) if both else None,
        'persona_monotonic_share': (_mean([1.0 if values == sorted(values) else 0.0 for values in persona_rows])
                                    if persona_rows else None),
        'persona_vs_real_spearman': (metrics.spearman([a for a, _ in pairs], [b for _, b in pairs])
                                     if len(pairs) >= 3 else float('nan')),
        'invalid_rate': (sum(row['invalid'] for row in table) / total) if total else None,
    }


def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '–'
    return f'{value:.2f}'


def _subsets(table):
    yield 'all', table
    for language in sorted({row['language'] for row in table}):
        yield language or '(no language)', [row for row in table if row['language'] == language]
    yield 'text only', [row for row in table if not row['has_images']]
    yield 'with images', [row for row in table if row['has_images']]


def write_report(items, rows, out_dir, rounds=2000):
    os.makedirs(out_dir, exist_ok=True)
    groups = sorted({(row['model'], row['context']) for row in rows})
    lines = [
        '# Simulated students vs real students', '',
        '| model | context | subset | items | Spearman [95% CI] | Pearson | MAE | mean JSD '
        '| top distractor agree (chance) | personas ordered | sim vs real by ability | invalid |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    all_rows = []
    summaries = []
    for model, context in groups:
        table = item_table(items, rows, model, context)
        all_rows.extend(table)
        for subset, part in _subsets(table):
            if not part:
                continue
            s = summarize(part, rounds=rounds)
            summaries.append({'model': model, 'context': context, 'subset': subset, **s})
            ci = f"{_fmt(s['spearman'])} [{_fmt(s['spearman_ci'][0])}, {_fmt(s['spearman_ci'][1])}]"
            lines.append(
                f"| {model} | {context} | {subset} | {s['items']} | {ci} | {_fmt(s['pearson'])} "
                f"| {_fmt(s['mae'])} | {_fmt(s['mean_jsd'])} "
                f"| {_fmt(s['distractor_agreement'])} ({_fmt(s['distractor_chance'])}) "
                f"| {_fmt(s['persona_monotonic_share'])} | {_fmt(s['persona_vs_real_spearman'])} "
                f"| {_fmt(s['invalid_rate'])} |"
            )
    lines += [
        '', 'Spearman/Pearson/MAE compare the real and simulated share of correct answers per activity. '
        'Mean JSD (0 = identical, 1 = disjoint) compares how answers spread over the options. '
        '"Personas ordered" is the share of activities where low <= mid <= high. '
        '"Sim vs real by ability" correlates each persona with the real students of that ability group.',
    ]
    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summaries, handle, indent=2, default=str)
    if all_rows:
        with open(os.path.join(out_dir, 'items.csv'), 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--items', required=True)
    parser.add_argument('--results', action='append', required=True, help='results JSONL, repeatable')
    parser.add_argument('--out', required=True, help='folder for report.md, summary.json, items.csv')
    parser.add_argument('--rounds', type=int, default=2000, help='bootstrap rounds')
    args = parser.parse_args(argv)
    from .simulate import load_items
    rows = []
    for path in args.results:
        with open(path, encoding='utf-8') as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    write_report(load_items(args.items), rows, args.out, rounds=args.rounds)
    print(f'Report written to {os.path.join(args.out, "report.md")}')


if __name__ == '__main__':
    main()
