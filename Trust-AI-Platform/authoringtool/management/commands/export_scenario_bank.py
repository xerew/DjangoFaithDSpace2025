"""Export whole scenarios and how real students went through them.

Used by the scenario walk-through of the simulated-students study
(research/simulated_students/walkthrough.py). Each line of the output is one
scenario: its activities, the routes between them exactly as Plato (Rasa)
follows them, and aggregate results of real students' runs: transitions,
evaluation outcomes, item statistics, score and path-length distributions,
and the paths that at least --min-path-students students took. No data about
individual students is written.

Simulated students walk the scenario as it is now, so the aggregates use only
the runs that match its current routes; runs from before a teacher changed the
routes are counted (follows_routes) but not summarised.
"""
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from django.core.management.base import BaseCommand

from authoringtool.models import (
    Activity, Answer, EvQuestionBranching, NextQuestionLogic, QuestionBunch, Scenario, UserAnswer,
)
from authoringtool.research_export import option_record, plain, text_with_images

GOALS_LIMIT = 3000


class RouteError(Exception):
    """The scenario is configured so that Plato cannot continue (it shows an error)."""


def route(nodes, node, option_id, answered):
    """Next activity id (None = end of scenario) and evaluation band, as Rasa decides.

    answered: (activity id, option id) of every answer so far, this one included.
    """
    if node['kind'] == 'page':
        return node['next'], None
    evaluation = node['evaluation']
    if evaluation is None:
        option = next((o for o in node['options'] if o['id'] == option_id), None)
        return (option['next'] if option else None), None
    if not evaluation['high'] or evaluation['high']['to'] is None or evaluation['mid']['to'] is None:
        raise RouteError(f'activity {node["id"]}: evaluation branching incomplete')
    weights = {o['id']: o['weight'] for n in nodes.values() for o in n['options']}
    bunch = set(evaluation['bunch'])
    scored = [(activity, option) for activity, option in answered if activity in bunch]
    average = sum(weights.get(option, 0) for _, option in scored) / (len({a for a, _ in scored}) or 1)
    for band in ('high', 'mid'):
        if average >= evaluation[band]['limit']:
            return evaluation[band]['to'], band
    return evaluation['low']['to'], 'low'


def follows_routes(nodes, run):
    """Whether every step of a real run is where the current routes lead."""
    for index, (activity_id, option_id) in enumerate(run[:-1]):
        node = nodes.get(activity_id)
        if node is None:
            return False
        try:
            following, _ = route(nodes, node, option_id, run[:index + 1])
        except RouteError:
            return False
        if following != run[index + 1][0]:
            return False
    return True


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


class Command(BaseCommand):
    help = 'Export scenario graphs and aggregate student runs for the scenario walk-through study.'

    def add_arguments(self, parser):
        parser.add_argument('--output', required=True, help='Path of the JSONL file to write.')
        parser.add_argument('--min-students', type=int, default=30,
                            help='Minimum number of students who started a scenario.')
        parser.add_argument('--min-path-students', type=int, default=5,
                            help='List a path only if at least this many students took it.')
        parser.add_argument('--visibility', action='append', choices=['public', 'org', 'private'],
                            help='Scenario visibility to include (default: public only).')

    def handle(self, *args, output, min_students, min_path_students, visibility, **options):
        visibility = visibility or ['public']
        out_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        scenarios = Scenario.objects.filter(
            visibility_status__in=visibility, archived_at__isnull=True,
        ).order_by('id')
        written, skipped_images = 0, 0
        with open(output, 'w', encoding='utf-8') as handle:
            for scenario in scenarios:
                runs = self.first_runs(scenario)
                if len(runs) < min_students:
                    continue
                nodes, skipped = self.graph(scenario, out_dir)
                used = [run for run in runs if follows_routes(nodes, run)]
                if len(used) < min_students:
                    continue
                real = {'students': len(runs), 'runs_used': len(used), 'follows_routes': len(used) / len(runs)}
                real.update(self.summarise(nodes, used, min_path_students))
                start = scenario.get_start_activity()
                ages = scenario.age_of_students
                record = {
                    'scenario_id': scenario.id,
                    'scenario': scenario.name,
                    'language': scenario.language or '',
                    'age_min': getattr(ages, 'lower', None),
                    'age_max': getattr(ages, 'upper', None),
                    'learning_goals': plain(scenario.learning_goals)[:GOALS_LIMIT],
                    'start': start.id if start else None,
                    'activities': {str(key): node for key, node in nodes.items()},
                    'real': real,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                written += 1
                skipped_images += skipped

        manifest = {
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'scenarios': written,
            'min_students': min_students,
            'min_path_students': min_path_students,
            'visibility': visibility,
            'external_images_skipped': skipped_images,
        }
        with open(output + '.manifest.json', 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
        self.stdout.write(self.style.SUCCESS(f'Exported {written} scenarios to {output}'))

    # ── the scenario as Plato routes it ───────────────────────────────────
    def graph(self, scenario, out_dir):
        activities = list(
            Activity.objects.filter(scenario=scenario).select_related('activity_type', 'phase').order_by('id')
        )
        ids = [a.id for a in activities]
        routes = defaultdict(dict)
        page_next = {}
        for logic in NextQuestionLogic.objects.filter(activity_id__in=ids).order_by('id'):
            if logic.answer_id:
                routes[logic.activity_id][logic.answer_id] = logic.next_activity_id
            else:
                page_next.setdefault(logic.activity_id, logic.next_activity_id)
        bunches = defaultdict(set)
        for bunch in QuestionBunch.objects.filter(activity_primary_id__in=ids):
            bunches[bunch.activity_primary_id].update(bunch.activity_ids)
        branching = {b.activity_id: b for b in EvQuestionBranching.objects.filter(activity_id__in=ids)}
        limits = dict(Activity.objects.filter(scenario=scenario).values_list('id', 'score_limit'))
        options = defaultdict(list)
        for option in Answer.objects.filter(activity_id__in=ids).order_by('id'):
            options[option.activity_id].append(option)

        def destination(activity_id, with_limit=True):
            result = {'to': activity_id}
            if with_limit:
                result['limit'] = limits.get(activity_id) if activity_id else None
            return result

        nodes, skipped_images = {}, 0
        for activity in activities:
            is_question = activity.activity_type is not None and activity.activity_type.name == 'Question'
            text, images, skipped = text_with_images(activity, out_dir)
            skipped_images += skipped
            evaluation = None
            if is_question and activity.is_evaluatable:
                b = branching.get(activity.id)
                evaluation = {
                    'bunch': sorted(bunches.get(activity.id, ())),
                    'high': destination(b.next_question_on_high_id) if b else None,
                    'mid': destination(b.next_question_on_mid_id) if b else None,
                    'low': destination(b.next_question_on_low_id, with_limit=False) if b else None,
                }
            option_records = []
            if is_question:
                for option in options[activity.id]:
                    record = option_record(activity, option, out_dir)
                    record.update(weight=option.answer_weight or 0, next=routes[activity.id].get(option.id))
                    option_records.append(record)
            nodes[activity.id] = {
                'id': activity.id,
                'kind': 'question' if is_question else 'page',
                'type': activity.activity_type.name if activity.activity_type else '',
                'phase': activity.phase.name if activity.phase_id else '',
                'text': text,
                'images': images,
                'missing_images': skipped,
                'interactive': bool(activity.simulation_id or activity.experiment_ll_id
                                    or activity.vr_ar_experiment_id),
                'options': option_records,
                'next': None if is_question else page_next.get(activity.id),
                'evaluation': evaluation,
            }
        return nodes, skipped_images

    # ── real students ─────────────────────────────────────────────────────
    def first_runs(self, scenario):
        """Each student's first run as [(activity id, option id)], teachers excluded.

        A run is one implementation; older answers without one are a single
        run per student, cut where the student went back to the start.
        Repeated answers to the same activity in a row (double clicks) count once.
        """
        start = scenario.get_start_activity()
        rows = (
            UserAnswer.objects.filter(activity__scenario=scenario)
            .exclude(user__groups__name='teachers')
            .order_by('id')
            .values_list('user_id', 'implementation_id', 'activity_id', 'answer_id')
        )
        runs = defaultdict(list)
        first_run = {}
        for user_id, implementation_id, activity_id, answer_id in rows:
            key = implementation_id or ('legacy', user_id)
            first_run.setdefault(user_id, key)
            if first_run[user_id] == key:
                runs[key].append((activity_id, answer_id))
        cleaned = []
        for steps in runs.values():
            run = []
            for step in steps:
                if run and start and step[0] == start.id:
                    break
                if run and run[-1][0] == step[0]:
                    continue
                run.append(step)
            cleaned.append(run)
        return cleaned

    def summarise(self, nodes, runs, min_path_students):
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
