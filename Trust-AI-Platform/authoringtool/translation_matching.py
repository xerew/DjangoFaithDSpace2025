"""Score possible 1:1 translations between scenarios for administrator review.

A translation must keep every activity, answer, correct option and route of
its source in the same flow order; only the text may be in another language.
Each candidate pair gets a confidence (0-100) and a list of the differences
found, so an administrator can confirm or reject it.
"""

from collections import Counter
from difflib import SequenceMatcher
import hashlib
from html import unescape
import json
import re

from django.db import transaction

from .models import (
    Activity,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
    TranslationMatch,
)


MIN_CONFIDENCE = 50
# Aligned activities sharing this much wording are a same-language copy.
SAME_LANGUAGE_WORD_OVERLAP = 0.5
BRANCH_LABELS = ['High branch', 'Moderate branch', 'Low branch']

_NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
_IMAGE = re.compile(r'src=["\']([^"\']+)["\']', re.I)
_WORD = re.compile(r'[^\W\d_]{3,}', re.U)


def _clean(html):
    text = re.sub(r'<[^>]+>', ' ', html or '')
    return re.sub(r'\s+', ' ', unescape(text)).strip()


def _short(activity, length=40):
    return _clean(activity.plain_text or activity.text)[:length]


# Every language-neutral setting that changes routing or scoring, with the
# label used when two otherwise matching activities differ in it.
FIELD_LABELS = {
    'type': 'activity type',
    'answers': 'number of answers',
    'correct': 'correct answers',
    'weights': 'answer weights',
    'evaluated': 'evaluated (branching) question',
    'primary': 'primary evaluation question',
    'must_wait': 'must wait',
    'score_limit': 'score limit',
    'phase': 'phase',
    'simulation': 'simulation',
    'lab': 'LabsLand experiment',
    'vr': 'VR/AR experiment',
}


def _fields(activity, answers, phase_rank):
    """Language-neutral settings of one activity."""
    return {
        'type': (
            activity.activity_type.name.strip().casefold()
            if activity.activity_type else '?'
        ),
        'answers': str(len(answers)),
        'correct': ''.join(
            '1' if answer.is_correct else '0' for answer in answers
        ),
        'weights': ','.join(str(answer.answer_weight) for answer in answers),
        'evaluated': 'yes' if activity.is_evaluatable else 'no',
        'primary': 'yes' if activity.is_primary_ev else 'no',
        'must_wait': 'yes' if activity.must_wait else 'no',
        'score_limit': f'{activity.score_limit:g}',
        'phase': str(phase_rank.get(activity.phase_id, 0)),
        'simulation': str(activity.simulation_id or ''),
        'lab': str(activity.experiment_ll_id or ''),
        'vr': str(activity.vr_ar_experiment_id or ''),
    }


def _token(fields):
    return '|'.join(fields[key] for key in FIELD_LABELS)


def _field_differences(fields_a, fields_b):
    return [
        f'{label} {fields_a[key] or "none"} vs {fields_b[key] or "none"}'
        for key, label in FIELD_LABELS.items()
        if fields_a[key] != fields_b[key]
    ]


def build_flow_profile(scenario):
    """Describe a scenario as students walk it, from its start activity."""
    activities = {
        activity.id: activity
        for activity in (
            Activity.objects
            .filter(scenario=scenario)
            .select_related('activity_type', 'phase')
            .prefetch_related('answers')
        )
    }
    answers = {
        activity_id: sorted(activity.answers.all(), key=lambda a: a.id)
        for activity_id, activity in activities.items()
    }
    answer_routes = {}
    direct_routes = {}
    for logic in NextQuestionLogic.objects.filter(
        activity__scenario=scenario,
        next_activity__isnull=False,
    ):
        if logic.answer_id:
            answer_routes[(logic.activity_id, logic.answer_id)] = (
                logic.next_activity_id
            )
        else:
            direct_routes[logic.activity_id] = logic.next_activity_id
    branching = {
        row.activity_id: (
            row.next_question_on_high_id,
            row.next_question_on_mid_id,
            row.next_question_on_low_id,
        )
        for row in EvQuestionBranching.objects.filter(
            activity__scenario=scenario,
        )
    }

    def targets(activity_id):
        out = [
            answer_routes.get((activity_id, answer.id))
            for answer in answers[activity_id]
        ]
        out.append(direct_routes.get(activity_id))
        out.extend(branching.get(activity_id, (None, None, None)))
        return [target if target in activities else None for target in out]

    start = scenario.get_start_activity()
    # An activity with no route in or out (no next-question link, no
    # High/Moderate/Low branch) is left-over authoring content: ignore it.
    linked = {start.id} if start else set()
    for activity_id in activities:
        outgoing = [target for target in targets(activity_id) if target]
        if outgoing:
            linked.add(activity_id)
            linked.update(outgoing)
    activities = {
        activity_id: activity
        for activity_id, activity in activities.items()
        if activity_id in linked
    }

    # Number only the phases that hold linked activities, so an empty or
    # left-over phase does not shift the numbering of the real ones.
    used_phases = {activity.phase_id for activity in activities.values()}
    phase_rank = {
        phase_id: rank
        for rank, phase_id in enumerate(
            (
                phase_id
                for phase_id in Phase.objects.filter(scenario=scenario)
                .order_by('created_on', 'id')
                .values_list('id', flat=True)
                if phase_id in used_phases
            ),
            start=1,
        )
    }
    bunch_members = {
        bunch.activity_primary_id: list(bunch.activity_ids or [])
        for bunch in QuestionBunch.objects.filter(
            activity_primary__scenario=scenario,
        )
    }

    order = []
    if start and start.id in activities:
        seen = {start.id}
        queue = [start.id]
        while queue:
            activity_id = queue.pop(0)
            order.append(activity_id)
            for target in targets(activity_id):
                if target and target not in seen:
                    seen.add(target)
                    queue.append(target)
    reachable = len(order)
    def leftover_key(activity_id):
        activity = activities[activity_id]
        return (
            activity.phase_id is None,
            activity.phase.created_on if activity.phase else activity.created_on,
            activity.phase_id or 0,
            activity.created_on,
            activity_id,
        )

    order += sorted(
        (activity_id for activity_id in activities if activity_id not in order),
        key=leftover_key,
    )
    position = {activity_id: index for index, activity_id in enumerate(order)}

    tokens, fields, edges, bunches = [], [], [], []
    words, anchors = [], Counter()
    for activity_id in order:
        activity = activities[activity_id]
        activity_fields = _fields(activity, answers[activity_id], phase_rank)
        fields.append(activity_fields)
        tokens.append(_token(activity_fields))
        edges.append(tuple(
            position[target] if target else None
            for target in targets(activity_id)
        ))
        bunches.append(tuple(
            position.get(member, -1)
            for member in bunch_members.get(activity_id, [])
        ))
        body = _clean(activity.plain_text or activity.text)
        words.append({word.casefold() for word in _WORD.findall(body)})
        anchors.update(_NUMBER.findall(body))
        anchors.update(
            source.rsplit('/', 1)[-1]
            for source in _IMAGE.findall(activity.text or '')
        )
        for answer in answers[activity_id]:
            anchors.update(_NUMBER.findall(_clean(answer.text)))
            if answer.image:
                anchors[answer.image.name.rsplit('/', 1)[-1]] += 1

    signature = hashlib.sha256(
        json.dumps(
            [tokens, edges, bunches],
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    phase_ids = {
        activities[activity_id].phase_id for activity_id in order[:reachable]
    }
    return {
        'scenario': scenario,
        'order': [activities[activity_id] for activity_id in order],
        'reachable': reachable,
        'tokens': tokens,
        'fields': fields,
        'edges': edges,
        'bunches': bunches,
        'words': words,
        'anchors': anchors,
        'phase_count': len(phase_ids),
        'question_count': sum(
            1 for token in tokens if token.startswith('question|')
        ),
        'signature': signature,
    }


def _jaccard(left, right):
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _go(position):
    return 'nowhere' if position is None else f'#{position + 1}'


def activity_text(activity, length=160):
    return _short(activity, length)


def _align(profile_a, profile_b):
    """Line up the reachable flows; return matcher, tokens and mapping."""
    tokens_a = profile_a['tokens'][:profile_a['reachable']]
    tokens_b = profile_b['tokens'][:profile_b['reachable']]
    matcher = SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    mapping = {}
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        # Same activities, or the same number of activities in the same
        # place with different settings: pair them so routes still compare.
        if op == 'equal' or (op == 'replace' and a2 - a1 == b2 - b1):
            for offset in range(a2 - a1):
                mapping[a1 + offset] = b1 + offset
    return matcher, tokens_a, tokens_b, mapping


def _routes_match(profile_a, profile_b, mapping, index_a, index_b):
    mapped = tuple(
        mapping.get(target, -1) if target is not None else None
        for target in profile_a['edges'][index_a]
    )
    return mapped == profile_b['edges'][index_b], mapped


def _bunch_matches(profile_a, profile_b, mapping, index_a, index_b):
    mapped = tuple(
        mapping.get(member, -1) if member >= 0 else -1
        for member in profile_a['bunches'][index_a]
    )
    return mapped == profile_b['bunches'][index_b]


def _members(positions):
    return ', '.join(_go(p) if p >= 0 else '?' for p in positions) or 'none'


def align_flows(profile_a, profile_b):
    """Rows for a side-by-side view: (index_a, index_b, status)."""
    matcher, _, _, mapping = _align(profile_a, profile_b)
    rows = []
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == 'equal':
            for offset in range(a2 - a1):
                same, _ = _routes_match(
                    profile_a, profile_b, mapping, a1 + offset, b1 + offset,
                )
                rows.append((
                    a1 + offset,
                    b1 + offset,
                    'same' if same else 'route differs',
                ))
        elif op == 'replace':
            for offset in range(max(a2 - a1, b2 - b1)):
                rows.append((
                    a1 + offset if a1 + offset < a2 else None,
                    b1 + offset if b1 + offset < b2 else None,
                    'differs',
                ))
        elif op == 'delete':
            rows.extend((index, None, 'only here') for index in range(a1, a2))
        else:
            rows.extend((None, index, 'only here') for index in range(b1, b2))
    rows.extend(
        (index, None, 'unreachable')
        for index in range(profile_a['reachable'], len(profile_a['order']))
    )
    rows.extend(
        (None, index, 'unreachable')
        for index in range(profile_b['reachable'], len(profile_b['order']))
    )
    return rows


def compare_flows(profile_a, profile_b):
    """Score how closely two flows match as a 1:1 translation."""
    id_a = profile_a['scenario'].id
    id_b = profile_b['scenario'].id
    matcher, tokens_a, tokens_b, mapping = _align(profile_a, profile_b)
    structure = matcher.ratio() if (tokens_a or tokens_b) else 0.0

    differences = []
    question_changed = False
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == 'equal':
            continue
        changed = tokens_a[a1:a2] + tokens_b[b1:b2]
        question_changed |= any(
            token.startswith('question|') for token in changed
        )
        if op == 'replace' and a2 - a1 == b2 - b1:
            for offset in range(a2 - a1):
                index = a1 + offset
                activity = profile_a['order'][index]
                kind = tokens_a[index].split('|')[0]
                details = '; '.join(_field_differences(
                    profile_a['fields'][index],
                    profile_b['fields'][b1 + offset],
                ))
                differences.append(
                    f'#{index + 1} {kind} "{_short(activity, 30)}": '
                    f'{details} ({id_a} vs {id_b})'
                )
            continue
        for index in range(a1, a2) if op != 'insert' else ():
            activity = profile_a['order'][index]
            kind = tokens_a[index].split('|')[0]
            differences.append(
                f'only in {id_a}: #{index + 1} {kind} "{_short(activity)}"'
                if op == 'delete' else
                f'#{index + 1} {kind} "{_short(activity)}" in {id_a} '
                f'differs from #{b1 + 1} in {id_b}'
            )
        if op == 'insert':
            for index in range(b1, b2):
                activity = profile_b['order'][index]
                kind = tokens_b[index].split('|')[0]
                differences.append(
                    f'only in {id_b}: #{index + 1} {kind} '
                    f'"{_short(activity)}"'
                )

    matching_routes = 0
    for index_a, index_b in sorted(mapping.items()):
        same, mapped = _routes_match(
            profile_a, profile_b, mapping, index_a, index_b,
        )
        same_bunch = _bunch_matches(
            profile_a, profile_b, mapping, index_a, index_b,
        )
        expected = profile_b['edges'][index_b]
        if same and same_bunch:
            matching_routes += 1
            continue
        activity = profile_a['order'][index_a]
        if not same_bunch:
            differences.append(
                f'question bunch at #{index_a + 1} "{_short(activity, 30)}": '
                f'members {_members(profile_a["bunches"][index_a])} in {id_a} '
                f'but {_members(profile_b["bunches"][index_b])} in {id_b}'
            )
        if same:
            continue
        answer_count = int(tokens_a[index_a].split('|')[1])
        slots = (
            [f'answer {number + 1}' for number in range(answer_count)]
            + ['next'] + BRANCH_LABELS
        )
        for slot, own, mine, theirs in zip(
            slots, profile_a['edges'][index_a], mapped, expected,
        ):
            if mine != theirs:
                differences.append(
                    f'route at #{index_a + 1} "{_short(activity, 30)}": '
                    f'{slot} goes to {_go(own)} in {id_a} '
                    f'but {_go(theirs)} in {id_b}'
                )
    longest = max(len(tokens_a), len(tokens_b), 1)
    routes = matching_routes / longest

    lexical = (
        sum(
            _jaccard(profile_a['words'][a], profile_b['words'][b])
            for a, b in mapping.items()
        ) / len(mapping)
        if mapping else 0.0
    )
    shared = sum((profile_a['anchors'] & profile_b['anchors']).values())
    anchor_total = max(
        sum(profile_a['anchors'].values()),
        sum(profile_b['anchors'].values()),
    )
    anchors = shared / anchor_total if anchor_total else 0.0

    leftovers = [
        f'unreachable in {profile["scenario"].id}: '
        f'{profile["tokens"][index].split("|")[0]} '
        f'"{_short(profile["order"][index])}"'
        for profile in (profile_a, profile_b)
        for index in range(profile['reachable'], len(profile['order']))
    ]
    exact = (
        bool(tokens_a)
        and structure == 1
        and routes == 1
        and profile_a['phase_count'] == profile_b['phase_count']
    )
    if exact:
        confidence = 90 + 10 * min(1.0, anchors * 2) if anchor_total else 95
        if leftovers:
            # Students see the same flow; only dead authoring content differs.
            confidence = min(confidence, 98) - min(len(leftovers) - 1, 3)
    else:
        confidence = 100 * structure * (0.5 + 0.5 * routes)
        confidence -= 4 * (len(differences) + len(leftovers))
        if question_changed:
            confidence = min(confidence, 60)
        confidence = min(confidence, 89)

    return {
        'confidence': int(round(max(confidence, 0))),
        'exact': exact,
        'structure': round(structure, 3),
        'routes': round(routes, 3),
        'anchors': round(anchors, 3),
        'lexical': round(lexical, 3),
        'differences': differences + leftovers,
        'activity_map': [
            [profile_a['order'][a].id, profile_b['order'][b].id]
            for a, b in sorted(mapping.items())
        ],
    }


def _could_match(profile_a, profile_b):
    if not profile_a['reachable'] or not profile_b['reachable']:
        return False
    if abs(len(profile_a['tokens']) - len(profile_b['tokens'])) > 2:
        return False
    if abs(profile_a['question_count'] - profile_b['question_count']) > 1:
        return False
    return SequenceMatcher(
        None,
        profile_a['tokens'],
        profile_b['tokens'],
        autojunk=False,
    ).quick_ratio() >= 0.9


def find_translation_candidates(scenarios=None):
    """Yield (profile_a, profile_b, result) for likely translation pairs."""
    scenarios = list(
        scenarios
        if scenarios is not None
        else Scenario.objects.order_by('id')
    )
    profiles = sorted(
        (build_flow_profile(scenario) for scenario in scenarios),
        key=lambda profile: profile['scenario'].id,
    )
    for index, profile_a in enumerate(profiles):
        for profile_b in profiles[index + 1:]:
            if not _could_match(profile_a, profile_b):
                continue
            result = compare_flows(profile_a, profile_b)
            if result['confidence'] < MIN_CONFIDENCE:
                continue
            if result['lexical'] >= SAME_LANGUAGE_WORD_OVERLAP:
                continue
            yield profile_a, profile_b, result


def scan_translation_matches(scenario_ids=None, dry_run=False):
    """Score all scenario pairs and store them for administrator review.

    Review decisions are kept on a rescan. A reviewed pair whose flow has
    changed since is marked ``changed`` so it can be checked again.
    """
    scenarios = Scenario.objects.order_by('id')
    if scenario_ids:
        scenarios = scenarios.filter(id__in=scenario_ids)
    candidates = list(find_translation_candidates(scenarios))
    if dry_run:
        return {'matches': len(candidates), 'candidates': [
            {
                'scenario_a': profile_a['scenario'].id,
                'scenario_b': profile_b['scenario'].id,
                'confidence': result['confidence'],
                'differences': result['differences'],
            }
            for profile_a, profile_b, result in candidates
        ]}

    seen = set()
    created = changed = 0
    with transaction.atomic():
        for profile_a, profile_b, result in candidates:
            pair = (profile_a['scenario'].id, profile_b['scenario'].id)
            seen.add(pair)
            values = {
                'confidence': result['confidence'],
                'exact_match': result['exact'],
                'structure_score': result['structure'],
                'route_score': result['routes'],
                'anchor_score': result['anchors'],
                'differences': result['differences'],
                'activity_map': result['activity_map'],
                'flow_signature_a': profile_a['signature'],
                'flow_signature_b': profile_b['signature'],
            }
            match = TranslationMatch.objects.filter(
                scenario_a_id=pair[0],
                scenario_b_id=pair[1],
            ).first()
            if match is None:
                TranslationMatch.objects.create(
                    scenario_a_id=pair[0],
                    scenario_b_id=pair[1],
                    **values,
                )
                created += 1
                continue
            flow_changed = (
                match.flow_signature_a != profile_a['signature']
                or match.flow_signature_b != profile_b['signature']
            )
            if match.status in {'confirmed', 'rejected'} and flow_changed:
                values['status'] = 'changed'
                changed += 1
            for field, value in values.items():
                setattr(match, field, value)
            match.save()

        stale = TranslationMatch.objects.all()
        if scenario_ids:
            stale = stale.filter(
                scenario_a_id__in=scenario_ids,
                scenario_b_id__in=scenario_ids,
            )
        removed = 0
        for match in stale:
            if (match.scenario_a_id, match.scenario_b_id) in seen:
                continue
            if match.status == 'pending':
                match.delete()
                removed += 1
            elif match.status in {'confirmed', 'rejected'}:
                match.status = 'changed'
                match.save(update_fields=['status', 'updated_at'])
                changed += 1
    return {
        'matches': len(candidates),
        'created': created,
        'changed_since_review': changed,
        'removed': removed,
    }
