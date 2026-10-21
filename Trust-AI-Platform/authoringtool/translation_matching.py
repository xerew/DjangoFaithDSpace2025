"""Score possible 1:1 translations between scenarios for administrator review.

A translation must keep every activity, answer, correct option and route of
its source in the same flow order; only the text may be in another language.
Each candidate pair gets a confidence (0-100) and a list of the differences
found, so an administrator can confirm or reject it.
"""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
from html import unescape
import json
import re

from django.db import transaction
from django.db.models import Count, Max, Q

from .models import (
    Activity,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
    TranslationMatch,
    UserAnswer,
)


MIN_CONFIDENCE = 50
# Scenarios with fewer activities across all their phases are not matched.
MIN_ACTIVITIES = 10
# Aligned activities sharing this much wording are a same-language copy.
SAME_LANGUAGE_WORD_OVERLAP = 0.5
# Pairs further apart than this are not scored at all.
MAX_ACTIVITY_GAP = 2
MAX_QUESTION_GAP = 1
MIN_SEQUENCE_SIMILARITY = 0.9
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
# label used when two otherwise matching activities differ in it. ("Must
# wait" is deliberately left out: it does not affect matching.)
FIELD_LABELS = {
    'type': 'activity type',
    'answers': 'number of answers',
    'correct': 'correct answers',
    'weights': 'answer weights',
    'evaluated': 'evaluated (branching) question',
    'primary': 'primary evaluation question',
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
        'score_limit': f'{activity.score_limit:g}',
        'phase': str(phase_rank.get(activity.phase_id, 0)),
        'simulation': str(activity.simulation_id or ''),
        'lab': str(activity.experiment_ll_id or ''),
        'vr': str(activity.vr_ar_experiment_id or ''),
    }


MEDIA_FIELDS_BLANK = {'simulation': '', 'lab': '', 'vr': ''}


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
    # Scoring only uses bunches of evaluated primary questions, and a bunch of
    # just the question itself scores like no bunch; compare only the rest.
    bunch_members = {}
    for bunch in QuestionBunch.objects.filter(
        activity_primary__scenario=scenario,
    ):
        primary = activities.get(bunch.activity_primary_id)
        members = list(bunch.activity_ids or [])
        if (
            primary is not None
            and primary.is_evaluatable
            and primary.is_primary_ev
            and members != [primary.id]
        ):
            bunch_members[primary.id] = members

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
    # The same flow without simulations, labs and VR experiments, which a
    # machine translation may swap for versions in its own language.
    structure_signature = hashlib.sha256(
        json.dumps(
            [
                [_token({**f, **MEDIA_FIELDS_BLANK}) for f in fields],
                edges,
                bunches,
            ],
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
        'structure_signature': structure_signature,
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


def _sequence_similarity(profile_a, profile_b):
    return SequenceMatcher(
        None,
        profile_a['tokens'],
        profile_b['tokens'],
        autojunk=False,
    ).quick_ratio()


def _could_match(profile_a, profile_b):
    if not profile_a['reachable'] or not profile_b['reachable']:
        return False
    gap = abs(len(profile_a['tokens']) - len(profile_b['tokens']))
    if gap > MAX_ACTIVITY_GAP:
        return False
    gap = abs(profile_a['question_count'] - profile_b['question_count'])
    if gap > MAX_QUESTION_GAP:
        return False
    return _sequence_similarity(profile_a, profile_b) >= MIN_SEQUENCE_SIMILARITY


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


CHECK_LABELS = {
    'activities': 'Enough activities',
    'start': 'Flow from the first activity',
    'size': 'Same number of activities',
    'questions': 'Same number of questions',
    'sequence': 'Same activity sequence',
    'confidence': 'Confidence',
    'wording': 'Different wording',
    'listed': 'In the review list',
}


def _describe(activity):
    kind = activity.activity_type.name if activity.activity_type else 'Activity'
    return f'activity {activity.id} ({kind} "{_short(activity)}")'


def _start_detail(profile):
    scenario = profile['scenario']
    if not profile['reachable']:
        return 'fail', f'{scenario.id} has no activities to walk from a first activity.'
    linked = len(profile['order'])
    detail = (
        f'{scenario.id}: {profile["reachable"]} of {linked} linked activities '
        f'are reachable from its first activity, {_describe(profile["order"][0])}.'
    )
    if profile['reachable'] == linked:
        return 'pass', detail
    # Unreachable activities that nothing routes to are where the rest of
    # the flow starts; one of them is often the real first activity.
    targets = {
        target for edges in profile['edges'] for target in edges
        if target is not None
    }
    entries = [
        _describe(profile['order'][index])
        for index in range(profile['reachable'], linked)
        if index not in targets
    ]
    if entries:
        detail += (
            ' Nothing routes to ' + ', '.join(entries[:3])
            + ('…' if len(entries) > 3 else '')
            + '. If one of these is the real start, open it and use '
            '"Make this the first activity".'
        )
    return 'warn', detail


def explain_translation_pair(first, second):
    """Run the scan's checks on one pair and say which one, if any, stops it.

    Every check is run, so all reasons show at once; ``stopped_by`` names
    the first failing one. ``result`` is the comparison the scan would
    store, or None when a scenario has no flow to compare.
    """
    scenario_a, scenario_b = sorted((first, second), key=lambda s: s.id)
    profile_a = build_flow_profile(scenario_a)
    profile_b = build_flow_profile(scenario_b)
    checks = []

    def check(key, state, detail):
        checks.append({
            'key': key,
            'label': CHECK_LABELS[key],
            'state': state,
            'detail': detail,
        })

    counts = [
        Activity.objects.filter(scenario=scenario, phase__isnull=False).count()
        for scenario in (scenario_a, scenario_b)
    ]
    check(
        'activities',
        'pass' if min(counts) >= MIN_ACTIVITIES else 'fail',
        f'{scenario_a.id} has {counts[0]} activities in phases and '
        f'{scenario_b.id} has {counts[1]}; each needs at least {MIN_ACTIVITIES}.',
    )

    starts = [_start_detail(profile_a), _start_detail(profile_b)]
    states = {state for state, _ in starts}
    check(
        'start',
        'fail' if 'fail' in states else 'warn' if 'warn' in states else 'pass',
        ' '.join(detail for _, detail in starts),
    )

    sizes = [len(profile_a['tokens']), len(profile_b['tokens'])]
    check(
        'size',
        'pass' if abs(sizes[0] - sizes[1]) <= MAX_ACTIVITY_GAP else 'fail',
        f'{scenario_a.id} has {sizes[0]} linked activities and '
        f'{scenario_b.id} has {sizes[1]}; at most {MAX_ACTIVITY_GAP} apart.',
    )

    questions = [profile_a['question_count'], profile_b['question_count']]
    check(
        'questions',
        'pass' if abs(questions[0] - questions[1]) <= MAX_QUESTION_GAP else 'fail',
        f'{scenario_a.id} has {questions[0]} questions and '
        f'{scenario_b.id} has {questions[1]}; at most {MAX_QUESTION_GAP} apart.',
    )

    similarity = _sequence_similarity(profile_a, profile_b)
    check(
        'sequence',
        'pass' if similarity >= MIN_SEQUENCE_SIMILARITY else 'fail',
        f'The activities agree {similarity:.0%} on type, answers, correct '
        f'options and settings, in flow order; at least '
        f'{MIN_SEQUENCE_SIMILARITY:.0%} is needed.',
    )

    result = None
    if profile_a['tokens'] and profile_b['tokens']:
        result = compare_flows(profile_a, profile_b)
        check(
            'confidence',
            'pass' if result['confidence'] >= MIN_CONFIDENCE else 'fail',
            f'Confidence {result["confidence"]}%; at least {MIN_CONFIDENCE}% '
            'is needed to be listed.',
        )
        check(
            'wording',
            'pass' if result['lexical'] < SAME_LANGUAGE_WORD_OVERLAP else 'fail',
            f'Matching activities share {result["lexical"]:.0%} of their '
            f'words; a translation must share less than '
            f'{SAME_LANGUAGE_WORD_OVERLAP:.0%}, or it is taken for a '
            'same-language copy.',
        )
    else:
        for key in ('confidence', 'wording'):
            check(key, 'skip', 'Not scored: a scenario has no activities to compare.')

    stopped_by = next(
        (item['key'] for item in checks if item['state'] == 'fail'),
        None,
    )
    match = TranslationMatch.objects.filter(
        scenario_a=scenario_a,
        scenario_b=scenario_b,
    ).first()
    if match is not None:
        check(
            'listed',
            'pass',
            f'Listed as {match.get_status_display()} with '
            f'{match.confidence}% confidence. The list\'s visibility filter '
            f'shows it only when both {scenario_a.visibility_status} and '
            f'{scenario_b.visibility_status} are ticked (or none).',
        )
    elif stopped_by is None:
        check(
            'listed',
            'warn',
            'Not listed yet: the scenarios changed after the last scan, or '
            'no scan has run since. Press "Scan for translations".',
        )
    else:
        check(
            'listed',
            'skip',
            f'Not listed: stopped by "{CHECK_LABELS[stopped_by]}".',
        )

    return {
        'scenario_a': scenario_a,
        'scenario_b': scenario_b,
        'profile_a': profile_a,
        'profile_b': profile_b,
        'checks': checks,
        'stopped_by': stopped_by,
        'result': result,
        'match': match,
    }


ENGLISH_LANGUAGE_NAMES = {
    'english', 'en', 'eng', 'inglês', 'ingles', 'inglés', 'anglais',
    'englisch', 'inglese', 'engleză', 'αγγλικά',
}


def is_english(language):
    return (language or '').strip().casefold() in ENGLISH_LANGUAGE_NAMES


def _confirmed_links():
    """Neighbour map of confirmed pairs: id -> {other id: {own act: other act}}."""
    links = defaultdict(dict)
    for first, second, pairs in TranslationMatch.objects.filter(
        status='confirmed',
    ).values_list('scenario_a_id', 'scenario_b_id', 'activity_map'):
        pairs = pairs or []
        links[first][second] = {a: b for a, b in pairs}
        links[second][first] = {b: a for a, b in pairs}
    return links


def translation_groups():
    """Each scenario in a confirmed translation group -> the whole group."""
    links = _confirmed_links()
    groups = {}
    for start in links:
        if start in groups:
            continue
        group = {start}
        queue = [start]
        while queue:
            for other in links[queue.pop()]:
                if other not in group:
                    group.add(other)
                    queue.append(other)
        frozen = frozenset(group)
        for member in group:
            groups[member] = frozen
    return groups


def representative(versions):
    """The version that stands for a translation group: English, else the oldest."""
    oldest_first = sorted(versions, key=lambda s: (s.created_on, s.id))
    return next((s for s in oldest_first if is_english(s.language)), oldest_first[0])


def language_order(versions):
    """English first, then by language and name."""
    return sorted(
        versions,
        key=lambda s: (not is_english(s.language), (s.language or '').casefold(), s.name),
    )


def bundle_translations(scenarios, groups=None):
    """Bundle the given scenarios that are confirmed translations of each other.

    Returns one list per bundle, in the order bundles first appear; each list
    starts with its representative, then the other languages.
    """
    groups = translation_groups() if groups is None else groups
    scenarios = list(scenarios)
    members = defaultdict(list)
    for scenario in scenarios:
        members[groups.get(scenario.id, scenario.id)].append(scenario)
    bundles = []
    for key, versions in members.items():
        first = representative(versions)
        bundles.append([first] + language_order(s for s in versions if s is not first))
    return bundles


def translation_group_ids(scenario):
    """Ids of scenarios joined to this one by confirmed translations.

    Confirmed pairs are exact 1:1 matches, so the relation carries over: if
    A translates B and B translates C, then C is listed for A as well.
    """
    return set(translation_groups().get(scenario.id, {scenario.id})) - {scenario.id}


def translation_activity_maps(scenario):
    """For every other member of the group: {its activity id: our activity id}.

    Members reached through another translation get the two mappings chained.
    """
    links = _confirmed_links()
    to_target = {scenario.id: None}
    queue = [scenario.id]
    while queue:
        current = queue.pop(0)
        current_map = to_target[current]
        for other, other_to_current in (
            (other, {theirs: ours for ours, theirs in pairs.items()})
            for other, pairs in links[current].items()
        ):
            if other in to_target:
                continue
            if current_map is None:
                to_target[other] = dict(other_to_current)
            else:
                to_target[other] = {
                    theirs: current_map[ours]
                    for theirs, ours in other_to_current.items()
                    if ours in current_map
                }
            queue.append(other)
    del to_target[scenario.id]
    return to_target


def pooled_implementation_count(scenario):
    """Implementations of the scenario plus all its confirmed translations."""
    members = Scenario.objects.filter(id__in=translation_group_ids(scenario))
    return scenario.metrics_implementation_count() + sum(
        member.metrics_implementation_count() for member in members
    )


def translation_pool_signature(scenario):
    """Changes when translation answers or confirmed pairs change; '' if none."""
    member_ids = sorted(translation_group_ids(scenario))
    if not member_ids:
        return ''
    state = UserAnswer.objects.filter(
        activity__scenario_id__in=member_ids,
    ).aggregate(count=Count('id'), latest=Max('id'))
    links = list(
        TranslationMatch.objects.filter(
            status='confirmed',
        ).filter(
            Q(scenario_a_id__in=member_ids + [scenario.id])
            | Q(scenario_b_id__in=member_ids + [scenario.id])
        ).order_by('id').values_list('id', 'updated_at')
    )
    payload = json.dumps(
        [member_ids, state['count'], state['latest'], links],
        default=str,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def _recheck_machine_translation_pairs(matches):
    """Mark machine-translation pairs changed only if their structure changed.

    Their activity map and scores come from the translation itself, so the
    scorer never overwrites them; swapped media do not count as a change.
    """
    changed = 0
    for match in matches.select_related('scenario_a', 'scenario_b'):
        if (
            build_flow_profile(match.scenario_a)['structure_signature']
            != match.structure_signature_a
            or build_flow_profile(match.scenario_b)['structure_signature']
            != match.structure_signature_b
        ):
            match.status = 'changed'
            match.save(update_fields=['status', 'updated_at'])
            changed += 1
    return changed


def scan_translation_matches(scenario_ids=None, dry_run=False):
    """Score all scenario pairs and store them for administrator review.

    Review decisions are kept on a rescan. A reviewed pair whose flow has
    changed since is marked ``changed`` so it can be checked again.
    """
    scenarios = Scenario.objects.annotate(
        phase_activity_count=Count(
            'activities',
            filter=Q(activities__phase__isnull=False),
        ),
    ).order_by('id')
    if scenario_ids:
        scenarios = scenarios.filter(id__in=scenario_ids)
    too_small = set(
        scenarios.filter(phase_activity_count__lt=MIN_ACTIVITIES)
        .values_list('id', flat=True)
    )
    candidates = list(find_translation_candidates(
        scenarios.filter(phase_activity_count__gte=MIN_ACTIVITIES)
    ))
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
            if match is not None and match.origin == 'machine_translation':
                continue  # checked by structure below
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
        changed += _recheck_machine_translation_pairs(
            stale.filter(origin='machine_translation', status='confirmed')
        )
        stale = stale.exclude(origin='machine_translation')
        removed = 0
        for match in stale:
            if (match.scenario_a_id, match.scenario_b_id) in seen:
                continue
            if match.status == 'pending':
                match.delete()
                removed += 1
            elif {match.scenario_a_id, match.scenario_b_id} & too_small:
                # No longer eligible for matching; keep the decision as is.
                continue
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
