"""Student runs of public scenarios, with their PISA label.

A run is one ScenarioImplementation: one student on one scenario. Answers
are the same eligible answers the analytics use (teachers, suspect and
excluded runs left out). A run gets a label only when it is finished:
the student answered an activity where their route ends. When the route
can't be followed (answers whose activity is gone, or a scenario without
any route end), the run counts as finished if the student answered
something in the scenario's last phase.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from authoringtool.help_time import finished_help_requests
from authoringtool.models import (
    Activity,
    Answer,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
    ScenarioImplementation,
)
from authoringtool.performance_formula import (
    categorize,
    phase_score,
    phase_weights,
    scored_activity_ids,
    weighted_percentage,
)
from authoringtool.translation_matching import translation_groups
from authoringtool.utils import get_eligible_user_answers, get_last_answers
from django.db.models import Max

ROUTE_END = 'route_end'
LAST_PHASE = 'last_phase'
UNFINISHED = 'unfinished'


@dataclass
class AnswerEvent:
    activity_id: int | None
    # 1-based position of the activity's phase; None when the activity is
    # gone or no longer in the scenario.
    phase_position: int | None
    created_on: datetime
    active_seconds: float | None
    chosen_weight: int | None
    is_last: bool


@dataclass
class Run:
    implementation_id: int
    user_id: int
    username: str
    scenario_id: int
    scenario_name: str
    language: str
    group_key: int
    is_current_version: bool
    phase_count: int
    phase_scores: list
    scored_by_position: dict
    answers: list
    help_by_position: dict
    final_percentage: float | None
    label: str | None
    finished_rule: str


@dataclass
class ScenarioContext:
    scenario: Scenario
    phases: list
    position_by_phase_id: dict
    position_by_activity_id: dict
    scored_ids_by_position: dict
    max_weight: dict
    terminal_ids: set
    activity_ids: set = field(default_factory=set)


def public_scenarios():
    """Public scenarios with at least one scored activity."""
    return list(
        Scenario.objects.filter(
            visibility_status='public',
            activities__is_evaluatable=True,
            activities__is_primary_ev=True,
        ).distinct().order_by('id')
    )


def terminal_activity_ids(scenario):
    """Activities where a student's route ends: no route leads on from them.

    Routes are next-activity rules (per answer or default) and the three
    branching targets, as in graph_validation. Non-primary members of a
    QuestionBunch are left out: the bunch's primary carries the route.
    """
    activity_ids = set(Activity.objects.filter(scenario=scenario).values_list('id', flat=True))
    has_route = set(
        NextQuestionLogic.objects.filter(
            activity_id__in=activity_ids, next_activity__isnull=False,
        ).values_list('activity_id', flat=True)
    )
    for branch in EvQuestionBranching.objects.filter(activity_id__in=activity_ids):
        if branch.next_question_on_high_id or branch.next_question_on_mid_id or branch.next_question_on_low_id:
            has_route.add(branch.activity_id)
    bunch_members = set()
    for bunch in QuestionBunch.objects.filter(activity_primary_id__in=activity_ids):
        bunch_members.update(
            member for member in (bunch.activity_ids or []) if member != bunch.activity_primary_id
        )
    return activity_ids - has_route - bunch_members


def scenario_context(scenario):
    phases = list(Phase.objects.filter(scenario=scenario).order_by('order', 'id'))
    position_by_phase_id = {phase.id: index + 1 for index, phase in enumerate(phases)}
    activities = list(Activity.objects.filter(scenario=scenario))
    activity_ids = {activity.id for activity in activities}
    position_by_activity_id = {
        activity.id: position_by_phase_id[activity.phase_id]
        for activity in activities
        if activity.phase_id in position_by_phase_id
    }
    primaries = defaultdict(list)
    for activity in sorted(activities, key=lambda a: a.id):
        if activity.is_evaluatable and activity.is_primary_ev and activity.phase_id in position_by_phase_id:
            primaries[position_by_phase_id[activity.phase_id]].append(activity.id)
    primary_ids = [aid for ids in primaries.values() for aid in ids]
    bunch_map = {
        bunch.activity_primary_id: bunch.activity_ids
        for bunch in QuestionBunch.objects.filter(activity_primary_id__in=primary_ids)
    }
    # Same scored set as the student CSV: bunch members that no longer
    # exist are dropped.
    member_ids = {aid for ids in bunch_map.values() for aid in ids}
    known = set(Activity.objects.filter(id__in=member_ids).values_list('id', flat=True)) | set(primary_ids)
    scored_ids_by_position = {
        position: scored_activity_ids(primaries.get(position, []), bunch_map, known_ids=known)
        for position in range(1, len(phases) + 1)
    }
    all_scored = {aid for ids in scored_ids_by_position.values() for aid in ids}
    max_weight = {
        row['activity_id']: row['max_weight']
        for row in Answer.objects.filter(activity_id__in=all_scored)
        .values('activity_id').annotate(max_weight=Max('answer_weight'))
    }
    return ScenarioContext(
        scenario=scenario,
        phases=phases,
        position_by_phase_id=position_by_phase_id,
        position_by_activity_id=position_by_activity_id,
        scored_ids_by_position=scored_ids_by_position,
        max_weight=max_weight,
        terminal_ids=terminal_activity_ids(scenario),
        activity_ids=activity_ids,
    )


def finished_rule(answered_ids, has_lost_answers, ctx):
    """How a run counts as finished, or UNFINISHED.

    `answered_ids` are the run's answered activities still in the scenario;
    `has_lost_answers` is True when some answers' activity is gone.
    """
    if answered_ids & ctx.terminal_ids:
        return ROUTE_END
    route_unknown = has_lost_answers or not ctx.terminal_ids
    last_position = len(ctx.phases)
    if route_unknown and any(
        ctx.position_by_activity_id.get(aid) == last_position for aid in answered_ids
    ):
        return LAST_PHASE
    return UNFINISHED


def _group_keys():
    groups = translation_groups()
    return lambda scenario_id: min(groups.get(scenario_id, {scenario_id}))


def _help(scenario, ctx):
    """Help per (user, activity) seconds, and per user and phase position."""
    by_user_activity = defaultdict(int)
    by_user_position = defaultdict(lambda: defaultdict(lambda: {'seconds': 0, 'requests': 0}))
    for user_id, activity_id, phase_id, seconds in finished_help_requests(scenario).values_list(
        'user_id', 'activity_id', 'phase_id', 'duration_seconds',
    ):
        seconds = seconds or 0
        if activity_id:
            by_user_activity[(user_id, activity_id)] += seconds
        position = ctx.position_by_phase_id.get(phase_id) or ctx.position_by_activity_id.get(activity_id)
        if position:
            bucket = by_user_position[user_id][position]
            bucket['seconds'] += seconds
            bucket['requests'] += 1
    return by_user_activity, by_user_position


def scenario_runs(scenario, group_key):
    ctx = scenario_context(scenario)
    if not ctx.phases:
        return []
    weights = phase_weights(len(ctx.phases))
    last_ids = set(get_last_answers(scenario.id).values_list('id', flat=True))
    help_by_pair, help_by_position = _help(scenario, ctx)

    answers_by_run = defaultdict(list)
    for row in (
        get_eligible_user_answers(scenario.id)
        .select_related('answer')
        .order_by('created_on', 'id')
    ):
        answers_by_run[row.implementation_id].append(row)

    implementations = {
        impl.id: impl
        for impl in ScenarioImplementation.objects.filter(id__in=answers_by_run).select_related('user')
    }
    runs = []
    for implementation_id, rows in answers_by_run.items():
        implementation = implementations[implementation_id]
        user_id = implementation.user_id
        events = []
        chosen = {}
        for row in rows:
            in_scenario = row.activity_id in ctx.activity_ids
            timing = row.timing or 0
            active = None
            if timing > 0:
                active = max(timing - help_by_pair.get((user_id, row.activity_id), 0), 0)
            is_last = row.id in last_ids
            weight = row.answer.answer_weight if row.answer else None
            events.append(AnswerEvent(
                activity_id=row.activity_id if in_scenario else None,
                phase_position=ctx.position_by_activity_id.get(row.activity_id) if in_scenario else None,
                created_on=row.created_on,
                active_seconds=active,
                chosen_weight=weight,
                is_last=is_last,
            ))
            if is_last and in_scenario and weight is not None:
                chosen[row.activity_id] = weight

        phase_scores = []
        scored_by_position = {}
        for position in range(1, len(ctx.phases) + 1):
            scored = ctx.scored_ids_by_position[position]
            phase_scores.append(phase_score(scored, chosen, ctx.max_weight))
            scored_by_position[position] = (sum(1 for aid in scored if aid in chosen), len(scored))

        answered_ids = {event.activity_id for event in events if event.activity_id is not None}
        has_lost = any(event.activity_id is None for event in events)
        rule = finished_rule(answered_ids, has_lost, ctx)
        final = weighted_percentage(phase_scores, weights)
        label = categorize(final) if rule != UNFINISHED and final is not None else None
        runs.append(Run(
            implementation_id=implementation_id,
            user_id=user_id,
            username=implementation.user.username,
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            language=(scenario.language or '').strip(),
            group_key=group_key,
            is_current_version=(
                implementation.version_confidence == 'exact'
                and implementation.scenario_version_id == scenario.current_version_id
            ),
            phase_count=len(ctx.phases),
            phase_scores=phase_scores,
            scored_by_position=scored_by_position,
            answers=events,
            help_by_position={pos: dict(v) for pos, v in help_by_position.get(user_id, {}).items()},
            final_percentage=final,
            label=label,
            finished_rule=rule,
        ))
    return sorted(runs, key=lambda run: run.implementation_id)


def load_runs(progress: Callable[[str], None] = lambda text: None):
    """Every run of every public scenario; labelled only when finished."""
    group_key = _group_keys()
    scenarios = public_scenarios()
    runs = []
    for index, scenario in enumerate(scenarios, start=1):
        progress(f'Reading answers: scenario {index} / {len(scenarios)}')
        runs.extend(scenario_runs(scenario, group_key(scenario.id)))
    return runs
