from collections import defaultdict, deque

from .models import (
    Activity,
    EvQuestionBranching,
    NextQuestionLogic,
    QuestionBunch,
)


class ScenarioGraphValidationError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        summary = "; ".join(issue["message"] for issue in self.issues)
        super().__init__(summary)


def _issue(code, message, activity_id=None):
    issue = {"code": code, "message": message}
    if activity_id is not None:
        issue["activity_id"] = activity_id
    return issue


def validate_scenario_graph(scenario):
    """Return structural issues found in a scenario's directed activity graph."""
    activities = list(
        Activity.objects.filter(scenario=scenario)
        .select_related("activity_type")
        .prefetch_related("answers")
        .order_by("id")
    )
    if not activities:
        return [_issue("empty_scenario", "The scenario has no activities.")]

    activity_by_id = {activity.id: activity for activity in activities}
    activity_ids = set(activity_by_id)
    issues = []

    start_id = scenario.start_activity_id
    if not start_id:
        issues.append(
            _issue(
                "missing_start_activity",
                "The scenario does not have an explicit start activity.",
            )
        )
    elif start_id not in activity_ids:
        issues.append(
            _issue(
                "invalid_start_activity",
                "The configured start activity does not belong to this scenario.",
                start_id,
            )
        )

    flow_edges = defaultdict(set)
    reachability_edges = defaultdict(set)
    logic_by_activity = defaultdict(list)
    logic_rows = NextQuestionLogic.objects.filter(
        activity_id__in=activity_ids
    ).select_related("answer", "next_activity")
    for logic in logic_rows:
        source_id = logic.activity_id
        logic_by_activity[source_id].append(logic)
        if logic.answer_id and logic.answer.activity_id != source_id:
            issues.append(
                _issue(
                    "answer_route_source_mismatch",
                    (
                        f"Route {logic.id} uses an answer that does not belong "
                        f"to activity '{activity_by_id[source_id].name}'."
                    ),
                    source_id,
                )
            )
        if not logic.next_activity_id:
            continue
        if logic.next_activity_id not in activity_ids:
            issues.append(
                _issue(
                    "cross_scenario_route",
                    (
                        f"Activity '{activity_by_id[source_id].name}' routes "
                        "outside this scenario."
                    ),
                    source_id,
                )
            )
            continue
        flow_edges[source_id].add(logic.next_activity_id)
        reachability_edges[source_id].add(logic.next_activity_id)

    branch_by_activity = {
        branch.activity_id: branch
        for branch in EvQuestionBranching.objects.filter(
            activity_id__in=activity_ids
        ).select_related(
            "next_question_on_high",
            "next_question_on_mid",
            "next_question_on_low",
        )
    }
    for activity in activities:
        branch = branch_by_activity.get(activity.id)
        if activity.is_evaluatable and not branch:
            issues.append(
                _issue(
                    "missing_branching",
                    (
                        f"Evaluatable activity '{activity.name}' has no "
                        "branching configuration."
                    ),
                    activity.id,
                )
            )
            continue
        if not branch:
            continue

        targets = {
            "High": branch.next_question_on_high_id,
            "Moderate": branch.next_question_on_mid_id,
            "Low": branch.next_question_on_low_id,
        }
        configured = [target for target in targets.values() if target]
        if configured and len(configured) != len(targets):
            missing = ", ".join(
                label for label, target in targets.items() if not target
            )
            issues.append(
                _issue(
                    "incomplete_branching",
                    (
                        f"Activity '{activity.name}' has incomplete branches: "
                        f"{missing} is missing."
                    ),
                    activity.id,
                )
            )
        for label, target_id in targets.items():
            if not target_id:
                continue
            if target_id not in activity_ids:
                issues.append(
                    _issue(
                        "cross_scenario_branch",
                        (
                            f"The {label} branch from '{activity.name}' "
                            "points outside this scenario."
                        ),
                        activity.id,
                    )
                )
                continue
            flow_edges[activity.id].add(target_id)
            reachability_edges[activity.id].add(target_id)

    for activity in activities:
        type_name = (
            activity.activity_type.name.lower()
            if activity.activity_type
            else ""
        )
        if type_name != "question" or activity.is_evaluatable:
            continue
        answers = list(activity.answers.all())
        answer_ids = {answer.id for answer in answers}
        rows = logic_by_activity.get(activity.id, [])
        generic_route_exists = any(
            row.answer_id is None and row.next_activity_id for row in rows
        )
        routed_answer_ids = {
            row.answer_id
            for row in rows
            if row.answer_id and row.next_activity_id
        }
        has_outgoing_route = bool(
            generic_route_exists
            or routed_answer_ids
            or flow_edges.get(activity.id)
        )
        if has_outgoing_route and not answers:
            issues.append(
                _issue(
                    "question_without_answers",
                    f"Question activity '{activity.name}' has no answers.",
                    activity.id,
                )
            )
        if routed_answer_ids and not generic_route_exists:
            missing_answer_ids = answer_ids - routed_answer_ids
            if missing_answer_ids:
                issues.append(
                    _issue(
                        "missing_answer_routes",
                        (
                            f"Question activity '{activity.name}' has "
                            f"{len(missing_answer_ids)} answer(s) without a "
                            "next-activity route."
                        ),
                        activity.id,
                    )
                )

    bunches = QuestionBunch.objects.filter(
        activity_primary_id__in=activity_ids
    )
    for bunch in bunches:
        for member_id in bunch.activity_ids or []:
            if member_id not in activity_ids:
                issues.append(
                    _issue(
                        "broken_question_bunch",
                        (
                            f"Question bunch for activity "
                            f"'{activity_by_id[bunch.activity_primary_id].name}' "
                            f"contains missing activity ID {member_id}."
                        ),
                        bunch.activity_primary_id,
                    )
                )
                continue
            if member_id != bunch.activity_primary_id:
                reachability_edges[bunch.activity_primary_id].add(member_id)

    if start_id in activity_ids:
        reachable = {start_id}
        queue = deque([start_id])
        while queue:
            source_id = queue.popleft()
            for target_id in reachability_edges.get(source_id, set()):
                if target_id not in reachable:
                    reachable.add(target_id)
                    queue.append(target_id)
        unreachable_ids = activity_ids - reachable
        for activity_id in sorted(unreachable_ids):
            issues.append(
                _issue(
                    "unreachable_activity",
                    (
                        f"Activity '{activity_by_id[activity_id].name}' is "
                        "unreachable from the scenario start."
                    ),
                    activity_id,
                )
            )

    color = {activity_id: 0 for activity_id in activity_ids}
    stack = []
    cycle_keys = set()

    def visit(activity_id):
        color[activity_id] = 1
        stack.append(activity_id)
        for target_id in flow_edges.get(activity_id, set()):
            if color[target_id] == 0:
                visit(target_id)
            elif color[target_id] == 1:
                start_index = stack.index(target_id)
                cycle = stack[start_index:] + [target_id]
                canonical_key = tuple(sorted(set(cycle)))
                if canonical_key not in cycle_keys:
                    cycle_keys.add(canonical_key)
                    names = " -> ".join(
                        activity_by_id[node_id].name for node_id in cycle
                    )
                    issues.append(
                        _issue(
                            "activity_cycle",
                            f"Unintended activity cycle detected: {names}.",
                            target_id,
                        )
                    )
        stack.pop()
        color[activity_id] = 2

    for activity_id in sorted(activity_ids):
        if color[activity_id] == 0:
            visit(activity_id)

    return issues


def assert_scenario_graph_integrity(scenario):
    issues = validate_scenario_graph(scenario)
    if issues:
        raise ScenarioGraphValidationError(issues)
    return True
