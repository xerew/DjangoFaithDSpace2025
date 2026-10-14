"""Help time: how long students kept a scenario paused to ask for help.

Only finished pauses are counted; a pause that is still open is in progress.
The same totals feed the dashboard, the time-spent chart (which subtracts
help from working time) and the downloadable CSV.
"""

from collections import defaultdict
from datetime import datetime, timedelta

from django.contrib.auth.models import User
from django.db.models import Count, Max, Q
from django.utils.dateparse import parse_date
from django.utils.timezone import make_aware

from usergroups.models import UserGroupMembership

from .models import Activity, HelpRequest, Phase
from .utils import get_last_answers


def _empty():
    return {'seconds': 0, 'requests': 0, 'students': set()}


def _finish(bucket):
    return {
        'seconds': bucket['seconds'],
        'requests': bucket['requests'],
        'students': len(bucket['students']),
    }


def finished_help_requests(scenario, user_ids=None, start_dt=None, end_dt=None):
    requests = HelpRequest.objects.filter(
        scenario=scenario,
        ended_at__isnull=False,
    )
    if user_ids is not None:
        requests = requests.filter(user_id__in=user_ids)
    if start_dt:
        requests = requests.filter(started_at__gte=start_dt)
    if end_dt:
        requests = requests.filter(started_at__lt=end_dt)
    return requests


def help_totals(scenario, user_ids=None, start_dt=None, end_dt=None):
    """Help seconds, request counts and students per activity, phase, user."""
    by_activity = defaultdict(_empty)
    by_phase = defaultdict(_empty)
    total = _empty()
    by_user_activity = defaultdict(int)
    by_user_phase = defaultdict(int)
    by_user = defaultdict(lambda: {'seconds': 0, 'requests': 0})

    for user_id, activity_id, phase_id, seconds in (
        finished_help_requests(scenario, user_ids, start_dt, end_dt)
        .values_list('user_id', 'activity_id', 'phase_id', 'duration_seconds')
    ):
        seconds = seconds or 0
        buckets = [total]
        if activity_id:
            buckets.append(by_activity[activity_id])
            by_user_activity[(user_id, activity_id)] += seconds
        if phase_id:
            buckets.append(by_phase[phase_id])
            by_user_phase[(user_id, phase_id)] += seconds
        for bucket in buckets:
            bucket['seconds'] += seconds
            bucket['requests'] += 1
            bucket['students'].add(user_id)
        by_user[user_id]['seconds'] += seconds
        by_user[user_id]['requests'] += 1

    return {
        'by_activity': {key: _finish(value) for key, value in by_activity.items()},
        'by_phase': {key: _finish(value) for key, value in by_phase.items()},
        'scenario': _finish(total),
        'by_user_activity': dict(by_user_activity),
        'by_user_phase': dict(by_user_phase),
        'by_user': {key: dict(value) for key, value in by_user.items()},
    }


def dashboard_student_ids(group_ids=None):
    """Students the dashboard counts: group members, never teachers."""
    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects
            .filter(group_id__in=group_ids)
            .values('user_id'),
        )
    else:
        users = User.objects.filter(
            Q(school_department__isnull=False)
            | Q(id__in=UserGroupMembership.objects.values('user_id'))
        )
    return list(
        users.exclude(groups__name='teachers')
        .values_list('id', flat=True)
        .distinct()
    )


def _day_start(value, add_days=0):
    day = parse_date(value) if value else None
    if not day:
        return None
    return make_aware(datetime.combine(day + timedelta(days=add_days), datetime.min.time()))


def help_time_chart(
    scenario,
    level='activity',
    group_ids=None,
    start_date=None,
    end_date=None,
):
    """Average help seconds per participating student, with request counts.

    ``level`` is ``activity``, ``phase`` or ``scenario``. A student takes part
    in an activity or phase when they answered in it, as in the time-spent
    chart.
    """
    user_ids = dashboard_student_ids(group_ids)
    start_dt = _day_start(start_date)
    end_dt = _day_start(end_date, add_days=1)
    totals = help_totals(scenario, user_ids, start_dt, end_dt)

    answers = get_last_answers(scenario.id).filter(user_id__in=user_ids)
    if start_dt:
        answers = answers.filter(created_on__gte=start_dt)
    if end_dt:
        answers = answers.filter(created_on__lt=end_dt)

    if level == 'scenario':
        items = [(None, scenario.name)]
        stats = {None: totals['scenario']}
        participants = {
            None: answers.values('user_id').distinct().count(),
        }
    elif level == 'phase':
        items = [
            (phase.id, phase.name)
            for phase in Phase.objects.filter(scenario=scenario)
            .order_by('created_on', 'id')
        ]
        stats = totals['by_phase']
        participants = {
            row['activity__phase_id']: row['students']
            for row in answers.values('activity__phase_id')
            .annotate(students=Count('user_id', distinct=True))
        }
    else:
        items = [
            (activity.id, activity.name)
            for activity in Activity.objects.filter(scenario=scenario)
            .order_by('phase__order', 'phase_id', 'order', 'created_on', 'id')
        ]
        stats = totals['by_activity']
        participants = {
            row['activity_id']: row['students']
            for row in answers.values('activity_id')
            .annotate(students=Count('user_id', distinct=True))
        }

    empty = {'seconds': 0, 'requests': 0, 'students': 0}
    data = {
        'categories': [],
        'average_help_seconds': [],
        'total_help_seconds': [],
        'requests': [],
        'students_asking': [],
    }
    for key, name in items:
        item = stats.get(key, empty)
        students = max(participants.get(key, 0), item['students'], 1)
        data['categories'].append(name)
        data['average_help_seconds'].append(round(item['seconds'] / students, 1))
        data['total_help_seconds'].append(item['seconds'])
        data['requests'].append(item['requests'])
        data['students_asking'].append(item['students'])
    return data


def help_cache_token(scenario):
    """Changes whenever a help period of the scenario starts or ends."""
    state = HelpRequest.objects.filter(scenario=scenario).aggregate(
        count=Count('id'),
        latest_start=Max('started_at'),
        latest_end=Max('ended_at'),
    )
    return (
        f"{state['count']}:"
        f"{state['latest_start'].timestamp() if state['latest_start'] else 0}:"
        f"{state['latest_end'].timestamp() if state['latest_end'] else 0}"
    )
