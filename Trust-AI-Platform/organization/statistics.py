"""Aggregated, privacy-preserving statistics for an organization."""

from collections import Counter

from django.db.models import Count

from authoringtool.models import ActivityType, Scenario, Subject, UserScenarioScore
from usergroups.models import UserGroup, UserGroupMembership


def get_organization_statistics(organization, include_group_rows=False):
    """Return usage statistics attributed to an organization's teachers.

    The current data model does not link a student group directly to an
    organization. Attribution therefore follows the owning teacher: teachers
    are organization members in the ``teachers`` auth group, and students are
    members of student groups created by those teachers.

    An implementation follows the convention used elsewhere in the platform:
    one distinct ``(student, scenario)`` pair in ``UserScenarioScore``.
    """
    teacher_ids = list(
        organization.members.filter(groups__name='teachers')
        .values_list('id', flat=True)
        .distinct()
    )
    groups = UserGroup.objects.filter(created_by_id__in=teacher_ids)
    group_ids = list(groups.values_list('id', flat=True))
    student_ids = (
        UserGroupMembership.objects.filter(group_id__in=group_ids)
        .values_list('user_id', flat=True)
        .distinct()
    )

    scores = UserScenarioScore.objects.filter(user_id__in=student_ids)
    implementation_pairs = scores.values('user_id', 'scenario_id').distinct()
    scenario_usage = list(
        scores.values(
            'scenario_id',
            'scenario__name',
            'scenario__visibility_status',
            'scenario__language',
        )
        .annotate(implementations=Count('user_id', distinct=True))
        .order_by('-implementations', 'scenario__name')
    )
    scenario_ids = [row['scenario_id'] for row in scenario_usage]

    assigned_scenario_count = (
        Scenario.objects.filter(assigned_groups__id__in=group_ids)
        .distinct()
        .count()
    )

    category_rows = list(
        Subject.objects.filter(scenarios__id__in=scenario_ids)
        .values('category')
        .annotate(scenario_count=Count('scenarios', distinct=True))
        .order_by('-scenario_count', 'category')
    )
    categorized_scenario_count = (
        Scenario.objects.filter(id__in=scenario_ids, subjects__isnull=False)
        .distinct()
        .count()
    )
    uncategorized_count = len(scenario_ids) - categorized_scenario_count
    if uncategorized_count:
        category_rows.append({
            'category': 'Uncategorized',
            'scenario_count': uncategorized_count,
        })

    activity_type_rows = list(
        ActivityType.objects.filter(activities__scenario_id__in=scenario_ids)
        .values('name')
        .annotate(
            activity_count=Count('activities', distinct=True),
            scenario_count=Count('activities__scenario', distinct=True),
        )
        .order_by('-scenario_count', '-activity_count', 'name')
    )

    visibility_counts = Counter(
        row['scenario__visibility_status'] or 'unspecified'
        for row in scenario_usage
    )
    language_counts = Counter(
        row['scenario__language'].strip()
        if row['scenario__language'] and row['scenario__language'].strip()
        else 'Unspecified'
        for row in scenario_usage
    )

    result = {
        'teacher_count': len(teacher_ids),
        'group_count': len(group_ids),
        'student_count': student_ids.count(),
        'active_student_count': scores.values('user_id').distinct().count(),
        'implementation_count': implementation_pairs.count(),
        'scenario_count': len(scenario_usage),
        'assigned_scenario_count': assigned_scenario_count,
        'scenario_usage': scenario_usage,
        'category_rows': category_rows,
        'activity_type_rows': activity_type_rows,
        'visibility_rows': [
            {'name': name.title(), 'scenario_count': count}
            for name, count in sorted(
                visibility_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        'language_rows': [
            {'name': name, 'scenario_count': count}
            for name, count in sorted(
                language_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }

    if include_group_rows:
        group_implementation_counts = Counter()
        for row in (
            scores.filter(user__usergroupmembership__group_id__in=group_ids)
            .values(
                'user__usergroupmembership__group_id',
                'user_id',
                'scenario_id',
            )
            .distinct()
        ):
            group_implementation_counts[
                row['user__usergroupmembership__group_id']
            ] += 1

        result['group_rows'] = [
            {
                'name': group.name,
                'teacher': group.created_by.get_full_name()
                or group.created_by.username,
                'students': group.student_count,
                'assigned_scenarios': group.assigned_scenario_count,
                'implementations': group_implementation_counts[group.id],
            }
            for group in groups.select_related('created_by')
            .annotate(
                student_count=Count('members', distinct=True),
                assigned_scenario_count=Count('assigned_scenarios', distinct=True),
            )
            .order_by('created_by__username', 'name')
        ]

    return result
