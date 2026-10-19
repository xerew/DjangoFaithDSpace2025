from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.db.models import Count
from datetime import date
from authoringtool.models import Scenario
from usergroups.models import UserGroup, UserGroupMembership


@login_required
def teacher_home(request):
    user = request.user
    # Students have no home page: send them to their scenarios, not a 403.
    if not user.groups.filter(name='teachers').exists():
        return redirect('studentScenarios')
    my_scenario_count = Scenario.objects.filter(created_by=user).count()
    my_group_count = UserGroup.objects.filter(created_by=user).count()
    total_students = UserGroupMembership.objects.filter(group__created_by=user).count()
    latest_public = Scenario.objects.filter(
        visibility_status='public', archived_at__isnull=True,
    ).order_by('-created_on')[:5]

    return render(request, 'home/home.html', {
        'my_scenario_count': my_scenario_count,
        'my_group_count': my_group_count,
        'total_students': total_students,
        'latest_public': latest_public,
        'show_get_started': my_scenario_count == 0,
        'today': date.today(),
    })
