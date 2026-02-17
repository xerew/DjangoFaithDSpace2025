from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import user_passes_test
from django.contrib import messages
from .models import UserGroup, UserGroupMembership
import openpyxl
from django.http import HttpResponse
from .forms import UserGroupForm
from authoringtool.models import Scenario, UserAnswer, MultilingualAnswer, MultilingualQuestion
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from functools import wraps
from openpyxl import Workbook
from django.contrib.auth.models import User
from django.db.models import Q, Count, Min, Max # 20 FEB
from collections import defaultdict # 24 FEB
from django.core.paginator import Paginator # 24 FEB
from django.utils.timezone import make_aware # 2
from datetime import datetime, timezone # 2
from django.http import HttpResponse
from django.conf import settings
import re
import csv

def group_required(group_name):
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped_view(request, *args, **kwargs):
            if request.user.groups.filter(name=group_name).exists():
                return view_func(request, *args, **kwargs)
            else:
                raise PermissionDenied
        return _wrapped_view
    return decorator

def get_next_suffix(prefix):
    """Calculate the next available suffix for users with the given prefix."""
    existing_users = User.objects.filter(username__startswith=prefix).values_list('username', flat=True)

    # Extract the numeric parts from the usernames (after the prefix)
    suffixes = []
    for username in existing_users:
        # Strip the prefix and convert the rest into an integer
        suffix = username[len(prefix):]
        if suffix.isdigit():
            suffixes.append(int(suffix))

    if suffixes:
        # Get the maximum suffix and increment by 1
        return max(suffixes) + 1
    return 1  # Start at 1 if no users with this prefix exist


@group_required('teachers')
def create_user_group(request):
    if request.method == "POST":
        form = UserGroupForm(request.POST)
        
        # Check if any scenarios are selected
        scenario_ids = request.POST.getlist('scenarios')
        # if not scenario_ids:
        #     form.add_error('assigned_scenarios', 'At least one scenario must be selected.')

        if form.is_valid():
            group = form.save(commit=False)
            group.created_by = request.user
            group.save()

            # Determine where to start numbering based on the shared prefix
            prefix = form.cleaned_data['prefix']
            next_suffix = get_next_suffix(prefix)

            # Save scenarios manually
            scenarios = Scenario.objects.filter(id__in=scenario_ids)
            print("SCENARIOS: ", scenarios)
            group.assigned_scenarios.set(scenarios)
            # group.save()

            # Create users for the group
            for i in range(next_suffix, next_suffix + group.number_of_users):
                username = f"{prefix}{i}"
                password = group.generate_password()
                user = User.objects.create_user(username=username, password=password)
                UserGroupMembership.objects.create(group=group, user=user, password=password)
            
            form.save # _m2m()  # Save the ManyToMany field for scenarios
            messages.success(request, f"Group '{group.name}' created successfully with {group.number_of_users} users.")
            return redirect('list_groups')
    else:
        form = UserGroupForm()

    # Filter scenarios based on the user's access rights
    user = request.user
    org_ids = user.member_of_organizations.values_list('id', flat=True)

    scenarios = Scenario.objects.filter(
        Q(created_by=user) |  # Private scenarios created by the user
        Q(visibility_status='public') |  # Public scenarios
        (Q(visibility_status='org') & Q(organizations__id__in=org_ids))  # Org-only scenarios where user belongs to the org
    ).distinct()

    return render(request, 'usergroups/create_group.html', {'form': form, 'scenarios': scenarios})

@group_required('teachers')
def list_groups(request):
    groups = UserGroup.objects.filter(created_by=request.user)
    is_dspace_partner = request.user.groups.filter(name="dspace_partners").exists()

    return render(request, 'usergroups/list_groups.html', {'groups': groups, 'is_dspace_partner': is_dspace_partner})

@group_required('teachers')
def download_credentials(request, group_id):
    if request.user.is_superuser:
        # Superusers can access any group
        group = get_object_or_404(UserGroup, id=group_id)
    else:
        # Normal users (e.g., teachers) can only access their own groups
        group = get_object_or_404(UserGroup, id=group_id, created_by=request.user)
    memberships = UserGroupMembership.objects.filter(group=group)
    
    # Sanitize group name to remove problematic characters for sheet name and filename
    safe_group_name = re.sub(r'[\\/*?[\]:]', '_', group.name)

    # Create an Excel workbook and worksheet
    wb = Workbook()
    ws = wb.active
    ws.title = f"{safe_group_name} Credentials"

    # Add headers
    ws.append(['Username', 'Password'])

    # Add member data
    for membership in memberships:
        ws.append([membership.user.username, membership.password])

    # Prepare the Excel response
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename="{group.name}_credentials.xlsx"'
    wb.save(response)

    return response

@group_required('teachers')
def delete_group(request, group_id):
    group = get_object_or_404(UserGroup, id=group_id, created_by=request.user)
    
    if request.method == "POST":
        # Fetch all memberships of the group
        memberships = UserGroupMembership.objects.filter(group=group)

        # Delete associated users
        for membership in memberships:
            membership.user.delete()  # Delete the actual user object

        # Now delete the group itself
        group.delete()

        # Redirect back to the list of groups
        return redirect('list_groups')
    
    return render(request, 'usergroups/confirm_delete.html', {'group': group})

@group_required('teachers')
def edit_group(request, group_id):
    group = get_object_or_404(UserGroup, id=group_id, created_by=request.user)

    # Get the current number of users BEFORE any changes
    current_number_of_users = group.number_of_users

    if request.method == "POST":
        form = UserGroupForm(request.POST, instance=group)

        if form.is_valid():
            # Get the new number of users from the form
            new_number_of_users = form.cleaned_data.get('number_of_users')
            print('NEW NUMBER IS: ', new_number_of_users)
            print('CURRENT NUMBER IS: ', current_number_of_users)

            # Compare and manage users
            if new_number_of_users > current_number_of_users:
                # Adding extra users
                diff = new_number_of_users - current_number_of_users
                prefix = form.cleaned_data['prefix']
                next_suffix = get_next_suffix(prefix)
                for i in range(next_suffix, next_suffix + diff):
                    username = f"{prefix}{i}"
                    password = group.generate_password()
                    user = User.objects.create_user(username=username, password=password)
                    UserGroupMembership.objects.create(group=group, user=user, password=password)
                # messages.success(request, f"Added {diff} new users to the group.")

            elif new_number_of_users < current_number_of_users:
                # Removing extra users
                diff = current_number_of_users - new_number_of_users
                memberships_to_delete = UserGroupMembership.objects.filter(group=group).order_by('-id')[:diff]
                for membership in memberships_to_delete:
                    membership.user.delete()  # Deletes the user and their membership
                    membership.delete()
                # messages.success(request, f"Removed {diff} users from the group.")

            # Update the group model with the new number of users
            group.number_of_users = new_number_of_users
            form.save()

            # Update assigned scenarios
            scenario_ids = request.POST.getlist('scenarios')
            scenarios = Scenario.objects.filter(id__in=scenario_ids)
            group.assigned_scenarios.set(scenarios)

            group.save()

            # messages.success(request, f"Group '{group.name}' updated successfully.")
            return redirect('view_group', group_id=group.id)

    else:
        form = UserGroupForm(instance=group)

    # Pass scenarios based on the user's access rights
    user = request.user
    org_ids = user.member_of_organizations.values_list('id', flat=True)
    scenarios = Scenario.objects.filter(
        Q(created_by=user) | 
        Q(visibility_status='public') | 
        (Q(visibility_status='org') & Q(organizations__id__in=org_ids))
    ).distinct()

    return render(request, 'usergroups/edit_group.html', {'form': form, 'group': group, 'scenarios': scenarios})

@group_required('teachers')
def view_group(request, group_id):
    if request.user.is_superuser:
        # Superusers can access any group
        group = get_object_or_404(UserGroup, id=group_id)
    else:
        # Normal users (e.g., teachers) can only access their own groups
        group = get_object_or_404(UserGroup, id=group_id, created_by=request.user)
    memberships = UserGroupMembership.objects.filter(group=group)
    scenarios = group.assigned_scenarios.all()

    return render(request, 'usergroups/view_group.html', {
        'group': group,
        'memberships': memberships,
        'scenarios': scenarios,
    })

# FEB 20
@group_required('dspace_partners')
def list_student_groups(request):
    user_groups = UserGroup.objects.all().prefetch_related('assigned_scenarios', 'members')

    student_extras_by_scenarios = {
        # scenario_id: extra_implementations
        23: 126, #pend GR
        57: 29, #gal GR
        56: 29, #gal EN
        29: 27, #photo EN
        26: 31, #photo GR
    }

    group_data = []
    total_scenario_implementations = defaultdict(int)

    for group in user_groups:
        assigned_scenarios = group.assigned_scenarios.all()
        teacher = group.created_by
        num_students = group.members.count()

        # # Calculate implementations (users who answered at least one activity)
        # implementations_count = UserAnswer.objects.filter(
        #     user__in=group.members.all()
        # ).values('user').distinct().count()

        # Get first and last implementation dates
        user_answers = UserAnswer.objects.filter(
            user__in=group.members.all()
        )

        # Count implementations per scenario
        scenario_implementations = defaultdict(int)
        for scenario in assigned_scenarios:
            count = UserAnswer.objects.filter(
                user__in=group.members.all(),
                activity__scenario=scenario
            ).values('user').distinct().count()
            scenario_implementations[scenario.name] = count
            total_scenario_implementations[scenario.name] += count # Added for scenario's implementation count
        
        total_implementations = sum(scenario_implementations.values())

        # implementations_count = user_answers.values('user').distinct().count()
        first_implementation = user_answers.aggregate(first=Min('created_on'))['first']
        last_implementation = user_answers.aggregate(last=Max('created_on'))['last']

        group_data.append({
            'group': group,
            'teacher': teacher,
            'num_students': num_students,
            'implementations': total_implementations, # implementations_count,
            'first_implementation': first_implementation,
            'last_implementation': last_implementation,
            'scenario_implementations': dict(scenario_implementations),
            'scenarios': ", ".join([s.name for s in assigned_scenarios]),
        })

    # Sorting logic
    sort_by = request.GET.get('sort', 'implementations')  # Default sorting
    order = request.GET.get('order', 'desc')  # Default to descending

    # Ensure datetime.min is timezone-aware
    datetime_min = make_aware(datetime.min, timezone.utc)

    reverse = order == 'desc'
    new_order = 'asc' if order == 'desc' else 'desc'
    
    
    # Sorting logic
    if sort_by == 'teacher':
        group_data.sort(key=lambda x: x['teacher'].username, reverse=reverse)
    # elif sort_by == 'scenario':
    #     group_data.sort(key=lambda x: x['scenarios'], reverse=reverse)
    elif sort_by == 'scenario':
        group_data.sort(key=lambda x: (x['scenarios'] is None, x['scenarios'] or ''), reverse=reverse)
    elif sort_by == 'first_implementation':
        group_data.sort(key=lambda x: x['first_implementation'] or datetime_min, reverse=reverse)
    elif sort_by == 'last_implementation':
        group_data.sort(key=lambda x: x['last_implementation'] or datetime_min, reverse=reverse)
    elif sort_by == 'implementations':
        # Make sure 0 appears at the top when sorting ascending
        group_data.sort(key=lambda x: (x['implementations'] is None, x['implementations']), reverse=reverse)
    else:
        # Default sorting by implementations
        group_data.sort(key=lambda x: x['implementations'], reverse=reverse)
    
    # Pagination: Show 10 groups per page
    paginator = Paginator(group_data, 10)  
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 1) Map scenario IDs -> names (define BEFORE using it)
    scenarios_map = {
        s.id: s.name
        for s in Scenario.objects.filter(id__in=student_extras_by_scenarios.keys())
    }

    # 2) Start from the real totals (currently keyed by scenario.name)
    adjusted_totals = dict(total_scenario_implementations)
    extras_applied = {}

    # 3) Add the hardcoded extras (by name)
    for sid, extra in student_extras_by_scenarios.items():
        name = scenarios_map.get(sid)
        if not name:
            continue
        adjusted_totals[name] = adjusted_totals.get(name, 0) + int(extra)
        extras_applied[name] = int(extra)

    # 4) Sort for display
    adjusted_totals_sorted = dict(
        sorted(adjusted_totals.items(), key=lambda item: item[1], reverse=True)
    )

    # 5) Grand total (REAL + EXTRAS)
    grand_total_implementations = sum(adjusted_totals.values())

    # 6) Build rows so the template is dead-simple (no dict lookups)
    totals_rows = [
        {"name": name, "total": total, "extra": extras_applied.get(name, 0)}
        for name, total in adjusted_totals_sorted.items()
    ]


    return render(request, 'usergroups/list_with_imp.html', {'group_data': group_data, 'sort_by': sort_by, 'order': new_order, 
                                                             #'total_scenario_implementations': dict(sorted(total_scenario_implementations.items(), key=lambda item: item[1], reverse=True)),
                                                             'totals_rows': totals_rows,  # <-- use adjusted rows
                                                            'grand_total_implementations': grand_total_implementations,})

def export_multilingual_answers_csv(request):
    # Set up the HTTP response with CSV headers
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="multilingual_answers.csv"'

    writer = csv.writer(response, delimiter=';')
    
    # Get all distinct questions in consistent order
    questions = MultilingualQuestion.objects.all().order_by('order', 'created_on')
    
    # Prepare headers
    header = ['User ID', 'Username', 'Scenario', 'Created On'] + [
        f"Q{index+1}" for index, _ in enumerate(questions)
    ]
    writer.writerow(header)

    # Group answers by user + scenario
    answers = MultilingualAnswer.objects.select_related('user', 'scenario', 'question').order_by('user__id', 'scenario__id')

    grouped_data = {}
    for answer in answers:
        key = (answer.user.id, answer.scenario.id if answer.scenario else None)
        if key not in grouped_data:
            grouped_data[key] = {
                'user_id': answer.user.id,
                'username': answer.user.username,
                'scenario_name': answer.scenario.name if answer.scenario else 'N/A',
                'created_on': answer.created_on.strftime('%Y-%m-%d %H:%M'),
                'answers': {}
            }
        grouped_data[key]['answers'][answer.question.id] = answer.answer_text

    # Write each row
    for entry in grouped_data.values():
        row = [
            entry['user_id'],
            entry['username'],
            entry['scenario_name'],
            entry['created_on']
        ]
        # Fill answers in order of questions
        for question in questions:
            row.append(entry['answers'].get(question.id, ''))  # Empty if no answer
        writer.writerow(row)

    return response