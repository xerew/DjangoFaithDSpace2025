from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse, HttpResponseForbidden, FileResponse, Http404
from django.template import loader
from .models import Scenario, Phase, ActivityType, Activity, Answer, AnswerFeedback, NextQuestionLogic, QuestionBunch, EvQuestionBranching, Simulation, UserAnswer, UserScenarioScore, SchoolDepartment, ExperimentLL, RemoteLabSession, VRARExperiment, ActivityProposal, UserProposalReview, Language, Subject
from psycopg2.extras import NumericRange
from django.urls import reverse
from django.utils.html import strip_tags
from html import unescape
from .forms import AnswerForm
from django.forms import formset_factory
from django.contrib import messages
import re
from django.contrib.auth.models import User
from django.db.models import Sum, Count, Q, Max, Min, F, Avg
from django.core.cache import cache
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from functools import wraps
from django.db.models.functions import Lower
from django.utils.dateparse import parse_date
from datetime import timedelta
from organization.models import Organization
from usergroups.models import UserGroupMembership, UserGroup
# For LTI
import hmac
import base64
import time
from hashlib import sha1
import urllib.parse
# Celery & Redis
from .tasks import compute_sankey_data, compute_time_spent_by_performer_type, compute_final_performance, compute_performance_data, compute_detailed_phase_scores_data, compute_performers_data, compute_activity_answers_data, compute_time_spent_data, compute_scenario_paths, compute_student_performance_metrics, compute_category_metrics_per_phase_activity, generate_llm_context_for_scenario, calculate_activities_in_risk, apply_user_proposals_to_new_scenario
from .utils import get_last_answers
from celery.result import AsyncResult
from django.conf import settings
import csv, os, json
from pathlib import Path
from django.views.decorators.http import require_POST
import markdown
from django.utils.safestring import mark_safe
from django.core.files.storage import FileSystemStorage
from django.utils._os import safe_join

def is_admin_user(user):
    """Returns True for superusers and staff — both get full scenario access."""
    return user.is_superuser or user.is_staff


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

def abbreviate_activity_name(name):
    # Assuming the format is "Δραστηριότητα X", extract the number and prepend "Δρ."
    parts = name.split(' ')
    if (len(parts) == 2 or len(parts) == 3) and parts[0] == "Δραστηριότητα":
        return f"Δρ. {parts[1]}" + (f" {parts[2]}" if len(parts) > 2 else "")
    return name

def sanitize_mermaid_text(text: str) -> str:
    if not text:
        return ""
    # Remove normal double quotes completely (or replace with ')
    text = text.replace('"', "'")
    # Also remove backslashes just in case
    text = text.replace("\\", "")
    # Optional: brackets can confuse Mermaid too, make them safe
    text = text.replace("[", "(").replace("]", ")")
    # Collapse whitespace/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def generate_flowchart(scenario_id):
    print(f"🧩 Starting flowchart generation for scenario {scenario_id}")
    try:
        phases = Phase.objects.filter(scenario_id=scenario_id).prefetch_related('activities')
        branching_logic = EvQuestionBranching.objects.filter(activity__scenario_id=scenario_id)
        next_activity_logic = NextQuestionLogic.objects.filter(activity__scenario_id=scenario_id)

        graph_definition = "graph TD\n"

        # Track node categories for classDef styling
        act_nodes = []   # regular activities
        ev_nodes  = []   # evaluatable/branching activities
        end_nodes = []   # END terminal nodes

        # Track edge insertion order for linkStyle colouring
        edge_idx    = 0
        high_edges  = []
        mid_edges   = []
        low_edges   = []

        # ── Phase subgraphs + node definitions ──────────────────────
        for phase in phases:
            safe_phase_name = sanitize_mermaid_text(phase.name)
            graph_definition += f"subgraph phase_{phase.id}[\"{safe_phase_name}\"]\n"

            for activity in phase.activities.all():
                safe_act_name = sanitize_mermaid_text(activity.name)
                print(f"   • Activity: {activity.name} (ID {activity.id})")
                graph_definition += f"A{activity.id}[\"{safe_act_name}\"]\n"
                graph_definition += (
                    f"click A{activity.id} href "
                    f"\"/authoringtool/scenarios/{scenario_id}/viewPhase/{phase.id}/viewActivity/{activity.id}/\"\n"
                )
                if activity.is_evaluatable:
                    ev_nodes.append(f"A{activity.id}")
                else:
                    act_nodes.append(f"A{activity.id}")

            graph_definition += "end\n"

        # ── Edges: branching + answer-based + direct ─────────────────
        for activity in Activity.objects.filter(scenario_id=scenario_id).prefetch_related('answers'):
            branching = branching_logic.filter(activity=activity).first()

            if branching:
                for branch, label in [
                    (branching.next_question_on_high, 'High'),
                    (branching.next_question_on_mid,  'Moderate'),
                    (branching.next_question_on_low,  'Low'),
                ]:
                    if branch:
                        safe_branch_name = sanitize_mermaid_text(branch.name)
                        graph_definition += (
                            f"A{activity.id} -->|{label}| "
                            f"A{branch.id}[\"{safe_branch_name}\"]\n"
                        )
                        if label == 'High':
                            high_edges.append(edge_idx)
                        elif label == 'Moderate':
                            mid_edges.append(edge_idx)
                        elif label == 'Low':
                            low_edges.append(edge_idx)
                        edge_idx += 1

            if activity.activity_type and activity.activity_type.name == 'Question':
                for answer in activity.answers.all():
                    answer_next_activity = next_activity_logic.filter(
                        activity=activity, answer=answer
                    ).first()
                    if answer_next_activity and answer_next_activity.next_activity:
                        match = re.search(r'\b[A-E]\b', answer.text)
                        answer_text = answer.text[:match.start()] + match.group() if match else answer.text[:20]
                        safe_label = sanitize_mermaid_text(f"Answer: {answer_text}")
                        safe_next_name = sanitize_mermaid_text(answer_next_activity.next_activity.name)
                        graph_definition += (
                            f"A{activity.id} -->|{safe_label}| "
                            f"A{answer_next_activity.next_activity.id}[\"{safe_next_name}\"]\n"
                        )
                        edge_idx += 1
            else:
                direct_next_activity = next_activity_logic.filter(
                    activity=activity, answer__isnull=True
                ).first()
                if direct_next_activity and direct_next_activity.next_activity:
                    safe_next_name = sanitize_mermaid_text(direct_next_activity.next_activity.name)
                    graph_definition += (
                        f"A{activity.id} --> "
                        f"A{direct_next_activity.next_activity.id}[\"{safe_next_name}\"]\n"
                    )
                    edge_idx += 1

        # ── END terminal nodes ────────────────────────────────────────
        for activity in Activity.objects.filter(scenario_id=scenario_id):
            has_next = False
            if branching_logic.filter(activity=activity).exists():
                has_next = True
            elif next_activity_logic.filter(
                activity=activity, answer__isnull=True, next_activity__isnull=False
            ).exists():
                has_next = True
            else:
                for answer in activity.answers.all():
                    if next_activity_logic.filter(
                        activity=activity, answer=answer, next_activity__isnull=False
                    ).exists():
                        has_next = True
                        break

            if not has_next:
                safe_end_label = sanitize_mermaid_text("END")
                graph_definition += f"A{activity.id} --> END{activity.id}([\"{safe_end_label}\"])\n"
                end_nodes.append(f"END{activity.id}")
                edge_idx += 1

        # ── Class definitions ─────────────────────────────────────────
        graph_definition += (
            "classDef actClass fill:#dbeafe,stroke:#1a56db,stroke-width:1.5px,color:#1e3a8a\n"
            "classDef evClass  fill:#ede9fe,stroke:#7c3aed,stroke-width:1.5px,color:#4c1d95\n"
            "classDef endClass fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#475569\n"
        )
        if act_nodes:
            graph_definition += f"class {','.join(act_nodes)} actClass\n"
        if ev_nodes:
            graph_definition += f"class {','.join(ev_nodes)} evClass\n"
        if end_nodes:
            graph_definition += f"class {','.join(end_nodes)} endClass\n"

        # ── Link colours for High / Moderate / Low branches ──────────
        for i in high_edges:
            graph_definition += f"linkStyle {i} stroke:green,stroke-width:2.5px\n"
        for i in mid_edges:
            graph_definition += f"linkStyle {i} stroke:orange,stroke-width:2.5px\n"
        for i in low_edges:
            graph_definition += f"linkStyle {i} stroke:red,stroke-width:2.5px\n"

        print("========== MERMAID OUTPUT ==========")
        print(graph_definition)
        print("====================================")
        print(f"✅ Flowchart generated successfully for scenario {scenario_id}")
        return graph_definition

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR generating flowchart for scenario {scenario_id}: {e}")
        traceback.print_exc()
        return None


def strip_html_tags(html_content):
    text = strip_tags(html_content)
    return unescape(text)

@group_required('teachers')
def index(request):
#     scenarios = Scenario.objects.all()
#     departments = SchoolDepartment.objects.all()
#     return render(request, 'authoringtool/index.html', {'scenarios': scenarios, 'departments': departments})
    user = request.user
    departments = SchoolDepartment.objects.all()
    # print('EINAI', request.user.is_staff)
    if is_admin_user(user):
        myScenarios = Scenario.objects.all()
    # Check if the user belongs to the 'teachers' group
    # if user.groups.filter(name='teachers').exists():
        # If the user is a teacher, show their scenarios, public ones, and org ones they belong to
    else:
        org_ids = user.member_of_organizations.values_list('id', flat=True)
        myScenarios = Scenario.objects.filter(
            Q(created_by=user) |  # Scenarios the user created
            Q(visibility_status='public') |  # Public scenarios
            Q(visibility_status='org', organizations__id__in=org_ids)  # Org-only scenarios the user is part of
        ).distinct()
    
    template = loader.get_template('authoringtool/index.html')
    context = {
        'scenarios': myScenarios,
        'departments': departments
    }
    return HttpResponse(template.render(context, request))

def _smart_page_range(page_obj):
    """Returns a list of page numbers with None representing an ellipsis."""
    current = page_obj.number
    total = page_obj.paginator.num_pages
    pages = set()
    pages.add(1)
    pages.add(total)
    for i in range(max(1, current - 2), min(total, current + 2) + 1):
        pages.add(i)
    result = []
    prev = None
    for p in sorted(pages):
        if prev is not None and p - prev > 1:
            result.append(None)  # ellipsis
        result.append(p)
        prev = p
    return result


@group_required('teachers')
def scenarios(request):
    query = request.GET.get('q', '').strip()  # Search query
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    language = request.GET.get('language')
    show_mine = request.GET.get('show_mine') == 'on'
    visibility_filter = request.GET.get('visibility', 'all')  # Visibility filter
    selected_subject_ids = [int(x) for x in request.GET.getlist('subject') if x.isdigit()]

    filters = Q()

    # Search by name, description, or creator username
    if query:
        filters &= Q(name__icontains=query) | Q(description__icontains=query) | Q(created_by__username__icontains=query)

    # Date filtering
    if start_date:
        filters &= Q(created_on__date__gte=start_date)
    if end_date:
        filters &= Q(created_on__date__lte=end_date)

    # Language filter
    if language:
        filters &= Q(language=language)

    # Subject filter
    if selected_subject_ids:
        filters &= Q(subjects__id__in=selected_subject_ids)

    # Show only the current user's scenarios if 'Show Mine' is selected
    if show_mine:
        filters &= Q(created_by=request.user)

    # Visibility filter based on the user's selection
    if visibility_filter == 'private':
        filters &= Q(visibility_status='private', created_by=request.user)
    elif visibility_filter == 'org':
        org_ids = request.user.member_of_organizations.values_list('id', flat=True)
        filters &= Q(visibility_status='org', organizations__id__in=org_ids)
    elif visibility_filter == 'public':
        filters &= Q(visibility_status='public')

    # Final filtering logic: user should only see their private scenarios, org scenarios they're a member of, or public ones
    if is_admin_user(request.user):
        # Admins (superuser/staff) can see all scenarios, no visibility restrictions
        visible_scenarios = Scenario.objects.filter(filters).distinct().order_by('-created_on')
    else:
        org_ids = request.user.member_of_organizations.values_list('id', flat=True)
        visible_scenarios = Scenario.objects.filter(
            Q(visibility_status='public') |  # Public scenarios visible to everyone
            Q(visibility_status='private', created_by=request.user) |  # Private scenarios visible only to the creator
            Q(visibility_status='org', organizations__id__in=org_ids)  # Org-only scenarios visible to members of the org
        ).filter(filters).distinct().order_by('-created_on')

    # Paginate before the can_edit loop (15 per page)
    paginator = Paginator(visible_scenarios, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Process only the current page's scenarios for can_edit
    user_org_ids = set(request.user.member_of_organizations.values_list('id', flat=True))
    for scenario in page_obj:
        scenario.can_edit = False
        if is_admin_user(request.user):
            scenario.can_edit = True
        if scenario.visibility_status == 'private' and scenario.created_by == request.user:
            scenario.can_edit = True
        elif scenario.visibility_status == 'org':
            scenario_org_ids = set(scenario.organizations.values_list('id', flat=True))
            if user_org_ids & scenario_org_ids and scenario.is_editable_by_org:
                scenario.can_edit = True
        elif scenario.visibility_status == 'public' and scenario.created_by == request.user:
            scenario.can_edit = True

    # Get distinct languages for the filter dropdown
    languages = Scenario.objects.values_list('language', flat=True).distinct().order_by('language')

    # Build query string without 'page' for pagination links
    get_params = request.GET.copy()
    get_params.pop('page', None)
    filter_querystring = get_params.urlencode()

    # Smart page range: always include first/last, ellipsis (None) for gaps
    page_range = _smart_page_range(page_obj)

    template = loader.get_template('authoringtool/scenarios.html')
    context = {
        'myScenarios': page_obj,
        'page_obj': page_obj,
        'page_range': page_range,
        'filter_querystring': filter_querystring,
        'query': query,
        'start_date': start_date,
        'end_date': end_date,
        'languages': languages,
        'selected_language': language,
        'show_mine': show_mine,
        'visibility_filter': visibility_filter,
        'all_subjects': Subject.objects.all(),
        'selected_subject_ids': selected_subject_ids,
    }
    return HttpResponse(template.render(context, request))

@group_required('teachers')
def createScenario(request):
    template = loader.get_template('authoringtool/createScenario.html')
    return HttpResponse(template.render({
        'languages': Language.objects.all(),
        'all_subjects': Subject.objects.all(),
    }, request))

@group_required('teachers')
def createScenarioData(request):
    name = request.POST.get('name')
    # Check if a scenario with the same name already exists
    if Scenario.objects.filter(name=name).exists():
        messages.error(request, 'A scenario with this name already exists.')
        return HttpResponseRedirect(reverse('createScenario'))  

    learning_goals = request.POST.get('learning_goals')
    description = request.POST.get('description')
    age_of_students_start = request.POST.get('min_age')
    age_of_students_end = request.POST.get('max_age')
    age_of_students_range = NumericRange(int(age_of_students_start), int(age_of_students_end))
    subject_domains = request.POST.get('subject', '')
    language = request.POST.get('language')
    suggested_learning_time = request.POST.get('suggested_learning_time')
    image = request.FILES.get('image_upload')
    if not image:
        messages.error(request, 'An image is required.')
        return HttpResponseRedirect(reverse('createScenario'))
    created_by = request.user

    newScenario = Scenario(name=name, learning_goals=learning_goals, description=description,
                           age_of_students=age_of_students_range, subject_domains=subject_domains,
                           language=language, suggested_learning_time=suggested_learning_time,
                           image=image, visibility_status='private',
                           is_editable_by_org=False, created_by=created_by)
    newScenario.save()
    selected_subject_ids = request.POST.getlist('subjects')
    if selected_subject_ids:
        newScenario.subjects.set(selected_subject_ids)
    messages.success(request, f"Scenario '{newScenario.name}' created successfully.")
    return HttpResponseRedirect(reverse('scenarios'))

@group_required('teachers')
def updateScenario(request, id):
    updateScenario = get_object_or_404(Scenario, id=id)
    if updateScenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    scenario_min_age = updateScenario.age_of_students.lower if updateScenario.age_of_students else None
    scenario_max_age = updateScenario.age_of_students.upper if updateScenario.age_of_students else None
    # Get the organizations the user is a member of
    user_organizations = request.user.member_of_organizations.all()
    template = loader.get_template('authoringtool/updateScenario.html')
    context = {
        'Scenario': updateScenario,
        'min_age': scenario_min_age,
        'max_age': scenario_max_age,
        'user_organizations': user_organizations,
        'languages': Language.objects.all(),
        'all_subjects': Subject.objects.all(),
        'selected_subject_ids': list(updateScenario.subjects.values_list('id', flat=True)),
    }
    return HttpResponse(template.render(context, request))

@group_required('teachers')
def updateScenarioData(request, id):
    updateScenario = get_object_or_404(Scenario, id=id)
    if updateScenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    visibility = request.POST.get('visibility')
    
    # Check if the visibility is 'org' and no organizations are selected
    selected_org_ids = request.POST.getlist('organizations')
    if visibility == 'org' and not selected_org_ids:
        messages.error(request, 'You must select at least one organization if the visibility is set to Organization.')
        return HttpResponseRedirect(reverse('updateScenario', args=[id]))
    
    name = request.POST.get('name')
    if Scenario.objects.filter(name=name).exclude(pk=id).exists():
        messages.error(request, 'A scenario with this name already exists.')
        return HttpResponseRedirect(reverse('updateScenario', args=[id])) 
    learning_goals = request.POST.get('learning_goals')
    # visibility = request.POST.get('visibility')
    editable = request.POST.get('is_editable_by_org') == 'on'
    description = request.POST.get('description')
    age_of_students_start = request.POST.get('min_age')
    print(age_of_students_start)
    age_of_students_end = request.POST.get('max_age')
    print(age_of_students_end)
    age_of_students_range = NumericRange(int(age_of_students_start), int(age_of_students_end))
    subject_domains = request.POST.get('subject', '')
    language = request.POST.get('language')
    suggested_learning_time = request.POST.get('suggested_learning_time')
    image = request.FILES.get('image_upload')
    if 'clear_image' in request.POST:
        image = None
    updateScenario.name = name
    updateScenario.learning_goals = learning_goals
    updateScenario.visibility_status = visibility
    updateScenario.is_editable_by_org = editable
    updateScenario.description = description
    updateScenario.age_of_students = age_of_students_range
    updateScenario.subject_domains = subject_domains
    updateScenario.language = language
    updateScenario.suggested_learning_time = suggested_learning_time
    updateScenario.updated_by = request.user
    if image is not None:
        updateScenario.image = image
    updateScenario.save()
    # Handle subjects
    updateScenario.subjects.set(request.POST.getlist('subjects'))
    # Handle organization visibility
    if visibility == 'org':
        selected_organizations = request.POST.getlist('organizations')
        updateScenario.organizations.set(selected_organizations)
    else:
        updateScenario.organizations.clear()
    return HttpResponseRedirect(reverse('scenarios'))

@group_required('teachers')
def deleteScenario(request, id):
    deleteScenario = get_object_or_404(Scenario, id=id)
    if deleteScenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    deleteScenario.delete()
    return HttpResponseRedirect(reverse('scenarios'))

def serve_rag_pdf(request, scenario_id, filename):
    try:
        folder = safe_join(settings.RAG_PDFS_ROOT, f"scenario_{scenario_id}")
        file_path = safe_join(folder, filename)
    except Exception:
        raise Http404("Invalid path")

    if not os.path.exists(file_path):
        raise Http404("File not found")

    return FileResponse(open(file_path, "rb"), content_type="application/pdf")

@group_required('teachers')
def viewScenario(request, id):
    myScenario = Scenario.objects.get(id=id)
    # Check if the user is the creator, an admin, or if the scenario is editable by the org and the user belongs to the org
    can_edit = False
    if is_admin_user(request.user) or myScenario.created_by == request.user:
        can_edit = True
    elif myScenario.visibility_status == 'org' and myScenario.is_editable_by_org:
        if myScenario.organizations.filter(members=request.user).exists():
            can_edit = True

    # ⬇️ Handle PDF upload for RAG
    if request.method == "POST" and request.FILES.get('rag_pdf'): # 2026
        rag_folder = os.path.join(settings.RAG_PDFS_ROOT, f"scenario_{id}")# os.path.join(settings.BASE_DIR, 'rag_pdfs', f'scenario_{id}')
        os.makedirs(rag_folder, exist_ok=True)
        uploaded_file = request.FILES['rag_pdf']
        fs = FileSystemStorage(location=rag_folder)
        fs.save(uploaded_file.name, uploaded_file)
        messages.success(request, f"Uploaded RAG PDF: {uploaded_file.name}")
        return HttpResponseRedirect(reverse('viewScenario', args=[id]))
    
    scenarioPhases = Phase.objects.filter(scenario=id) # Me
    mermaid_graph_definition = generate_flowchart(id)
    # print(f'MERMAID: \n', mermaid_graph_definition)
    scenario_min_age = myScenario.age_of_students.lower if myScenario.age_of_students else None
    scenario_max_age = myScenario.age_of_students.upper if myScenario.age_of_students else None

    # List existing RAG files # 2026
    rag_folder = os.path.join(settings.RAG_PDFS_ROOT, f"scenario_{id}")# os.path.join(settings.BASE_DIR, 'rag_pdfs', f'scenario_{id}')
    if os.path.exists(rag_folder):
        rag_files = [f for f in os.listdir(rag_folder) if f.lower().endswith('.pdf')]
    else:
        rag_files = []


    implementation_count = UserScenarioScore.objects.filter(scenario=myScenario).values('user').distinct().count()

    template = loader.get_template('authoringtool/viewScenario.html')
    context = {
        'myScenario': myScenario,
        'min_age': scenario_min_age,
        'max_age': scenario_max_age,
        'Phases': scenarioPhases,
        'phase_count': scenarioPhases.count(),
        'mermaid_graph_definition': mermaid_graph_definition,
        'can_edit': can_edit,
        'rag_files': rag_files,
        "llm_html": mark_safe(markdown.markdown(myScenario.llm_context or "")),
        'implementation_count': implementation_count,
    }
    return HttpResponse(template.render(context, request))

def delete_rag_pdf(request, scenario_id, filename):
    scenario = Scenario.objects.get(id=scenario_id)
    if request.user != scenario.created_by and not is_admin_user(request.user):
        messages.error(request, "Only the scenario owner can delete RAG PDFs.")
        return HttpResponseRedirect(reverse('viewScenario', args=[scenario_id]))

    folder = os.path.join(settings.RAG_PDFS_ROOT, f"scenario_{scenario_id}") # os.path.join(settings.BASE_DIR, 'rag_pdfs', f'scenario_{scenario_id}')
    safe_path = (Path(folder) / filename).resolve()
    if not str(safe_path).startswith(str(Path(folder).resolve())):
        return HttpResponseForbidden("Invalid path")
    file_path = safe_path
    if os.path.exists(file_path):
        os.remove(file_path)
        messages.success(request, f"Deleted {filename}")
    else:
        messages.error(request, "File not found.")

    return HttpResponseRedirect(reverse('viewScenario', args=[scenario_id]))

@group_required('teachers')
def createPhase(request, id):
    scenario = get_object_or_404(Scenario, id=id)
    context = {'scenario_id': id, 'myScenario': scenario}
    return render(request, 'authoringtool/createPhase.html', context)

@group_required('teachers')
def createPhaseData(request, id):
    name = request.POST.get('name')
    description = request.POST.get('description')
    image = request.FILES.get('image_upload')
    scenario = request.POST.get('scenario_id')
    created_by = request.user

    scenario_instance = get_object_or_404(Scenario, id=scenario)

    if Phase.objects.filter(scenario=scenario_instance).count() >= 5:
        messages.error(request, "A scenario cannot have more than 5 phases.")
        return HttpResponseRedirect(reverse('viewScenario', args=[scenario]))

    newPhase = Phase(name=name, description=description, image=image, scenario=scenario_instance, created_by=created_by)
    newPhase.save()
    return HttpResponseRedirect(reverse('viewScenario', args=[scenario]))

@group_required('teachers')
def updatePhase(request, scenario_id, phase_id):
    updatePhase = get_object_or_404(Phase, id=phase_id)
    scenario = get_object_or_404(Scenario, id=scenario_id)
    if scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    template = loader.get_template('authoringtool/updatePhase.html')
    context = {
        'Phase': updatePhase,
        'scenario_id': scenario_id,
        'myScenario': scenario,
    }
    return render(request, 'authoringtool/updatePhase.html', context)

@group_required('teachers')
def updatePhaseData(request, scenario_id, phase_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    if scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    name = request.POST.get('name')
    description = request.POST.get('description')
    image = request.FILES.get('image_upload')
    updatePhase = Phase.objects.get(id=phase_id)
    updatePhase.name = name
    updatePhase.description = description
    if image:
        updatePhase.image = image
    updatePhase.updated_by = request.user
    updatePhase.save()
    return HttpResponseRedirect(reverse('viewScenario', args=[scenario_id]))

@group_required('teachers')
def deletePhase(request, scenario_id, phase_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    if scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    if Phase.objects.filter(scenario=scenario).count() <= 1:
        messages.error(request, "A scenario must have at least 1 phase.")
        return HttpResponseRedirect(reverse('viewScenario', args=[scenario_id]))
    deletePhase = get_object_or_404(Phase, id=phase_id)
    deletePhase.delete()
    return HttpResponseRedirect(reverse('viewScenario', args=[scenario_id]))

def viewPhase(request, scenario_id, phase_id):
    myPhase = Phase.objects.get(id=phase_id)
    myScenario = Scenario.objects.get(id=scenario_id)
    # Check if the user is the creator, an admin, or if the scenario is editable by the org and the user belongs to the org
    can_edit = False
    if is_admin_user(request.user) or myScenario.created_by == request.user:
        can_edit = True
    elif myScenario.visibility_status == 'org' and myScenario.is_editable_by_org:
        if myScenario.organizations.filter(members=request.user).exists():
            can_edit = True
    activities = Activity.objects.filter(phase=phase_id).order_by('id')
    # scenarioPhases = Phase.objects.filter(scenario=id) # Me
    template = loader.get_template('authoringtool/viewPhase.html')
    context = {
        'myPhase': myPhase,
        'myScenario': myScenario,
        'activities': activities,
        'can_edit': can_edit,
        "llm_html": mark_safe(markdown.markdown(myPhase.llm_context or "")),
    }
    return HttpResponse(template.render(context, request))

@group_required('teachers')
def createActivity(request, scenario_id, phase_id):
    activityTypes = ActivityType.objects.all()
    scenario = Scenario.objects.get(id=scenario_id)
    phase = Phase.objects.get(id=phase_id)
    simulations = Simulation.objects.all()
    # Fetch LabsLand experiments from the ExperimentLL model
    remote_labs = ExperimentLL.objects.all()
    # Fetch VR/AR Labs
    vr_ar_exp = VRARExperiment.objects.all()
    # linked_activities = NextQuestionLogic.objects.values_list('next_activity', flat=True)
    eligible_activities = Activity.objects.select_related('phase').filter(
        scenario=scenario_id
    ).order_by('phase__created_on', 'created_on')
    # eval_activities: current phase + later phases only (for evaluation multi-picker)
    eval_activities = Activity.objects.select_related('phase').filter(
        scenario=scenario_id,
        phase__created_on__gte=phase.created_on
    ).order_by('phase__created_on', 'created_on')
    context = {
        'activityTypes': activityTypes,
        'myScenario': scenario,
        'myPhase': phase,
        'eligible_activities': eligible_activities,
        'eval_activities': eval_activities,
        'simulations': simulations,
        'remote_labs': remote_labs,
        'vr_ar_exp': vr_ar_exp,
    }
    # template = loader.get_template('authoringtool/createPhase.html')
    return render(request, 'authoringtool/createActivity.html', context)

@group_required('teachers')
def createActivityData(request, scenario_id, phase_id):
    activity_name = request.POST.get('activity_name')
    activity_text = request.POST.get('activity_text')
    plain_text_content = strip_html_tags(activity_text)
    is_evaluatable = request.POST.get('is_evaluatable') == 'on'
    is_primary_ev = request.POST.get('is_primary_ev') == 'on'
    activity_type = request.POST.get('activity_type')
    helping_quote = request.POST.get('helping_quote')
    next_activity_id = request.POST.get('next_activity_id')
    simulation_id = request.POST.get('simulation')
    remote_lab_id = request.POST.get('remote_lab')  # Get the selected remote lab URL
    vr_ar_lab_id = request.POST.get('VRAR_lab')
    experiment_type = request.POST.get('experiment_type')  # Get the experiment type
    
    scenario_instance = get_object_or_404(Scenario, id=scenario_id)
    phase_instance = get_object_or_404(Phase, id=phase_id)
    activity_type_instance = get_object_or_404(ActivityType, id=activity_type)
    
    # Initialize all experiment instances as None
    simulation_instance = None
    experiment_instance = None
    vr_ar_instance = None
    
    if activity_type_instance.name == 'Experiment':
        # Only set the appropriate instance based on experiment_type
        if experiment_type == 'simulation' and simulation_id:
            simulation_instance = get_object_or_404(Simulation, id=simulation_id)
        elif experiment_type == 'remote_lab' and remote_lab_id:
            experiment_instance = get_object_or_404(ExperimentLL, id=remote_lab_id)
        elif experiment_type == 'vr_ar_exp' and vr_ar_lab_id:
            vr_ar_instance = get_object_or_404(VRARExperiment, id=vr_ar_lab_id)
            
    created_by = request.user

    newActivity = Activity(
        name=activity_name,
        text=activity_text,
        plain_text=plain_text_content,
        is_evaluatable=is_evaluatable,
        activity_type=activity_type_instance,
        helper=helping_quote,
        scenario=scenario_instance,
        phase=phase_instance,
        created_by=created_by,
        simulation=simulation_instance,
        is_primary_ev=is_primary_ev,
        experiment_ll=experiment_instance,
        vr_ar_experiment=vr_ar_instance
    )
    newActivity.save()
    if next_activity_id:
            if next_activity_id == 'create_new':
                current_activity = get_object_or_404(Activity, id=newActivity.id)
                new_activity = Activity.objects.create(
                    name=f"Next activity of {current_activity.name}",
                    scenario=current_activity.scenario,
                    phase=current_activity.phase,
                    text=f"Activity created by {current_activity.name}",
                    plain_text=f"Activity created by {current_activity.name}",
                    activity_type=activity_type_instance,
                    created_by = created_by
                )
                NextQuestionLogic.objects.create(
                activity=current_activity,
                next_activity=new_activity,
                )
            else:
                next_activity_instance = get_object_or_404(Activity, id=next_activity_id)
                newNextQuestionLogic = NextQuestionLogic(activity=newActivity, next_activity=next_activity_instance)
                # NextQuestionLogic.objects.create(activity=newActivity, next_activity=next_activity_instance)
                newNextQuestionLogic.save()
    selected_activities_ids = request.POST.getlist('selected_activities')
    activity_primary_id = newActivity.id
    activity_primary = get_object_or_404(Activity, id=activity_primary_id)
    selected_activities_ids = [int(id) for id in selected_activities_ids if id.isdigit()]
    # 11/05/24 - Start #
    selected_activities_ids.append(activity_primary_id)
    # 11/05/24 - End #
    newQuestionBunch = QuestionBunch(activity_primary=activity_primary, activity_ids = selected_activities_ids)
    newQuestionBunch.save()
    if newActivity.is_evaluatable:
        print('I GOT HERE 1')
        return HttpResponseRedirect(reverse('updateCriterion', args=[scenario_id, phase_id, newActivity.id]))
    elif newActivity.activity_type.name == 'Question':
        print('I GOT HERE 2')
        return HttpResponseRedirect(reverse('updateAnswers', args=[scenario_id, phase_id, newActivity.id]))
    else:
        print('I GOT HERE 3')
        return HttpResponseRedirect(reverse('phase', args=[scenario_id, phase_id]))

@group_required('teachers')
def updateActivity(request, scenario_id, phase_id, activity_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    if scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    updateActivity = Activity.objects.get(id=activity_id)
    scenario = Scenario.objects.get(id=scenario_id)
    phase = Phase.objects.get(id=phase_id)
    activityTypes = ActivityType.objects.all()
    simulations = Simulation.objects.all()
    remote_labs = ExperimentLL.objects.all()
    vr_ar_exp = VRARExperiment.objects.all()
    
    # Initialize experiment instances
    existing_sim = None
    existing_remote = None
    existing_vr_ar = None
    current_experiment_type = None
    
    # Determine which experiment type is set and get the corresponding instance
    if updateActivity.simulation:
        existing_sim = Simulation.objects.get(id=updateActivity.simulation.id)
        current_experiment_type = 'simulation'
    elif updateActivity.experiment_ll:
        existing_remote = ExperimentLL.objects.get(id=updateActivity.experiment_ll.id)
        current_experiment_type = 'remote_lab'
    elif updateActivity.vr_ar_experiment:
        existing_vr_ar = VRARExperiment.objects.get(id=updateActivity.vr_ar_experiment.id)
        current_experiment_type = 'vr_ar_exp'
    
    # next_activity_logic = NextQuestionLogic.objects.filter(activity=activity_id).first()
    next_activity_logic = NextQuestionLogic.objects.filter(
        activity=activity_id, answer__isnull=True
    ).first()
    nextActivityIds = QuestionBunch.objects.filter(activity_primary=activity_id)
    if nextActivityIds.exists():
        nextActivityIdsList = nextActivityIds.first().activity_ids
    else:
        nextActivityIdsList = []
    nextActivityIdsList = [id for bunch in nextActivityIds for id in bunch.activity_ids]
    if next_activity_logic:
        next_activity_data = next_activity_logic.next_activity
    else:
        next_activity_data = None
    eligible_activities = Activity.objects.select_related('phase').filter(
        scenario=scenario_id
    ).exclude(id=activity_id).order_by('phase__created_on', 'created_on')
    # eval_activities: current phase + later phases only (for evaluation multi-picker)
    eval_activities = Activity.objects.select_related('phase').filter(
        scenario=scenario_id,
        phase__created_on__gte=phase.created_on
    ).exclude(id=activity_id).order_by('phase__created_on', 'created_on')
    template = loader.get_template('authoringtool/updateActivity.html')
    context = {
        'Activity': updateActivity,
        'myPhase': phase,
        'myScenario': scenario,
        'activityTypes': activityTypes,
        'simulations': simulations,
        'remote_labs': remote_labs,
        'vr_ar_exp': vr_ar_exp,
        'current_activity_type_id': updateActivity.activity_type.id,
        'current_activity_type_name': updateActivity.activity_type.name,
        'eligible_activities': eligible_activities,
        'eval_activities': eval_activities,
        'nextActivity': next_activity_data,
        'nextActivityIdsList': nextActivityIdsList,
        'existing_sim': existing_sim,
        'existingRemote': existing_remote,
        'existingVR_AR': existing_vr_ar,
        'current_experiment_type': current_experiment_type,
    }
    return render(request, 'authoringtool/updateActivity.html', context)

@group_required('teachers')
def updateActivityData(request, scenario_id, phase_id, activity_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    if scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    formerActivity = Activity.objects.get(id = activity_id)
    activity_name = request.POST.get('activity_name')
    activity_text = request.POST.get('activity_text')
    plain_text_content = strip_html_tags(activity_text)
    is_evaluatable = request.POST.get('is_evaluatable') == 'on'
    is_primary_ev = request.POST.get('is_primary_ev') == 'on'
    activity_type = request.POST.get('activity_type')
    helping_quote = request.POST.get('helping_quote')
    next_activity_id = request.POST.get('next_activity_id')
    simulation_id = request.POST.get('simulation')
    experimentll_id = request.POST.get('remote_lab')
    vr_ar_id = request.POST.get('VRAR_lab')
    experiment_type = request.POST.get('experiment_type')
    
    updateActivity = Activity.objects.get(id=activity_id)
    updateActivity.name = activity_name
    updateActivity.text = activity_text
    updateActivity.plain_text = plain_text_content
    updateActivity.is_evaluatable = is_evaluatable
    updateActivity.is_primary_ev = is_primary_ev
    activity_type_instance = get_object_or_404(ActivityType, id=activity_type)
    
    # First, clear all experiment types
    updateActivity.simulation = None
    updateActivity.experiment_ll = None
    updateActivity.vr_ar_experiment = None
    
    if activity_type_instance.name == 'Experiment':
        # Set only the selected experiment type
        if experiment_type == 'simulation' and simulation_id:
            updateActivity.simulation = get_object_or_404(Simulation, id=simulation_id)
        elif experiment_type == 'remote_lab' and experimentll_id:
            updateActivity.experiment_ll = get_object_or_404(ExperimentLL, id=experimentll_id)
        elif experiment_type == 'vr_ar_exp' and vr_ar_id:
            updateActivity.vr_ar_experiment = get_object_or_404(VRARExperiment, id=vr_ar_id)
            
    updateActivity.activity_type = activity_type_instance
    updateActivity.helper = helping_quote
    updateActivity.updated_by = request.user
    updateActivity.save()
    if updateActivity.is_evaluatable:
        next_activity_id = None
        selected_activities_ids = request.POST.getlist('selected_activities')
        activity_primary_id = updateActivity.id
        activity_primary = get_object_or_404(Activity, id=activity_primary_id)
        selected_activities_ids = [int(id) for id in selected_activities_ids if id.isdigit()]
        # 11/05/24 - Start #
        selected_activities_ids.append(activity_primary_id)
        # 11/05/24 - End #
        question_bunch, created = QuestionBunch.objects.update_or_create(
            activity_primary=activity_primary,
            defaults={'activity_ids': selected_activities_ids}
        )
        question_bunch.save()
    if next_activity_id:
        if next_activity_id == 'create_new':
            if NextQuestionLogic.objects.filter(activity=updateActivity).exists():
                NextQuestionLogic.objects.filter(activity=updateActivity).delete()
            if activity_type_instance.name == 'Experiment':
                activity_type_instance = get_object_or_404(ActivityType, name='Explanation')
            current_activity = get_object_or_404(Activity, id=updateActivity.id)
            new_activity = Activity.objects.create(
                name=f"Next activity of {current_activity.name}",
                scenario=current_activity.scenario,
                phase=current_activity.phase,
                text=f"Activity created by {current_activity.name}",
                plain_text=f"Activity created by {current_activity.name}",
                activity_type=activity_type_instance,
                created_by = request.user
            )
            NextQuestionLogic.objects.create(
            activity=current_activity,
            next_activity=new_activity,
            )
        else:
            next_activity_ac = get_object_or_404(Activity, id=next_activity_id)
            former_activity_exists = NextQuestionLogic.objects.filter(activity=updateActivity).exists()
            if former_activity_exists:
                updateNextQuestionLogic = NextQuestionLogic.objects.filter(activity=updateActivity)
                for next_logic in updateNextQuestionLogic:
                    next_logic.activity = updateActivity
                    next_logic.next_activity = next_activity_ac
                    next_logic.save()
                # updateNextQuestionLogic.activity = updateActivity
                # updateNextQuestionLogic.next_activity = next_activity_ac
                # updateNextQuestionLogic.save()
            else:
                newNextQuestionLogic = NextQuestionLogic(activity=updateActivity, next_activity=next_activity_ac)
                newNextQuestionLogic.save()
    else:
        # If no next activity is selected, and a NextQuestionLogic instance exists, delete it
        # NextQuestionLogic.objects.filter(activity=updateActivity).delete()
        NextQuestionLogic.objects.filter(activity=updateActivity, answer__isnull=True).delete()
    # return HttpResponseRedirect(reverse('phase', args=[scenario_id, phase_id]))
    print(f'FORMER: ', formerActivity.activity_type.name)
    print(f'NEXT: ', updateActivity.activity_type.name)
    print(f'ACTIVITY ID: ', updateActivity.id)
    if formerActivity.is_evaluatable != updateActivity.is_evaluatable:
        if formerActivity.is_evaluatable:
            print(f'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEformer: ', formerActivity.is_evaluatable)
            print(f'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEElatter: ', updateActivity.is_evaluatable)
            QuestionBunch.objects.get(activity_primary = formerActivity).delete()
            if EvQuestionBranching.objects.filter(activity=formerActivity).exists():
                EvQuestionBranching.objects.get(activity = formerActivity).delete()
    if formerActivity.activity_type.name != updateActivity.activity_type.name:
        if updateActivity.activity_type.name == 'Question':
            NextQuestionLogic.objects.filter(activity=updateActivity).delete()
            return HttpResponseRedirect(reverse('updateAnswers', args=[scenario_id, phase_id, updateActivity.id]))
        elif formerActivity.activity_type.name == 'Question':
            Answer.objects.filter(activity=updateActivity).delete()
            if not next_activity_id:
                next_activity_instance = None
            else:
                next_activity_instance = get_object_or_404(Activity, id=next_activity_id)
            newNextQuestionLogic = NextQuestionLogic(activity=updateActivity, next_activity=next_activity_instance)
            newNextQuestionLogic.save()
            return HttpResponseRedirect(reverse('phase', args=[scenario_id, phase_id]))
        elif not formerActivity.is_evaluatable and updateActivity.is_evaluatable:
            print(f'I WENT HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE1')
            NextQuestionLogic.objects.filter(activity=updateActivity).delete()
            return HttpResponseRedirect(reverse('createCriterion', args=[scenario_id, phase_id, updateActivity.id]))
        else:
            return HttpResponseRedirect(reverse('phase', args=[scenario_id, phase_id]))
    elif not formerActivity.is_evaluatable and updateActivity.is_evaluatable:
        print(f'I WENT HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
        print(updateActivity.id)
        NextQuestionLogic.objects.filter(activity=updateActivity).delete()
        return HttpResponseRedirect(reverse('updateCriterion', args=[scenario_id, phase_id, updateActivity.id]))
    else:
        return HttpResponseRedirect(reverse('phase', args=[scenario_id, phase_id]))

@group_required('teachers')
def deleteActivity(request, scenario_id, phase_id, activity_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    if scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    deleteActivity = get_object_or_404(Activity, id=activity_id)
    deleteActivity.delete()
    return HttpResponseRedirect(reverse('phase', args=[scenario_id, phase_id]))

def viewActivity(request, scenario_id, phase_id, activity_id):
    myActivity = Activity.objects.get(id=activity_id)
    next_activity_logic = NextQuestionLogic.objects.filter(activity=activity_id).first()
    myScenario = Scenario.objects.get(id=scenario_id)
    # Check if the user is the creator, an admin, or if the scenario is editable by the org and the user belongs to the org
    can_edit = False
    if is_admin_user(request.user) or myScenario.created_by == request.user:
        can_edit = True
    elif myScenario.visibility_status == 'org' and myScenario.is_editable_by_org:
        if myScenario.organizations.filter(members=request.user).exists():
            can_edit = True
    myPhase = Phase.objects.get(id=phase_id)
    
    # Initialize experiment instances as None
    existingSim = None
    existingRemote = None
    existingVR_AR = None
    
    # Check which experiment type is set and get the corresponding instance
    if myActivity.simulation:
        existingSim = Simulation.objects.get(id=myActivity.simulation.id)
    elif myActivity.experiment_ll:
        existingRemote = ExperimentLL.objects.get(id=myActivity.experiment_ll.id)
    elif myActivity.vr_ar_experiment:
        existingVR_AR = VRARExperiment.objects.get(id=myActivity.vr_ar_experiment.id)

    activityQuestionBunch = QuestionBunch.objects.filter(activity_primary=myActivity.id).first()
    if myActivity.is_evaluatable:
        if activityQuestionBunch is None:
            return HttpResponseRedirect(reverse('updateActivity', args=[scenario_id, phase_id, activity_id]))
        activities = Activity.objects.filter(id__in=activityQuestionBunch.activity_ids)
        myEVQB = EvQuestionBranching.objects.filter(activity=myActivity).first()
        print(f'EVQB:',myEVQB)
        if myEVQB:
            high_activity = get_object_or_404(Activity, id=myEVQB.next_question_on_high.id) if myEVQB.next_question_on_high else None
            mid_activity = get_object_or_404(Activity, id=myEVQB.next_question_on_mid.id) if myEVQB.next_question_on_mid else None
            low_activity = get_object_or_404(Activity, id=myEVQB.next_question_on_low.id) if myEVQB.next_question_on_low else None
            print(f'HIGH:',high_activity, 'MID:', mid_activity, 'LOW:', low_activity)
        else:
            high_activity = mid_activity = low_activity = None
    else:
        activities = myEVQB = high_activity = mid_activity = low_activity = None

    if next_activity_logic:
        next_activity_data = next_activity_logic.next_activity
    else:
        next_activity_data = None
    answers = Answer.objects.filter(activity=activity_id).order_by('created_on')
    for answer in answers:
        next_question_logic = NextQuestionLogic.objects.filter(answer=answer).first()
        if next_question_logic:
            answer.next_activity = next_question_logic.next_activity
        else:
            answer.next_activity = None

    template = loader.get_template('authoringtool/viewActivity.html')
    context = {
        'activity': myActivity,
        'answers': answers,
        'nextActivity': next_activity_data,
        'activities': activities,
        'high_activity': high_activity,
        'mid_activity': mid_activity,
        'low_activity': low_activity,
        'myPhase': myPhase,
        'myScenario': myScenario,
        'existingSim': existingSim,
        'existingRemote': existingRemote,
        'existingVR_AR': existingVR_AR,
        'can_edit': can_edit,
        "llm_html": mark_safe(markdown.markdown(myActivity.llm_context or "")),
    }
    return HttpResponse(template.render(context, request))

@group_required('teachers')
def createAnswers(request, scenario_id, phase_id, activity_id):
    if request.method == 'POST':
        activity = Activity.objects.get(id=activity_id)

        # Initialize an empty list to hold the parsed answers
        parsed_answers = []

        # Iterate through the POST data
        for key in request.POST.keys():
            # Look for keys that represent an answer text
            if 'text]' in key and key.startswith('answers['):
                # Extract the index from the key
                index = key.split('[')[1].split(']')[0]

                # Build the keys for the other properties based on the index
                text_key = f'answers[{index}][text]'
                correct_key = f'answers[{index}][is_correct]'
                weight_key = f'answers[{index}][answer_weight]'
                # For image and vid_url, we'll need to check in request.FILES and request.POST respectively
                image_key = f'answers[{index}][image]'
                vid_url_key = f'answers[{index}][vid_url]'

                # Extract the values using the built keys
                text = request.POST.get(text_key, '').strip()
                is_correct = correct_key in request.POST
                answer_weight = max(1, min(3, int(request.POST.get(weight_key, 1) or 1)))
                image = request.FILES.get(image_key)
                vid_url = request.POST.get(vid_url_key, '').strip()
                created_by = request.user

                # Append this answer's data as a dict to the parsed_answers list
                parsed_answers.append({
                    'text': text,
                    'is_correct': is_correct,
                    'answer_weight': answer_weight,
                    'image': image,
                    'vid_url': vid_url,
                    'created_by': created_by,
                })

        # Now, parsed_answers contains all the answers submitted
        print(f"Number of answers parsed: {len(parsed_answers)}")

        # Process each parsed answer
        for answer_data in parsed_answers:
            if answer_data['text']:  # Ensure there's text before saving
                answer = Answer(
                    activity=activity,
                    text=answer_data['text'],
                    is_correct=answer_data['is_correct'],
                    answer_weight=answer_data['answer_weight'],
                    image=answer_data['image'],
                    vid_url=answer_data['vid_url'],
                    created_by=answer_data['created_by']
                )
                answer.save()

        return HttpResponseRedirect(reverse('activity', args=[scenario_id, phase_id, activity_id]))

    return render(request, 'authoringtool/createAnswer.html', {'activity_id': activity_id, 'scenario_id': scenario_id, 'phase_id': phase_id})

@group_required('teachers')
def deleteAnswer(request, scenario_id, phase_id, activity_id, answer_id):
    deleteAnswer = Answer.objects.get(id=answer_id)
    deleteAnswer.delete()
    return HttpResponseRedirect(reverse('activity', args=[scenario_id, phase_id, activity_id]))

@group_required('teachers')
def updateAnswers(request, scenario_id, phase_id, activity_id):
    activity = get_object_or_404(Activity, id=activity_id)
    all_activities_in_scenario = Activity.objects.filter(scenario=scenario_id) # ME
    # current_next_activity = NextQuestionLogic.objects.filter(activity=activity).values_list('next_activity', flat=True) # ME
    # ineligible_activity_ids = NextQuestionLogic.objects.exclude(next_activity__in=current_next_activity).values_list('next_activity', flat=True) # ME
    # eligible_activities = all_activities_in_scenario.exclude(id__in=ineligible_activity_ids) # ME

    if request.method == 'POST':
        existing_answer_ids = [answer.id for answer in activity.answers.all()]
        submitted_answer_ids = []

        for key in request.POST.keys():
            if 'text]' in key and key.startswith('answers['):
                index = key.split('[')[1].split(']')[0]
                answer_id = request.POST.get(f'answers[{index}][id]', None)
                next_activity_id = request.POST.get(f'next_activity_{index}', '')
                
                answer = Answer(activity=activity) if not answer_id else Answer.objects.get(id=answer_id)
                submitted_answer_ids.append(answer.id) if answer_id else None

                text_key = f'answers[{index}][text]'
                correct_key = f'answers[{index}][is_correct]'
                weight_key = f'answers[{index}][answer_weight]'
                clear_image_key = f'clear_image_{index}'
                vid_url_key = f'answers[{index}][vid_url]'

                text = request.POST.get(text_key, '').strip()
                is_correct = correct_key in request.POST
                answer_weight = max(1, min(3, int(request.POST.get(weight_key, 1) or 1)))
                clear_image = request.POST.get(clear_image_key, '') == 'on'
                vid_url = request.POST.get(vid_url_key, '').strip()
                updated_by = request.user

                if clear_image:
                    answer.image.delete()
                else:
                    image_file = request.FILES.get(f'answers[{index}][image]', None)
                    if image_file:
                        answer.image = image_file

                answer.text = text
                answer.is_correct = is_correct
                answer.answer_weight = answer_weight
                answer.vid_url = vid_url
                answer.updated_by = updated_by
                answer.save()

                if next_activity_id:
                    if next_activity_id == 'create_new':
                        NextQuestionLogic.objects.filter(activity=activity, answer=answer).delete()
                        current_activity = get_object_or_404(Activity, id=activity_id)
                        current_type = get_object_or_404(ActivityType, name='Question')
                        new_activity = Activity.objects.create(
                            name=f"Next activity of {current_activity.name} by answer {index}",
                            scenario=current_activity.scenario,
                            phase=current_activity.phase,
                            text=f"Activity created by {current_activity.name}",
                            plain_text=f"Activity created by {current_activity.name}",
                            activity_type=current_type,
                            created_by = request.user
                        )
                        NextQuestionLogic.objects.create(
                        activity=current_activity,
                        answer = answer,
                        next_activity=new_activity,
                        )
                    else:
                        NextQuestionLogic.objects.filter(activity=activity, answer=answer).delete()
                        next_activity = get_object_or_404(Activity, id=next_activity_id)
                        NextQuestionLogic.objects.update_or_create(
                            activity=activity,
                            answer=answer,
                            defaults={'next_activity': next_activity}
                        )
                else:
                    # If no next activity is selected, remove existing NextQuestionLogic for this answer
                    NextQuestionLogic.objects.filter(activity=activity, answer=answer).delete()

        # Delete any existing answers not included in the submission
        for answer_id in existing_answer_ids:
            if answer_id not in submitted_answer_ids:
                Answer.objects.get(id=answer_id).delete()

        messages.success(request, 'Answers updated successfully!')
        return redirect(reverse('activity', args=[scenario_id, phase_id, activity_id]))

    answers = activity.answers.all()

    for answer in answers:
        next_question_logic = NextQuestionLogic.objects.filter(answer=answer).first()
        answer.next_activity_id = next_question_logic.next_activity.id if next_question_logic else None

    return render(request, 'authoringtool/updateAnswers.html', {
        'activity': activity,
        'answers': answers,
        'scenario': scenario_id,
        'phase': phase_id,
        'eligible_activities': all_activities_in_scenario,
    })

@group_required('teachers')
def createCriterion(request, scenario_id, phase_id, activity_id):
    activityPrimary = Activity.objects.get(id=activity_id)
    activityQuestionBunch = QuestionBunch.objects.filter(activity_primary=activity_id).first()
    myScenario = Scenario.objects.get(id=scenario_id)
    myPhase = Phase.objects.get(id=phase_id)

    eligible_activities = Activity.objects.select_related('phase').filter(scenario=scenario_id).exclude(id=activity_id).order_by('phase__created_on', 'created_on')
    
    context = {
        'myScenario': myScenario,
        'myPhase': myPhase, 
        'eligible_activities': eligible_activities,
        'myActivity': activityPrimary
    }
    # template = loader.get_template('authoringtool/createPhase.html')
    return render(request, 'authoringtool/createCriterion.html', context)

@group_required('teachers')
def createCriterionData(request, scenario_id, phase_id, activity_id):
    activityPrimary = Activity.objects.get(id=activity_id)
    if request.POST:
        high_performers_activity_id = request.POST.get('highPerformersSelect')
        mid_performers_activity_id = request.POST.get('midPerformersSelect')
        low_performers_activity_id = request.POST.get('lowPerformersSelect')

        if high_performers_activity_id:
            print(f'WTF DO I PRINT: ', high_performers_activity_id)
            high_performers_activity = get_object_or_404(Activity, id=high_performers_activity_id)
        else:
            high_performers_activity = None

        if mid_performers_activity_id:
            mid_performers_activity = get_object_or_404(Activity, id=mid_performers_activity_id)
        else:
            mid_performers_activity = None

        if low_performers_activity_id:
            low_performers_activity = get_object_or_404(Activity, id=low_performers_activity_id)
        else:
            low_performers_activity = None

        high_performers_score_limit = request.POST.get('high_score_limit')
        print(high_performers_score_limit)
        mid_performers_score_limit = request.POST.get('mid_score_limit')
        print(mid_performers_score_limit)
        low_performers_score_limit = request.POST.get('low_score_limit')
        print(low_performers_score_limit)

        newEvQuestionBranching = EvQuestionBranching(
            activity = activityPrimary,
            next_question_on_high = high_performers_activity,
            next_question_on_mid = mid_performers_activity,
            next_question_on_low = low_performers_activity
        )
        newEvQuestionBranching.save()

        if high_performers_activity_id:
            high_performers_activity = get_object_or_404(Activity, id=high_performers_activity_id)
            high_performers_activity.score_limit = high_performers_score_limit
            high_performers_activity.save()
        if mid_performers_activity_id:
            mid_performers_activity = get_object_or_404(Activity, id=mid_performers_activity_id)
            mid_performers_activity.score_limit = mid_performers_score_limit
            mid_performers_activity.save()
        if low_performers_activity_id:
            low_performers_activity = get_object_or_404(Activity, id=low_performers_activity_id)
            low_performers_activity.score_limit = low_performers_score_limit
            low_performers_activity.save()

        return HttpResponseRedirect(reverse('activity', args=[scenario_id, phase_id, activity_id]))

@group_required('teachers')
def updateCriterion(request, scenario_id, phase_id, activity_id):
    activityPrimary = Activity.objects.get(id=activity_id)
    myScenario = Scenario.objects.get(id=scenario_id)
    myPhase = Phase.objects.get(id=phase_id)
    activityQuestionBunch = QuestionBunch.objects.filter(activity_primary=activityPrimary.id).first()
    eligible_activities = Activity.objects.select_related('phase').filter(scenario=scenario_id).exclude(id=activity_id).order_by('phase__created_on', 'created_on')
    context_creation = {
        'myScenario': myScenario,
        'myPhase': myPhase, 
        'eligible_activities': eligible_activities,
        'myActivity': activityPrimary
    }
    if not EvQuestionBranching.objects.filter(activity=activityPrimary.id).exists():
        return render(request, 'authoringtool/createCriterion.html', context_creation)
    activityEvQuestionBranching = EvQuestionBranching.objects.get(activity=activityPrimary.id)

    high_id = activityEvQuestionBranching.next_question_on_high
    high_performers_activity = get_object_or_404(Activity, id=high_id.id) if high_id else None
    mid_id = activityEvQuestionBranching.next_question_on_mid
    mid_performers_activity = get_object_or_404(Activity, id=mid_id.id) if mid_id else None
    low_id = activityEvQuestionBranching.next_question_on_low
    low_performers_activity = get_object_or_404(Activity, id=low_id.id) if low_id else None
    
    context = {
        'myScenario': myScenario,
        'myPhase': myPhase, 
        'eligible_activities': eligible_activities,
        'myActivity': activityPrimary,
        'high_activity': high_performers_activity,
        'mid_activity': mid_performers_activity,
        'low_activity': low_performers_activity
    }
    # template = loader.get_template('authoringtool/createPhase.html')
    return render(request, 'authoringtool/updateCriterion.html', context)

@group_required('teachers')
def updateCriterionData(request, scenario_id, phase_id, activity_id):
    activityPrimary = Activity.objects.get(id=activity_id)
    if request.POST:
        high_performers_activity_id = request.POST.get('highPerformersSelect')
        mid_performers_activity_id = request.POST.get('midPerformersSelect')
        low_performers_activity_id = request.POST.get('lowPerformersSelect')

        if high_performers_activity_id:
            high_performers_activity = get_object_or_404(Activity, id=high_performers_activity_id)
        else:
            high_performers_activity = None

        if mid_performers_activity_id:
            mid_performers_activity = get_object_or_404(Activity, id=mid_performers_activity_id)
        else:
            mid_performers_activity = None

        if low_performers_activity_id:
            low_performers_activity = get_object_or_404(Activity, id=low_performers_activity_id)
        else:
            low_performers_activity = None

        high_performers_score_limit = request.POST.get('high_score_limit')
        print(high_performers_score_limit)
        mid_performers_score_limit = request.POST.get('mid_score_limit')
        print(mid_performers_score_limit)
        low_performers_score_limit = request.POST.get('low_score_limit')
        print(low_performers_score_limit)

        updateEvQuestionBranching = EvQuestionBranching.objects.get(activity=activityPrimary)
        updateEvQuestionBranching.next_question_on_high = high_performers_activity
        updateEvQuestionBranching.next_question_on_mid = mid_performers_activity
        updateEvQuestionBranching.next_question_on_low = low_performers_activity
        updateEvQuestionBranching.save()

        high_performers_activity = get_object_or_404(Activity, id=high_performers_activity_id)
        high_performers_activity.score_limit = high_performers_score_limit
        high_performers_activity.save()
        mid_performers_activity = get_object_or_404(Activity, id=mid_performers_activity_id)
        mid_performers_activity.score_limit = mid_performers_score_limit
        mid_performers_activity.save()
        low_performers_activity = get_object_or_404(Activity, id=low_performers_activity_id)
        low_performers_activity.score_limit = low_performers_score_limit
        low_performers_activity.save()

        return HttpResponseRedirect(reverse('activity', args=[scenario_id, phase_id, activity_id]))

def sankey_data(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')

    # Trigger Celery task
    result = compute_sankey_data.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_sankey_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def check_for_duplicate_answers(scenario_id):
    # Get all answers for the scenario
    duplicate_answers = UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id) \
        .values('user_id', 'activity_id') \
        .annotate(answer_count=Count('id')) \
        .filter(answer_count__gt=1)
    
    # If there are duplicate answers, print or log them
    if duplicate_answers.exists():
        for duplicate in duplicate_answers:
            print(f"User {duplicate['user_id']} has {duplicate['answer_count']} answers for activity {duplicate['activity_id']}")
    else:
        print("No duplicate answers found.")

# Helper function to get the last answer for each user/activity
def get_last_answers(scenario_id):
    # Fetch the last answers for each user and activity based on the created_on timestamp
    last_answers = UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id) \
        .values('user_id', 'activity_id') \
        .annotate(last_answer_id=Max('id'))  # Get the last answer ID for each user and activity
    
    # Use the last answer IDs to retrieve the corresponding UserAnswer objects
    last_answer_objects = UserAnswer.objects.filter(id__in=[entry['last_answer_id'] for entry in last_answers])
    
    # Count distinct users for each activity
    user_counts = last_answer_objects.values('activity_id').annotate(user_count=Count('user_id', distinct=True))

    # Convert user_counts to a dictionary for easier lookup
    user_count_dict = {entry['activity_id']: entry['user_count'] for entry in user_counts}

    # print('THE USER COUNT IS: ', user_count_dict)

    # Use the last answer IDs to retrieve the corresponding UserAnswer objects
    return UserAnswer.objects.filter(id__in=[entry['last_answer_id'] for entry in last_answers])

# Helper function to get the last answers for each user/activity
def get_last_answers_only_for_users(scenario_id, start_date=None, end_date=None):
    # Fetch the last answers for each user and activity based on the created_on timestamp
    last_answers = UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id)
    
    # Apply date filters if provided
    if start_date:
        last_answers = last_answers.filter(created_on__gte=start_date)
    if end_date:
        last_answers = last_answers.filter(created_on__lte=end_date + timedelta(days=1))
    
    # Get the latest answers by selecting the max created_on timestamp for each user-activity combination
    last_answers = last_answers.values('user_id', 'activity_id') \
        .annotate(last_answer_id=Max('id'))  # Get the last answer ID for each user and activity

    # Use the last answer IDs to retrieve the corresponding UserAnswer objects
    return UserAnswer.objects.filter(id__in=[entry['last_answer_id'] for entry in last_answers])

def final_performance(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_final_performance.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_final_performance_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def activity_answers_data(request, scenario_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    data_type = request.GET.get('type', 'activities')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_activity_answers_data.delay(scenario_id, group_ids, start_date, end_date, data_type)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_activity_answers_data_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def performance_data(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_performance_data.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_performance_data_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def time_spent_data(request, scenario_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    data_type = request.GET.get('type', 'activities')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_time_spent_data.delay(scenario_id, group_ids, start_date, end_date, data_type)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_time_spent_data_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def detailed_phase_scores_data(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_detailed_phase_scores_data.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_detailed_phase_scores_data_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def performers_data(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_performers_data.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_performers_data_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def time_spent_by_performer_type(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_time_spent_by_performer_type.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_time_spent_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

# TODO: Move performance_by_department computation to a Celery task to avoid blocking the web worker.
@group_required('teachers')
def performance_by_department(request, scenario_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    # Create a unique cache key based on scenario_id
    cache_key = f'performance_by_department_{scenario_id}_{start_date}_{end_date}'
    
    # Try to get the data from the cache
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return JsonResponse(cached_data)
    
    phases = Phase.objects.filter(scenario=scenario)
    departments = SchoolDepartment.objects.all()
    data = {'departments': [], 'phases': [], 'performance': {}}

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)

    # Get the minimum activity ID in the scenario
    min_activity_id = Activity.objects.filter(scenario=scenario).order_by('id').first()

    if not min_activity_id:
        return JsonResponse({"error": "No activities found for this scenario"}, status=400)

    # Apply start_date and end_date filters to last answers
    if start_date:
        start_date = parse_date(str(start_date))
        last_answers = last_answers.filter(created_on__gte=start_date)
    if end_date:
        end_date = parse_date(str(end_date))
        last_answers = last_answers.filter(created_on__lte=end_date + timedelta(days=1))

    for department in departments:
        department_users = User.objects.filter(school_department=department, userscenarioscore__scenario=scenario).distinct()

        if not department_users.exists():
            continue  # Skip this department if no users in the department

        # Filter valid users who started with the minimum activity
        valid_users = []  # List of users who started with the minimum activity

        for user in department_users:
            # Check if user has answered the minimum activity
            if last_answers.filter(user=user, activity=min_activity_id).exists():
                valid_users.append(user)

        data['departments'].append(department.name)
        data['performance'][department.name] = {'High': [], 'Mid': [], 'Low': []}

        for phase in phases:
            high_performers = 0
            mid_performers = 0
            low_performers = 0

            activities = Activity.objects.filter(phase=phase)

            for user in valid_users:
                total_primary_score = 0
                total_primary_max_score = 0

                # Track processed activities to avoid duplicates
                processed_activities = set()

                # Process only primary evaluatable activities via QuestionBunch
                primary_evaluatable_activities = activities.filter(is_evaluatable=True, is_primary_ev=True)
                if primary_evaluatable_activities.exists():
                    primary_count = primary_evaluatable_activities.count()
                    primary_weight_share = 100 / primary_count  # Equal distribution for each primary evaluatable activity

                    for primary_activity in primary_evaluatable_activities:
                        if primary_activity.id in processed_activities:
                            continue  # Skip already processed activity
                        try:
                            question_bunch = QuestionBunch.objects.get(activity_primary=primary_activity)
                            bunch_activities = Activity.objects.filter(id__in=question_bunch.activity_ids)
                        except QuestionBunch.DoesNotExist:
                            bunch_activities = [primary_activity]

                        for bunch_activity in bunch_activities:
                            if bunch_activity.id in processed_activities:
                                continue  # Skip already processed bunch activity
                            processed_activities.add(bunch_activity.id)  # Mark activity as processed

                            user_last_answer = last_answers.filter(user=user, activity=bunch_activity).first()
                            if user_last_answer and user_last_answer.answer:
                                total_primary_score += (user_last_answer.answer.answer_weight * primary_weight_share) / 100

                            # Calculate max score for each bunch activity
                            highest_answer_weight = Answer.objects.filter(activity=bunch_activity).order_by('-answer_weight').first()
                            if highest_answer_weight:
                                total_primary_max_score += (highest_answer_weight.answer_weight * primary_weight_share) / 100

                # Calculate performance for this user in this phase based on primary activities only
                if total_primary_max_score > 0:
                    percentage_score = (total_primary_score / total_primary_max_score) * 100

                    if percentage_score >= 83.3:
                        high_performers += 1
                    elif percentage_score >= 49.7:
                        mid_performers += 1
                    else:
                        low_performers += 1

            # Calculate and store performance percentages for the department in this phase
            total_users = high_performers + mid_performers + low_performers
            if total_users > 0:
                data['performance'][department.name]['High'].append((high_performers / total_users) * 100)
                data['performance'][department.name]['Mid'].append((mid_performers / total_users) * 100)
                data['performance'][department.name]['Low'].append((low_performers / total_users) * 100)
            else:
                data['performance'][department.name]['High'].append(0)
                data['performance'][department.name]['Mid'].append(0)
                data['performance'][department.name]['Low'].append(0)

    # Collect phase names
    data['phases'] = [phase.name for phase in phases]

    # Store the data in the cache with a 6-month timeout
    cache.set(cache_key, data, timeout=6 * 30 * 24 * 60 * 60)

    return JsonResponse(data)

@group_required('teachers')
def duplicate_scenario(request, scenario_id):
    try:
        original_scenario = get_object_or_404(Scenario, pk=scenario_id)
        user = request.user
        new_scenario_name_base = f"{original_scenario.name} Copy Made by {user.username}"
        new_scenario_name = new_scenario_name_base
        
        # Check if scenario with the new name already exists and modify the name if needed
        counter = 1
        while Scenario.objects.filter(name=new_scenario_name).exists():
            new_scenario_name = f"{new_scenario_name_base} {counter}"
            counter += 1
        
        new_scenario = Scenario.objects.create(
            name=new_scenario_name,
            learning_goals=original_scenario.learning_goals,
            description=original_scenario.description,
            age_of_students=original_scenario.age_of_students,
            subject_domains=original_scenario.subject_domains,
            language=original_scenario.language,
            suggested_learning_time=original_scenario.suggested_learning_time,
            image=original_scenario.image,
            created_by=user,
            updated_by=user,
            origin_scenario=original_scenario,
        )

        # Map for original activities to their duplicates
        activity_mapping = {}
        answer_mapping = {}

        # Duplicate Phases and Activities
        for phase in original_scenario.phases.all():
            new_phase = Phase.objects.create(
                name=phase.name,
                description=phase.description,
                image=phase.image,
                scenario=new_scenario,
                created_by=user,
                updated_by=user
            )

            for activity in phase.activities.all():
                new_activity = Activity.objects.create(
                    name=activity.name,
                    text=activity.text,
                    plain_text=activity.plain_text,
                    correct_count=activity.correct_count,
                    incorrect_count=activity.incorrect_count,
                    is_evaluatable=activity.is_evaluatable,
                    is_primary_ev=activity.is_primary_ev,
                    must_wait=activity.must_wait,
                    score_limit=activity.score_limit,
                    scenario=new_scenario,
                    phase=new_phase,
                    activity_type=activity.activity_type,
                    helper=activity.helper,
                    simulation=activity.simulation,
                    experiment_ll=activity.experiment_ll,
                    vr_ar_experiment = activity.vr_ar_experiment,
                    created_by=user,
                    updated_by=user
                )

                # Copy answers
                for answer in activity.answers.all():
                    new_answer = Answer.objects.create(
                        activity=new_activity,
                        text=answer.text,
                        is_correct=answer.is_correct,
                        answer_weight=answer.answer_weight,
                        image=answer.image,
                        vid_url=answer.vid_url,
                        created_by=user,
                        updated_by=user
                    )
                    answer_mapping[answer.id] = new_answer

                # Add activity to mapping
                activity_mapping[activity.id] = new_activity

        # Duplicate Next Question Logic
        for original_activity_id, new_activity in activity_mapping.items():
            original_activity = Activity.objects.get(pk=original_activity_id)
            for logic in original_activity.next_logic.all():
                NextQuestionLogic.objects.create(
                    activity=new_activity,
                    answer=answer_mapping.get(logic.answer.id, None) if logic.answer else None,
                    next_activity=activity_mapping.get(logic.next_activity.id, None) if logic.next_activity else None
                )

        # Duplicate EvQuestionBranching and QuestionBunch for Evaluatable Activities
        for original_activity_id, new_activity in activity_mapping.items():
            original_activity = Activity.objects.get(pk=original_activity_id)
            if original_activity.is_evaluatable:
                if hasattr(original_activity, 'branching'):
                    branching = original_activity.branching
                    EvQuestionBranching.objects.create(
                        activity=new_activity,
                        next_question_on_high=activity_mapping.get(branching.next_question_on_high.id, None) if branching.next_question_on_high else None,
                        next_question_on_high_feedback=branching.next_question_on_high_feedback,
                        next_question_on_mid=activity_mapping.get(branching.next_question_on_mid.id, None) if branching.next_question_on_mid else None,
                        next_question_on_mid_feedback=branching.next_question_on_mid_feedback,
                        next_question_on_low=activity_mapping.get(branching.next_question_on_low.id, None) if branching.next_question_on_low else None,
                        next_question_on_low_feedback=branching.next_question_on_low_feedback,
                    )

                # Duplicate QuestionBunch
                question_bunch = QuestionBunch.objects.filter(activity_primary=original_activity).first()
                if question_bunch:
                    new_bunch = QuestionBunch.objects.create(
                        activity_primary=new_activity,
                        activity_ids=[activity_mapping[aid].id for aid in question_bunch.activity_ids]
                    )

        messages.success(request, f'Successfully duplicated scenario: {new_scenario_name}')
        return redirect('updateScenario', id=new_scenario.id)

    except Exception as e:
        messages.error(request, f'Failed to duplicate scenario: {str(e)}')
        return redirect('scenarios')  # Redirect to an appropriate error view or page
    
# LTI Operations
LTI_CONSUMER_KEY = os.environ.get('LTI_CONSUMER_KEY', 'dspace')
LTI_SHARED_SECRET = os.environ.get('LTI_SHARED_SECRET', 'FUQYeguEf7WoIJ-f-_U_Eg')

def double_encode(value):
    """Encodes a value twice for OAuth."""
    return urllib.parse.quote_plus(urllib.parse.quote_plus(value))

def single_encode(value):
    """Encodes a value once for OAuth."""
    return urllib.parse.quote_plus(str(value))

def generate_oauth_signature(secret, params, url, method='POST'):
    """Generates the OAuth signature for the LTI request and prints the URL-encoded base string."""
    
    # Percent-encode the URL
    encoded_url = single_encode(url)

    # Sort the parameters alphabetically and manually encode them
    encoded_params = []
    
    for k, v in params.items():
        if k in ['oauth_nonce', 'roles']:  # Double-encode only these fields
            encoded_value = single_encode(v)
        else:  # Single-encode for all other fields
            encoded_value = single_encode(v)
        encoded_params.append(f"{single_encode(k)}={encoded_value}")

    # Sort the list of encoded key-value pairs
    encoded_params.sort()

    # Concatenate the sorted parameters into the base string format
    encoded_params_str = '&'.join(encoded_params)

    # Prepare the signature base string
    base_string = '&'.join([
        method.upper(),
        encoded_url,  # Percent-encode the URL
        single_encode(encoded_params_str)  # Percent-encode the sorted parameters as a single string
    ])

    print(f"Base String (Correctly Encoded): {base_string}")
    
    # Create the signing key (shared secret and an empty token secret)
    signing_key = f"{secret}&" # f"{LTI_SHARED_SECRET}&"

    # Calculate the HMAC-SHA1 signature
    hashed = hmac.new(signing_key.encode(), base_string.encode(), sha1)
    oauth_signature = base64.b64encode(hashed.digest()).decode()

    return oauth_signature

def lms_lti_launch(request, experiment_id): #=None
    """Simulates an LMS sending an LTI launch request to an external tool."""

    # Fetch the correct experiment by its ID
    experiment = get_object_or_404(ExperimentLL, id=experiment_id)
    
    # Define the URL of the LTI tool (LabsLand in this case)
    launch_url = experiment.launch_url#'https://labsland.com/lti/v2/fgguyzvdkk92273929/pendulum/'
    consumer_key = experiment.consumer_key
    shared_secret = experiment.shared_secret

    # Generate the current timestamp
    timestamp = str(int(time.time()))  # Current Unix timestamp in seconds

    # Getting user id & username from session
    user_id_s = request.user
    username_s = request.user.username
    email_s = request.user.email

    # Encode the timestamp in Base64 for the nonce
    base64_nonce = base64.b64encode(timestamp.encode('utf-8')).decode('utf-8')

    # Generate the LTI and OAuth parameters for the LTI launch
    params = {
        'lti_message_type': 'basic-lti-launch-request',
        'lti_version': 'LTI-1p0',
        'user_id': user_id_s,
        'ext_user_username': username_s,
        'roles': 'urn:lti:role:ims/lis/Learner',  # Double-encoded
        'email': email_s,  # Single-encoded
        'context_id': 'dspace',  # Single-encoded
        'resource_link_id': 'pendulum',  # Single-encoded
        'launch_presentation_locale': 'en',  # Single-encoded
        'oauth_consumer_key': consumer_key, # LTI_CONSUMER_KEY,  # Single-encoded
        'oauth_nonce': base64_nonce,  # Double-encoded
        'oauth_signature_method': 'HMAC-SHA1',  # Single-encoded
        'oauth_timestamp': timestamp,  # Single-encoded
        'oauth_version': '1.0',  # Single-encoded
    }

    # Generate the OAuth signature
    oauth_signature = generate_oauth_signature(shared_secret, params, launch_url)
    params['oauth_signature'] = oauth_signature

    # Print out the generated OAuth signature and Base64-encoded nonce for debugging
    print(f"OAuth Signature: {oauth_signature}")
    print(f"Nonce (Base64 Encoded Timestamp): {base64_nonce}")

    # Render a form that will auto-submit the POST request to the LTI tool
    return render(request, 'lti_integration.html', {'launch_url': launch_url, 'params': params})

def lab_sessions_data(request, scenario_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    
    print("SCENARIO :", scenario, scenario.id)
    # Optional date filtering from request parameters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # Parse dates and handle potential errors
    start_date = parse_date(str(start_date)) if str(start_date) else None
    end_date = parse_date(str(end_date)) if str(end_date) else None
    
    # Create a unique cache key using scenario and date range
    cache_key = f'lab_sessions_data_{scenario_id}_{start_date}_{end_date}'
    cached_data = cache.get(cache_key)
    # if cached_data:
    #    return JsonResponse(cached_data)
    
    # Filter users based on scenario and department/group conditions
    users = User.objects.filter(
        Q(userscenarioscore__scenario=scenario) &
        (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
    ).distinct()
    users = users.exclude(groups__name='teachers')

    # Define the lab sessions and filter by date if provided
    lab_sessions = RemoteLabSession.objects.filter(scenario=scenario, user__in=users)
    
    # Apply start_date and end_date filters
    if start_date:
        start_date = parse_date(start_date)
        lab_sessions = lab_sessions.filter(start__date__gte=start_date)
    
    if end_date:
        end_date = parse_date(end_date)
        lab_sessions = lab_sessions.filter(start__date__lte=end_date + timedelta(days=1))
    
    print("LAB: ", lab_sessions)
    lab_sessions = list(lab_sessions)

    # Initialize data structure for response
    data = {
        'user_count_by_duration': {
            '0-30s': 0,
            '30s-1m': 0,
            '1-2m': 0,
            '2-3m': 0,
            '3-5m': 0,
            '5m+': 0
        },
        'avg_exec_duration_by_mass': {},
        'session_count_by_angle': {},
        'sessions_over_time': {}
    }

    # Calculate user count by total duration
    for session in lab_sessions:
        total_duration_seconds = abs(session.pre_duration).total_seconds() + abs(session.exec_duration).total_seconds()
        
        if total_duration_seconds <= 30:
            data['user_count_by_duration']['0-30s'] += 1
        elif total_duration_seconds <= 60:
            data['user_count_by_duration']['30s-1m'] += 1
        elif total_duration_seconds <= 120:
            data['user_count_by_duration']['1-2m'] += 1
        elif total_duration_seconds <= 180:
            data['user_count_by_duration']['2-3m'] += 1
        elif total_duration_seconds <= 300:
            data['user_count_by_duration']['3-5m'] += 1
        else:
            data['user_count_by_duration']['5m+'] += 1

    # Calculate average execution duration by mass type
    mass_durations = {}
    for session in lab_sessions:
        mass = session.mass
        exec_duration = abs(session.exec_duration).total_seconds() / 60  # Convert to minutes
        if mass in mass_durations:
            mass_durations[mass].append(exec_duration)
        else:
            mass_durations[mass] = [exec_duration]
    
    for mass, durations in mass_durations.items():
        data['avg_exec_duration_by_mass'][mass] = sum(durations) / len(durations)

    # Count sessions by angle
    for session in lab_sessions:
        angle = session.angle
        data['session_count_by_angle'][angle] = data['session_count_by_angle'].get(angle, 0) + 1

    # Count sessions over time
    for session in lab_sessions:
        day = session.start.date().isoformat()
        data['sessions_over_time'][day] = data['sessions_over_time'].get(day, 0) + 1

    # Cache the data with a timeout of 6 months
    cache.set(cache_key, data, timeout=6 * 30 * 24 * 60 * 60)
    return JsonResponse(data)

def get_first_answers(scenario_id):
    # Fetch the earliest answer for each user and activity based on the created_on timestamp
    first_answers = (
        UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id)
        .values('user_id', 'activity_id')
        .annotate(first_answer_id=Min('id'))  # Get the first answer ID for each user and activity
    )

    # Use the first answer IDs to retrieve the corresponding UserAnswer objects
    return UserAnswer.objects.filter(id__in=[entry['first_answer_id'] for entry in first_answers])

def scenario_paths(request, scenario_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    
    # Parse start and end dates from request
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    # Convert group_ids to a list of integers (if it's not empty)
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []  # Ensure it's an empty list, not None

    # Trigger Celery task
    result = compute_scenario_paths.delay(scenario_id, group_ids, start_date, end_date)

    # Return task ID to the client
    return JsonResponse({'task_id': result.id})

def get_scenario_paths_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def student_performance_metrics(request, scenario_id):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    group_ids = request.GET.get('group_ids', '')
    include_activity_detail = request.GET.get('include_activity_detail', '0') == '1'
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',') if g.isdigit()]
    else:
        group_ids = []

    result = compute_student_performance_metrics.delay(
        scenario_id, group_ids, start_date, end_date, include_activity_detail
    )
    return JsonResponse({'task_id': result.id})

def get_student_performance_metrics_task_status(request, task_id):
    result = AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return JsonResponse({"status": "completed", "data": result.result})
    elif result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.info)})

def get_teacher_groups(request, scenario_id):
    groups = UserGroup.objects.filter(
        created_by=request.user,
        assigned_scenarios__id=scenario_id
    ).values('id', 'name')
    return JsonResponse({'groups': list(groups)})

def ai_metrics(request, scenario_id):
    scenario = get_object_or_404(Scenario, id=scenario_id)

    can_edit = False
    if is_admin_user(request.user) or scenario.created_by == request.user:
        can_edit = True
    elif getattr(scenario, 'visibility_status', None) == 'org' and getattr(scenario, 'is_editable_by_org', False):
        if scenario.organizations.filter(members=request.user).exists():
            can_edit = True

    phases_data = []
    for phase in Phase.objects.filter(scenario=scenario).order_by('id'):
        primaries = Activity.objects.filter(
            phase=phase,
            is_evaluatable=True,
            is_primary_ev=True
        ).values('id', 'name')
        phases_data.append({
            'id':         phase.id,
            'name':       phase.name,
            'activities': list(primaries),
        })

    # Pre-load cached CSVs so tables render immediately on page load
    from collections import defaultdict as _dd
    metrics_csv = os.path.join(settings.AI_METRICS_CACHE_ROOT, f"scenario_{scenario_id}_combined_activity_metrics.csv")
    flags_csv   = os.path.join(settings.AI_METRICS_CACHE_ROOT, f"scenario_{scenario_id}_flagged_activities_with_reasons.csv")

    metrics_grouped = []
    flags_grouped   = []

    if os.path.exists(metrics_csv):
        raw = []
        with open(metrics_csv, newline='', encoding='utf-8') as f:
            raw = list(csv.DictReader(f))
        seen = {}
        for row in raw:
            # Normalise keys that contain spaces / % signs for Django templates
            safe_row = {
                'Phase':        row.get('Phase', ''),
                'Activity':     row.get('Activity', ''),
                'Type':         row.get('Type', ''),
                'Category':     row.get('Category', ''),
                'Total':        row.get('Total', ''),
                'Correct':      row.get('Correct', ''),
                'Wrong':        row.get('Wrong', ''),
                'PctCorrect':   row.get('% Correct', ''),
                'PctWrong':     row.get('% Wrong', ''),
                'AvgTime':      row.get('Avg Time', ''),
                'NextLow':      row.get('Next Low', ''),
                'NextModerate': row.get('Next Moderate', ''),
                'NextHigh':     row.get('Next High', ''),
            }
            key = f"{safe_row['Phase']}|||{safe_row['Activity']}"
            if key not in seen:
                seen[key] = {'phase': safe_row['Phase'], 'activity': safe_row['Activity'],
                             'type': safe_row['Type'], 'rows': []}
                metrics_grouped.append(seen[key])
            seen[key]['rows'].append(safe_row)

    if os.path.exists(flags_csv):
        raw = []
        with open(flags_csv, newline='', encoding='utf-8') as f:
            raw = list(csv.DictReader(f))
        seen = {}
        for row in raw:
            act = row.get('Activity', '')
            if act not in seen:
                seen[act] = {'activity': act, 'rows': []}
                flags_grouped.append(seen[act])
            seen[act]['rows'].append(row)

    return render(request, 'authoringtool/ai_metrics_scenario.html', {
        'myScenario':      scenario,
        'PhasesData':      phases_data,
        'can_edit':        can_edit,
        'metrics_grouped': metrics_grouped,
        'flags_grouped':   flags_grouped,
        'has_metrics':     bool(metrics_grouped),
        'has_flags':       bool(flags_grouped),
    })

def category_metrics_view(request, scenario_id):
    start = request.GET.get('start_date')
    end = request.GET.get('end_date')
    group_ids = request.GET.getlist('group_ids[]', None)
    task = compute_category_metrics_per_phase_activity.delay(scenario_id, group_ids, start, end)
    return JsonResponse({'task_id': task.id})

def category_metrics_status(request, task_id):
    r = AsyncResult(task_id)
    if r.state == 'SUCCESS':
        scenario_id = r.result.get('scenario_id') if isinstance(r.result, dict) else None
        if not scenario_id:
            return JsonResponse({'status': 'error', 'error': 'Missing scenario_id from result'})

        # csv_path = os.path.join(settings.BASE_DIR, 'ai_metrics_cache', f'scenario_{scenario_id}_combined_activity_metrics.csv')
        csv_path = os.path.join(settings.AI_METRICS_CACHE_ROOT, f"scenario_{scenario_id}_combined_activity_metrics.csv")

        if not os.path.exists(csv_path):
            return JsonResponse({'status': 'error', 'error': 'CSV not found'})

        data = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return JsonResponse({'status': 'done', 'data': data})
    
    return JsonResponse({'status': r.state})

def risk_flags_view(request, scenario_id):
    task = calculate_activities_in_risk.delay(scenario_id)
    return JsonResponse({'task_id': task.id})

def risk_flags_status(request, task_id):
    r = AsyncResult(task_id)
    if r.state == 'SUCCESS':
        scenario_id = r.result.get('scenario_id') if isinstance(r.result, dict) else None
        if not scenario_id:
            return JsonResponse({'status': 'error', 'error': 'Missing scenario_id from result'})

        # Path for the risk flags CSV
        flags_csv_path = os.path.join(
            settings.AI_METRICS_CACHE_ROOT,#BASE_DIR, 'ai_metrics_cache',
            f'scenario_{scenario_id}_flagged_activities_with_reasons.csv'
        )
        if not os.path.exists(flags_csv_path):
            return JsonResponse({'status': 'error', 'error': 'Flags CSV not found'})

        data = []
        with open(flags_csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return JsonResponse({'status': 'done', 'data': data})
    
    return JsonResponse({'status': r.state})

@login_required
def trigger_llm_context_task(request, scenario_id):
    if request.method == "POST":
        try:
            scenario = Scenario.objects.get(id=scenario_id)
            force = request.GET.get("force", "false").lower() == "true"
            task = generate_llm_context_for_scenario.delay(scenario.id, force_rebuild=force)
            return JsonResponse({"status": "started", "task_id": task.id})
        except Scenario.DoesNotExist:
            return JsonResponse({"error": "Scenario not found"}, status=404)
    return JsonResponse({"error": "Invalid request"}, status=400)

@login_required
def get_llm_context_task_status(request, task_id):
    task = AsyncResult(task_id)
    return JsonResponse({"state": task.state, "result": task.result})

# @login_required
# def proposal_list_view(request, scenario_id):
#     myScenario = Scenario.objects.get(id=scenario_id)
#     proposals = ActivityProposal.objects.select_related('activity', 'phase', 'scenario', 'reviewer')\
#                                         .prefetch_related('flag', 'categories_in_risk')\
#                                         .order_by('-created_at')
#     return render(request, 'authoringtool/proposal_list.html', {
#         'proposals': proposals,
#         'myScenario' : myScenario
#     })

@login_required
def proposal_list_view(request, scenario_id):
    myScenario = get_object_or_404(Scenario, id=scenario_id)

    # 1. Fetch all shared proposals for the scenario
    proposals = ActivityProposal.objects.filter(scenario=myScenario)\
        .select_related('activity', 'phase', 'scenario')\
        .prefetch_related('flag', 'categories_in_risk')\
        .order_by('-created_at')

    # 2. For each proposal, ensure there's a UserProposalReview
    user_reviews = {
        review.proposal_id: review
        for review in UserProposalReview.objects.filter(
            user=request.user,
            proposal__in=proposals
        )
    }

    # All proposals for this scenario
    total = proposals.count()

    # User-specific decisions
    reviewed_count = sum(1 for review in user_reviews.values() if review.status in ["accepted", "rejected"])

    # Show button only if all are reviewed
    show_create_button = (total > 0 and reviewed_count == total)

    for prop in proposals:
        if prop.id not in user_reviews:
            user_reviews[prop.id] = UserProposalReview.objects.create(
                proposal=prop,
                user=request.user
            )

    accepted_count = sum(1 for r in user_reviews.values() if r.status == "accepted")
    rejected_count = sum(1 for r in user_reviews.values() if r.status == "rejected")
    pending_count  = total - accepted_count - rejected_count

    # JSONField returns a dict — serialise to a proper JSON string so the
    # template can embed it safely inside a <script type="application/json">.
    import json as _json
    for review in user_reviews.values():
        if review.teacher_edited_json:
            review.teacher_edited_json_str = _json.dumps(
                review.teacher_edited_json, ensure_ascii=False
            )
        else:
            review.teacher_edited_json_str = None

    return render(request, 'authoringtool/proposal_list.html', {
        'proposals':         proposals,
        'myScenario':        myScenario,
        'user_reviews':      user_reviews,
        'show_create_button': show_create_button,
        'total_count':       total,
        'accepted_count':    accepted_count,
        'rejected_count':    rejected_count,
        'pending_count':     pending_count,
    })

# @login_required
# def accept_proposal(request, pk, scenario_id):
#     proposal = get_object_or_404(ActivityProposal, pk=pk)
#     proposal.accept(request.user)
#     return redirect('proposal_list', scenario_id=scenario_id)

# @login_required
# def reject_proposal(request, pk, scenario_id):
#     proposal = get_object_or_404(ActivityProposal, pk=pk)
#     proposal.reject(request.user)
#     return redirect('proposal_list', scenario_id=scenario_id)

@require_POST
@group_required('teachers')
def accept_proposal(request, pk, scenario_id):
    proposal = get_object_or_404(ActivityProposal, pk=pk)
    if proposal.scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    review, _ = UserProposalReview.objects.get_or_create(
        proposal=proposal,
        user=request.user
    )
    review.accept()
    return redirect('proposal_list', scenario_id=scenario_id)

@require_POST
@group_required('teachers')
def reject_proposal(request, pk, scenario_id):
    proposal = get_object_or_404(ActivityProposal, pk=pk)
    if proposal.scenario.created_by != request.user and not is_admin_user(request.user):
        return HttpResponseForbidden("You don't own this scenario.")
    review, _ = UserProposalReview.objects.get_or_create(
        proposal=proposal,
        user=request.user
    )
    review.reject()
    return redirect('proposal_list', scenario_id=scenario_id)

@login_required
def create_personal_scenario(request, scenario_id):
    print(f"SCENARIO IS : {scenario_id}")
    apply_user_proposals_to_new_scenario.delay(scenario_id, request.user.id)
    messages.success(request, "Your personalized scenario is being created. It will appear in your scenarios shortly.")
    return redirect('proposal_list', scenario_id=scenario_id)

@group_required('teachers')
def edit_proposal_json(request, scenario_id, pk):
    proposal = get_object_or_404(ActivityProposal, pk=pk)
    user = request.user

    # Get or create the review
    review, created = UserProposalReview.objects.get_or_create(
        proposal=proposal,
        user=user,
        defaults={'status': 'new'}
    )

    # Build JSON from POST data
    data = {
        "activity_name": request.POST.get("activity_name", ""),
        "content": request.POST.get("content", ""),
        "explanation": request.POST.get("explanation", ""),
        "answers": []
    }

    for i in range(1, 20):  # allow up to 20 answers
        key = f"answer_text_{i}"
        val = request.POST.get(key)
        if val:
            data["answers"].append({"text": val.strip()})

    review.teacher_edited_json = data

    print("✅ Type of final data:", type(data))  # should be <class 'dict'>
    print("✅ Dumped JSON string:", json.dumps(data, ensure_ascii=False))

    # ✅ Keep review visible after editing
    if review.status not in ['accepted', 'rejected']:
        review.status = 'new'

    review.save()
    print("RAW saved value:", review.teacher_edited_json)
    return redirect("proposal_list", scenario_id=scenario_id)


@group_required('teachers')
def download_template(request):
    import io
    from .template_generator import generate_blank_template
    sims   = list(Simulation.objects.values_list('name', flat=True).order_by('name'))
    rlabs  = list(ExperimentLL.objects.values_list('name', flat=True).order_by('name'))
    vrlabs = list(VRARExperiment.objects.values_list('name', flat=True).order_by('name'))
    subjs  = list(Subject.objects.values_list('name', flat=True).order_by('name'))
    wb = generate_blank_template(simulations=sims, remote_labs=rlabs, vr_labs=vrlabs, subjects=subjs)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    response = HttpResponse(
        buf.read(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    response['Content-Disposition'] = 'attachment; filename="scenario_template.xlsx"'
    return response


@require_POST
@group_required('teachers')
def import_scenario(request):
    from .importer import ScenarioImporter
    uploaded = request.FILES.get('template_file')
    if not uploaded:
        return JsonResponse({'success': False, 'errors': [
            {'sheet': 'File', 'row': 0, 'column': '-', 'message': 'No file uploaded.'}
        ]})
    if not uploaded.name.endswith('.xlsx'):
        return JsonResponse({'success': False, 'errors': [
            {'sheet': 'File', 'row': 0, 'column': '-', 'message': 'Only .xlsx files are supported.'}
        ]})
    importer = ScenarioImporter(uploaded, request.user)
    scenario, errors = importer.run()
    if errors:
        return JsonResponse({'success': False, 'errors': errors})
    image_file = request.FILES.get('scenario_image')
    if image_file:
        scenario.image = image_file
        scenario.save(update_fields=['image'])
    return JsonResponse({
        'success': True,
        'scenario_id': scenario.id,
        'redirect': reverse('viewScenario', args=[scenario.id]),
    })