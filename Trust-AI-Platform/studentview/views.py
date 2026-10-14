from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist, PermissionDenied
from django.db.models import Q
from django.http import JsonResponse, HttpResponse, Http404
from django.core.paginator import Paginator
from studentview.help_labels import help_labels
from authoringtool.models import (
    Scenario,
    Activity,
    NextQuestionLogic,
    Answer,
    Simulation,
    ExperimentLL,
    RemoteLabSession,
    Phase,
    VRARExperiment,
    MultilingualAnswer,
    MultilingualQuestion,
    ScenarioImplementation,
    UserScenarioScore,
    HelpRequest,
)
from django.views.decorators.http import require_POST
from authoringtool.models import Subject
from authoringtool.html_safety import sanitize_html
from authoringtool.views import user_can_view_scenario
from usergroups.models import UserGroupMembership
import json
import logging
import re
import requests
from django.conf import settings
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from django.contrib.auth.models import User
from django.contrib import messages

logger = logging.getLogger(__name__)

@login_required
def scenario_viewer(request, scenario_id):
    scenario = get_object_or_404(Scenario, pk=scenario_id)
    user = request.user
    check_scenario_access(user, scenario)
    is_teacher = user.groups.filter(name="teachers").exists()
    if not is_teacher and hasattr(scenario, 'revision_draft'):
        return HttpResponse(
            (
                'This scenario is temporarily unavailable while its teacher '
                'prepares a new published revision. Please try again later.'
            ),
            status=503,
        )
    first_activity = scenario.get_start_activity()
    if not first_activity:
        return HttpResponse(
            'No activities found in this scenario.',
            status=404,
        )

    current_version = scenario.ensure_current_version()
    implementation = None
    if not is_teacher:
        implementation, _ = ScenarioImplementation.start_or_resume(
            user,
            scenario,
        )
        existing_score = UserScenarioScore.objects.filter(
            implementation=implementation,
        ).first()
        if existing_score is None:
            transitional_score = (
                UserScenarioScore.objects
                .filter(
                    user=user,
                    scenario=scenario,
                    implementation__isnull=True,
                    scenario_version=current_version,
                    version_confidence='exact',
                )
                .order_by('id')
                .first()
            )
            if transitional_score:
                transitional_score.implementation = implementation
                transitional_score.save()
            else:
                UserScenarioScore.objects.create(
                    user=user,
                    scenario=scenario,
                    implementation=implementation,
                    scenario_version=current_version,
                    version_confidence='exact',
                    data_quality_status='unreviewed',
                )

    feedback_form_json = None
    if not is_teacher:
        from feedback.utils import get_applicable_form, serialize_form, user_has_responded
        fb_form = get_applicable_form(scenario, 'student')
        if fb_form and not user_has_responded(fb_form, user, scenario):
            feedback_form_json = json.dumps(serialize_form(fb_form), ensure_ascii=False)

    help_paused = (
        not is_teacher
        and HelpRequest.objects.filter(
            user=user,
            scenario=scenario,
            ended_at__isnull=True,
        ).exists()
    )

    return render(request, 'studentview/scenarioView.html', {
        'activity': first_activity, 'myScenario': scenario, 'user': user,
        'is_teacher': is_teacher,
        'implementation': implementation,
        'feedback_form_json': feedback_form_json,
        'help_paused': help_paused,
        'help_text': help_labels(scenario.language),
    })


def _is_teacher(user):
    return user.groups.filter(name="teachers").exists()


def check_scenario_access(user, scenario):
    """Raise unless the user may open this scenario.

    Students may only open scenarios assigned to one of their groups (exactly
    what their scenario list shows); teachers may preview any scenario they
    can view.
    """
    if _is_teacher(user) or user.is_staff or user.is_superuser:
        if not user_can_view_scenario(user, scenario):
            raise PermissionDenied
    elif scenario.archived_at or not Scenario.objects.filter(
        pk=scenario.pk, assigned_groups__members=user,
    ).exists():
        raise Http404('This scenario is not available.')


@login_required
@require_POST
def help_start(request, scenario_id):
    """Pause the scenario: the student asks for help on the current activity."""
    scenario = get_object_or_404(Scenario, pk=scenario_id)
    check_scenario_access(request.user, scenario)
    if _is_teacher(request.user):
        return JsonResponse({'error': 'Only students can ask for help.'}, status=403)
    activity = Activity.objects.filter(
        pk=request.POST.get('activity_id') or None,
        scenario=scenario,
    ).select_related('phase').first()
    if activity is None:
        return JsonResponse(
            {'error': 'The activity does not belong to this scenario.'},
            status=400,
        )
    open_request = HelpRequest.objects.filter(
        user=request.user,
        scenario=scenario,
        ended_at__isnull=True,
    ).first()
    if open_request is None:
        HelpRequest.objects.create(
            user=request.user,
            scenario=scenario,
            implementation=ScenarioImplementation.objects.filter(
                user=request.user,
                scenario=scenario,
            ).exclude(status='abandoned').order_by('-started_at', '-id').first(),
            activity=activity,
            phase=activity.phase,
        )
    return JsonResponse({'paused': True})


@login_required
@require_POST
def help_end(request, scenario_id):
    """Resume the scenario and store how long the help pause lasted."""
    scenario = get_object_or_404(Scenario, pk=scenario_id)
    check_scenario_access(request.user, scenario)
    seconds = 0
    for open_request in HelpRequest.objects.filter(
        user=request.user,
        scenario=scenario,
        ended_at__isnull=True,
    ):
        open_request.close()
        seconds += open_request.duration_seconds or 0
    return JsonResponse({'paused': False, 'seconds': seconds})

@login_required
def scenario_form(request, scenario_id):
    return redirect('studentView', scenario_id=scenario_id)

    scenario = get_object_or_404(Scenario, pk=scenario_id)

    # Map canonical scenario language to language code (see Scenario.LANGUAGE_CHOICES)
    language_mapping = {
        'Greek':      'gr',
        'Portuguese': 'pt',
        'German':     'de',
        'Spanish':    'es',
        'French':     'fr',
        'Italian':    'it',
        'Romanian':   'ro',
        'Estonian':   'et',
        'Ukrainian':  'uk',
        'Persian':    'fa',
        'English':    'en',
    }
    
    # Get the appropriate language code, default to English if not found
    scenario_language = scenario.language.strip() if scenario.language else 'English'
    language_code = language_mapping.get(scenario_language, 'en')
    
    # Get all questions ordered by their order field
    questions = MultilingualQuestion.objects.all().order_by('order')
    
    # Check if all questions have been answered for this scenario
    all_questions_answered = True
    questions_with_text = []
    
    for question in questions:
        # Get the appropriate language field based on scenario's language
        question_text = getattr(question, f'question_text_{language_code}', None)
        # If no translation exists, fall back to English
        if not question_text:
            question_text = question.question_text_en
            
        # Get existing answer if any for this specific scenario
        existing_answer = None
        try:
            existing_answer = question.answers.get(user=request.user, scenario=scenario)
        except MultilingualAnswer.DoesNotExist:
            all_questions_answered = False
            
        questions_with_text.append({
            'question': question,
            'text': question_text,
            'existing_answer': existing_answer.answer_text if existing_answer else '',
            'is_required': question.is_required
        })
    
    answered_count = sum(1 for q in questions_with_text if q['existing_answer'])
    total_questions = len(questions_with_text)

    context = {
        'myScenario': scenario,
        'questions': questions_with_text,
        'display_language': scenario_language,
        'all_questions_answered': all_questions_answered,
        'answered_count': answered_count,
        'total_questions': total_questions,
    }
    
    return render(request, 'studentview/scenarioForm.html', context)

@login_required
def submit_answers(request, scenario_id):
    if request.method == 'POST':
        scenario = get_object_or_404(Scenario, pk=scenario_id)
        check_scenario_access(request.user, scenario)

        for question in MultilingualQuestion.objects.all():
            # Get the answer text, even if it's empty
            answer_text = request.POST.get(f'question_{question.id}', '')
            
            # Get or create the answer for this specific scenario
            answer, created = MultilingualAnswer.objects.get_or_create(
                question=question,
                user=request.user,
                scenario=scenario,
                defaults={
                    'answer_text': answer_text,
                    'created_by': request.user,
                    'updated_by': request.user
                }
            )
            if not created:
                answer.answer_text = answer_text
                answer.updated_by = request.user
                answer.save()
        
        # messages.success(request, 'Your feedback has been saved successfully!')
        return redirect('studentView', scenario_id=scenario_id)
    
    return redirect('scenario_questions', scenario_id=scenario_id)

def get_activity(request, activity_id):
    activity = get_object_or_404(Activity, id=activity_id)
    if activity.scenario is None:
        raise Http404('This activity is not available.')
    check_scenario_access(request.user, activity.scenario)
    is_teacher = (
        request.user.is_authenticated
        and request.user.groups.filter(name="teachers").exists()
    )
    if not is_teacher and hasattr(activity.scenario, 'revision_draft'):
        return JsonResponse(
            {
                'error': (
                    'This scenario is temporarily unavailable while a new '
                    'revision is being prepared.'
                )
            },
            status=503,
        )
    simulation_data = None
    experimentLL_data = None
    vr_ar_data = None
    # simulation_iframe = None
    # simulation = None
    try:
        # Check if the activity has a simulation linked before attempting to fetch
        if activity.simulation:
            simulation = get_object_or_404(Simulation, id=activity.simulation.id)
            simulation_data = {
                'id': simulation.id,
                'iframe_url': simulation.iframe_url  # Assuming Simulation model has an 'iframe_url' field
            }
            # print("SIMULATION ID --> : ", simulation.id, "SIMULATION URL --> : ", simulation.iframe_url)
            #simulation_iframe = simulation.iframe_url
        elif activity.experiment_ll:
            experimentLL = get_object_or_404(ExperimentLL, id=activity.experiment_ll.id)
            experimentLL_data = {
                'id': experimentLL.id,
                'iframe_url': reverse('lti_integration_in_student', args=[experimentLL.id])  # LTI integration URL
            }
        elif activity.vr_ar_experiment:
            vr_ar_experiment = get_object_or_404(VRARExperiment, id=activity.vr_ar_experiment.id)
            vr_ar_data = {
                'id': vr_ar_experiment.id,
                'name': vr_ar_experiment.name,
                'qr_code_url': vr_ar_experiment.qr_code.url if vr_ar_experiment.qr_code else None,
                'picture_url': vr_ar_experiment.picture.url if vr_ar_experiment.picture else None,
                'launch_url': vr_ar_experiment.launch_url
            }
        else:
            simulation_data = None
            experimentLL_data = None
            vr_ar_data = None

        return JsonResponse({
            'activity_id': activity.id,
            'activity_type_name': activity.activity_type.name,
            'name': activity.name,
            'content': sanitize_html(activity.text),
            'simulation': simulation_data,
            #'simulation_iframe_url': simulation.iframe_url,
            'experimentLL': experimentLL_data,
            'vr_ar_data': vr_ar_data
        })

    except Activity.DoesNotExist:
        return JsonResponse({'error': 'Activity not found'}, status=404)
    except (Simulation.DoesNotExist, ExperimentLL.DoesNotExist):
        # If the activity is supposed to have a simulation but doesn't find it, handle it gracefully
        return JsonResponse({
            'activity_id': activity.id,
            'error': 'Linked simulation not found',
            'activity_type_name': activity.activity_type.name,
            'name': activity.name,
            'content': sanitize_html(activity.text)
        }, status=404)


RASA_CLIENT_IDENTITY_KEYS = ('user_id', 'userId', 'scenarioId')


@require_POST
def rasa_chat(request):
    """Forward a chat message to Rasa as the logged-in user.

    Rasa is only reachable on the internal network, so this is the only way
    in. The user id comes from the session, never from the browser, and the
    conversation id is namespaced by it, so nobody can answer or reset a
    scenario on another student's behalf.
    """
    try:
        payload = json.loads(request.body)
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({'error': 'Invalid JSON.'}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({'error': 'Invalid payload.'}, status=400)

    message = payload.get('message')
    if not isinstance(message, str) or not message.strip():
        return JsonResponse({'error': 'Missing message.'}, status=400)

    metadata = payload.get('metadata') or {}
    if not isinstance(metadata, dict):
        return JsonResponse({'error': 'Invalid metadata.'}, status=400)
    metadata = {
        key: value for key, value in metadata.items()
        if key not in RASA_CLIENT_IDENTITY_KEYS
    }
    scenario_id = str(metadata.get('scenario_id') or '').strip()
    if scenario_id:
        if not scenario_id.isdigit():
            return JsonResponse({'error': 'Invalid scenario.'}, status=400)
        metadata['scenario_id'] = scenario_id
    metadata['user_id'] = str(request.user.id)

    client_sender = re.sub(r'[^A-Za-z0-9-]', '', str(payload.get('sender') or ''))
    sender = f"user-{request.user.id}-{client_sender[:64] or 'chat'}"

    try:
        response = requests.post(
            settings.RASA_WEBHOOK_URL,
            json={'sender': sender, 'message': message, 'metadata': metadata},
            timeout=settings.RASA_TIMEOUT_SECONDS,
        )
        replies = response.json()
    except (requests.RequestException, ValueError):
        logger.exception('Rasa did not answer the chat proxy')
        return JsonResponse([], safe=False, status=502)
    if response.status_code != 200 or not isinstance(replies, list):
        logger.error('Rasa returned HTTP %s to the chat proxy', response.status_code)
        return JsonResponse([], safe=False, status=502)
    return JsonResponse(replies, safe=False)


@login_required
def chatbot_interaction(request):
    # Assuming POST request with JSON body
    data = json.loads(request.body)
    user_message = data['message']
    response_message = process_chatbot_message(user_message)
    return JsonResponse({'response': response_message})

def process_chatbot_message(message):
    # Placeholder for integrating with actual chatbot logic
    return "Response to the user message"

def chat_interface(request):
    return render(request, 'studentview/chatBot.html')

@login_required
def scenarios_view(request):
    user = request.user
    is_teacher = user.groups.filter(name='teachers').exists()
    query = request.GET.get('q', '').strip()
    language = request.GET.get('language', '')
    selected_subject_ids = [int(x) for x in request.GET.getlist('subject') if x.isdigit()]

    if is_teacher:
        org_ids = user.member_of_organizations.values_list('id', flat=True)
        base_qs = Scenario.objects.filter(
            Q(created_by=user) |
            Q(visibility_status='public') |
            Q(visibility_status='org', organizations__id__in=org_ids)
        ).distinct()
    else:
        user_memberships = UserGroupMembership.objects.filter(user=user)
        group_ids = user_memberships.values_list('group_id', flat=True)
        base_qs = Scenario.objects.filter(
            assigned_groups__id__in=group_ids, archived_at__isnull=True,
        ).distinct()

    base_qs = base_qs.filter(archived_at__isnull=True)
    if query:
        base_qs = base_qs.filter(Q(name__icontains=query) | Q(description__icontains=query))
    if selected_subject_ids:
        base_qs = base_qs.filter(subjects__id__in=selected_subject_ids)
    if language:
        base_qs = base_qs.filter(language=language)

    base_qs = base_qs.order_by('-created_on')

    # Annotate all scenarios (evaluate queryset once before pagination)
    all_question_ids = set(MultilingualQuestion.objects.values_list('id', flat=True))
    total_questions = len(all_question_ids)
    scenario_list = list(base_qs)

    answered_pairs = set(
        MultilingualAnswer.objects.filter(
            user=user,
            scenario_id__in=[s.id for s in scenario_list]
        ).values_list('scenario_id', 'question_id')
    )
    answered_map = {}
    for scenario_id, question_id in answered_pairs:
        answered_map.setdefault(scenario_id, set()).add(question_id)
    for scenario in scenario_list:
        answered_for_scenario = answered_map.get(scenario.id, set())
        scenario.has_answered_all_questions = (
            total_questions > 0 and answered_for_scenario >= all_question_ids
        )

    paginator = Paginator(scenario_list, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    get_params = request.GET.copy()
    get_params.pop('page', None)
    filter_querystring = get_params.urlencode()

    # Smart page range
    def _smart_page_range(po):
        cur, total = po.number, po.paginator.num_pages
        pages = set([1, total])
        for i in range(max(1, cur - 2), min(total, cur + 2) + 1):
            pages.add(i)
        result, prev = [], None
        for p in sorted(pages):
            if prev is not None and p - prev > 1:
                result.append(None)
            result.append(p)
            prev = p
        return result

    languages = Scenario.objects.values_list('language', flat=True).distinct().order_by('language')

    template = loader.get_template('studentview/scenarioSelection.html')
    context = {
        'myScenarios': page_obj,
        'page_obj': page_obj,
        'page_range': _smart_page_range(page_obj),
        'filter_querystring': filter_querystring,
        'is_teacher': is_teacher,
        'query': query,
        'selected_language': language,
        'languages': languages,
        'all_subjects': Subject.objects.all(),
        'selected_subject_ids': selected_subject_ids,
    }
    return HttpResponse(template.render(context, request))
    
def pendulum_lab_view(request):
    return render(request, 'studentview/pendulum_modified_imu.html')

@csrf_exempt
def save_iteration_remlab(request):
    if request.method == 'POST':
        # Authentication check: require a valid session user.
        # @csrf_exempt is intentional — LabsLand posts from an external origin
        # where CSRF tokens are impossible. However we still enforce auth.
        if not request.user.is_authenticated:
            logger.warning(
                "save_iteration_remlab: unauthenticated request from %s",
                request.META.get('REMOTE_ADDR', 'unknown')
            )
            return JsonResponse({'error': 'unauthorized'}, status=401)

        try:
            data = json.loads(request.body)

            # Extract and validate data — user_id from the POST body is ignored;
            # we always use the authenticated session user to prevent spoofing.
            user_id = request.user.id
            activity_id = data.get("activity_id")
            iteration = data.get("iteration")
            start_timestamp_str = data.get('start_timestamp')
            end_timestamp_str = data.get('end_timestamp')
            angle = data.get("angle")
            mass_type = data.get("mass_type")
            total_pre_duration = data.get("total_pre_duration")
            total_exec_duration = data.get("total_exec_duration")
            lab_name = "Pendulum Lab"

            # Convert timestamps
            start_timestamp = datetime.fromisoformat(start_timestamp_str.replace("Z", "+00:00")) if isinstance(start_timestamp_str, str) else None
            end_timestamp = datetime.fromisoformat(end_timestamp_str.replace("Z", "+00:00")) if isinstance(end_timestamp_str, str) else None

            # Retrieve user and activity, handle exceptions if not found
            try:
                user = User.objects.get(id=user_id)
                activity = Activity.objects.get(id=activity_id)
            except ObjectDoesNotExist as e:
                print(f"User or Activity not found: {e}")
                return JsonResponse({"error": "User or Activity not found"}, status=404)

            # Retrieve related scenario and phase
            scenario = activity.scenario
            phase = activity.phase
            
            # print(f"Data for saving iteration: user_id={user_id}, activity_id={activity_id}, iteration={iteration}, start={start_timestamp}, end={end_timestamp}, angle={angle}, mass={mass_type}, pre_duration={total_pre_duration}, exec_duration={total_exec_duration}")
            if activity.experiment_ll:
                try:
                    session = RemoteLabSession(
                        user_id=user.id,
                        activity=activity,
                        phase=phase,
                        scenario=scenario,
                        iteration=iteration,
                        lab_name=lab_name,
                        start=start_timestamp,
                        end=end_timestamp,
                        angle=angle,
                        mass=mass_type,
                        pre_duration=total_pre_duration,
                        exec_duration=total_exec_duration
                    )
                    session.save()  # Explicitly save
                    # Verify the session was saved
                    if RemoteLabSession.objects.filter(
                        user=user,
                        activity=activity,
                        iteration=iteration,
                        start=start_timestamp,
                        end=end_timestamp
                    ).exists():
                        print("Session saved and verified in the database.")
                        return JsonResponse({"status": "success"}, status=201)
                    else:
                        print("Session save failed - could not verify in database.")
                        return JsonResponse({"error": "Save verification failed"}, status=500)
                    # print("Session saved successfully!")
                    # return JsonResponse({"status": "success"}, status=201)
                except Exception as e:
                    print(f"Error during explicit save: {e}")
                """
                # Save data to the model
                RemoteLabSession.objects.create(
                    user_id=user.id,
                    activity=activity,
                    phase=phase,
                    scenario=scenario,
                    iteration=iteration,
                    lab_name=lab_name,
                    start=start_timestamp,
                    end=end_timestamp,
                    angle=angle,
                    mass=mass_type,
                    pre_duration=total_pre_duration,
                    exec_duration=total_exec_duration
                )

                return JsonResponse({"status": "success"}, status=201)
                """
            else:
                print("No experiment linked; skipping save.")
                return JsonResponse({"status": "pass"}, status=201)

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

        except Exception as e:
            print(f"Error saving iteration data: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)
