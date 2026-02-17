from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import JsonResponse, HttpResponse
from authoringtool.models import Scenario, Activity, NextQuestionLogic, Answer, Simulation, ExperimentLL, RemoteLabSession, Phase, Scenario, VRARExperiment, MultilingualAnswer, MultilingualQuestion
from usergroups.models import UserGroupMembership
import json
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from django.contrib.auth.models import User
from django.contrib import messages

def scenario_viewer(request, scenario_id):
    scenario = get_object_or_404(Scenario, pk=scenario_id)
    first_activity = scenario.phases.first().activities.first() if scenario.phases.exists() else None
    user = request.user
    is_teacher = user.groups.filter(name="teachers").exists()
    if not first_activity:
        return render(request, 'error_page.html', {'error': 'No activities found in this scenario.'})
    return render(request, 'studentview/scenarioView.html', {'activity': first_activity, 'myScenario': scenario, 'user': user, 'is_teacher': is_teacher})
    
@login_required
def scenario_form(request, scenario_id):
    scenario = get_object_or_404(Scenario, pk=scenario_id)
    
    # Map scenario language to language code
    language_mapping = {
        'Ελληνικά': 'gr',
        'Greek': 'gr',
        'Português': 'pt',
        'Portuguese': 'pt',
        'German': 'de',
        'Deutsch': 'de',
        'English': 'en'
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
    
    context = {
        'myScenario': scenario,
        'questions': questions_with_text,
        'display_language': scenario_language,
        'all_questions_answered': all_questions_answered
    }
    
    return render(request, 'studentview/scenarioForm.html', context)

@login_required
def submit_answers(request, scenario_id):
    if request.method == 'POST':
        scenario = get_object_or_404(Scenario, pk=scenario_id)
        
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
            'content': activity.text,
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
            'content': activity.text
        }, status=404)

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
    # Check if the user belongs to the 'teachers' group
    if user.groups.filter(name='teachers').exists():
        # If the user is a teacher, show their scenarios, public ones, and org ones they belong to
        org_ids = user.member_of_organizations.values_list('id', flat=True)
        
        myScenarios = Scenario.objects.filter(
            Q(created_by=user) |  # Scenarios the user created
            Q(visibility_status='public') |  # Public scenarios
            Q(visibility_status='org', organizations__id__in=org_ids)  # Org-only scenarios the user is part of
        ).distinct()
    else:
        # If the user is not a teacher, only show scenarios assigned to their group
        user_memberships = UserGroupMembership.objects.filter(user=user)
        group_ids = user_memberships.values_list('group_id', flat=True)

        myScenarios = Scenario.objects.filter(
            assigned_groups__id__in=group_ids  # Scenarios assigned to the groups the user is part of
        ).distinct()

    # Get all questions
    all_questions = MultilingualQuestion.objects.all()
    # Get all answers for this user
    user_answers = MultilingualAnswer.objects.filter(user=user)
    
    for scenario in myScenarios:
    # Get answers for this specific scenario
        scenario_answers = user_answers.filter(scenario=scenario)
    
        all_answered = True
        for question in all_questions:
            try:
                scenario_answers.get(question=question)  # Just existence is enough
            except MultilingualAnswer.DoesNotExist:
                all_answered = False
                break

        # Add custom attribute to scenario
        scenario.has_answered_all_questions = all_answered

    template = loader.get_template('studentview/scenarioSelection.html')
    context = {
        'myScenarios': myScenarios,
        'is_teacher': is_teacher
    }
    return HttpResponse(template.render(context, request))
    
def pendulum_lab_view(request):
    return render(request, 'studentview/pendulum_modified_imu.html')

@csrf_exempt
def save_iteration_remlab(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract and validate data
            user_id = data.get("user_id")
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