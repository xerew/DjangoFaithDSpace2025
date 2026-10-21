"""Text-only editing of machine translations.

A machine translation keeps its source's structure, so the editor may change
its text and media (simulations, labs, VR experiments, images, videos) only.
"""

from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.urls import reverse

from authoringtool.models import (
    Activity,
    Answer,
    ExperimentLL,
    Phase,
    Scenario,
    ScenarioTranslationJob,
    Simulation,
    VRARExperiment,
)

from .units import strip_html

PENDING_MESSAGE = 'This translation is still being prepared.'
STRUCTURE_MESSAGE = (
    'Machine translations keep the original structure. Only text and media '
    'can be changed.'
)


def machine_translation_job(scenario):
    """The job that made this scenario, or None for any other scenario."""
    try:
        return scenario.translation_job
    except ScenarioTranslationJob.DoesNotExist:
        return None


def original_of(job, obj):
    """The source row a translated scenario, phase, activity or answer came
    from, or None when it no longer exists."""
    if job is None:
        return None
    if isinstance(obj, Scenario):
        return job.source_scenario
    kind = {Phase: 'phases', Activity: 'activities', Answer: 'answers'}[type(obj)]
    source_id = job.id_map.get(kind, {}).get(str(obj.id))
    if source_id is None:
        return None
    return type(obj).objects.filter(pk=source_id).first()


def mt_context(scenario, obj=None):
    """Template context for the text-only editor of a machine translation."""
    job = machine_translation_job(scenario)
    return {
        'is_machine_translation': job is not None,
        'original': original_of(job, obj) if job is not None and obj is not None else None,
    }


def translation_pending_guard(request, scenario):
    job = machine_translation_job(scenario)
    if job is None or job.status == ScenarioTranslationJob.STATUS_DONE:
        return None
    messages.error(request, PENDING_MESSAGE)
    return HttpResponseRedirect(reverse('scenarios'))


def translation_structure_guard(request, scenario):
    if machine_translation_job(scenario) is None:
        return None
    pending = translation_pending_guard(request, scenario)
    if pending:
        return pending
    messages.error(request, STRUCTURE_MESSAGE)
    return HttpResponseRedirect(reverse('viewScenario', args=[scenario.id]))


def save_translation_activity_text(request, activity):
    """Save an activity's text and media; ignore every structural field."""
    post = request.POST
    activity.name = post.get('activity_name', activity.name)
    activity.text = post.get('activity_text', activity.text)
    activity.plain_text = strip_html(activity.text)
    activity.helper = post.get('helping_quote', activity.helper) or ''
    if activity.activity_type and activity.activity_type.name == 'Experiment':
        experiment_type = post.get('experiment_type')
        chosen = {
            'simulation': (Simulation, post.get('simulation')),
            'remote_lab': (ExperimentLL, post.get('remote_lab')),
            'vr_ar_exp': (VRARExperiment, post.get('VRAR_lab')),
        }.get(experiment_type)
        if chosen and chosen[1]:
            model, pk = chosen
            activity.simulation = activity.experiment_ll = activity.vr_ar_experiment = None
            selected = get_object_or_404(model, pk=pk)
            if model is Simulation:
                activity.simulation = selected
            elif model is ExperimentLL:
                activity.experiment_ll = selected
            else:
                activity.vr_ar_experiment = selected
    activity.updated_by = request.user
    activity.save()


def save_translation_answer_text(request, activity):
    """Save existing answers' text and media; never add, remove or rescore."""
    post = request.POST
    answers = {str(answer.id): answer for answer in activity.answers.all()}
    for key in post:
        if not (key.startswith('answers[') and key.endswith('][text]')):
            continue
        index = key[len('answers['):-len('][text]')]
        answer = answers.get(post.get(f'answers[{index}][id]', ''))
        if answer is None:
            continue
        answer.text = post.get(key, answer.text).strip() or answer.text
        answer.vid_url = post.get(f'answers[{index}][vid_url]', answer.vid_url or '').strip()
        if post.get(f'clear_image_{index}') == 'on':
            answer.image.delete(save=False)
        else:
            image = request.FILES.get(f'answers[{index}][image]')
            if image:
                answer.image = image
        answer.updated_by = request.user
        answer.save()
