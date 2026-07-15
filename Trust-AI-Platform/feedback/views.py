import json

from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.contrib import messages
from django.db.models import Count
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from functools import wraps

from authoringtool.models import Scenario
from .models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse


def staff_required(view_func):
    @wraps(view_func)
    @login_required
    def _wrapped(request, *args, **kwargs):
        if not (request.user.is_staff or request.user.is_superuser):
            return HttpResponseForbidden("Access denied.")
        return view_func(request, *args, **kwargs)
    return _wrapped


def _clean_answer(answers, question):
    raw = answers.get(str(question.id))
    return raw.strip() if isinstance(raw, str) else ''


@require_POST
@login_required
def submit_feedback(request, form_id, scenario_id):
    form = get_object_or_404(FeedbackForm, id=form_id)
    scenario = get_object_or_404(Scenario, id=scenario_id)

    is_teacher = request.user.groups.filter(name='teachers').exists()
    if form.audience == 'student' and is_teacher:
        return JsonResponse({'success': False, 'error': 'Student forms cannot be submitted by teachers.'}, status=403)
    if form.audience == 'teacher' and not is_teacher:
        return JsonResponse({'success': False, 'error': 'Teacher forms can only be submitted by teachers.'}, status=403)
    if not form.applies_to(scenario):
        return JsonResponse({'success': False, 'error': 'This form does not apply to this scenario.'}, status=403)

    if FeedbackResponse.objects.filter(form=form, user=request.user, scenario=scenario).exists():
        return JsonResponse({'success': False, 'error': 'You have already submitted this form.'}, status=400)

    try:
        payload = json.loads(request.body or '{}')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON.'}, status=400)
    answers = payload.get('answers') or {}
    if not isinstance(answers, dict):
        return JsonResponse({'success': False, 'error': 'Invalid answers payload.'}, status=400)

    questions = list(form.questions.all())
    for question in questions:
        raw = _clean_answer(answers, question)
        if question.is_required and not raw:
            return JsonResponse({'success': False, 'error': f'Question "{question.text}" is required.'}, status=400)
        if raw and question.question_type == 'choice' and raw not in question.options:
            return JsonResponse({'success': False, 'error': f'Invalid option for "{question.text}".'}, status=400)

    try:
        with transaction.atomic():
            response = FeedbackResponse.objects.create(form=form, user=request.user, scenario=scenario)
            for question in questions:
                raw = _clean_answer(answers, question)
                if raw:
                    FeedbackAnswer.objects.create(response=response, question=question, answer_text=raw)
    except IntegrityError:
        return JsonResponse({'success': False, 'error': 'You have already submitted this form.'}, status=400)

    return JsonResponse({'success': True})


def _parse_questions_json(raw):
    """Returns (questions_list, error_message). Valid shape: list of
    {text, type in {choice,text}, options list, required bool}."""
    try:
        data = json.loads(raw or '[]')
    except json.JSONDecodeError:
        return None, 'Invalid question data.'
    if not isinstance(data, list) or not data:
        return None, 'At least one question is required.'
    cleaned = []
    for item in data:
        text = (item.get('text') or '').strip()
        qtype = item.get('type')
        options = item.get('options') or []
        if not text:
            return None, 'Every question needs text.'
        if qtype not in ('choice', 'text'):
            return None, 'Invalid question type.'
        options = [str(o).strip() for o in options if str(o).strip()]
        if qtype == 'choice' and len(options) < 2:
            return None, f'Question "{text}" needs at least two options.'
        cleaned.append({
            'text': text, 'type': qtype, 'options': options if qtype == 'choice' else [],
            'required': bool(item.get('required')),
        })
    return cleaned, None


def _save_form_from_post(request, form=None):
    """Shared create/edit POST handling. Returns (form, error_message)."""
    title = (request.POST.get('title') or '').strip()
    if not title:
        return None, 'Title is required.'
    audience = request.POST.get('audience')
    if audience not in ('teacher', 'student'):
        return None, 'Invalid audience.'
    questions, error = _parse_questions_json(request.POST.get('questions_json'))
    if error:
        return None, error

    assign_to_all = request.POST.get('assign_to_all') == 'on'
    checked_ids = set()
    for raw_id in request.POST.getlist('scenarios'):
        if raw_id.isdigit():
            checked_ids.add(int(raw_id))

    with transaction.atomic():
        if form is None:
            form = FeedbackForm(created_by=request.user)
        form.title = title
        form.description = (request.POST.get('description') or '').strip()
        form.audience = audience
        form.is_active = request.POST.get('is_active') == 'on'
        form.assign_to_all = assign_to_all
        form.save()

        all_ids = set(Scenario.objects.values_list('id', flat=True))
        if assign_to_all:
            form.excluded_scenarios.set(all_ids - checked_ids)
            form.included_scenarios.clear()
        else:
            form.included_scenarios.set(checked_ids & all_ids)
            form.excluded_scenarios.clear()

        form.questions.all().delete()
        for index, q in enumerate(questions):
            FeedbackQuestion.objects.create(
                form=form, text=q['text'], question_type=q['type'],
                options=q['options'], is_required=q['required'], order=index,
            )
    return form, None


@staff_required
def feedback_form_list(request):
    forms = FeedbackForm.objects.annotate(
        question_count=Count('questions', distinct=True),
        response_count=Count('responses', distinct=True),
    )
    return render(request, 'feedback/form_list.html', {'forms': forms})


@staff_required
def feedback_form_create(request):
    if request.method == 'POST':
        form, error = _save_form_from_post(request)
        if error is None:
            messages.success(request, 'Feedback form created.')
            return redirect('feedback_form_list')
        return render(request, 'feedback/form_edit.html', {
            'form_obj': None, 'error': error, 'scenarios': Scenario.objects.order_by('name'),
            'questions_json': request.POST.get('questions_json') or '[]',
            'posted': request.POST,
        })
    return render(request, 'feedback/form_edit.html', {
        'form_obj': None, 'scenarios': Scenario.objects.order_by('name'), 'questions_json': '[]',
    })


@staff_required
def feedback_form_edit(request, form_id):
    form = get_object_or_404(FeedbackForm, id=form_id)
    if request.method == 'POST':
        _, error = _save_form_from_post(request, form=form)
        if error is None:
            messages.success(request, 'Feedback form updated.')
            return redirect('feedback_form_list')
        return render(request, 'feedback/form_edit.html', {
            'form_obj': form, 'error': error, 'scenarios': Scenario.objects.order_by('name'),
            'questions_json': request.POST.get('questions_json') or '[]',
            'posted': request.POST,
        })
    questions_json = json.dumps([
        {'text': q.text, 'type': q.question_type, 'options': q.options, 'required': q.is_required}
        for q in form.questions.all()
    ])
    return render(request, 'feedback/form_edit.html', {
        'form_obj': form, 'scenarios': Scenario.objects.order_by('name'), 'questions_json': questions_json,
    })


@require_POST
@staff_required
def feedback_form_delete(request, form_id):
    form = get_object_or_404(FeedbackForm, id=form_id)
    form.delete()
    messages.success(request, 'Feedback form deleted.')
    return redirect('feedback_form_list')


@staff_required
def feedback_form_responses(request, form_id):
    form = get_object_or_404(FeedbackForm, id=form_id)
    return render(request, 'feedback/form_responses.html', {'form_obj': form, 'responses': form.responses.select_related('user', 'scenario')})
