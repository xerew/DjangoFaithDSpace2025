import json

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_POST
from functools import wraps

from authoringtool.models import Scenario
from .models import FeedbackAnswer, FeedbackForm, FeedbackResponse


def staff_required(view_func):
    @wraps(view_func)
    @login_required
    def _wrapped(request, *args, **kwargs):
        if not (request.user.is_staff or request.user.is_superuser):
            return HttpResponseForbidden("Access denied.")
        return view_func(request, *args, **kwargs)
    return _wrapped


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

    questions = list(form.questions.all())
    for question in questions:
        raw = (answers.get(str(question.id)) or '').strip()
        if question.is_required and not raw:
            return JsonResponse({'success': False, 'error': f'Question "{question.text}" is required.'}, status=400)
        if raw and question.question_type == 'choice' and raw not in question.options:
            return JsonResponse({'success': False, 'error': f'Invalid option for "{question.text}".'}, status=400)

    with transaction.atomic():
        response = FeedbackResponse.objects.create(form=form, user=request.user, scenario=scenario)
        for question in questions:
            raw = (answers.get(str(question.id)) or '').strip()
            if raw:
                FeedbackAnswer.objects.create(response=response, question=question, answer_text=raw)

    return JsonResponse({'success': True})
