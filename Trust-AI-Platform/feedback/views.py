import json

from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.contrib import messages
from django.db.models import Count
from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from functools import wraps

from authoringtool.models import Scenario
from .models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse
from .languages import FEEDBACK_LANGUAGES, LANGUAGE_CODES, language_name
from .utils import display_language, serialize_form


def staff_required(view_func):
    @wraps(view_func)
    @login_required
    def _wrapped(request, *args, **kwargs):
        if not (request.user.is_staff or request.user.is_superuser):
            return HttpResponseForbidden("Access denied.")
        return view_func(request, *args, **kwargs)
    return _wrapped


# Returned by _clean_answer when the chosen option is no longer where the page showed it.
FORM_CHANGED = object()


def _is_position(value, count):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count


def _clean_answer(answers, question, shown_options=()):
    """The text to store for one answer: '' when unanswered, None when invalid,
    FORM_CHANGED when the form was edited after the page was opened.

    The popup sends a choice as {"position": i, "label": shown text}; it is
    stored in the original wording, whatever language the form was shown in.
    The label must still be what the form shows at that position, so a page
    opened before the options were reordered cannot store the wrong option.
    A bare position and, for pages opened before positions were sent, the
    original wording as a string are also accepted.
    """
    raw = answers.get(str(question.id))
    if raw is None:
        return ''
    if question.question_type == 'choice':
        if isinstance(raw, dict):
            position, label = raw.get('position'), raw.get('label')
            if not _is_position(position, len(question.options)) or not isinstance(label, str):
                return None
            if position >= len(shown_options) or shown_options[position] != label:
                return FORM_CHANGED
            return question.options[position]
        if isinstance(raw, int) and not isinstance(raw, bool):
            return question.options[raw] if 0 <= raw < len(question.options) else None
        if isinstance(raw, str):
            raw = raw.strip()
            return raw if raw == '' or raw in question.options else None
        return None
    return raw.strip() if isinstance(raw, str) else None


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

    language = display_language(form, scenario)
    shown = {q['id']: q for q in serialize_form(form, language)['questions']}
    questions = list(form.questions.all())
    cleaned = {}
    for question in questions:
        shown_question = shown.get(question.id, {})
        raw = _clean_answer(answers, question, shown_question.get('options', ()))
        text = shown_question.get('text', question.text)
        if raw is FORM_CHANGED:
            return JsonResponse({
                'success': False,
                'error': 'This form has changed since you opened it. Please reload the page and answer again.',
            }, status=400)
        if raw is None:
            return JsonResponse({'success': False, 'error': f'Invalid option for "{text}".'}, status=400)
        if question.is_required and not raw:
            return JsonResponse({'success': False, 'error': f'Question "{text}" is required.'}, status=400)
        cleaned[question.id] = raw

    try:
        with transaction.atomic():
            response = FeedbackResponse.objects.create(
                form=form, user=request.user, scenario=scenario, language=language,
            )
            for question in questions:
                raw = cleaned[question.id]
                if raw:
                    FeedbackAnswer.objects.create(response=response, question=question, answer_text=raw)
    except IntegrityError:
        return JsonResponse({'success': False, 'error': 'You have already submitted this form.'}, status=400)

    return JsonResponse({'success': True})


def _clean_translations(raw, fields, option_count=None, label=''):
    """Clean {code: {field: text, 'options': [text]}} from the editor.

    Returns (translations, error). Strings are stripped; languages whose
    entries are all empty are dropped.
    """
    if raw in (None, ''):
        return {}, None
    if not isinstance(raw, dict):
        return None, 'Invalid question data.'
    cleaned = {}
    for code, entry in raw.items():
        if code not in LANGUAGE_CODES:
            return None, f'Unknown translation language: {code}.'
        if not isinstance(entry, dict):
            return None, 'Invalid question data.'
        item = {field: str(entry.get(field) or '').strip() for field in fields}
        if option_count is not None:
            options = entry.get('options') or []
            if not isinstance(options, list):
                return None, 'Invalid question data.'
            options = [str(o if o is not None else '').strip() for o in options]
            if len(options) > option_count:
                return None, f'The {language_name(code)} translation of "{label}" has more options than the original.'
            if any(options):
                item['options'] = options
            elif 'options' in entry:
                item['options'] = []
        if any(v for v in item.values()):
            cleaned[code] = item
    return cleaned, None


def _parse_questions_json(raw):
    """Returns (questions_list, error_message). Valid shape: list of
    {id int|null, text, type in {choice,text}, options list, required bool,
    translations {code: {text, options}}}."""
    try:
        data = json.loads(raw or '[]')
    except json.JSONDecodeError:
        return None, 'Invalid question data.'
    if not isinstance(data, list) or not data:
        return None, 'At least one question is required.'
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            return None, 'Invalid question data.'
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
        if qtype != 'choice':
            options = []
        qid = item.get('id')
        if qid is not None and (isinstance(qid, bool) or not isinstance(qid, int)):
            return None, 'Invalid question data.'
        translations, error = _clean_translations(
            item.get('translations'), ('text',),
            option_count=len(options) if qtype == 'choice' else 0, label=text,
        )
        if error:
            return None, error
        cleaned.append({
            'id': qid, 'text': text, 'type': qtype, 'options': options,
            'required': bool(item.get('required')), 'translations': translations,
        })
    return cleaned, None


def _check_existing_questions(form, questions):
    """Ids must belong to this form; answered questions keep type and options."""
    posted_ids = [q['id'] for q in questions if q['id'] is not None]
    if not posted_ids:
        return None
    if len(set(posted_ids)) != len(posted_ids):
        return 'Invalid question data.'
    if form is None:
        return 'Invalid question data.'
    existing = {
        q.id: q for q in form.questions.annotate(answer_count=Count('answers')).filter(id__in=posted_ids)
    }
    for q in questions:
        if q['id'] is None:
            continue
        current = existing.get(q['id'])
        if current is None:
            return 'Invalid question data.'
        if current.answer_count and (q['type'] != current.question_type or q['options'] != list(current.options)):
            return (f'Question "{current.text}" already has answers, so its type and options can\'t be changed. '
                    'Remove it and add a new question instead.')
    return None


def _save_form_from_post(request, form=None):
    """Shared create/edit POST handling. Returns (form, error_message).

    Questions are updated in place, so answers survive edits; only questions
    left out of the post are deleted, with their answers.
    """
    title = (request.POST.get('title') or '').strip()
    if not title:
        return None, 'Title is required.'
    audience = request.POST.get('audience')
    if audience not in ('teacher', 'student'):
        return None, 'Invalid audience.'
    source_language = request.POST.get('source_language') or 'en'
    if source_language not in LANGUAGE_CODES:
        return None, 'Invalid original language.'
    questions, error = _parse_questions_json(request.POST.get('questions_json'))
    if error:
        return None, error
    try:
        raw_form_translations = json.loads(request.POST.get('form_translations_json') or '{}')
    except json.JSONDecodeError:
        return None, 'Invalid question data.'
    form_translations, error = _clean_translations(raw_form_translations, ('title', 'description'))
    if error:
        return None, error
    error = _check_existing_questions(form, questions)
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
        form.source_language = source_language
        form.translations = form_translations
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

        kept_ids = {q['id'] for q in questions if q['id'] is not None}
        form.questions.exclude(id__in=kept_ids).delete()
        for index, q in enumerate(questions):
            values = {
                'text': q['text'], 'question_type': q['type'], 'options': q['options'],
                'is_required': q['required'], 'order': index, 'translations': q['translations'],
            }
            if q['id'] is None:
                FeedbackQuestion.objects.create(form=form, **values)
            else:
                FeedbackQuestion.objects.filter(id=q['id'], form=form).update(**values)
    return form, None


def _editor_questions_json(form):
    return json.dumps([
        {
            'id': q.id, 'text': q.text, 'type': q.question_type, 'options': q.options,
            'required': q.is_required, 'translations': q.translations or {},
            'answer_count': q.answer_count,
        }
        for q in form.questions.annotate(answer_count=Count('answers'))
    ])


@staff_required
def feedback_form_list(request):
    forms = FeedbackForm.objects.annotate(
        question_count=Count('questions', distinct=True),
        response_count=Count('responses', distinct=True),
    )
    return render(request, 'feedback/form_list.html', {'forms': forms})


def _editor_context(request, form, error=None):
    """Context for the form editor. After a failed save it re-shows what was
    posted, with answer counts taken from the database so locks stay on."""
    context = {
        'form_obj': form, 'scenarios': Scenario.objects.order_by('name'),
        'languages': FEEDBACK_LANGUAGES,
    }
    if request.method != 'POST':
        context.update({
            'questions_json': _editor_questions_json(form) if form else '[]',
            'form_translations_json': json.dumps(form.translations or {}) if form else '{}',
            'source_language': form.source_language if form else 'en',
        })
        return context
    try:
        posted = json.loads(request.POST.get('questions_json') or '[]')
    except json.JSONDecodeError:
        posted = []
    if not isinstance(posted, list):
        posted = []
    counts = {}
    if form is not None:
        counts = dict(form.questions.annotate(n=Count('answers')).values_list('id', 'n'))
    for item in posted:
        if isinstance(item, dict):
            item['answer_count'] = counts.get(item.get('id'), 0)
    context.update({
        'error': error,
        'questions_json': json.dumps(posted),
        'form_translations_json': request.POST.get('form_translations_json') or '{}',
        'source_language': request.POST.get('source_language') or 'en',
        'posted': request.POST,
        'posted_scenario_ids': {int(s) for s in request.POST.getlist('scenarios') if s.isdigit()},
    })
    return context


@staff_required
def feedback_form_create(request):
    if request.method == 'POST':
        form, error = _save_form_from_post(request)
        if error is None:
            messages.success(request, 'Feedback form created.')
            return redirect('feedback_form_list')
        return render(request, 'feedback/form_edit.html', _editor_context(request, None, error))
    return render(request, 'feedback/form_edit.html', _editor_context(request, None))


@staff_required
def feedback_form_edit(request, form_id):
    form = get_object_or_404(FeedbackForm, id=form_id)
    if request.method == 'POST':
        _, error = _save_form_from_post(request, form=form)
        if error is None:
            messages.success(request, 'Feedback form updated.')
            return redirect('feedback_form_list')
        return render(request, 'feedback/form_edit.html', _editor_context(request, form, error))
    return render(request, 'feedback/form_edit.html', _editor_context(request, form))


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
    questions = list(form.questions.all())
    responses = form.responses.select_related('user', 'scenario').prefetch_related('answers__question')
    response_rows = []
    for response in responses:
        answer_map = {a.question_id: a.answer_text for a in response.answers.all()}
        response_rows.append({
            'response': response,
            'language': language_name(response.language or form.source_language),
            'answers': [(q, answer_map.get(q.id, '')) for q in questions],
        })
    return render(request, 'feedback/form_responses.html', {
        'form_obj': form,
        'questions': questions,
        'response_rows': response_rows,
    })


@require_POST
@staff_required
def feedback_response_delete(request, response_id):
    response = get_object_or_404(FeedbackResponse, id=response_id)
    form_id = response.form_id
    response.delete()
    messages.success(request, 'Response deleted.')
    return redirect('feedback_form_responses', form_id=form_id)


def _export_rows(form):
    """Header row + one row per response, question columns in question order."""
    questions = list(form.questions.all())
    header = ['Username', 'Scenario', 'Submitted', 'Language'] + [q.text for q in questions]
    rows = [header]
    responses = form.responses.select_related('user', 'scenario').prefetch_related('answers')
    for response in responses:
        answer_map = {a.question_id: a.answer_text for a in response.answers.all()}
        rows.append(
            [response.user.username, response.scenario.name, response.submitted_at.strftime('%Y-%m-%d %H:%M'),
             response.language or form.source_language]
            + [answer_map.get(q.id, '') for q in questions]
        )
    return rows


@staff_required
def feedback_form_export_csv(request, form_id):
    import csv
    form = get_object_or_404(FeedbackForm, id=form_id)
    http_response = HttpResponse(content_type='text/csv')
    http_response['Content-Disposition'] = f'attachment; filename="feedback_form_{form.id}_responses.csv"'
    writer = csv.writer(http_response)  # default delimiter: comma (deliberate; spec requirement)
    for row in _export_rows(form):
        writer.writerow(row)
    return http_response


@staff_required
def feedback_form_export_xlsx(request, form_id):
    import io
    import openpyxl
    form = get_object_or_404(FeedbackForm, id=form_id)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Responses'
    for row in _export_rows(form):
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    http_response = HttpResponse(
        buf.read(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    http_response['Content-Disposition'] = f'attachment; filename="feedback_form_{form.id}_responses.xlsx"'
    return http_response
