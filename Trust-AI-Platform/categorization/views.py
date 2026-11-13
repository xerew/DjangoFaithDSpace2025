"""Admin-only dashboard of the categorization models (Django admin area)."""

from functools import wraps

from django.contrib import admin, messages
from django.contrib.auth.views import redirect_to_login
from django.http import Http404, HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from . import chatbot_report
from .export import chatbot_workbook, workbook
from .models import ChatbotEvaluation, TrainingRun
from .report import early_view, filter_options, final_view, over_time_view
from .tasks import train_models_task


def admin_only(view):
    """Staff or superusers; anonymous users go to the admin login."""
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path(), reverse('admin:login'))
        if not (request.user.is_staff or request.user.is_superuser):
            return HttpResponseForbidden(_('Only administrators can open this page.'))
        return view(request, *args, **kwargs)
    return wrapped


def _filters(request):
    version = request.GET.get('version', '')
    return {
        'scenario': request.GET.get('scenario', ''),
        'language': request.GET.get('language', ''),
        'version': version if version in ('current', 'older') else '',
    }


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tab_links(request):
    links = {}
    for tab in ('final', 'early', 'time'):
        query = request.GET.copy()
        query['tab'] = tab
        query.pop('checkpoint', None)
        query.pop('prediction', None)
        query.pop('evaluation', None)
        links[tab] = '?' + query.urlencode()
    # The categorization filters mean nothing on the chatbot tab.
    links['chatbot'] = '?tab=chatbot'
    return links


def _chart_text(chart, labels=True):
    if labels:
        chart['labels'] = [_('After phase %(k)s') % {'k': k} for k in chart['checkpoints']]
    chart['names'] = {
        'xgb_accuracy': _('XGBoost accuracy'),
        'baseline_accuracy': _('Formula so far accuracy'),
        'xgb_f1': _('XGBoost macro F1'),
        'baseline_f1': _('Formula so far macro F1'),
        'trained_on': _('Runs learned from'),
        'tested': _('Runs tested'),
    }
    return chart


def selected_run(request):
    finished = TrainingRun.objects.filter(status='finished')
    return finished.filter(pk=_int(request.GET.get('run'))).first() or finished.first()


def _active_and_failed(model, selected, heavy):
    """The job in progress, and the last failure if newer than what is shown."""
    model.expire_stale()
    light = model.objects.defer(*heavy)
    active = light.filter(status__in=model.ACTIVE).first()
    last_failed = light.filter(status='failed').first()
    if last_failed and selected and last_failed.created_at < selected.created_at:
        last_failed = None
    return active, last_failed


# Fields too big to load just to list or poll evaluations.
EVALUATION_RESULTS = ('report', 'errors', 'successes', 'nlu_text', 'log_tail')


def _chatbot_context(request):
    finished = ChatbotEvaluation.objects.filter(status='finished')
    selected = finished.filter(pk=_int(request.GET.get('evaluation'))).first() or finished.first()
    active, last_failed = _active_and_failed(ChatbotEvaluation, selected, EVALUATION_RESULTS)
    context = {
        'evaluation': selected,
        'evaluations': finished.only('id', 'finished_at', 'folds')[:50],
        'chatbot_active': active,
        'chatbot_failed': last_failed,
        'fold_choices': ChatbotEvaluation.FOLD_CHOICES,
    }
    if selected:
        view = chatbot_report.build(selected)
        minutes = None
        if selected.finished_at and selected.started_at:
            minutes = max(round((selected.finished_at - selected.started_at).total_seconds() / 60), 1)
        context.update({
            'chatbot': view,
            'chatbot_minutes': minutes,
            'chatbot_charts': {
                'per_intent': [{'intent': r['intent'], 'f1': r['f1']} for r in view['per_intent']],
                'confusion': view['confusion'],
                'confidence': view['confidence'],
                'names': {
                    'f1': _('F1'), 'right': _('Right'), 'wrong': _('Wrong'),
                    'fallback': _('Fallback threshold'),
                },
            },
        })
    return context


@admin_only
def dashboard(request):
    tab = request.GET.get('tab')
    tab = tab if tab in ('early', 'time', 'chatbot') else 'final'
    context = {
        **admin.site.each_context(request),
        'title': _('Student categorization models'),
        # The admin's app sidebar adds nothing on this page.
        'is_nav_sidebar_enabled': False,
        'tab': tab,
        'tab_links': _tab_links(request),
    }
    if tab == 'chatbot':
        context.update(_chatbot_context(request))
        return render(request, 'categorization/dashboard.html', context)

    selected = selected_run(request)
    active, last_failed = _active_and_failed(TrainingRun, selected, ('results',))
    filters = _filters(request)
    context.update({
        'selected': selected,
        # Only what the dropdown shows: each run's results can be megabytes.
        'finished_runs': TrainingRun.objects.filter(status='finished').only('id', 'finished_at')[:50],
        'active': active,
        'last_failed': last_failed,
        'filters': filters,
    })
    if selected:
        checkpoint = _int(request.GET.get('checkpoint'))
        duration = None
        if selected.finished_at and selected.started_at:
            duration = round((selected.finished_at - selected.started_at).total_seconds())
        early = early_view(selected.results, filters, checkpoint)
        _chart_text(early['chart'])
        over_time = over_time_view(selected.results, request.GET.get('prediction'))
        if over_time:
            _chart_text(over_time['chart'], labels=False)
        context.update({
            'over_time': over_time,
            'final': final_view(selected.results, filters),
            'early': early,
            'options': filter_options(selected.results),
            'duration': duration,
        })
    return render(request, 'categorization/dashboard.html', context)


@require_POST
@admin_only
def train(request):
    try:
        run = TrainingRun.start(request.user)
    except TrainingRun.AlreadyRunning:
        messages.warning(request, _('A training run is already in progress.'))
    else:
        train_models_task.delay(run.id)
        messages.success(request, _('Training started.'))
    return redirect('categorization_dashboard')


@admin_only
def status(request):
    TrainingRun.expire_stale()
    run = TrainingRun.objects.defer('results').first()
    if not run:
        return JsonResponse({})
    return JsonResponse({
        'id': run.id,
        'status': run.status,
        'progress': run.progress,
        'error': run.error,
    })


@admin_only
def export(request, run_id, tab):
    if tab not in ('final', 'early', 'time'):
        raise Http404
    run = get_object_or_404(TrainingRun, pk=run_id, status='finished')
    content = workbook(run, tab, _filters(request), _int(request.GET.get('checkpoint')))
    response = HttpResponse(
        content,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    response['Content-Disposition'] = f'attachment; filename="categorization-run-{run.id}-{tab}.xlsx"'
    return response


@require_POST
@admin_only
def chatbot_run(request):
    folds = _int(request.POST.get('folds'))
    back = reverse('categorization_dashboard') + '?tab=chatbot'
    if folds not in ChatbotEvaluation.FOLD_CHOICES:
        messages.error(request, _('Choose 3 or 5 folds.'))
        return redirect(back)
    try:
        ChatbotEvaluation.start(request.user, folds=folds)
    except ChatbotEvaluation.AlreadyRunning:
        messages.warning(request, _('A chatbot evaluation is already in progress.'))
    else:
        messages.success(request, _('Chatbot evaluation queued.'))
    return redirect(back)


@admin_only
def chatbot_status(request):
    ChatbotEvaluation.expire_stale()
    evaluation = ChatbotEvaluation.objects.defer(*EVALUATION_RESULTS).first()
    if not evaluation:
        return JsonResponse({})
    return JsonResponse({
        'id': evaluation.id,
        'status': evaluation.status,
        'progress': evaluation.progress,
        'error': evaluation.error,
    })


@admin_only
def chatbot_export(request, evaluation_id):
    evaluation = get_object_or_404(ChatbotEvaluation, pk=evaluation_id, status='finished')
    response = HttpResponse(
        chatbot_workbook(evaluation),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    response['Content-Disposition'] = f'attachment; filename="chatbot-intents-{evaluation.id}.xlsx"'
    return response
