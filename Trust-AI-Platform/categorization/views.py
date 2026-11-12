"""Admin-only dashboard of the categorization models (Django admin area)."""

from functools import wraps

from django.contrib import admin, messages
from django.contrib.auth.views import redirect_to_login
from django.http import Http404, HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from .export import workbook
from .models import TrainingAlreadyRunning, TrainingRun
from .report import early_view, filter_options, final_view
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
    for tab in ('final', 'early'):
        query = request.GET.copy()
        query['tab'] = tab
        query.pop('checkpoint', None)
        links[tab] = '?' + query.urlencode()
    return links


def _chart_text(chart):
    chart['labels'] = [_('After phase %(k)s') % {'k': k} for k in chart['checkpoints']]
    chart['names'] = {
        'xgb_accuracy': _('XGBoost accuracy'),
        'baseline_accuracy': _('Formula so far accuracy'),
        'xgb_f1': _('XGBoost macro F1'),
        'baseline_f1': _('Formula so far macro F1'),
    }
    return chart


def selected_run(request):
    finished = TrainingRun.objects.filter(status='finished')
    return finished.filter(pk=_int(request.GET.get('run'))).first() or finished.first()


@admin_only
def dashboard(request):
    TrainingRun.expire_stale()
    selected = selected_run(request)
    light = TrainingRun.objects.defer('results')
    active = light.filter(status__in=TrainingRun.ACTIVE).first()
    last_failed = light.filter(status='failed').first()
    if last_failed and selected and last_failed.created_at < selected.created_at:
        last_failed = None
    tab = 'early' if request.GET.get('tab') == 'early' else 'final'
    filters = _filters(request)
    context = {
        **admin.site.each_context(request),
        'title': _('Student categorization models'),
        # The admin's app sidebar adds nothing on this page.
        'is_nav_sidebar_enabled': False,
        'selected': selected,
        # Only what the dropdown shows: each run's results can be megabytes.
        'finished_runs': TrainingRun.objects.filter(status='finished').only('id', 'finished_at')[:50],
        'active': active,
        'last_failed': last_failed,
        'tab': tab,
        'tab_links': _tab_links(request),
        'filters': filters,
    }
    if selected:
        checkpoint = _int(request.GET.get('checkpoint'))
        duration = None
        if selected.finished_at and selected.started_at:
            duration = round((selected.finished_at - selected.started_at).total_seconds())
        early = early_view(selected.results, filters, checkpoint)
        _chart_text(early['chart'])
        context.update({
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
    except TrainingAlreadyRunning:
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
    if tab not in ('final', 'early'):
        raise Http404
    run = get_object_or_404(TrainingRun, pk=run_id, status='finished')
    content = workbook(run, tab, _filters(request), _int(request.GET.get('checkpoint')))
    response = HttpResponse(
        content,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    response['Content-Disposition'] = f'attachment; filename="categorization-run-{run.id}-{tab}.xlsx"'
    return response
