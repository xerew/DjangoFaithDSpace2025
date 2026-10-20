from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib.auth.models import User, Group
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.http import require_POST
from django.db import transaction
from django.core.paginator import Paginator
from django.db.models import Count, ProtectedError, Q
from django.conf import settings
from django.utils.html import strip_tags
from django.utils import timezone
from functools import wraps
from authoringtool.models import Language, Scenario, Simulation, ExperimentLL, VRARExperiment
from organization.models import Organization
from studentview.help_labels import HELP_LABELS

from .bulk_email import (
    absolutize_content_urls,
    eligible_teacher_users,
    resolve_campaign_recipients,
    unique_valid_email_users,
)
from .models import BulkEmailCampaign, BulkEmailDelivery, EmailOutboxState, MaintenanceNotice


def staff_required(view_func):
    @wraps(view_func)
    @login_required
    def _wrapped(request, *args, **kwargs):
        if not (request.user.is_staff or request.user.is_superuser):
            return HttpResponseForbidden("Access denied.")
        return view_func(request, *args, **kwargs)
    return _wrapped


@staff_required
def admin_dashboard(request):
    now = timezone.now()
    users = list(User.objects.filter(
        Q(groups__isnull=False) | Q(is_staff=True) | Q(is_superuser=True)
    ).distinct().prefetch_related(
        'groups', 'admin_of_organizations', 'member_of_organizations',
    ).order_by('username'))
    for user in users:
        # Organizations the user administers or belongs to, for the user table.
        admin_ids = {org.id for org in user.admin_of_organizations.all()}
        orgs = {
            org.id: org
            for org in [*user.admin_of_organizations.all(), *user.member_of_organizations.all()]
        }
        user.organization_rows = [
            {'org': org, 'is_admin': org.id in admin_ids}
            for org in sorted(orgs.values(), key=lambda org: org.short_name.casefold())
        ]
    groups = Group.objects.annotate(member_count=Count('user')).order_by('name')
    agg = User.objects.aggregate(
        total=Count('id'),
        active=Count('id', filter=Q(is_active=True)),
        staff=Count('id', filter=Q(is_staff=True)),
        teachers=Count('id', filter=Q(groups__name__iexact='teachers')),
        no_role=Count('id', filter=Q(groups__isnull=True, is_staff=False, is_superuser=False)),
    )
    stats = {
        'total': agg['total'],
        'active': agg['active'],
        'staff': agg['staff'],
        'inactive': agg['total'] - agg['active'],
        'teachers': agg['teachers'],
        'no_role': agg['no_role'],
    }
    is_superuser = request.user.is_superuser
    simulations = list(Simulation.objects.all().order_by('language', 'name'))
    remote_labs = list(ExperimentLL.objects.all().order_by('name'))
    vr_labs = list(VRARExperiment.objects.all().order_by('name'))
    sim_languages = sorted({s.language for s in simulations if s.language})
    email_teachers = list(
        eligible_teacher_users()
        .prefetch_related('member_of_organizations')
        .order_by('first_name', 'last_name', 'username')
    )
    email_teacher_count = len(unique_valid_email_users(eligible_teacher_users()))
    email_organizations = (
        Organization.objects.annotate(
            email_teacher_count=Count(
                'members',
                filter=(
                    Q(members__is_active=True)
                    & Q(members__groups__name__iexact='teachers')
                    & ~Q(members__email='')
                ),
                distinct=True,
            )
        )
        .order_by('name')
    )
    recent_email_campaigns = (
        BulkEmailCampaign.objects.select_related('created_by')
        .prefetch_related('organizations')[:10]
    )
    outbox_state = EmailOutboxState.load()
    maintenance_notices = list(
        MaintenanceNotice.objects.select_related('created_by')
        .order_by('-starts_at', '-id')[:10]
    )
    return render(request, 'accounts/admin_dashboard.html', {
        'all_users': users,
        'groups': groups,
        'stats': stats,
        'is_superuser': is_superuser,
        'simulations': simulations,
        'remote_labs': remote_labs,
        'vr_labs': vr_labs,
        'lab_count': len(simulations) + len(remote_labs) + len(vr_labs),
        'sim_languages': sim_languages,
        'email_teachers': email_teachers,
        'email_teacher_count': email_teacher_count,
        'email_organizations': email_organizations,
        'recent_email_campaigns': recent_email_campaigns,
        'outbox_state': outbox_state,
        'outbox_paused': outbox_state.is_paused(now),
        'email_hourly_limit': settings.EMAIL_CAMPAIGN_HOURLY_LIMIT,
        'maintenance_notices': maintenance_notices,
        'active_maintenance_count': MaintenanceNotice.objects.active(now).count(),
        'languages': language_rows(),
    })


def _bulk_email_selection(request):
    target_type = request.POST.get('target_type', '').strip()
    recipients, organizations = resolve_campaign_recipients(
        target_type,
        teacher_ids=request.POST.getlist('teacher_ids'),
        organization_ids=request.POST.getlist('organization_ids'),
    )
    return target_type, recipients, organizations


@require_POST
@staff_required
def admin_bulk_email_recipient_count(request):
    try:
        target_type, recipients, organizations = _bulk_email_selection(request)
    except ValueError as exc:
        return JsonResponse({'success': False, 'error': str(exc)}, status=400)
    return JsonResponse({
        'success': True,
        'target_type': target_type,
        'recipient_count': len(recipients),
        'organization_count': organizations.count(),
    })


@require_POST
@staff_required
def admin_send_bulk_email(request):
    if request.POST.get('confirmed') != 'true':
        return JsonResponse({
            'success': False,
            'error': 'Confirm the recipient count before sending.',
        }, status=400)

    subject = request.POST.get('subject', '').strip()
    body_html = request.POST.get('body_html', '').strip()
    if not subject:
        return JsonResponse({'success': False, 'error': 'Subject is required.'}, status=400)
    if len(subject) > 200:
        return JsonResponse({
            'success': False,
            'error': 'Subject must be 200 characters or fewer.',
        }, status=400)
    if '\r' in subject or '\n' in subject:
        return JsonResponse({'success': False, 'error': 'Subject contains invalid characters.'}, status=400)
    if not strip_tags(body_html).strip() and '<img' not in body_html.lower():
        return JsonResponse({'success': False, 'error': 'Message content is required.'}, status=400)

    try:
        target_type, recipients, organizations = _bulk_email_selection(request)
    except ValueError as exc:
        return JsonResponse({'success': False, 'error': str(exc)}, status=400)
    if not recipients:
        return JsonResponse({
            'success': False,
            'error': 'No active teachers with valid email addresses match this selection.',
        }, status=400)

    configured_site_url = getattr(settings, 'SITE_URL', '').strip()
    site_url = configured_site_url or request.build_absolute_uri('/').rstrip('/')
    body_html = absolutize_content_urls(body_html, site_url)
    with transaction.atomic():
        campaign = BulkEmailCampaign.objects.create(
            created_by=request.user,
            target_type=target_type,
            subject=subject,
            body_html=body_html,
            site_url=site_url,
            recipient_count=len(recipients),
        )
        campaign.recipients.set(recipients)
        campaign.organizations.set(organizations)
        # One outbox row per recipient; the beat task sends them within the
        # hourly cap (accounts.outbox.drain_outbox).
        BulkEmailDelivery.objects.bulk_create([
            BulkEmailDelivery(
                campaign=campaign,
                user=user,
                email=user.email.strip(),
                recipient_name=user.get_full_name() or user.get_username(),
            )
            for user in recipients
        ])

    return JsonResponse({
        'success': True,
        'campaign_id': campaign.id,
        'recipient_count': campaign.recipient_count,
        'status': campaign.status,
        'sent_count': campaign.sent_count,
        'failed_count': campaign.failed_count,
    })


DELIVERY_FILTERS = {
    'waiting': BulkEmailDelivery.PENDING_STATUSES,
    'sent': (BulkEmailDelivery.STATUS_SENT,),
    'failed': (BulkEmailDelivery.STATUS_FAILED,),
}


@staff_required
def admin_bulk_email_campaign(request, campaign_id):
    """Who received a campaign, who is waiting, and who failed (with why)."""
    campaign = get_object_or_404(
        BulkEmailCampaign.objects.select_related('created_by'), pk=campaign_id,
    )
    deliveries = campaign.deliveries.all()
    counts = {'all': deliveries.count()}
    for name, statuses in DELIVERY_FILTERS.items():
        counts[name] = deliveries.filter(status__in=statuses).count()

    status_filter = request.GET.get('status', 'all')
    if status_filter in DELIVERY_FILTERS:
        deliveries = deliveries.filter(status__in=DELIVERY_FILTERS[status_filter])
    else:
        status_filter = 'all'
    page_obj = Paginator(deliveries.order_by('id'), 100).get_page(request.GET.get('page'))

    try:
        resent = max(0, int(request.GET.get('resent', 0)))
    except ValueError:
        resent = 0
    outbox_state = EmailOutboxState.load()
    return render(request, 'accounts/admin_email_campaign.html', {
        'campaign': campaign,
        'page_obj': page_obj,
        'status_filter': status_filter,
        'counts': counts,
        'outbox_state': outbox_state,
        'outbox_paused': outbox_state.is_paused(timezone.now()),
        'resent': resent,
    })


@require_POST
@staff_required
def admin_bulk_email_resend(request, campaign_id):
    """Put chosen recipients back in the queue. Nothing is resent automatically."""
    campaign = get_object_or_404(BulkEmailCampaign, pk=campaign_id)
    rows = campaign.deliveries.filter(user__isnull=False)
    if request.POST.get('mode') == 'selected':
        rows = rows.filter(
            pk__in=[value for value in request.POST.getlist('delivery_ids') if value.isdigit()],
        ).exclude(status__in=BulkEmailDelivery.PENDING_STATUSES)
    else:
        rows = rows.filter(status=BulkEmailDelivery.STATUS_FAILED)

    resent = rows.update(status=BulkEmailDelivery.STATUS_WAITING, error='')
    if resent:
        campaign.refresh_counts()
    url = reverse('admin_bulk_email_campaign', args=[campaign.id])
    return redirect(f'{url}?resent={resent}')


@require_POST
@staff_required
def admin_edit_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    first_name = request.POST.get('first_name', '').strip()
    last_name = request.POST.get('last_name', '').strip()
    email = request.POST.get('email', '').strip()
    is_staff = request.POST.get('is_staff') == 'true'
    is_superuser = request.POST.get('is_superuser') == 'true'
    group_ids = request.POST.getlist('groups')

    if is_superuser and not request.user.is_superuser:
        is_superuser = False

    if user == request.user and not is_staff:
        return JsonResponse({'success': False, 'error': 'Cannot remove your own staff status.'})

    user.first_name = first_name
    user.last_name = last_name
    user.email = email
    user.is_staff = is_staff
    update_fields = ['first_name', 'last_name', 'email', 'is_staff']
    if request.user.is_superuser:
        user.is_superuser = is_superuser
        update_fields.append('is_superuser')
    user.save(update_fields=update_fields)
    user.groups.set(Group.objects.filter(id__in=group_ids))
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot delete your own account.'})
    if user.is_superuser and not request.user.is_superuser:
        return JsonResponse({'success': False, 'error': 'Cannot delete a superuser.'})
    try:
        with transaction.atomic():
            user.delete()
    except ProtectedError:
        return JsonResponse({
            'success': False,
            'error': (
                'This account has student work (answers, scores or feedback) that '
                'must be kept. Use Suspend to block the login instead.'
            ),
        })
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_toggle_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot suspend your own account.'})
    if user.is_superuser and not request.user.is_superuser:
        return JsonResponse({'success': False, 'error': 'Cannot suspend a superuser account.'})
    user.is_active = not user.is_active
    user.save(update_fields=['is_active'])
    return JsonResponse({'success': True, 'is_active': user.is_active})


@require_POST
@staff_required
def admin_create_role(request):
    name = request.POST.get('name', '').strip()
    if not name:
        return JsonResponse({'success': False, 'error': 'Role name is required.'})
    if Group.objects.filter(name__iexact=name).exists():
        return JsonResponse({'success': False, 'error': 'A role with this name already exists.'})
    group = Group.objects.create(name=name)
    return JsonResponse({'success': True, 'id': group.id, 'name': group.name})


@require_POST
@staff_required
def admin_rename_role(request, role_id):
    group = get_object_or_404(Group, id=role_id)
    name = request.POST.get('name', '').strip()
    if not name:
        return JsonResponse({'success': False, 'error': 'Role name is required.'})
    if Group.objects.filter(name__iexact=name).exclude(id=role_id).exists():
        return JsonResponse({'success': False, 'error': 'A role with this name already exists.'})
    group.name = name
    group.save()
    return JsonResponse({'success': True, 'name': group.name})


@require_POST
@staff_required
def admin_delete_role(request, role_id):
    group = get_object_or_404(Group, id=role_id)
    if group.user_set.exists():
        return JsonResponse({'success': False, 'error': 'Remove all members before deleting this role.'})
    group.delete()
    return JsonResponse({'success': True})


# ── Languages ───────────────────────────────────────────────────────────────
# Scenarios and organizations store a language by name, so renaming a
# language also renames it there. Languages with built-in chat and help texts
# (studentview/help_labels.py) are matched by name and cannot be renamed.

LANGUAGE_NAME_MAX_LENGTH = 50  # Organization.language is the shortest column


def _built_in(name):
    return name.casefold() in {label.casefold() for label in HELP_LABELS}


def _usage_counts(model):
    counts = {}
    for row in model.objects.exclude(language__isnull=True).values('language').annotate(n=Count('id')):
        key = (row['language'] or '').strip().casefold()
        counts[key] = counts.get(key, 0) + row['n']
    return counts


def language_rows():
    scenarios = _usage_counts(Scenario)
    organizations = _usage_counts(Organization)
    return [
        {
            'language': language,
            'scenario_count': scenarios.get(language.name.casefold(), 0),
            'organization_count': organizations.get(language.name.casefold(), 0),
            'built_in': _built_in(language.name),
        }
        for language in Language.objects.order_by('name')
    ]


def _clean_language_name(raw, exclude_id=None):
    """Return (name, error)."""
    name = ' '.join((raw or '').split())
    if not name:
        return None, 'Language name is required.'
    if len(name) > LANGUAGE_NAME_MAX_LENGTH:
        return None, f'Language name must be at most {LANGUAGE_NAME_MAX_LENGTH} characters.'
    if Language.objects.filter(name__iexact=name).exclude(id=exclude_id).exists():
        return None, f'A language called "{name}" already exists.'
    return name, None


def _count(number, noun):
    return f"{number} {noun}{'' if number == 1 else 's'}"


@require_POST
@staff_required
def admin_create_language(request):
    name, error = _clean_language_name(request.POST.get('name'))
    if error:
        return JsonResponse({'success': False, 'error': error})
    language = Language.objects.create(name=name)
    return JsonResponse({
        'success': True, 'id': language.id, 'name': language.name,
        'built_in': _built_in(language.name),
    })


@require_POST
@staff_required
def admin_rename_language(request, language_id):
    language = get_object_or_404(Language, id=language_id)
    if _built_in(language.name):
        return JsonResponse({'success': False, 'error': (
            f'"{language.name}" has built-in chat and help texts tied to its name, '
            'so it cannot be renamed.'
        )})
    name, error = _clean_language_name(request.POST.get('name'), exclude_id=language.id)
    if error:
        return JsonResponse({'success': False, 'error': error})
    old_name = language.name
    with transaction.atomic():
        scenarios = Scenario.objects.filter(language__iexact=old_name).update(language=name)
        organizations = Organization.objects.filter(language__iexact=old_name).update(language=name)
        language.name = name
        language.save(update_fields=['name'])
    return JsonResponse({
        'success': True, 'name': name,
        'message': (
            f'Renamed to "{name}" and updated {_count(scenarios, "scenario")} '
            f'and {_count(organizations, "organization")}.'
        ),
    })


@require_POST
@staff_required
def admin_delete_language(request, language_id):
    language = get_object_or_404(Language, id=language_id)
    scenarios = Scenario.objects.filter(language__iexact=language.name).count()
    organizations = Organization.objects.filter(language__iexact=language.name).count()
    if scenarios or organizations:
        return JsonResponse({'success': False, 'error': (
            f'"{language.name}" is used by {_count(scenarios, "scenario")} and '
            f'{_count(organizations, "organization")}. Change their language first.'
        )})
    language.delete()
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_impersonate(request, user_id):
    if request.session.get('impersonator_id'):
        return JsonResponse({'success': False, 'error': 'Cannot impersonate while already impersonating.'}, status=403)
    target = get_object_or_404(User, id=user_id)
    if target.is_superuser:
        return JsonResponse({'success': False, 'error': 'Cannot impersonate a superuser.'}, status=403)
    if target.is_staff and not request.user.is_superuser:
        return JsonResponse({'success': False, 'error': 'Staff can only impersonate non-staff users.'}, status=403)
    if target == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot impersonate yourself.'}, status=403)
    impersonator_id = request.user.id          # save before login() may flush session
    target.backend = 'django.contrib.auth.backends.ModelBackend'
    auth_login(request, target)                # switches session to target user
    request.session['impersonator_id'] = impersonator_id
    return JsonResponse({'success': True, 'redirect': '/'})


@require_POST
@login_required
def admin_impersonate_exit(request):
    impersonator_id = request.session.get('impersonator_id')
    if not impersonator_id:
        return redirect('index')
    original = User.objects.filter(id=impersonator_id).first()
    if original is None:
        # Impersonator account was deleted; clear the flag and redirect to login
        request.session.pop('impersonator_id', None)
        return redirect('login')
    original.backend = 'django.contrib.auth.backends.ModelBackend'
    auth_login(request, original)              # session flush normally clears impersonator_id
    request.session.pop('impersonator_id', None)  # defensive: clear if flush was skipped
    return redirect('admin_dashboard')
