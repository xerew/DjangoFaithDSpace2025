from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.contrib import messages
from functools import wraps
from django.utils.html import strip_tags
from html import unescape
from .models import Organization, JoinRequest, Announcement, OrgChatMessage
from django.contrib.auth.models import User
from .forms import OrganizationForm, AnnouncementForm
from authoringtool.models import Language
from django.utils.text import slugify

from .reports import build_organization_report
from .statistics import get_organization_statistics


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


def strip_html_tags(html_content):
    text = strip_tags(html_content)
    return unescape(text)

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

@login_required
def create_organization(request):
    if request.method == 'POST':
        form = OrganizationForm(request.POST, request.FILES)
        if form.is_valid():
            organization = form.save(commit=False)
            organization.created_by = request.user  # Set the user as the creator
            organization.save()
            
            # Add the creator as both admin and member
            organization.admins.add(request.user)  # Add the creator as an admin
            organization.members.add(request.user)  # Add the creator as a member
            messages.success(request, f"Organization '{organization.name}' created successfully.")
            return redirect('organization_detail', org_id=organization.id)
    else:
        form = OrganizationForm()
    
    return render(request, 'organization/create_organization.html', {
        'form': form,
        'languages': Language.objects.all(),
    })

@login_required
def organization_detail(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    is_site_admin = request.user.is_staff or request.user.is_superuser
    is_member = organization.members.filter(id=request.user.id).exists()
    is_admin = organization.admins.filter(id=request.user.id).exists() or is_site_admin

    join_request = None
    if not is_member:
        join_request = JoinRequest.objects.filter(user=request.user, organization=organization).first()

    pending_requests = None
    if is_admin:
        pending_requests = JoinRequest.objects.filter(
            organization=organization, status='pending'
        ).select_related('user')

    announcements_qs = organization.announcements.select_related('created_by')
    announcements_paginator = Paginator(announcements_qs, 5)
    announcements_page = announcements_paginator.get_page(request.GET.get('page'))
    announcement_page_range = _smart_page_range(announcements_page)
    organization_stats = get_organization_statistics(organization)

    return render(request, 'organization/organization_detail.html', {
        'organization': organization,
        'is_member': is_member,
        'is_admin': is_admin,
        'join_request': join_request,
        'pending_requests': pending_requests,
        'announcements': announcements_page,
        'announcement_page_range': announcement_page_range,
        'organization_stats': organization_stats,
    })


@require_GET
@login_required
def download_organization_report(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    report = build_organization_report(organization)
    filename = slugify(organization.short_name or organization.name) or f'organization-{org_id}'
    response = HttpResponse(
        report.getvalue(),
        content_type=(
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ),
    )
    response['Content-Disposition'] = (
        f'attachment; filename="{filename}-data-report.xlsx"'
    )
    return response


@require_POST
@login_required
def request_to_join(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    if organization.members.filter(id=request.user.id).exists():
        return redirect('organization_detail', org_id=org_id)

    join_request, created = JoinRequest.objects.get_or_create(
        user=request.user,
        organization=organization,
        defaults={'status': 'pending'}
    )
    if not created and join_request.status == 'rejected':
        join_request.status = 'pending'
        join_request.reviewed_by = None
        join_request.reviewed_at = None
        join_request.save()
        messages.success(request, "Your join request has been re-submitted.")
    elif created:
        messages.success(request, "Your request to join has been submitted.")
    return redirect('organization_detail', org_id=org_id)


@require_POST
@login_required
def approve_join_request(request, request_id):
    from django.utils import timezone
    join_req = get_object_or_404(JoinRequest, id=request_id)
    organization = join_req.organization

    is_site_admin = request.user.is_staff or request.user.is_superuser
    if request.user not in organization.admins.all() and not is_site_admin:
        return redirect('organization_detail', org_id=organization.id)

    join_req.status = 'approved'
    join_req.reviewed_by = request.user
    join_req.reviewed_at = timezone.now()
    join_req.save()
    organization.members.add(join_req.user)
    messages.success(request, f"{join_req.user.get_full_name() or join_req.user.username} has been added to {organization.name}.")
    return redirect('organization_detail', org_id=organization.id)


@require_POST
@login_required
def reject_join_request(request, request_id):
    from django.utils import timezone
    join_req = get_object_or_404(JoinRequest, id=request_id)
    organization = join_req.organization

    is_site_admin = request.user.is_staff or request.user.is_superuser
    if request.user not in organization.admins.all() and not is_site_admin:
        return redirect('organization_detail', org_id=organization.id)

    join_req.status = 'rejected'
    join_req.reviewed_by = request.user
    join_req.reviewed_at = timezone.now()
    join_req.save()
    messages.info(request, f"Request from {join_req.user.get_full_name() or join_req.user.username} has been rejected.")
    return redirect('organization_detail', org_id=organization.id)

@login_required
def add_member_to_org(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    if request.user not in organization.admins.all() and not is_site_admin:
        return redirect('organization_detail', org_id=org_id)

    users = None
    search_performed = False

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()

        # Perform the search
        if username or first_name or last_name:
            users = User.objects.all()
            if username:
                users = users.filter(username__icontains=username)
            if first_name:
                users = users.filter(first_name__icontains=first_name)
            if last_name:
                users = users.filter(last_name__icontains=last_name)

            search_performed = True

    return render(request, 'organization/add_member.html', {
        'organization': organization,
        'users': users,
        'search_performed': search_performed
    })

@require_POST
@login_required
def add_member_to_org_confirm(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    can_manage = request.user in organization.admins.all() or is_site_admin
    if can_manage and user not in organization.members.all():
        organization.members.add(user)
        messages.success(request, f"{user.get_full_name() or user.username} has been added to {organization.name}.")
        return redirect('organization_detail', org_id=org_id)

    messages.error(request, "Could not add member. They may already be a member.")
    return redirect('organization_detail', org_id=org_id)


@require_POST
@login_required
def make_admin(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    if request.user in organization.admins.all() or is_site_admin:
        organization.admins.add(user)
        messages.success(request, f"{user.get_full_name() or user.username} is now an admin.")
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)


@require_POST
@login_required
def delete_organization(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    if request.user in organization.admins.all() or is_site_admin:
        org_name = organization.name
        organization.delete()
        messages.success(request, f"Organization '{org_name}' has been deleted.")
        return redirect('list_organizations')

    return redirect('organization_detail', org_id=org_id)

@require_POST
@login_required
def promote_admin(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    can_manage = request.user in organization.admins.all() or is_site_admin
    if can_manage and user in organization.members.all():
        organization.admins.add(user)
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)

@require_POST
@login_required
def demote_admin(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    can_manage = request.user in organization.admins.all() or is_site_admin
    if can_manage and user != request.user:
        organization.admins.remove(user)
        messages.success(request, f"{user.get_full_name() or user.username} has been demoted from admin.")
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)

@require_POST
@login_required
def remove_member(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    can_manage = request.user in organization.admins.all() or is_site_admin
    if can_manage and user != request.user:
        if user in organization.admins.all():
            return redirect('organization_detail', org_id=org_id)
        organization.members.remove(user)
        messages.success(request, f"{user.get_full_name() or user.username} has been removed from {organization.name}.")
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)

@login_required
def edit_organization(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)

    is_site_admin = request.user.is_staff or request.user.is_superuser
    if request.user not in organization.admins.all() and not is_site_admin:
        return redirect('organization_detail', org_id=org_id)

    if request.method == 'POST':
        form = OrganizationForm(request.POST, request.FILES, instance=organization)
        if form.is_valid():
            form.save()
            messages.success(request, "Organization updated successfully.")
            return redirect('organization_detail', org_id=org_id)
    else:
        form = OrganizationForm(instance=organization)

    return render(request, 'organization/edit_organization.html', {
        'form': form,
        'organization': organization,
        'languages': Language.objects.all(),
    })

@login_required
def list_organizations(request):
    from django.db.models import Count, Q as Qdb
    query    = request.GET.get('q', '').strip()
    language = request.GET.get('language', '').strip()
    sort     = request.GET.get('sort', 'name')

    qs = Organization.objects.annotate(member_count=Count('members', distinct=True))

    if query:
        qs = qs.filter(Qdb(name__icontains=query) | Qdb(short_name__icontains=query) | Qdb(country__icontains=query))
    if language:
        qs = qs.filter(language__iexact=language)

    sort_options = {'name': 'name', '-name': '-name', 'country': 'country', '-members': '-member_count', 'members': 'member_count'}
    qs = qs.order_by(sort_options.get(sort, 'name'))

    user = request.user
    all_orgs   = list(qs)
    my_orgs    = [o for o in all_orgs if o.members.filter(id=user.id).exists()]
    other_orgs = [o for o in all_orgs if not o.members.filter(id=user.id).exists()]

    languages = Organization.objects.values_list('language', flat=True).distinct().order_by('language')

    return render(request, 'organization/list_organizations.html', {
        'my_orgs': my_orgs,
        'other_orgs': other_orgs,
        'languages': languages,
        'query': query,
        'selected_language': language,
        'sort': sort,
    })


def _is_org_admin(user, organization):
    is_site_admin = user.is_staff or user.is_superuser
    return organization.admins.filter(id=user.id).exists() or is_site_admin


@login_required
def create_announcement(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    if not _is_org_admin(request.user, organization):
        return redirect('organization_detail', org_id=org_id)

    if request.method == 'POST':
        form = AnnouncementForm(request.POST)
        if form.is_valid():
            announcement = form.save(commit=False)
            announcement.organization = organization
            announcement.created_by = request.user
            announcement.plain_text = strip_html_tags(announcement.body)
            announcement.save()
            messages.success(request, "Announcement posted.")
            return redirect('organization_detail', org_id=org_id)
    else:
        form = AnnouncementForm()

    return render(request, 'organization/create_announcement.html', {
        'form': form,
        'organization': organization,
    })


@login_required
def edit_announcement(request, org_id, announcement_id):
    organization = get_object_or_404(Organization, id=org_id)
    announcement = get_object_or_404(Announcement, id=announcement_id, organization=organization)
    if not _is_org_admin(request.user, organization):
        return redirect('organization_detail', org_id=org_id)

    if request.method == 'POST':
        form = AnnouncementForm(request.POST, instance=announcement)
        if form.is_valid():
            updated = form.save(commit=False)
            updated.plain_text = strip_html_tags(updated.body)
            updated.save()
            messages.success(request, "Announcement updated.")
            return redirect('organization_detail', org_id=org_id)
    else:
        form = AnnouncementForm(instance=announcement)

    return render(request, 'organization/edit_announcement.html', {
        'form': form,
        'organization': organization,
        'announcement': announcement,
    })


@require_POST
@login_required
def delete_announcement(request, org_id, announcement_id):
    organization = get_object_or_404(Organization, id=org_id)
    announcement = get_object_or_404(Announcement, id=announcement_id, organization=organization)
    if _is_org_admin(request.user, organization):
        announcement.delete()
        messages.success(request, "Announcement deleted.")
    return redirect('organization_detail', org_id=org_id)


@login_required
def org_chat(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    if not organization.members.filter(id=request.user.id).exists():
        return redirect('organization_detail', org_id=org_id)

    chat_messages = organization.chat_messages.select_related('sender')
    return render(request, 'organization/org_chat.html', {
        'organization': organization,
        'chat_messages': chat_messages,
    })


@require_POST
@login_required
def send_org_chat_message(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    if not organization.members.filter(id=request.user.id).exists():
        return JsonResponse({'success': False, 'error': 'Not a member.'}, status=403)

    body = (request.POST.get('body') or '').strip()
    if not body:
        return JsonResponse({'success': False, 'error': 'Message cannot be empty.'}, status=400)

    msg = OrgChatMessage.objects.create(organization=organization, sender=request.user, body=body)
    return JsonResponse({
        'success': True,
        'message': {
            'id': msg.id,
            'body': msg.body,
            'created_at': msg.created_at.strftime('%d %b, %H:%M'),
            'sender_id': msg.sender_id,
            'sender_name': msg.sender.get_full_name() or msg.sender.username,
        },
    })


@require_GET
@login_required
def org_chat_poll(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)
    if not organization.members.filter(id=request.user.id).exists():
        return JsonResponse({'success': False, 'error': 'Not a member.'}, status=403)

    try:
        since_id = int(request.GET.get('since_id', '0'))
    except ValueError:
        since_id = 0

    new_messages = organization.chat_messages.select_related('sender').filter(id__gt=since_id)
    return JsonResponse({
        'success': True,
        'messages': [
            {
                'id': m.id,
                'body': m.body,
                'created_at': m.created_at.strftime('%d %b, %H:%M'),
                'sender_id': m.sender_id,
                'sender_name': m.sender.get_full_name() or m.sender.username,
            }
            for m in new_messages
        ],
    })


@login_required
def announcement_detail(request, org_id, announcement_id):
    organization = get_object_or_404(Organization, id=org_id)
    announcement = get_object_or_404(Announcement, id=announcement_id, organization=organization)
    is_admin = _is_org_admin(request.user, organization)

    return render(request, 'organization/announcement_detail.html', {
        'organization': organization,
        'announcement': announcement,
        'is_admin': is_admin,
    })
