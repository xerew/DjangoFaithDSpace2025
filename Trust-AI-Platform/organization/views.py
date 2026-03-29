from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST
from django.contrib import messages
from functools import wraps
from .models import Organization
from django.contrib.auth.models import User
from .forms import OrganizationForm
from authoringtool.models import Language

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
    if not organization.members.filter(id=request.user.id).exists() and not is_site_admin:
        return HttpResponseForbidden("You are not a member of this organization.")
    return render(request, 'organization/organization_detail.html', {'organization': organization})

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