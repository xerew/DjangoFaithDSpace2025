from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from functools import wraps
from .models import Organization
from django.contrib.auth.models import User
from .forms import OrganizationForm

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

            return redirect('organization_detail', org_id=organization.id)
    else:
        form = OrganizationForm()
    
    return render(request, 'organization/create_organization.html', {'form': form})

@login_required
def organization_detail(request, org_id):
    organization = Organization.objects.get(id=org_id)
    return render(request, 'organization/organization_detail.html', {'organization': organization})

@login_required
def add_member_to_org(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)

    if request.user not in organization.admins.all():
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

@login_required
def add_member_to_org_confirm(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    if request.user in organization.admins.all() and user not in organization.members.all():
        organization.members.add(user)  # Add the user as a member
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)


@login_required
def make_admin(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    if request.user in organization.admins.all():
        organization.admins.add(user)  # Promote the user to admin
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_list')


@login_required
def delete_organization(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)

    if request.user in organization.admins.all():
        organization.delete()
        return redirect('list_organizations')

    return redirect('organization_detail', org_id=org_id)

@login_required
def promote_admin(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    # Only allow current admins to promote other members to admin
    if request.user in organization.admins.all() and user in organization.members.all():
        organization.admins.add(user)  # Promote the user to admin
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)

@login_required
def demote_admin(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    if request.user in organization.admins.all() and user != request.user:
        organization.admins.remove(user)  # Demote the user from admin
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)

@login_required
def remove_member(request, org_id, user_id):
    organization = get_object_or_404(Organization, id=org_id)
    user = get_object_or_404(User, id=user_id)

    if request.user in organization.admins.all() and user != request.user:
        # Prevent removing an admin directly
        if user in organization.admins.all():
            return redirect('organization_detail', org_id=org_id)  # Redirect if the user is still an admin
        
        # Remove the user from the organization's members
        organization.members.remove(user)
        return redirect('organization_detail', org_id=org_id)

    return redirect('organization_detail', org_id=org_id)

@login_required
def edit_organization(request, org_id):
    organization = get_object_or_404(Organization, id=org_id)

    # Check if the user is an admin of the organization
    if request.user not in organization.admins.all():
        return redirect('organization_detail', org_id=org_id)

    if request.method == 'POST':
        form = OrganizationForm(request.POST, request.FILES, instance=organization)
        if form.is_valid():
            form.save()
            return redirect('organization_detail', org_id=org_id)
    else:
        form = OrganizationForm(instance=organization)

    return render(request, 'organization/edit_organization.html', {'form': form, 'organization': organization})

@login_required
def list_organizations(request):
    organizations = Organization.objects.all()

    return render(request, 'organization/list_organizations.html', {
        'organizations': organizations
    })