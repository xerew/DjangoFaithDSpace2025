from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User, Group
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.http import require_POST
from functools import wraps


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
    users = User.objects.all().prefetch_related('groups').order_by('username')
    groups = Group.objects.all().prefetch_related('user_set').order_by('name')
    stats = {
        'total': users.count(),
        'active': users.filter(is_active=True).count(),
        'staff': users.filter(is_staff=True).count(),
        'inactive': users.filter(is_active=False).count(),
    }
    is_superuser = request.user.is_superuser
    return render(request, 'accounts/admin_dashboard.html', {
        'all_users': users,
        'groups': groups,
        'stats': stats,
        'is_superuser': is_superuser,
    })


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
    if request.user.is_superuser:
        user.is_superuser = is_superuser
    user.save(update_fields=['first_name', 'last_name', 'email', 'is_staff', 'is_superuser'])
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
    user.delete()
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_toggle_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot suspend your own account.'})
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


@require_POST
@staff_required
def admin_impersonate(request, user_id):
    target = get_object_or_404(User, id=user_id)
    if target.is_superuser:
        return JsonResponse({'success': False, 'error': 'Cannot impersonate a superuser.'})
    if target == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot impersonate yourself.'})
    impersonator_id = request.user.id          # save before login() may flush session
    target.backend = 'django.contrib.auth.backends.ModelBackend'
    auth_login(request, target)                # switches session to target user
    request.session['impersonator_id'] = impersonator_id
    return JsonResponse({'success': True, 'redirect': '/'})


@require_POST
def admin_impersonate_exit(request):
    impersonator_id = request.session.get('impersonator_id')
    if not impersonator_id:
        return redirect('index')
    original = get_object_or_404(User, id=impersonator_id)
    original.backend = 'django.contrib.auth.backends.ModelBackend'
    auth_login(request, original)              # session flush clears impersonator_id
    return redirect('admin_dashboard')
