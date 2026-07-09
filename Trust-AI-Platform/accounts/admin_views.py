from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User, Group
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.http import require_POST
from django.db.models import Count, Q
from functools import wraps
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment


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
    groups = Group.objects.annotate(member_count=Count('user')).order_by('name')
    agg = User.objects.aggregate(
        total=Count('id'),
        active=Count('id', filter=Q(is_active=True)),
        staff=Count('id', filter=Q(is_staff=True)),
    )
    stats = {
        'total': agg['total'],
        'active': agg['active'],
        'staff': agg['staff'],
        'inactive': agg['total'] - agg['active'],
    }
    is_superuser = request.user.is_superuser
    simulations = list(Simulation.objects.all().order_by('language', 'name'))
    remote_labs = list(ExperimentLL.objects.all().order_by('name'))
    vr_labs = list(VRARExperiment.objects.all().order_by('name'))
    sim_languages = sorted({s.language for s in simulations if s.language})
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
    user.delete()
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
