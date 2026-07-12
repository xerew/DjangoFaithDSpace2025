from functools import wraps

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .models import Message


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


def _is_valid_target(user):
    return user.is_staff or user.is_superuser or user.groups.filter(name='teachers').exists()


@group_required('teachers')
def message_threads(request):
    from organization.models import Organization

    me = request.user
    sent_to = Message.objects.filter(sender=me).values_list('recipient_id', flat=True)
    received_from = Message.objects.filter(recipient=me).values_list('sender_id', flat=True)
    partner_ids = set(sent_to) | set(received_from)

    threads = []
    for partner in User.objects.filter(id__in=partner_ids):
        latest = Message.objects.filter(
            Q(sender=me, recipient=partner) | Q(sender=partner, recipient=me)
        ).order_by('-created_at').first()
        unread = Message.objects.filter(sender=partner, recipient=me, read_at__isnull=True).count()
        threads.append({'partner': partner, 'latest': latest, 'unread': unread})
    threads.sort(key=lambda t: (t['latest'].created_at, t['latest'].id), reverse=True)

    organizations = Organization.objects.filter(members=me).order_by('name')

    return render(request, 'messaging/thread_list.html', {'threads': threads, 'organizations': organizations})


@group_required('teachers')
def thread(request, user_id):
    partner = get_object_or_404(User, pk=user_id)
    if partner == request.user or not _is_valid_target(partner):
        return redirect('message_threads')

    Message.objects.filter(
        sender=partner, recipient=request.user, read_at__isnull=True
    ).update(read_at=timezone.now())

    thread_messages = Message.objects.filter(
        Q(sender=request.user, recipient=partner) | Q(sender=partner, recipient=request.user)
    )
    return render(request, 'messaging/thread.html', {'partner': partner, 'thread_messages': thread_messages})


@require_POST
@group_required('teachers')
def send_message(request):
    recipient_id = request.POST.get('recipient_id')
    body = (request.POST.get('body') or '').strip()

    if not body:
        return JsonResponse({'success': False, 'error': 'Message cannot be empty.'}, status=400)

    recipient = get_object_or_404(User, pk=recipient_id)
    if recipient == request.user or not _is_valid_target(recipient):
        return JsonResponse({'success': False, 'error': 'Invalid recipient.'}, status=400)

    msg = Message.objects.create(sender=request.user, recipient=recipient, body=body)
    return JsonResponse({
        'success': True,
        'message': {
            'id': msg.id,
            'body': msg.body,
            'created_at': msg.created_at.strftime('%d %b %Y, %H:%M'),
            'sender_id': msg.sender_id,
        },
    })


@require_GET
@group_required('teachers')
def unread_status(request):
    unread_qs = Message.objects.filter(recipient=request.user, read_at__isnull=True)
    latest = unread_qs.order_by('-created_at').first()

    data = {'unread_count': unread_qs.count(), 'latest': None}
    if latest:
        data['latest'] = {
            'id': latest.id,
            'sender_id': latest.sender_id,
            'sender_name': latest.sender.get_full_name() or latest.sender.username,
            'snippet': latest.body[:80],
            'created_at': latest.created_at.isoformat(),
        }
    return JsonResponse(data)
