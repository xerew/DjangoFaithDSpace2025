"""Admin page for choosing which NTUA address the Ollama tunnel uses."""

from django.contrib import admin, messages
from django.http import HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils import timezone

from . import ollama_tunnel
from .machine_translation.health import server_status


def _when(value):
    return timezone.localtime(value).strftime('%d %b %Y, %H:%M:%S') if value else ''


def _connection_summary(status):
    """One sentence about the tunnel, plus the last SSH error if useful."""
    if not status['reporting']:
        last = (
            f" Its last report was at {_when(status['updated'])}."
            if status['updated'] else ''
        )
        return {
            'level': 'error',
            'text': f'The tunnel is not reporting.{last}',
            'hint': (
                'Check that the ollama-tunnel container is running '
                '(docker compose ps ollama-tunnel).'
            ),
        }
    if status['state'] == 'connected':
        since = f" since {_when(status['since'])}" if status['since'] else ''
        return {
            'level': 'success',
            'text': f"Connected to {status['host']}{since}.",
            'hint': '',
        }
    if status['state'] == 'stopped':
        return {
            'level': 'error',
            'text': 'The tunnel has stopped.',
            'hint': 'Start it with: docker compose up -d ollama-tunnel',
        }
    if status['state'] == 'connecting':
        return {
            'level': 'warning',
            'text': f"Trying {status['host']}...",
            'hint': '',
        }
    return {
        'level': 'error',
        'text': 'None of the addresses can be reached right now. '
                'The tunnel keeps trying.',
        'hint': '',
    }


def ai_server_connection_view(request):
    url = reverse('ai_server_connection')
    status = ollama_tunnel.read_status()

    if request.method == 'POST':
        target = request.POST.get('target', '')
        try:
            ollama_tunnel.set_target(target, request.user.get_username())
        except ValueError:
            messages.error(
                request,
                f'"{target}" is not one of the addresses this server allows. '
                'Allowed addresses come from OLLAMA_SSH_HOSTS in the server\'s '
                '.env file.',
            )
        except OSError as error:
            messages.error(request, f'Could not save the choice: {error}')
        else:
            choice = (
                'Automatic' if target == ollama_tunnel.AUTO else target
            )
            messages.success(
                request,
                f'Saved: {choice}. The tunnel switches within about '
                '10 seconds; refresh this page to see it.',
            )
        return HttpResponseRedirect(url)

    target = ollama_tunnel.read_target()
    ollama = server_status([])
    context = {
        **admin.site.each_context(request),
        'title': 'AI server connection',
        'status': status,
        'summary': _connection_summary(status),
        'last_message': status['message'],
        'target': target,
        'target_changed_at': _when(target['changed_at']),
        'pending': (
            status['reporting'] and status['mode']
            and status['mode'] != target['target']
        ),
        'auto': ollama_tunnel.AUTO,
        'ollama_ok': ollama['state'] != 'offline',
        'ollama_checked_at': _when(ollama['checked_at']),
        'ollama_offline_since': _when(ollama['offline_since']),
    }
    return TemplateResponse(
        request, 'admin/ai_server_connection.html', context,
    )
