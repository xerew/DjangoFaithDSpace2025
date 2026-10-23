"""Whether the AI server (Ollama on the NTUA machine) can be used right now.

Only the Celery containers reach the NTUA machine (through the SSH tunnel),
so Celery checks the server every minute and caches what it found; the web
side reads it. Machine translation and AI proposals share this one check.
"""

from datetime import datetime, timedelta

import requests
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from django.utils.translation import gettext as _, ngettext

STATUS_KEY = 'authoringtool:translation-service-status'
# Kept long so "offline since" survives; freshness is judged by checked_at.
STATUS_TIMEOUT = 7 * 24 * 3600
STALE_AFTER = timedelta(seconds=180)

OK = 'ok'
OFFLINE = 'offline'
MISSING_MODELS = 'missing_models'


def proposal_models():
    """Models a proposal / LLM context run may need, in the order they're used."""
    return [
        settings.LLM_PROPOSAL_MODEL,
        settings.LLM_IMAGE_DESCRIPTION_MODEL,
        settings.LLM_TRANSLATION_MODEL,
    ]


def record_service_status(reachable, model_present=False, models=None):
    """Cache what a check found. `models` is the server's model list; when
    omitted, `model_present` stands for the machine-translation model only."""
    if models is None:
        models = [settings.LLM_MACHINE_TRANSLATION_MODEL] if model_present else []
    now = timezone.now()
    previous = cache.get(STATUS_KEY) or {}
    offline_since = None
    if not reachable:
        offline_since = previous.get('offline_since') or now.isoformat()
    cache.set(STATUS_KEY, {
        'reachable': reachable,
        'models': sorted(models),
        'checked_at': now.isoformat(),
        'offline_since': offline_since,
    }, STATUS_TIMEOUT)


def check_ollama():
    """Ask Ollama which models it has (Celery side) and cache the answer."""
    try:
        response = requests.get(f'{settings.OLLAMA_URL}/api/tags', timeout=3)
        response.raise_for_status()
        models = [model.get('name') for model in response.json().get('models', [])]
        reachable = True
    except (requests.RequestException, ValueError):
        models = []
        reachable = False
    record_service_status(reachable, models=models)
    return {'reachable': reachable, 'models': models}


def check_translation_service():
    status = check_ollama()
    return {
        'reachable': status['reachable'],
        'model_present': settings.LLM_MACHINE_TRANSLATION_MODEL in status['models'],
    }


def _parse(value):
    return datetime.fromisoformat(value) if value else None


def server_status(required_models, now=None):
    """State of the AI server for a feature needing `required_models`."""
    now = now or timezone.now()
    status = cache.get(STATUS_KEY) or {}
    checked_at = _parse(status.get('checked_at'))
    offline_since = _parse(status.get('offline_since'))
    missing = [model for model in required_models if model not in status.get('models', [])]

    if not checked_at or now - checked_at > STALE_AFTER or not status.get('reachable'):
        state = OFFLINE
    elif missing:
        state = MISSING_MODELS
    else:
        state = OK
    return {
        'state': state,
        'missing': missing if state == MISSING_MODELS else [],
        'checked_at': checked_at,
        'offline_since': offline_since,
    }


def _when(value):
    return timezone.localtime(value).strftime('%d %b %Y, %H:%M')


def proposal_server_status(now=None):
    """Server state plus the banner text for the Metrics & AI page."""
    status = server_status(proposal_models(), now)
    if status['state'] == MISSING_MODELS:
        status['title'] = ngettext(
            'The AI server is missing a model proposals need: %(models)s.',
            'The AI server is missing models proposals need: %(models)s.',
            len(status['missing']),
        ) % {'models': ', '.join(status['missing'])}
    else:
        status['title'] = _(
            "The AI server is offline, so proposals and LLM context "
            "can't be generated right now."
        )

    since = (
        _when(status['offline_since'])
        if status['state'] == OFFLINE and status['offline_since'] else None
    )
    checked = _when(status['checked_at']) if status['checked_at'] else None
    if since and checked:
        first = _('Offline since %(since)s · last checked %(checked)s.') % {
            'since': since, 'checked': checked}
    elif since:
        first = _('Offline since %(since)s · not checked yet.') % {'since': since}
    elif checked:
        first = _('Last checked %(checked)s.') % {'checked': checked}
    else:
        first = _('Not checked yet.')
    again = _("It's checked again every minute.")
    status['detail'] = f'{first} {again}'
    return status


def translation_service_available(now=None):
    """True when a recent check found Ollama with the translation model."""
    return server_status([settings.LLM_MACHINE_TRANSLATION_MODEL], now)['state'] == OK
