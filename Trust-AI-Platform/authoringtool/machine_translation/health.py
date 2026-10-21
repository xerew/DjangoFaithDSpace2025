"""Whether the translation service can be used right now.

Only the Celery containers reach the NTUA machine (through the SSH tunnel),
so Celery checks the service and caches the result; the web side reads it.
"""

from datetime import datetime, timedelta

import requests
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

STATUS_KEY = 'authoringtool:translation-service-status'
STATUS_TIMEOUT = 600
STALE_AFTER = timedelta(seconds=180)


def record_service_status(reachable, model_present):
    cache.set(STATUS_KEY, {
        'reachable': reachable,
        'model_present': model_present,
        'checked_at': timezone.now().isoformat(),
    }, STATUS_TIMEOUT)


def check_translation_service():
    """Ask Ollama which models it has (Celery side) and cache the answer."""
    try:
        response = requests.get(f'{settings.OLLAMA_URL}/api/tags', timeout=3)
        response.raise_for_status()
        names = {model.get('name') for model in response.json().get('models', [])}
        reachable = True
        model_present = settings.LLM_MACHINE_TRANSLATION_MODEL in names
    except (requests.RequestException, ValueError):
        reachable = model_present = False
    record_service_status(reachable, model_present)
    return {'reachable': reachable, 'model_present': model_present}


def translation_service_available(now=None):
    """True when a recent check found Ollama with the translation model."""
    status = cache.get(STATUS_KEY)
    if not status:
        return False
    now = now or timezone.now()
    checked_at = datetime.fromisoformat(status['checked_at'])
    if now - checked_at > STALE_AFTER:
        return False
    return bool(status['reachable'] and status['model_present'])
