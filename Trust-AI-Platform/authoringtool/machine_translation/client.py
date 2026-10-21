"""Ollama requests for machine translation of one unit at a time."""

import json

import requests
from django.conf import settings


class TranslationServiceUnavailable(Exception):
    """Ollama or the translation model cannot be reached right now."""


class TranslationTimeout(Exception):
    """The request took longer than MACHINE_TRANSLATION_TIMEOUT."""


class TranslationOutputInvalid(Exception):
    """The model answered, but not with the fields that were asked for."""


SYSTEM_PROMPT = (
    'You are a professional translator of school science lessons. '
    'Translate the values of the JSON object from {source} into {target}. '
    'Translate only human-readable text. Keep every HTML tag and attribute, '
    'every number, unit and formula (such as $...$ or \\(...\\)) exactly as '
    'written. Keep the same keys. Return only the JSON object.'
)


def _schema(fields):
    return {
        'type': 'object',
        'properties': {key: {'type': 'string'} for key in fields},
        'required': list(fields),
        'additionalProperties': False,
    }


def translate_fields(fields, source_language, target_language, context):
    """Translate the values of fields; raise one of the errors above on failure."""
    payload = {
        'model': settings.LLM_MACHINE_TRANSLATION_MODEL,
        'stream': False,
        'options': {'temperature': 0.2},
        'format': _schema(fields),
        'messages': [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT.format(
                    source=source_language, target=target_language,
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'Lesson context: {context}\n\n'
                    f'{json.dumps(fields, ensure_ascii=False)}'
                ),
            },
        ],
    }
    try:
        response = requests.post(
            f'{settings.OLLAMA_URL}/api/chat',
            json=payload,
            timeout=settings.MACHINE_TRANSLATION_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise TranslationTimeout(str(exc)) from exc
    except requests.RequestException as exc:
        raise TranslationServiceUnavailable(str(exc)) from exc

    try:
        envelope = response.json()
    except ValueError:
        envelope = {}
    if response.status_code >= 500 or response.status_code == 404:
        raise TranslationServiceUnavailable(
            envelope.get('error') or f'HTTP {response.status_code}'
        )
    if envelope.get('error'):
        if 'model' in envelope['error']:
            raise TranslationServiceUnavailable(envelope['error'])
        raise TranslationOutputInvalid(envelope['error'])

    content = (envelope.get('message') or {}).get('content', '')
    try:
        translated = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise TranslationOutputInvalid('the reply was not JSON') from exc
    if not isinstance(translated, dict) or not all(
        isinstance(value, str) for value in translated.values()
    ):
        raise TranslationOutputInvalid('the reply was not a set of text fields')
    return translated
