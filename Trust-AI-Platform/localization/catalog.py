"""Helpers for reading and checking the gettext catalogs in ``locale/``."""
import re
from collections import Counter
from pathlib import Path

import polib
from django.conf import settings

_PY_FORMAT = re.compile(r'%(?:\([A-Za-z_]\w*\))?[#0\- +]*\d*(?:\.\d+)?[sdifrc]')
_BRACE = re.compile(r'\{[A-Za-z_0-9]*\}')
_TAG = re.compile(r'<\s*/?\s*([A-Za-z][A-Za-z0-9-]*)[^<>]*?>')

_cache = {}


def po_path(code):
    return Path(settings.BASE_DIR) / 'locale' / code / 'LC_MESSAGES' / 'django.po'


def load_po(code):
    return polib.pofile(str(po_path(code)), encoding='utf-8')


def placeholders(text):
    """Count python-format, brace and HTML tag-name tokens in ``text``."""
    text = (text or '').replace('%%', '')
    tokens = Counter(_PY_FORMAT.findall(text))
    tokens.update(_BRACE.findall(text))
    tokens.update(m.lower() for m in _TAG.findall(text))
    return tokens


def placeholder_problems(msgid, msgstr):
    source, target = placeholders(msgid), placeholders(msgstr)
    problems = []
    for token in sorted((source - target).elements()):
        problems.append(f'missing {token}')
    for token in sorted((target - source).elements()):
        problems.append(f'unexpected {token}')
    return problems


def marked_msgids(code='el'):
    """Every msgid and msgid_plural in the catalog (cached per file mtime)."""
    path = po_path(code)
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _cache:
        ids = set()
        for entry in polib.pofile(str(path), encoding='utf-8'):
            ids.add(entry.msgid)
            if entry.msgid_plural:
                ids.add(entry.msgid_plural)
        _cache.clear()
        _cache[key] = ids
    return _cache[key]


def untranslated(code):
    """Entries with an empty translation (any plural form) or flagged fuzzy."""
    result = []
    for entry in load_po(code):
        if entry.obsolete:
            continue
        if 'fuzzy' in entry.flags:
            result.append(entry)
        elif entry.msgid_plural:
            if not all(entry.msgstr_plural.values()):
                result.append(entry)
        elif not entry.msgstr:
            result.append(entry)
    return result
