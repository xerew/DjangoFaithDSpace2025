"""Helpers for reading and checking the gettext catalogs in ``locale/``."""
import re
import string
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


_FULL_TAG = re.compile(r'<[^<>]*>')

# Problem texts translation_problems adds besides 'missing X' / 'unexpected X'.
UNBALANCED_BRACKETS = 'unbalanced < and >'
EXTRA_DOUBLE_QUOTE = 'extra straight double quote'
STRAY_PERCENT = 'stray percent sign'
UNEXPECTED_LINE_BREAK = 'unexpected line break'
UNSAFE_BRACES = 'curly braces differ from the source'


def _full_tags(text):
    return Counter(' '.join(tag.split()) for tag in _FULL_TAG.findall(text or ''))


def _is_python_format(text):
    """True when Python %-formatting is applied to ``text`` (tokens or an escaped %%)."""
    return '%%' in text or bool(_PY_FORMAT.search(text.replace('%%', '')))


def _brace_fields(text):
    """The field names of ``text`` as a format string, or None if it has anything more.

    None covers text the format syntax cannot read (a lone brace) and fields
    with a format spec or a conversion. Attribute access and indexes stay in
    the field name, so they can only match a source that has them.
    """
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError:
        return None
    fields = Counter()
    for _literal, name, spec, conversion in parsed:
        if name is None:
            continue
        if spec or conversion:
            return None
        fields[name] += 1
    return fields


def _braces_match(msgid, msgstr):
    """True when ``msgstr`` has the source's brace fields, in any order, and no other braces.

    Call sites such as format_html raise KeyError on an unknown field, and a
    field like {name.__class__} prints internals. A source the format syntax
    cannot read has literal braces only, so just their number has to match.
    """
    if (msgstr.count('{'), msgstr.count('}')) != (msgid.count('{'), msgid.count('}')):
        return False
    source = _brace_fields(msgid)
    return source is None or _brace_fields(msgstr) == source


def translation_problems(msgid, msgstr):
    """``placeholder_problems`` plus the checks that keep a translation safe to render.

    Translations are output unescaped, and Python call sites %-format them,
    so besides the placeholders a translation must keep every HTML tag whole
    (attributes included), add no straight double quote (it could break out of
    an HTML attribute), add no lone % when the text is %-formatted, add no
    line break, and keep its curly braces as the source has them.
    """
    msgid, msgstr = msgid or '', msgstr or ''
    problems = placeholder_problems(msgid, msgstr)
    source_tags, target_tags = _full_tags(msgid), _full_tags(msgstr)
    for tag in sorted((source_tags - target_tags).elements()):
        problems.append(f'missing {tag}')
    for tag in sorted((target_tags - source_tags).elements()):
        problems.append(f'unexpected {tag}')
    if (msgstr.count('<'), msgstr.count('>')) != (msgid.count('<'), msgid.count('>')):
        problems.append(UNBALANCED_BRACKETS)
    if msgstr.count('"') > msgid.count('"'):
        problems.append(EXTRA_DOUBLE_QUOTE)
    if _is_python_format(msgid):
        rest = _PY_FORMAT.sub('', msgstr.replace('%%', ''))
        if '%' in rest:
            problems.append(STRAY_PERCENT)
    if '\n' in msgstr and '\n' not in msgid:
        problems.append(UNEXPECTED_LINE_BREAK)
    if not _braces_match(msgid, msgstr):
        problems.append(UNSAFE_BRACES)
    return problems


_TERMS_WRAPPERS = {
    'accounts/templates/accounts/tos.html',
    'accounts/templates/accounts/tos_public.html',
}
_TERMS_BODIES = 'accounts/templates/accounts/tos/'


def _is_terms_template(path):
    path = path.replace('\\', '/')
    if path.startswith('./'):
        path = path[2:]
    return path in _TERMS_WRAPPERS or path.startswith(_TERMS_BODIES)


def is_terms_only(entry):
    """True when only Terms of Service templates use this text, going by its #: references.

    makemessages writes those references for every new string, so new Terms
    text is covered without a list to maintain. They use the separator of the
    system that ran it, so both are accepted.
    """
    return bool(entry.occurrences) and all(
        _is_terms_template(path) for path, _line in entry.occurrences)


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
