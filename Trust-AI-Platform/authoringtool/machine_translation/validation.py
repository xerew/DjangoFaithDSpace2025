"""Checks that a translated unit kept everything that must not change."""

import re
from collections import Counter
from html.parser import HTMLParser

_DIGITS = re.compile(r'\d+')
_FORMULA = re.compile(r'\$\$.+?\$\$|\$.+?\$|\\\(.+?\\\)|\\\[.+?\\\]', re.DOTALL)


class _TagCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(('start', tag, tuple(sorted(attrs, key=lambda a: a[0]))))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.tags.append(('end', tag))


def _tags(value):
    collector = _TagCollector()
    collector.feed(value)
    collector.close()
    return collector.tags


def validate_unit(source, translated):
    """Return readable problems with a translation; empty when it is valid."""
    if not isinstance(translated, dict):
        return ['the reply was not a set of fields']
    errors = []
    missing = sorted(set(source) - set(translated))
    extra = sorted(set(translated) - set(source))
    if missing:
        errors.append(f'missing {", ".join(missing)}')
    if extra:
        errors.append(f'unexpected {", ".join(extra)}')
    for key in sorted(set(source) & set(translated)):
        original, result = source[key], translated[key]
        if not isinstance(result, str):
            errors.append(f'{key} is not text')
            continue
        if original.strip() and not result.strip():
            errors.append(f'{key} is empty')
            continue
        if _tags(original) != _tags(result):
            errors.append(f'{key} changed the HTML tags or attributes')
        if Counter(_DIGITS.findall(original)) != Counter(_DIGITS.findall(result)):
            errors.append(f'{key} changed the numbers')
        if _FORMULA.findall(original) != _FORMULA.findall(result):
            errors.append(f'{key} changed a formula')
    return errors
