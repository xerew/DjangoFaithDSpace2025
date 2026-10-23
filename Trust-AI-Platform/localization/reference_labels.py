"""Interface-language labels for the built-in reference values.

The database keeps the English values (they are used in logic, URLs and
form submissions). ``ui_label`` only changes what a reader sees.
"""
from django.utils.encoding import force_str
from django.utils.translation import pgettext_lazy

from localization.languages import LANGUAGE_META, code_for_table_name, meta_for

SUBJECTS = {
    'Physics': pgettext_lazy('subject', 'Physics'),
    'Mathematics': pgettext_lazy('subject', 'Mathematics'),
    'Chemistry': pgettext_lazy('subject', 'Chemistry'),
    'Biology': pgettext_lazy('subject', 'Biology'),
    'Computer Science': pgettext_lazy('subject', 'Computer Science'),
    'Environmental Science': pgettext_lazy('subject', 'Environmental Science'),
    'Earth & Space Science': pgettext_lazy('subject', 'Earth & Space Science'),
    'Engineering & Technology': pgettext_lazy('subject', 'Engineering & Technology'),
    'Statistics & Data Science': pgettext_lazy('subject', 'Statistics & Data Science'),
}

CATEGORIES = {
    'STEM': pgettext_lazy('subject category', 'STEM'),
    'Humanities': pgettext_lazy('subject category', 'Humanities'),
    'Social Sciences': pgettext_lazy('subject category', 'Social Sciences'),
    'Arts': pgettext_lazy('subject category', 'Arts'),
    'Other': pgettext_lazy('subject category', 'Other'),
}

ACTIVITY_TYPES = {
    'Explanation': pgettext_lazy('activity type', 'Explanation'),
    'Experiment': pgettext_lazy('activity type', 'Experiment'),
    'Guidance': pgettext_lazy('activity type', 'Guidance'),
    'Question': pgettext_lazy('activity type', 'Question'),
}

GROUPS = {
    'High': pgettext_lazy('performance group', 'High'),
    'Moderate': pgettext_lazy('performance group', 'Moderate'),
    'Low': pgettext_lazy('performance group', 'Low'),
}

# Persian is not an interface language but can still be a scenario language.
EXTRA_LANGUAGE_NAMES = {'persian': 'فارسی'}

_LOOKUPS = (SUBJECTS, CATEGORIES, ACTIVITY_TYPES, GROUPS)


def ui_label(value):
    """Return the interface-language label for a known English value."""
    if value is None or value == '':
        return value
    if not isinstance(value, str):
        return value
    for table in _LOOKUPS:
        if value in table:
            return force_str(table[value])
    key = value.strip().casefold()
    if key in EXTRA_LANGUAGE_NAMES:
        return EXTRA_LANGUAGE_NAMES[key]
    code = code_for_table_name(value)
    if code and any(m['table_name'].casefold() == key for m in LANGUAGE_META):
        return meta_for(code)['native']
    return value
