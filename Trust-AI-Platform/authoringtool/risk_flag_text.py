"""Risk-flag sentences as a code plus numbers, rendered in any language.

The stored ``flag_type`` and ``flag_reason`` stay English: ``flag_type`` is a
key into the QValue reinforcement table and the sentences are fed to the AI
proposal prompt. ``reason_code`` and ``reason_params`` let the pages show the
same sentence in the reader's language.

This module must not import models: migrations import ``parse_reason``.
"""
import re
from contextlib import nullcontext

from django.utils import translation
from django.utils.translation import gettext_lazy as _
from django.utils.translation import pgettext_lazy

# Placeholders hold values that are already formatted exactly as the English
# sentences always showed them (for example '31.0' or '23.40'), so English
# output is byte-identical to the old f-strings. A literal percent sign is
# written %% in every template.
REASONS = {
    'engagement_slow': _(
        "Engagement risk: %(group)s group's Avg Time (%(t)ss) is above "
        "mean+std (%(mean)ss ± %(std)ss, CV=%(cv)s). "
        "Significantly slower than the other groups — possible confusion "
        "or disengagement."
    ),
    'engagement_fast': _(
        "Engagement risk: %(group)s group's Avg Time (%(t)ss) is below "
        "mean-std (%(mean)ss ± %(std)ss, CV=%(cv)s). "
        "Suspiciously fast relative to other groups — may indicate "
        "guessing or skipping."
    ),
    'v_shape': _(
        "Unexpected V-shape: Moderate group (%(moderate)s%% wrong) performs "
        "worse than both High (%(high)s%%) and Low (%(low)s%%) groups. "
        "Activity may be poorly calibrated for mid-range learners."
    ),
    'low_beats_moderate': _(
        "Low group (%(low)s%% wrong) outperforms Moderate (%(moderate)s%% "
        "wrong) — unexpected reversal in performance order. "
        "Activity content or answer choices may not align with the expected "
        "difficulty level."
    ),
    'high_worse_than_moderate': _(
        "High group (%(high)s%% wrong) performs worse than Moderate "
        "(%(moderate)s%% wrong). "
        "High-category students should outperform Moderate — this reversal "
        "may indicate activity miscalibration or a mismatch between the "
        "activity and the scoring rubric."
    ),
    'timing_all_wrong': _(
        "%(group)s group answered everything incorrectly (100%% wrong) yet "
        "spent %(time)ss — ≥%(ratio)sx and ≥%(gap)ss more than the other "
        "groups (%(other1)ss, %(other2)ss). "
        "Prolonged time with total failure indicates active confusion or "
        "cognitive overload, not disengagement."
    ),
    'timing_all_right': _(
        "%(group)s group answered everything correctly (0%% wrong) yet spent "
        "%(time)ss — ≥%(ratio)sx and ≥%(gap)ss more than the other groups "
        "(%(other1)ss, %(other2)ss). "
        "Perfect outcome with excessive time may indicate effortful "
        "retrieval at the boundary of the group's ability range, rather "
        "than confident mastery."
    ),
    'systemic_failure': _(
        "All performance categories individually exceed the PISA "
        "Low-performance boundary (%(base)s%% wrong): "
        "High %(high)s%%, Moderate %(moderate)s%%, Low %(low)s%% wrong. "
        "All groups score below the Low-performance threshold on this "
        "activity, indicating a content-level failure independent of learner "
        "category — review the question wording, answer options, or "
        "prerequisite content."
    ),
    'dynamic_correctness': _(
        "Dynamic correctness: %(group)s group's %%Wrong (%(wrong)s%%) exceeds "
        "mean+std threshold (%(threshold)s%%) with a category spread of "
        "%(spread)spp. "
        "This group is struggling significantly more than the others on this "
        "activity."
    ),
}

# The stored flag types, one entry per kind. The engagement ones are stored
# once per activity kind, e.g. "Engagement risk (too slow, question)".
FLAG_TYPES = {
    'Dynamic correctness threshold': _('Dynamic correctness threshold'),
    'Extreme correctness pattern': _('Extreme correctness pattern'),
    'Engagement risk (too slow)': _(
        'Engagement risk (too slow, %(activity_type)s)'),
    'Engagement risk (too fast)': _(
        'Engagement risk (too fast, %(activity_type)s)'),
    'Timing discrepancy under extreme correctness': _(
        'Timing discrepancy under extreme correctness'),
    'Systemic failure': _('Systemic failure'),
}

ACTIVITY_KINDS = {
    'question': pgettext_lazy('activity kind', 'question'),
    'non-question': pgettext_lazy('activity kind', 'non-question'),
}

# Which stored flag types each code can belong to.
_CODE_FLAG_TYPES = {
    'engagement_slow': ('Engagement risk (too slow, question)',
                        'Engagement risk (too slow, non-question)'),
    'engagement_fast': ('Engagement risk (too fast, question)',
                        'Engagement risk (too fast, non-question)'),
    'v_shape': ('Extreme correctness pattern',),
    'low_beats_moderate': ('Extreme correctness pattern',),
    'high_worse_than_moderate': ('Extreme correctness pattern',),
    'timing_all_wrong': ('Timing discrepancy under extreme correctness',),
    'timing_all_right': ('Timing discrepancy under extreme correctness',),
    'systemic_failure': ('Systemic failure',),
    'dynamic_correctness': ('Dynamic correctness threshold',),
}

# Params that hold performance-group names (High / Moderate / Low).
GROUP_PARAMS = {'group'}

_PLACEHOLDER = re.compile(r'(%\((\w+)\)s|%%)')
_ENGAGEMENT_TYPE = re.compile(
    r'^Engagement risk \((too slow|too fast), (question|non-question)\)$')


def _english(lazy_text):
    with translation.override('en'):
        return str(lazy_text)


def _template_regex(template):
    """Build a regex from an English template: literal parts escaped,
    each %(name)s a named non-greedy group, %% a literal percent sign."""
    pattern = []
    pos = 0
    for match in _PLACEHOLDER.finditer(template):
        pattern.append(re.escape(template[pos:match.start()]))
        if match.group(1) == '%%':
            pattern.append('%')
        else:
            pattern.append('(?P<%s>.+?)' % match.group(2))
        pos = match.end()
    pattern.append(re.escape(template[pos:]))
    return re.compile(''.join(pattern), re.DOTALL)


_REGEXES = {}


def _regex_for(code):
    if code not in _REGEXES:
        _REGEXES[code] = _template_regex(_english(REASONS[code]))
    return _REGEXES[code]


def _label(value):
    from localization.reference_labels import ui_label
    return ui_label(value)


def reason_text(code, params, language=None):
    """Render a reason in ``language`` (default: the active language).

    ``params`` are the already-formatted strings. With ``language='en'`` the
    result is exactly the sentence the analysis always produced.
    """
    template = REASONS[code]
    if language == 'en':
        with translation.override('en'):
            return str(template) % params
    context = translation.override(language) if language else nullcontext()
    with context:
        values = {
            key: _label(value) if key in GROUP_PARAMS else value
            for key, value in params.items()
        }
        return str(template) % values


def parse_reason(flag_type, reason):
    """Return ``(code, params)`` for an English stored reason, else None."""
    if not isinstance(reason, str):
        return None
    for code, flag_types in _CODE_FLAG_TYPES.items():
        if flag_type not in flag_types:
            continue
        match = _regex_for(code).fullmatch(reason)
        if match:
            return code, match.groupdict()
    return None


def flag_type_text(flag_type, language=None):
    """The flag type in the active language; unknown values unchanged."""
    match = _ENGAGEMENT_TYPE.match(flag_type or '')
    context = translation.override(language) if language else nullcontext()
    with context:
        if match:
            key = 'Engagement risk (%s)' % match.group(1)
            kind = ACTIVITY_KINDS[match.group(2)]
            return str(FLAG_TYPES[key]) % {'activity_type': str(kind)}
        if flag_type in FLAG_TYPES and '%(' not in _english(
                FLAG_TYPES[flag_type]):
            return str(FLAG_TYPES[flag_type])
    return flag_type


def display_flag(flag_type, reason, code='', params=None):
    """``(flag_type, reason)`` as shown to the reader in the active language.

    Uses the stored code and numbers when there are any, else parses the
    English sentence, else returns the stored text unchanged.
    """
    shown_type = flag_type_text(flag_type)
    parsed = (code, params) if code and params and code in REASONS else (
        parse_reason(flag_type, reason))
    if parsed:
        try:
            return shown_type, reason_text(parsed[0], parsed[1])
        except (KeyError, TypeError, ValueError):
            pass
    return shown_type, reason
