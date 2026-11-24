"""Languages a feedback form can be translated into.

The same seven languages as the platform interface. Replace with
localization.languages.code_for_table_name once the interface translation is
merged into main.
"""

FEEDBACK_LANGUAGES = [
    ('en', 'English', 'English'),
    ('el', 'Greek', 'Ελληνικά'),
    ('fr', 'French', 'Français'),
    ('de', 'German', 'Deutsch'),
    ('es', 'Spanish', 'Español'),
    ('ro', 'Romanian', 'Română'),
    ('pt', 'Portuguese', 'Português'),
]

LANGUAGE_CODES = frozenset(code for code, _english, _native in FEEDBACK_LANGUAGES)

LANGUAGE_CHOICES = [(code, english) for code, english, _native in FEEDBACK_LANGUAGES]


def language_code(name):
    """Map a scenario language (code, English or native name) to a code, or None."""
    key = str(name or '').strip().casefold()
    if not key:
        return None
    for code, english, native in FEEDBACK_LANGUAGES:
        if key in (code, english.casefold(), native.casefold()):
            return code
    return None


def language_name(code):
    """English name of a language code; '' for blank or unknown codes."""
    for known, english, _native in FEEDBACK_LANGUAGES:
        if known == code:
            return english
    return ''
