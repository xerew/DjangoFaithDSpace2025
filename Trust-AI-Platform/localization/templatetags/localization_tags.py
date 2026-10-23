from django import template
from django.utils import translation

from localization import reference_labels
from localization.languages import LANGUAGE_META, meta_for

register = template.Library()


@register.inclusion_tag('includes/language_switcher.html', takes_context=True)
def language_switcher(context):
    request = context.get('request')
    code = (translation.get_language() or 'en').split('-')[0]
    return {
        'current': meta_for(code),
        'languages': LANGUAGE_META,
        'next': request.get_full_path() if request else '/',
        'csrf_token': context.get('csrf_token', ''),
    }


@register.filter
def ui_label(value):
    """Show a built-in subject, type, group or language name in the interface language."""
    return reference_labels.ui_label(value)
