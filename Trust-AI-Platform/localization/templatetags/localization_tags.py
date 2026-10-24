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


@register.filter
def can_review_translations(user):
    """True for staff and for users assigned at least one language to review."""
    if not getattr(user, 'is_authenticated', False):
        return False
    cached = getattr(user, '_can_review_translations', None)
    if cached is None:
        from localization.reviewers import reviewable_codes
        cached = bool(reviewable_codes(user))
        user._can_review_translations = cached
    return cached
