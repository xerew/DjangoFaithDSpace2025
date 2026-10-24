"""Who may correct which interface translation."""
from django.conf import settings

from .languages import LANGUAGE_META
from .models import TranslationReviewer

# English is the source language: every other interface language can be reviewed.
REVIEW_LANGUAGES = [m for m in LANGUAGE_META if m['code'] != settings.LANGUAGE_CODE]
REVIEW_CODES = [m['code'] for m in REVIEW_LANGUAGES]


def is_admin(user):
    return user.is_staff or user.is_superuser


def can_be_reviewer(user):
    """Only active teachers and administrators can be assigned a language."""
    return user.is_active and (is_admin(user) or user.groups.filter(name='teachers').exists())


def reviewable_codes(user):
    """The languages ``user`` may review, in display order (all of them for staff)."""
    if not getattr(user, 'is_authenticated', False):
        return []
    if is_admin(user):
        return list(REVIEW_CODES)
    assigned = set(TranslationReviewer.objects.filter(user=user).values_list('language', flat=True))
    return [code for code in REVIEW_CODES if code in assigned]


def can_review(user, code):
    return code in reviewable_codes(user)
