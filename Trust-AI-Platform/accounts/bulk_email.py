"""Recipient selection and content helpers for administrator email campaigns."""

import re

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.validators import validate_email

from organization.models import Organization

from .models import BulkEmailCampaign


def eligible_teacher_users():
    """Active Teacher-role users with a non-empty email address."""
    return (
        User.objects.filter(is_active=True, groups__name__iexact='teachers')
        .exclude(email='')
        .distinct()
    )


def normalize_ids(values):
    ids = set()
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            ids.add(parsed)
    return sorted(ids)


def unique_valid_email_users(queryset):
    """Return one user per case-insensitive valid email address."""
    recipients = []
    seen_emails = set()
    for user in queryset.only('id', 'email').order_by('id'):
        email = (user.email or '').strip()
        normalized = email.casefold()
        if not normalized or normalized in seen_emails:
            continue
        try:
            validate_email(email)
        except ValidationError:
            continue
        seen_emails.add(normalized)
        recipients.append(user)
    return recipients


def resolve_campaign_recipients(target_type, teacher_ids=None, organization_ids=None):
    """Resolve and deduplicate an administrator's selected recipients."""
    teachers = eligible_teacher_users()
    selected_organizations = Organization.objects.none()

    if target_type == BulkEmailCampaign.TARGET_ALL_TEACHERS:
        queryset = teachers
    elif target_type == BulkEmailCampaign.TARGET_SELECTED_TEACHERS:
        queryset = teachers.filter(id__in=normalize_ids(teacher_ids or []))
    elif target_type == BulkEmailCampaign.TARGET_ORGANIZATIONS:
        selected_organizations = Organization.objects.filter(
            id__in=normalize_ids(organization_ids or [])
        ).order_by('name')
        queryset = teachers.filter(member_of_organizations__in=selected_organizations)
    else:
        raise ValueError('Choose a valid recipient group.')

    return unique_valid_email_users(queryset), selected_organizations


_RELATIVE_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href)\s*=\s*)(?P<quote>["\'])(?P<url>/(?!/)[^"\']*)',
    flags=re.IGNORECASE,
)


def absolutize_content_urls(html, site_url):
    """Make TinyMCE media and relative links usable outside the website."""
    base = (site_url or '').rstrip('/')
    if not base:
        return html

    def replace(match):
        return (
            f"{match.group('prefix')}{match.group('quote')}"
            f"{base}{match.group('url')}"
        )

    return _RELATIVE_URL_RE.sub(replace, html)
