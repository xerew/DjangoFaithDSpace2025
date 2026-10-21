"""Reviewers of machine translations, kept as general scenario collaborators.

Only admins assign a reviewer, who must be a teacher or an admin; the
reviewer stays on as a collaborator until the creator or an admin removes
them. Collaborators edit; sharing and deleting stay with the owner.
"""

import logging

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.mail import send_mail
from django.db.models import Q
from django.urls import reverse
from django.utils import timezone

from authoringtool.models import ScenarioCollaborator, ScenarioTranslationJob

logger = logging.getLogger(__name__)

REVIEW_SUBJECT = 'You have been asked to review a translation'


def _is_admin(user):
    return user.is_superuser or user.is_staff


def can_be_reviewer(user):
    return _is_admin(user) or user.groups.filter(name='teachers').exists()


def is_collaborator(user, scenario):
    return ScenarioCollaborator.objects.filter(scenario=scenario, user=user).exists()


def _send_review_email(scenario, job, reviewer, added_by):
    link = settings.SITE_URL + reverse('viewScenario', args=[scenario.id])
    body = (
        f'{added_by.get_full_name() or added_by.username} asked you to review the '
        f'{job.target_language} translation "{scenario.name}" on Trust AI Lab. '
        f'Open it here: {link}'
    )
    try:
        send_mail(REVIEW_SUBJECT, body, None, [reviewer.email])
    except Exception:  # the assignment stands even if the email fails
        logger.exception('Could not email translation reviewer %s', reviewer.pk)


def assign_reviewer(scenario, reviewer, added_by):
    if not _is_admin(added_by) or not can_be_reviewer(reviewer):
        raise PermissionDenied
    collaborator, created = ScenarioCollaborator.objects.get_or_create(
        scenario=scenario,
        user=reviewer,
        defaults={
            'role': ScenarioCollaborator.ROLE_TRANSLATION_REVIEWER,
            'added_by': added_by,
        },
    )
    if created and reviewer.email:
        _send_review_email(scenario, scenario.translation_job, reviewer, added_by)
    return collaborator


def can_remove_collaborators(user, scenario):
    return _is_admin(user) or scenario.created_by_id == user.id


def remove_collaborator(scenario, user, removed_by):
    if not can_remove_collaborators(removed_by, scenario):
        raise PermissionDenied
    ScenarioCollaborator.objects.filter(scenario=scenario, user=user).delete()


def can_verify(user, scenario):
    job = getattr(scenario, 'translation_job', None)
    if job is None or job.status != ScenarioTranslationJob.STATUS_DONE:
        return False
    return _is_admin(user) or is_collaborator(user, scenario)


def verify_translation(scenario, user):
    if not can_verify(user, scenario):
        raise PermissionDenied
    ScenarioTranslationJob.objects.filter(target_scenario=scenario).update(
        verified_by=user, verified_at=timezone.now(),
    )


def unverify_translation(scenario, admin):
    if not _is_admin(admin):
        raise PermissionDenied
    ScenarioTranslationJob.objects.filter(target_scenario=scenario).update(
        verified_by=None, verified_at=None,
    )


def reviewer_candidates(scenario, query, limit=20):
    """Teachers and admins matching query; same-language organizations first."""
    from django.contrib.auth.models import User

    users = User.objects.filter(
        Q(groups__name='teachers') | Q(is_staff=True) | Q(is_superuser=True),
        is_active=True,
    )
    if query:
        users = users.filter(
            Q(username__icontains=query)
            | Q(first_name__icontains=query)
            | Q(last_name__icontains=query)
            | Q(email__icontains=query)
        )
    users = list(users.distinct().prefetch_related('member_of_organizations').select_related('profile')[:200])
    language = scenario.translation_job.target_language.casefold()

    def suggested(user):
        return any(
            (org.language or '').strip().casefold() == language
            for org in user.member_of_organizations.all()
        )

    rows = [
        {
            'id': user.id,
            'username': user.username,
            'name': user.get_full_name(),
            'institution': getattr(getattr(user, 'profile', None), 'institution', '') or '',
            'suggested': suggested(user),
        }
        for user in users
    ]
    rows.sort(key=lambda row: (not row['suggested'], row['username'].casefold()))
    return rows[:limit]
