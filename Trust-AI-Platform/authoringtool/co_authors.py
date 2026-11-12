"""Co-authors a scenario's creator picks from the teachers of any organization.

The creator or an admin adds and removes them, whatever the scenario's
visibility. Co-authors edit and can use the scenario with their own groups;
sharing and deleting stay with the owner (see views.user_manages_scenario).
"""

import logging

from django.conf import settings
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.core.mail import send_mail
from django.db.models import Q
from django.urls import reverse
from django.utils import translation
from django.utils.encoding import force_str
from django.utils.translation import gettext as _, gettext_lazy

from authoringtool.machine_translation.collaborators import can_remove_collaborators
from authoringtool.models import ScenarioCollaborator
from localization.languages import recipient_language
from organization.models import Organization

logger = logging.getLogger(__name__)

CANDIDATE_LIMIT = 50
ADDED_SUBJECT = gettext_lazy('You were added as a co-author')

_SEARCH_FIELDS = (
    'username', 'first_name', 'last_name', 'email',
    'member_of_organizations__name', 'member_of_organizations__short_name',
)
_OWN_ORGANIZATION, _OTHER_ORGANIZATION, _NO_ORGANIZATION = 0, 1, 2


class NotEligible(Exception):
    """The chosen user cannot be a co-author of this scenario."""


def co_authored_q(user):
    """Scenario filter for the scenarios ``user`` co-authors. Add ``.distinct()``."""
    return Q(
        collaborators__user=user,
        collaborators__role=ScenarioCollaborator.ROLE_CO_AUTHOR,
    )


def co_authored_ids(user, scenario_ids=None):
    rows = ScenarioCollaborator.objects.filter(
        user=user, role=ScenarioCollaborator.ROLE_CO_AUTHOR,
    )
    if scenario_ids is not None:
        rows = rows.filter(scenario_id__in=scenario_ids)
    return set(rows.values_list('scenario_id', flat=True))


def _person(user):
    return {'id': user.id, 'username': user.username, 'name': user.get_full_name()}


def co_author_rows(scenario):
    rows = scenario.collaborators.filter(
        role=ScenarioCollaborator.ROLE_CO_AUTHOR,
    ).select_related('user')
    return [_person(row.user) for row in rows]


def _send_added_email(scenario, co_author, added_by):
    link = settings.SITE_URL + reverse('viewScenario', args=[scenario.id])
    try:
        with translation.override(recipient_language(co_author)):
            body = _(
                '%(sender)s added you as a co-author of the scenario "%(name)s" '
                'on Trust AI Lab. Open it here: %(link)s'
            ) % {
                'sender': added_by.get_full_name() or added_by.username,
                'name': scenario.name,
                'link': link,
            }
            send_mail(force_str(ADDED_SUBJECT), body, None, [co_author.email])
    except Exception:  # they stay a co-author even if the email fails
        logger.exception('Could not email co-author %s', co_author.pk)


def add(scenario, user, added_by):
    """Make an active teacher a co-author. Someone already on the scenario stays as they are."""
    if not can_remove_collaborators(added_by, scenario):
        raise PermissionDenied
    if (
        user.pk == scenario.created_by_id
        or not user.is_active
        or not user.groups.filter(name='teachers').exists()
    ):
        raise NotEligible
    _, created = ScenarioCollaborator.objects.get_or_create(
        scenario=scenario,
        user=user,
        defaults={'role': ScenarioCollaborator.ROLE_CO_AUTHOR, 'added_by': added_by},
    )
    if created and user.email:
        _send_added_email(scenario, user, added_by)


def _search(users, query):
    for word in query.split():
        matches_word = Q()
        for field in _SEARCH_FIELDS:
            matches_word |= Q(**{f'{field}__icontains': word})
        users = users.filter(matches_word)
    return users


def candidates(scenario, query, viewer, limit=CANDIDATE_LIMIT):
    """Teachers who can still be added, grouped by organization.

    The creator's organizations come first, then the others from A to Z, then
    teachers with no organization. A teacher in two organizations appears under
    both. Returns ``(groups, truncated)``; ``limit`` counts teachers, not rows.
    """
    users = _search(
        User.objects.filter(groups__name='teachers', is_active=True)
        .exclude(pk=scenario.created_by_id)
        .exclude(pk__in=scenario.collaborators.values('user_id')),
        query,
    ).distinct().prefetch_related('member_of_organizations')

    own_ids = set(
        Organization.objects.filter(members__id=scenario.created_by_id).values_list('id', flat=True)
    ) if scenario.created_by_id else set()
    viewer_ids = set(viewer.member_of_organizations.values_list('id', flat=True))

    def organization_key(organization):
        rank = _OWN_ORGANIZATION if organization.id in own_ids else _OTHER_ORGANIZATION
        return (rank, organization.name.casefold())

    def user_key(user):
        keys = [organization_key(org) for org in user.member_of_organizations.all()]
        best = min(keys) if keys else (_NO_ORGANIZATION, '')
        return (best, user.username.casefold())

    ordered = sorted(users, key=user_key)
    shown = ordered[:limit]

    groups = {}
    for user in shown:
        for organization in user.member_of_organizations.all() or [None]:
            if organization is None:
                key, name, yours = (_NO_ORGANIZATION, ''), None, False
            else:
                key, name, yours = (
                    organization_key(organization), organization.name, organization.id in viewer_ids,
                )
            group = groups.setdefault(key, {'name': name, 'yours': yours, 'users': []})
            group['users'].append(_person(user))
    return [groups[key] for key in sorted(groups)], len(ordered) > limit
