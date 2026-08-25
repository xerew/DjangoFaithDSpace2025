from .evidence import (
    get_evidence_answers,
    get_evidence_signature,
    normalize_evidence_language,
    normalize_evidence_scope,
)
from .models import Scenario
from django.conf import settings
from django.db.models import Max, Min, OuterRef, Subquery
import os
import re


def get_eligible_user_answers(
    scenario_id,
    scope='local',
    language='',
):
    """Answers from implementations eligible for the requested scope."""
    scenario = Scenario.objects.get(pk=scenario_id)
    return get_evidence_answers(
        scenario,
        scope=scope,
        language=language,
    )


def get_scenario_evidence_cache_paths(
    scenario,
    scope='local',
    language='',
):
    """Return cache files isolated to the complete evidence source set."""
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    version = scenario.ensure_current_version()
    signature = get_evidence_signature(
        scenario,
        scope,
        language,
    )[:16]
    language_slug = (
        re.sub(r'[^a-z0-9]+', '-', language.casefold()).strip('-')
        if language
        else 'all-languages'
    )
    prefix = (
        f'scenario_{scenario.id}_v{version.id}_{scope}_'
        f'{language_slug}_{signature}'
    )
    return {
        'metrics': os.path.join(
            settings.AI_METRICS_CACHE_ROOT,
            f'{prefix}_combined_activity_metrics.csv',
        ),
        'flags': os.path.join(
            settings.AI_METRICS_CACHE_ROOT,
            f'{prefix}_flagged_activities_with_reasons.csv',
        ),
    }


def get_last_answers(scenario_id, scope='local', language=''):
    # Correlated subquery: for each (implementation, activity) row, pick the
    # answer with the highest id.
    # This avoids pulling all IDs into Python memory and lets PostgreSQL resolve it in one query.
    latest_id = (
        get_eligible_user_answers(scenario_id, scope, language)
        .filter(
            implementation_id=OuterRef('implementation_id'),
            activity_id=OuterRef('activity_id'),
        )
        .order_by('-id')
        .values('id')[:1]
    )
    return (
        get_eligible_user_answers(scenario_id, scope, language)
        .filter(id=Subquery(latest_id))
        .select_related('answer', 'activity', 'user')
    )

# def get_first_answers(scenario_id):
#     # Fetch the earliest answer for each user and activity based on the created_on timestamp
#     first_answers = (
#         UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id)
#         .values('user_id', 'activity_id')
#         .annotate(first_answer_id=Min('id'))  # Get the first answer ID for each user and activity
#     )
def get_first_answers(scenario_id):
    # Correlated subquery: for each (user, activity) row, pick the one with the lowest id.
    # This avoids pulling all IDs into Python memory and lets the database resolve it in one query.
    first_subquery = (
        get_eligible_user_answers(scenario_id)
        .filter(activity=OuterRef('activity'), user=OuterRef('user'))
        .order_by('id')
        .values('id')[:1]
    )
    return get_eligible_user_answers(scenario_id).filter(
        id=Subquery(first_subquery)
    )
