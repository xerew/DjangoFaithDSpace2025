import hashlib
import json

from django.db.models import Count, Exists, Max, OuterRef, Sum

from .models import (
    Scenario,
    ScenarioVersionCompatibility,
    UserAnswer,
    UserScenarioScore,
)


EVIDENCE_SCOPES = {'local', 'compatible', 'historical'}


def normalize_evidence_scope(scope):
    return scope if scope in EVIDENCE_SCOPES else 'compatible'


def normalize_evidence_language(language):
    return (language or '').strip()


def get_evidence_versions(scenario, scope='compatible', language=''):
    """Return current versions that may safely contribute to this analysis."""
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    current = scenario.ensure_current_version()
    if scope == 'historical':
        return scenario.versions.none()
    if scope == 'local':
        versions = scenario.versions.filter(pk=current.pk)
    else:
        versions = scenario.compatible_current_versions()
    if language:
        versions = versions.filter(scenario__language__iexact=language)
    return versions


def get_evidence_scores(scenario, scope='compatible', language=''):
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    if scope == 'historical':
        scores = UserScenarioScore.objects.filter(
            implementation__scenario=scenario,
            implementation__version_confidence='legacy_unknown',
            implementation__data_quality_status__in=['unreviewed', 'clean'],
        )
        if language and (
            (scenario.language or '').strip().casefold()
            != language.casefold()
        ):
            return scores.none()
        return scores.exclude(
            implementation__user__groups__name='teachers',
        )
    versions = get_evidence_versions(scenario, scope, language)
    return UserScenarioScore.objects.filter(
        implementation__scenario_version__in=versions,
        implementation__version_confidence='exact',
        implementation__data_quality_status__in=['unreviewed', 'clean'],
    ).exclude(
        implementation__user__groups__name='teachers',
    )


def get_evidence_implementation_count(
    scenario,
    scope='compatible',
    language='',
):
    return (
        get_evidence_scores(scenario, scope, language)
        .values('implementation_id')
        .distinct()
        .count()
    )


def get_evidence_answers(scenario, scope='compatible', language=''):
    """Return answers with an eligible implementation in an evidence version."""
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    if scope == 'historical':
        if language and (
            (scenario.language or '').strip().casefold()
            != language.casefold()
        ):
            return UserAnswer.objects.none()
        return (
            UserAnswer.objects
            .filter(
                implementation__scenario=scenario,
                implementation__version_confidence='legacy_unknown',
                implementation__data_quality_status__in=[
                    'unreviewed',
                    'clean',
                ],
            )
            .exclude(implementation__user__groups__name='teachers')
        )

    versions = get_evidence_versions(scenario, scope, language)
    return (
        UserAnswer.objects
        .filter(
            implementation__scenario_version__in=versions,
            implementation__version_confidence='exact',
            implementation__data_quality_status__in=[
                'unreviewed',
                'clean',
            ],
        )
        .exclude(implementation__user__groups__name='teachers')
    )


def get_evidence_breakdown(scenario, scope='compatible', language=''):
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    scores = get_evidence_scores(scenario, scope, language)
    if scope == 'historical':
        implementation_count = (
            scores.values('implementation_id').distinct().count()
        )
        return [{
            'scenario_id': scenario.id,
            'scenario_name': scenario.name,
            'language': (scenario.language or '').strip() or 'Unspecified',
            'version_id': None,
            'version_number': None,
            'implementation_count': implementation_count,
            'is_legacy': True,
        }]

    rows = (
        scores
        .values(
            'implementation__scenario_id',
            'implementation__scenario__name',
            'implementation__scenario__language',
            'implementation__scenario_version_id',
            'implementation__scenario_version__version_number',
        )
        .annotate(implementation_count=Count('implementation_id', distinct=True))
        .order_by(
            'implementation__scenario__language',
            'implementation__scenario__name',
        )
    )
    return [
        {
            'scenario_id': row['implementation__scenario_id'],
            'scenario_name': row['implementation__scenario__name'],
            'language': (
                row['implementation__scenario__language'] or ''
            ).strip()
            or 'Unspecified',
            'version_id': row['implementation__scenario_version_id'],
            'version_number': (
                row['implementation__scenario_version__version_number']
            ),
            'implementation_count': row['implementation_count'],
        }
        for row in rows
    ]


def _get_evidence_source_payload(
    scenario,
    scope='compatible',
    language='',
):
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    if scope == 'historical':
        scenario.ensure_current_version()
        return {
            'scope': scope,
            'target_version': scenario.current_version_id,
            'scenario_id': scenario.id,
            'versions': [],
            'memberships': [],
            'legacy_evidence': True,
            'language': language,
        }

    versions = list(
        get_evidence_versions(scenario, scope, language)
        .values_list('id', flat=True)
        .order_by('id')
    )
    membership_rows = list(
        ScenarioVersionCompatibility.objects
        .filter(scenario_version_id__in=versions)
        .values_list(
            'scenario_version_id',
            'cluster_id',
            'status',
            'updated_at',
        )
        .order_by('scenario_version_id')
    )
    return {
        'scope': scope,
        'language': language,
        'target_version': scenario.current_version_id,
        'versions': versions,
        'memberships': [
            [
                version_id,
                cluster_id,
                status,
                updated_at.isoformat() if updated_at else '',
            ]
            for version_id, cluster_id, status, updated_at in membership_rows
        ],
    }


def get_evidence_source_signature(
    scenario,
    scope='compatible',
    language='',
):
    """Fingerprint structural sources and review decisions, not student rows."""
    payload = _get_evidence_source_payload(scenario, scope, language)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()


def get_evidence_signature(scenario, scope='compatible', language=''):
    """Fingerprint source decisions plus all mutable eligible evidence."""
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    score_state = get_evidence_scores(
        scenario,
        scope,
        language,
    ).aggregate(
        eligible_row_count=Count('id'),
        highest_score_id=Max('id'),
        eligible_id_sum=Sum('id'),
        eligible_user_id_sum=Sum('user_id'),
        eligible_user_score_sum=Sum('user_score'),
        eligible_duration_sum=Sum('timeDoingScenario'),
        eligible_last_activity_id_sum=Sum('last_activity_id'),
        eligible_department_id_sum=Sum('user__school_department_id'),
    )
    answer_state = get_evidence_answers(
        scenario,
        scope,
        language,
    ).aggregate(
        eligible_row_count=Count('id'),
        highest_answer_id=Max('id'),
        eligible_id_sum=Sum('id'),
        eligible_user_id_sum=Sum('user_id'),
        eligible_activity_id_sum=Sum('activity_id'),
        eligible_answer_id_sum=Sum('answer_id'),
        eligible_timing_sum=Sum('timing'),
    )
    payload = {
        'source_signature': get_evidence_source_signature(
            scenario,
            scope,
            language,
        ),
        'score_state': score_state,
        'answer_state': answer_state,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()


def get_evidence_context(scenario, scope='compatible', language=''):
    scope = normalize_evidence_scope(scope)
    language = normalize_evidence_language(language)
    if scope == 'historical':
        scenario.ensure_current_version()
        breakdown = get_evidence_breakdown(scenario, scope, language)
        source_language = (
            (scenario.language or '').strip() or 'Unspecified'
        )
        return {
            'scope': scope,
            'language_filter': language,
            'source_signature': get_evidence_source_signature(
                scenario,
                scope,
                language,
            ),
            'signature': get_evidence_signature(
                scenario,
                scope,
                language,
            ),
            'version_ids': [],
            'version_count': 0,
            'scenario_count': 1,
            'implementation_count': sum(
                row['implementation_count'] for row in breakdown
            ),
            'languages': [source_language] if breakdown else [],
            'sources': breakdown,
            'is_legacy': True,
        }

    versions = list(
        get_evidence_versions(scenario, scope, language)
        .select_related('scenario')
        .order_by('scenario__language', 'scenario__name')
    )
    breakdown = get_evidence_breakdown(scenario, scope, language)
    languages = sorted({
        (version.scenario.language or '').strip() or 'Unspecified'
        for version in versions
    }, key=str.casefold)
    return {
        'scope': scope,
        'language_filter': language,
        'source_signature': get_evidence_source_signature(
            scenario,
            scope,
            language,
        ),
        'signature': get_evidence_signature(
            scenario,
            scope,
            language,
        ),
        'version_ids': [version.id for version in versions],
        'version_count': len(versions),
        'scenario_count': len({version.scenario_id for version in versions}),
        'implementation_count': sum(
            row['implementation_count'] for row in breakdown
        ),
        'languages': languages,
        'sources': breakdown,
    }


def evidence_source_version_ids_for_activity(
    scenario,
    activity,
    scope,
    language='',
):
    """Versions that can contribute to one target activity by lineage."""
    versions = get_evidence_versions(scenario, scope, language)
    return list(
        versions
        .filter(
            scenario__activities__lineage_key=activity.lineage_key,
        )
        .values_list('id', flat=True)
        .distinct()
    )
