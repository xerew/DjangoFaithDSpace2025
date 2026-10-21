"""The machine-translation queue: one job at a time, oldest first.

An outage puts the job back in the queue with its finished units kept; only
output that keeps failing the checks, or an unexpected error, fails a job,
and then visibly. A running job renews its claim and the queue lock after
every unit, so a worker that dies frees the queue within STUCK_AFTER.
"""

import logging
import uuid
from datetime import timedelta

from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import F
from django.utils import timezone

from authoringtool.models import Scenario, ScenarioTranslationJob, TranslationMatch
from authoringtool.translation_matching import build_flow_profile

from .client import (
    TranslationOutputInvalid,
    TranslationServiceUnavailable,
    TranslationTimeout,
    translate_fields,
)
from .health import check_translation_service, record_service_status
from .units import apply_unit, pending_units
from .validation import validate_unit

logger = logging.getLogger(__name__)

LOCK_KEY = 'authoringtool:drain-translation-queue'
MAX_UNIT_ATTEMPTS = 3

Job = ScenarioTranslationJob

_OK = 'ok'
_DONE = 'done'
_FAILED = 'failed'
_REQUEUED = 'requeued'
_GONE = 'gone'
_LOST = 'lost'


def stuck_after():
    """Longest one unit may take (all its attempts) plus a margin."""
    return timedelta(seconds=settings.MACHINE_TRANSLATION_TIMEOUT * MAX_UNIT_ATTEMPTS + 600)


def _lock_timeout():
    return int(stuck_after().total_seconds()) + 300


def release_stuck(now):
    """Jobs whose worker died mid-run go back to the queue."""
    return Job.objects.filter(
        status=Job.STATUS_TRANSLATING,
        claimed_at__lt=now - stuck_after(),
    ).update(status=Job.STATUS_WAITING, claimed_at=None)


def _claim(job_id, now):
    claimed = Job.objects.filter(pk=job_id, status=Job.STATUS_WAITING).update(
        status=Job.STATUS_TRANSLATING,
        claimed_at=now,
        attempts=F('attempts') + 1,
    ) == 1
    if claimed:
        Job.objects.filter(pk=job_id, started_at__isnull=True).update(started_at=now)
    return claimed


class _Run:
    """One drain run: owns the queue lock and the claim of the current job."""

    def __init__(self):
        self.token = uuid.uuid4().hex
        self.claimed_at = None

    def acquire(self):
        return cache.add(LOCK_KEY, self.token, _lock_timeout())

    def release(self):
        if cache.get(LOCK_KEY) == self.token:
            cache.delete(LOCK_KEY)

    def heartbeat(self, job):
        """Renew the claim and lock; False if another run took the job over."""
        now = timezone.now()
        renewed = Job.objects.filter(
            pk=job.pk, status=Job.STATUS_TRANSLATING, claimed_at=self.claimed_at,
        ).update(claimed_at=now) == 1
        if renewed:
            self.claimed_at = now
            cache.touch(LOCK_KEY, _lock_timeout())
        return renewed


def _requeue(job):
    Job.objects.filter(pk=job.pk).update(status=Job.STATUS_WAITING, claimed_at=None)


def _fail(job, error):
    Job.objects.filter(pk=job.pk).update(
        status=Job.STATUS_FAILED, error=error, finished_at=timezone.now(),
    )


def retry_job(job):
    """Put a failed job back in the queue; finished units are kept."""
    Job.objects.filter(pk=job.pk, status=Job.STATUS_FAILED).update(
        status=Job.STATUS_WAITING,
        error='',
        claimed_at=None,
        finished_at=None,
        notified_at=None,
    )


def _context(job):
    target = Scenario.objects.get(pk=job.target_scenario_id)
    return f'{target.name}: {target.learning_goals}'


def _service_up():
    status = check_translation_service()
    return status['reachable'] and status['model_present']


def _translate_unit(job, unit):
    if not unit.fields:
        apply_unit(job, unit, {})
        return _OK
    errors = []
    for _ in range(MAX_UNIT_ATTEMPTS):
        try:
            translated = translate_fields(
                unit.fields, job.source_language, job.target_language, _context(job),
            )
        except (TranslationServiceUnavailable, TranslationTimeout) as exc:
            # Only a real outage requeues; an error Ollama keeps returning for
            # this unit while it is up counts as a failed attempt.
            if not _service_up():
                record_service_status(False, False)
                _requeue(job)
                return _REQUEUED
            errors = [
                'the translation took too long'
                if isinstance(exc, TranslationTimeout)
                else f'the translation service refused it ({exc})'
            ]
            continue
        except TranslationOutputInvalid as exc:
            errors = [str(exc)]
            continue
        errors = validate_unit(unit.fields, translated, unit.max_lengths)
        if not errors:
            apply_unit(job, unit, translated)
            return _OK
    _fail(job, f'{unit.label}: {"; ".join(errors)}')
    return _FAILED


@transaction.atomic
def finish_job(job):
    """Version the translation, pair it with its source and mark it done."""
    target = Scenario.objects.get(pk=job.target_scenario_id)
    target.ensure_current_version(
        created_by=job.requested_by, change_summary='Machine translation',
    )
    source = Scenario.objects.filter(pk=job.source_scenario_id).first()
    if source is not None:
        pairs = [
            [old_id, int(new_id)]
            for new_id, old_id in job.id_map.get('activities', {}).items()
        ]
        first, second = source, target
        if first.id > second.id:
            first, second = second, first
            pairs = [[new_id, old_id] for old_id, new_id in pairs]
        profile_a, profile_b = build_flow_profile(first), build_flow_profile(second)
        TranslationMatch.objects.update_or_create(
            scenario_a=first,
            scenario_b=second,
            defaults={
                'origin': 'machine_translation',
                'status': 'confirmed',
                'exact_match': True,
                'confidence': 100,
                'structure_score': 1.0,
                'route_score': 1.0,
                'anchor_score': 1.0,
                'differences': [],
                'activity_map': sorted(pairs),
                'flow_signature_a': profile_a['signature'],
                'flow_signature_b': profile_b['signature'],
                'structure_signature_a': profile_a['structure_signature'],
                'structure_signature_b': profile_b['structure_signature'],
                'review_notes': 'Created by machine translation.',
            },
        )
    Job.objects.filter(pk=job.pk).update(
        status=Job.STATUS_DONE,
        finished_at=timezone.now(),
        model=settings.LLM_MACHINE_TRANSLATION_MODEL,
    )


def _run(job, run):
    step = 'Preparing the translation'
    try:
        job.refresh_from_db()
        for unit in pending_units(job):
            step = unit.label
            outcome = _translate_unit(job, unit)
            if outcome != _OK:
                return outcome
            if not run.heartbeat(job):
                return _LOST
        step = 'Finishing the translation'
        finish_job(job)
        return _DONE
    except ObjectDoesNotExist:
        return _GONE  # the translation was deleted while it ran
    except Exception as exc:  # never leave a job stuck in "translating"
        logger.exception('Machine translation job %s failed', job.pk)
        _fail(job, f'{step}: {exc}')
        return _FAILED


def drain_translation_queue(now=None):
    """Translate waiting jobs one at a time until none is left."""
    result = {_DONE: 0, _FAILED: 0, _REQUEUED: 0}
    run = _Run()
    if not run.acquire():
        return result
    try:
        release_stuck(now or timezone.now())
        while not Job.objects.filter(status=Job.STATUS_TRANSLATING).exists():
            job = Job.objects.filter(status=Job.STATUS_WAITING).order_by('created_at', 'id').first()
            run.claimed_at = timezone.now()
            if job is None or not _claim(job.pk, run.claimed_at):
                break
            outcome = _run(job, run)
            if outcome in result:
                result[outcome] += 1
            if outcome in (_REQUEUED, _LOST):
                break
    finally:
        run.release()
    return result
