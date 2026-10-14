"""Keep student evidence when teachers delete content.

Teachers may remove scenarios, phases, activities and answer options, but
what students did must survive for analytics and research:

* a scenario with student data is archived (hidden, restorable), not deleted;
* a removed phase, activity or answer option leaves the students' answers in
  place, detached, with a snapshot in ``UserAnswer.detached_from``.

The foreign keys back this up: evidence uses PROTECT or SET_NULL, so nothing
cascades it away even outside these helpers.
"""
from django.db import transaction
from django.db.models import ProtectedError, Q
from django.utils import timezone

from .models import (
    ActivityProposal,
    HelpRequest,
    PhetLabSessions,
    RemoteLabSession,
    ScenarioImplementation,
    UserAnswer,
    UserScenarioScore,
)

TEACHERS = Q(user__groups__name='teachers')


def answers_touching(activities=(), answers=()):
    """Students' answers that a removal would detach (teachers excluded)."""
    return UserAnswer.objects.filter(
        Q(activity__in=list(activities)) | Q(answer__in=list(answers))
    ).exclude(TEACHERS)


def removal_impact(activities=(), answers=()):
    """Return how many answers, and from how many students, a removal touches."""
    touched = answers_touching(activities, answers)
    return {
        'answers': touched.count(),
        'students': touched.values('user').distinct().count(),
    }


def scenario_impact(scenario):
    """Return how many students used a scenario and how many answers they gave."""
    answers = UserAnswer.objects.filter(activity__scenario=scenario).exclude(TEACHERS)
    students = set(answers.values_list('user', flat=True))
    students.update(
        ScenarioImplementation.objects.filter(scenario=scenario)
        .exclude(TEACHERS).values_list('user', flat=True)
    )
    students.update(
        UserScenarioScore.objects.filter(scenario=scenario)
        .exclude(TEACHERS).values_list('user', flat=True)
    )
    return {'answers': answers.count(), 'students': len(students)}


def _answer_snapshot(answer):
    return {
        'id': answer.id,
        'text': answer.text,
        'is_correct': answer.is_correct,
        'weight': answer.answer_weight,
    }


def _activity_snapshot(activity):
    return {
        'id': activity.id,
        'name': activity.name,
        'type': activity.activity_type.name if activity.activity_type_id else None,
        'phase_id': activity.phase_id,
        'phase': activity.phase.name if activity.phase_id else None,
        'scenario_id': activity.scenario_id,
    }


def detach_student_evidence(activities=(), answers=(), removed_by=None):
    """Snapshot what is about to be removed onto the evidence that uses it.

    Covers students' answers and lab sessions, and teachers' AI proposals
    (with their reviews). Call just before deleting: the delete then sets the
    foreign keys to NULL and the records themselves are kept.
    """
    stamp = {
        'detached_at': timezone.now().isoformat(),
        'detached_by': removed_by.id if removed_by else None,
    }
    for activity in activities:
        info = {**stamp, 'activity': _activity_snapshot(activity)}
        ActivityProposal.objects.filter(activity=activity).update(detached_from=info)
        RemoteLabSession.objects.filter(activity=activity).update(detached_from=info)
        PhetLabSessions.objects.filter(activity=activity).update(detached_from=info)
        chosen = {a.id: _answer_snapshot(a) for a in activity.answers.all()}
        rows = UserAnswer.objects.filter(activity=activity)
        for answer_id in set(rows.values_list('answer_id', flat=True)):
            detached = dict(info)
            if answer_id in chosen:
                detached['answer'] = chosen[answer_id]
            rows.filter(answer_id=answer_id).update(detached_from=detached)
    for answer in answers:
        UserAnswer.objects.filter(answer=answer).update(
            detached_from={**stamp, 'answer': _answer_snapshot(answer)}
        )


def _delete_teacher_previews(scenario):
    """Teachers' own test runs are not evidence and may go with the scenario."""
    UserAnswer.objects.filter(activity__scenario=scenario).filter(TEACHERS).delete()
    HelpRequest.objects.filter(scenario=scenario).filter(TEACHERS).delete()
    PhetLabSessions.objects.filter(activity__scenario=scenario).filter(TEACHERS).delete()
    RemoteLabSession.objects.filter(scenario=scenario).filter(TEACHERS).delete()
    UserScenarioScore.objects.filter(scenario=scenario).filter(TEACHERS).delete()
    ScenarioImplementation.objects.filter(scenario=scenario).filter(TEACHERS).delete()


def archive_scenario(scenario, user):
    scenario.archived_at = timezone.now()
    scenario.archived_by = user
    scenario.save(update_fields=['archived_at', 'archived_by'])


def restore_scenario(scenario):
    scenario.archived_at = None
    scenario.archived_by = None
    scenario.save(update_fields=['archived_at', 'archived_by'])


def delete_or_archive_scenario(scenario, user):
    """Delete a scenario nobody used; archive one with student data.

    Returns 'deleted' or 'archived'.
    """
    if not scenario.has_student_evidence():
        try:
            with transaction.atomic():
                _delete_teacher_previews(scenario)
                scenario.delete()
            return 'deleted'
        except ProtectedError:
            pass  # evidence of another kind (feedback, questionnaire, labs)
    archive_scenario(scenario, user)
    return 'archived'
