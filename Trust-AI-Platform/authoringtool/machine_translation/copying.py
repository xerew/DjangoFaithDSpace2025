"""Structural copy of a scenario that a machine translation then fills in.

The copy keeps the source text until each unit is translated, so the
structure is identical by construction and progress is easy to count.
"""

from django.db import transaction

from authoringtool.models import (
    Activity,
    Answer,
    AnswerFeedback,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
    ScenarioTranslationJob,
)


def _free_name(base, user):
    name, counter = base, 2
    while Scenario.objects.filter(created_by=user, name=name).exists():
        name = f'{base} ({counter})'
        counter += 1
    return name


@transaction.atomic
def copy_for_translation(source, user, target_language, source_language):
    """Copy source's structure for user and queue its translation."""
    family = source.ensure_family()
    target = Scenario.objects.create(
        name=_free_name(f'{source.name} ({target_language})', user),
        learning_goals=source.learning_goals,
        description=source.description,
        age_of_students=source.age_of_students,
        subject_domains=source.subject_domains,
        language=target_language,
        suggested_learning_time=source.suggested_learning_time,
        image=source.image,
        visibility_status='private',
        created_by=user,
        updated_by=user,
        origin_scenario=source,
        family=family,
        variant_type='translation',
        ai_metrics_min_implementations=source.ai_metrics_min_implementations,
    )
    target.subjects.set(source.subjects.all())

    phases, activities, answers = {}, {}, {}
    for phase in source.phases.all():
        phases[phase.id] = Phase.objects.create(
            name=phase.name,
            description=phase.description,
            image=phase.image,
            order=phase.order,
            scenario=target,
            created_by=user,
            updated_by=user,
        )

    for activity in source.activities.order_by('id'):
        copied = Activity.objects.create(
            name=activity.name,
            text=activity.text,
            plain_text=activity.plain_text,
            is_evaluatable=activity.is_evaluatable,
            is_primary_ev=activity.is_primary_ev,
            must_wait=activity.must_wait,
            score_limit=activity.score_limit,
            order=activity.order,
            scenario=target,
            phase=phases.get(activity.phase_id),
            activity_type=activity.activity_type,
            helper=activity.helper,
            simulation=activity.simulation,
            experiment_ll=activity.experiment_ll,
            vr_ar_experiment=activity.vr_ar_experiment,
            lineage_key=activity.lineage_key,
            concept=activity.concept,
            created_by=user,
            updated_by=user,
        )
        activities[activity.id] = copied
        for answer in activity.answers.order_by('id'):
            copied_answer = Answer.objects.create(
                activity=copied,
                text=answer.text,
                is_correct=answer.is_correct,
                answer_weight=answer.answer_weight,
                image=answer.image,
                vid_url=answer.vid_url,
                created_by=user,
                updated_by=user,
            )
            answers[answer.id] = copied_answer
            for feedback in answer.feedbacks.order_by('id'):
                AnswerFeedback.objects.create(
                    answer=copied_answer,
                    text=feedback.text,
                    image=feedback.image,
                    vid_url=feedback.vid_url,
                    created_by=user,
                    updated_by=user,
                )

    for logic in NextQuestionLogic.objects.filter(activity__scenario=source):
        NextQuestionLogic.objects.create(
            activity=activities[logic.activity_id],
            answer=answers.get(logic.answer_id),
            next_activity=activities.get(logic.next_activity_id),
        )
    for branching in EvQuestionBranching.objects.filter(activity__scenario=source):
        EvQuestionBranching.objects.create(
            activity=activities[branching.activity_id],
            next_question_on_high=activities.get(branching.next_question_on_high_id),
            next_question_on_high_feedback=branching.next_question_on_high_feedback,
            next_question_on_mid=activities.get(branching.next_question_on_mid_id),
            next_question_on_mid_feedback=branching.next_question_on_mid_feedback,
            next_question_on_low=activities.get(branching.next_question_on_low_id),
            next_question_on_low_feedback=branching.next_question_on_low_feedback,
        )
    for bunch in QuestionBunch.objects.filter(activity_primary__scenario=source):
        QuestionBunch.objects.create(
            activity_primary=activities[bunch.activity_primary_id],
            activity_ids=[
                activities[activity_id].id
                for activity_id in bunch.activity_ids
                if activity_id in activities
            ],
        )

    start = activities.get(source.start_activity_id)
    if start:
        target.start_activity = start
        target.save(update_fields=['start_activity'])

    return ScenarioTranslationJob.objects.create(
        source_scenario=source,
        target_scenario=target,
        source_language=source_language,
        target_language=target_language,
        requested_by=user,
        id_map={
            'phases': {str(new.id): old for old, new in phases.items()},
            'activities': {str(new.id): old for old, new in activities.items()},
            'answers': {str(new.id): old for old, new in answers.items()},
        },
        units_total=1 + len(phases) + len(activities),
    )
