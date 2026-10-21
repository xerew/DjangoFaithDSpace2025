"""Shared fixtures for the machine-translation tests (not a test module)."""

from django.contrib.auth.models import Group, User

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    AnswerFeedback,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
)


def make_teacher(username, admin=False):
    user = User.objects.create_user(username, f'{username}@example.com', 'pw')
    group, _ = Group.objects.get_or_create(name='teachers')
    user.groups.add(group)
    if admin:
        user.is_staff = True
        user.is_superuser = True
        user.save()
    return user


def build_source_scenario(owner, name='Ohm\'s Law', language='English'):
    """Two phases, three activities: a question with routed answers (one with
    feedback), an explanation, and an evaluated question with branching."""
    question, _ = ActivityType.objects.get_or_create(name='Question', defaults={'created_by': owner, 'updated_by': owner})
    explanation, _ = ActivityType.objects.get_or_create(name='Explanation', defaults={'created_by': owner, 'updated_by': owner})
    scenario = Scenario.objects.create(
        name=name,
        language=language,
        description='Learn how current depends on resistance.',
        learning_goals='Use V = IR.',
        visibility_status='public',
        created_by=owner,
        updated_by=owner,
    )
    phase_1 = Phase.objects.create(
        name='Explore', description='First look', scenario=scenario,
        created_by=owner, updated_by=owner,
    )
    phase_2 = Phase.objects.create(
        name='Check', description='Test yourself', scenario=scenario,
        created_by=owner, updated_by=owner,
    )
    ask = Activity.objects.create(
        name='Doubling resistance',
        text='<p>What happens to the current if the resistance doubles from 2 to 4 ohms?</p>',
        plain_text='What happens to the current if the resistance doubles from 2 to 4 ohms?',
        helper='Think of $V=IR$',
        scenario=scenario, phase=phase_1, activity_type=question,
        llm_context='old context', correct_count=7, incorrect_count=3,
        created_by=owner, updated_by=owner,
    )
    explain = Activity.objects.create(
        name='Why it halves',
        text='<p>The current <b>halves</b>.</p><p><img src="/media/images/ohm.png"></p>',
        plain_text='The current halves.',
        scenario=scenario, phase=phase_1, activity_type=explanation,
        created_by=owner, updated_by=owner,
    )
    check = Activity.objects.create(
        name='Final check',
        text='<p>Pick the right formula.</p>',
        plain_text='Pick the right formula.',
        is_evaluatable=True, is_primary_ev=True, score_limit=0.5,
        scenario=scenario, phase=phase_2, activity_type=question,
        created_by=owner, updated_by=owner,
    )
    halves = Answer.objects.create(
        activity=ask, text='It halves', is_correct=True, answer_weight=2,
        created_by=owner, updated_by=owner,
    )
    doubles = Answer.objects.create(
        activity=ask, text='It doubles', is_correct=False, answer_weight=1,
        created_by=owner, updated_by=owner,
    )
    AnswerFeedback.objects.create(
        answer=doubles, text='Not quite.', created_by=owner, updated_by=owner,
    )
    Answer.objects.create(
        activity=check, text='V = IR', is_correct=True,
        created_by=owner, updated_by=owner,
    )
    NextQuestionLogic.objects.create(activity=ask, answer=halves, next_activity=explain)
    NextQuestionLogic.objects.create(activity=ask, answer=doubles, next_activity=explain)
    NextQuestionLogic.objects.create(activity=explain, next_activity=check)
    EvQuestionBranching.objects.create(
        activity=check,
        next_question_on_low=ask,
        next_question_on_low_feedback='Try again.',
    )
    QuestionBunch.objects.create(activity_primary=check, activity_ids=[ask.id, check.id])
    scenario.start_activity = ask
    scenario.save(update_fields=['start_activity'])
    return scenario
