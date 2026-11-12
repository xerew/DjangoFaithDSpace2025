"""Small scenarios with scored activities and student answers, for tests.

Each phase has one primary scored activity whose answers weigh 0, 1 and 2.
Phase 3 also has a QuestionBunch: its primary plus one more scored activity.
Activities are chained with default routes; the last phase's activity ends
the route.
"""

from types import SimpleNamespace

from django.contrib.auth.models import Group, User

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
    ScenarioImplementation,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)


def make_teacher(username='ml_teacher'):
    teachers, _ = Group.objects.get_or_create(name='teachers')
    teacher = User.objects.create_user(username, password='pass')
    teacher.groups.add(teachers)
    return teacher


def _question_type(teacher):
    kind = ActivityType.objects.filter(name='Question').first()
    return kind or ActivityType.objects.create(
        name='Question', created_by=teacher, updated_by=teacher,
    )


def make_activity(scenario, phase, teacher, name, primary=True):
    activity = Activity.objects.create(
        name=name, text=name, plain_text=name,
        scenario=scenario, phase=phase, activity_type=_question_type(teacher),
        is_evaluatable=True, is_primary_ev=primary,
        created_by=teacher, updated_by=teacher,
    )
    for weight in (0, 1, 2):
        Answer.objects.create(
            activity=activity, text=f'{name} w{weight}', is_correct=weight == 2,
            answer_weight=weight, created_by=teacher, updated_by=teacher,
        )
    return activity


def route(source, target):
    NextQuestionLogic.objects.create(activity=source, answer=None, next_activity=target)


def make_scenario(teacher, name='Pendulum', phases=4, visibility='public',
                  language='English', bunch_phase=3):
    """A scenario whose phase k holds `primaries[k-1]` (and the bunch)."""
    scenario = Scenario.objects.create(
        name=name, language=language, visibility_status=visibility,
        created_by=teacher, updated_by=teacher,
    )
    ns = SimpleNamespace(scenario=scenario, teacher=teacher, phases=[],
                         primaries=[], bunch_member=None)
    for index in range(phases):
        add_phase(ns)
        if index + 1 == bunch_phase:
            primary = ns.primaries[-1]
            member = make_activity(scenario, ns.phases[-1], teacher,
                                   f'{name} P{index + 1} extra', primary=False)
            QuestionBunch.objects.create(
                activity_primary=primary, activity_ids=[primary.id, member.id],
            )
            ns.bunch_member = member
    scenario.start_activity = ns.primaries[0]
    scenario.save()
    return ns


def add_phase(ns):
    """Append a phase with one primary scored activity, routed from the last."""
    number = len(ns.phases) + 1
    phase = Phase.objects.create(
        name=f'Phase {number}', scenario=ns.scenario, order=number,
        created_by=ns.teacher, updated_by=ns.teacher,
    )
    activity = make_activity(ns.scenario, phase, ns.teacher, f'{ns.scenario.name} P{number}')
    if ns.primaries:
        route(ns.primaries[-1], activity)
    ns.phases.append(phase)
    ns.primaries.append(activity)
    return activity


def make_student(username):
    department, _ = SchoolDepartment.objects.get_or_create(name='Physics')
    student = User.objects.create_user(username, password='pass')
    student.school_department = department
    student.save()
    return student


def answer(student, scenario, activity, weight, timing=30, created_on=None):
    implementation, _ = ScenarioImplementation.start_or_resume(student, scenario)
    UserScenarioScore.objects.get_or_create(
        user=student, scenario=scenario, implementation=implementation,
    )
    row = UserAnswer.objects.create(
        user=student, activity=activity,
        answer=activity.answers.get(answer_weight=weight),
        implementation=implementation, timing=timing,
    )
    if created_on is not None:
        UserAnswer.objects.filter(pk=row.pk).update(created_on=created_on)
    return row


def answer_all(student, ns, weights, bunch_weight=None):
    """Answer phase k's primary with weights[k-1]; the bunch member too."""
    for index, weight in enumerate(weights):
        if weight is None:
            continue
        answer(student, ns.scenario, ns.primaries[index], weight)
        if ns.bunch_member and ns.bunch_member.phase_id == ns.phases[index].id:
            answer(student, ns.scenario, ns.bunch_member,
                   weight if bunch_weight is None else bunch_weight)


def three_students(ns, prefix='st'):
    """High (100%), Moderate (67.5%) and Low (10%) students on a 4-phase scenario.

    Phase %: high 100/100/100/100; moderate 100/50/50/100; low 0/50/0/0.
    """
    high = make_student(f'{prefix}_high')
    moderate = make_student(f'{prefix}_moderate')
    low = make_student(f'{prefix}_low')
    answer_all(high, ns, [2, 2, 2, 2])
    answer_all(moderate, ns, [2, 1, 1, 2])
    answer_all(low, ns, [0, 1, 0, 0])
    return high, moderate, low
