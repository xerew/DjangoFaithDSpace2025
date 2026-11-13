import csv
import io
from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from authoringtool.models import (
    Activity,
    Answer,
    HelpRequest,
    NextQuestionLogic,
    ScenarioImplementation,
    TranslationMatch,
    UserAnswer,
)
from authoringtool.tasks import compute_student_performance_metrics
from categorization.dataset import load_runs, terminal_activity_ids
from categorization.tests.fixtures import (
    answer,
    answer_all,
    make_activity,
    make_scenario,
    make_student,
    make_teacher,
    route,
    three_students,
)


def runs_by_username(runs):
    return {run.username: run for run in runs}


class LoadRunsTests(TestCase):
    def setUp(self):
        self.teacher = make_teacher()
        self.ns = make_scenario(self.teacher)

    def test_only_public_scenarios(self):
        for visibility in ('private', 'org'):
            hidden = make_scenario(self.teacher, name=f'Hidden {visibility}', visibility=visibility)
            answer_all(make_student(f'student_{visibility}'), hidden, [2, 2, 2, 2])
        answer_all(make_student('public_student'), self.ns, [2, 2, 2, 2])
        runs = load_runs()
        self.assertEqual({run.username for run in runs}, {'public_student'})
        self.assertEqual({run.scenario_id for run in runs}, {self.ns.scenario.id})

    def test_label_matches_csv(self):
        three_students(self.ns)
        content = compute_student_performance_metrics(self.ns.scenario.id, None, None, None)['csv_content']
        csv_final = {row['Username']: row['Final Categorization']
                     for row in csv.DictReader(io.StringIO(content))}
        runs = runs_by_username(load_runs())
        self.assertEqual(len(csv_final), 3)
        for username, category in csv_final.items():
            self.assertEqual(runs[username].label, category, username)
            self.assertEqual(runs[username].finished_rule, 'route_end')

    def test_branch_ending_early_counts_as_finished(self):
        ns = make_scenario(self.teacher, name='Branching')
        second = ns.primaries[1]
        remediation = Activity.objects.create(
            name='Remediation', text='Read again', plain_text='Read again',
            scenario=ns.scenario, phase=ns.phases[1], is_evaluatable=False,
            created_by=self.teacher, updated_by=self.teacher,
        )
        Answer.objects.create(activity=remediation, text='Done',
                              created_by=self.teacher, updated_by=self.teacher)
        NextQuestionLogic.objects.create(
            activity=second, answer=second.answers.get(answer_weight=0), next_activity=remediation,
        )
        self.assertIn(remediation.id, terminal_activity_ids(ns.scenario))
        student = make_student('remediated')
        answer(student, ns.scenario, ns.primaries[0], 0)
        answer(student, ns.scenario, second, 0)
        UserAnswer.objects.create(
            user=student, activity=remediation, answer=remediation.answers.get(),
            implementation=ScenarioImplementation.objects.get(user=student), timing=20,
        )
        run = runs_by_username(load_runs())['remediated']
        self.assertEqual(run.finished_rule, 'route_end')
        self.assertEqual(run.label, 'Low')

    def test_terminal_ids_skip_bunch_members(self):
        terminals = terminal_activity_ids(self.ns.scenario)
        self.assertEqual(terminals, {self.ns.primaries[3].id})

    def test_dropout_is_scored_by_the_points_gathered(self):
        # Unanswered activities count as 0: phase 1 at 100% with weight 0.2 is 20%.
        answer_all(make_student('dropout'), self.ns, [2, None, None, None])
        run = runs_by_username(load_runs())['dropout']
        self.assertEqual(run.finished_rule, 'unfinished')
        self.assertAlmostEqual(run.final_percentage, 20.0)
        self.assertEqual(run.label, 'Low')

    def test_dropout_with_enough_points_can_be_high(self):
        # Phases 1-3 at 100%: 0.2 + 0.2 + 0.45 = 85%, above the High threshold.
        answer_all(make_student('late_dropout'), self.ns, [2, 2, 2, None])
        run = runs_by_username(load_runs())['late_dropout']
        self.assertEqual(run.finished_rule, 'unfinished')
        self.assertEqual(run.label, 'High')

    def test_deleted_activity_answers_fall_back(self):
        ns = make_scenario(self.teacher, name='Lost route')
        last = ns.primaries[3]
        closing = make_activity(ns.scenario, ns.phases[3], self.teacher, 'Closing', primary=False)
        route(last, closing)
        student = make_student('legacy_student')
        answer_all(student, ns, [2, 2, 2, 2])
        # The first activity was deleted later: its answers lost their activity.
        UserAnswer.objects.filter(user=student, activity=ns.primaries[0]).update(activity=None)
        run = runs_by_username(load_runs())['legacy_student']
        self.assertEqual(run.finished_rule, 'last_phase')
        self.assertIsNotNone(run.label)

    def test_without_lost_answers_the_same_run_is_unfinished(self):
        ns = make_scenario(self.teacher, name='Closing missed')
        closing = make_activity(ns.scenario, ns.phases[3], self.teacher, 'Closing', primary=False)
        route(ns.primaries[3], closing)
        answer_all(make_student('missed_closing'), ns, [2, 2, 2, 2])
        self.assertEqual(runs_by_username(load_runs())['missed_closing'].finished_rule, 'unfinished')

    def test_teacher_and_excluded_runs_left_out(self):
        answer_all(make_student('kept'), self.ns, [2, 2, 2, 2])
        answer_all(self.teacher, self.ns, [2, 2, 2, 2])
        excluded = make_student('excluded')
        answer_all(excluded, self.ns, [2, 2, 2, 2])
        ScenarioImplementation.objects.filter(user=excluded).update(data_quality_status='excluded')
        self.assertEqual({run.username for run in load_runs()}, {'kept'})

    def test_translation_group_key(self):
        greek = make_scenario(self.teacher, name='Ekkremes', language='Greek')
        TranslationMatch.objects.create(
            scenario_a=self.ns.scenario, scenario_b=greek.scenario, status='confirmed',
        )
        answer_all(make_student('en_student'), self.ns, [2, 2, 2, 2])
        answer_all(make_student('el_student'), greek, [2, 2, 2, 2])
        runs = runs_by_username(load_runs())
        self.assertEqual(runs['en_student'].group_key, runs['el_student'].group_key)
        self.assertEqual(runs['el_student'].group_key, min(self.ns.scenario.id, greek.scenario.id))
        self.assertEqual(runs['el_student'].language, 'Greek')

    def test_current_version_flag(self):
        answer_all(make_student('current'), self.ns, [2, 2, 2, 2])
        legacy = make_student('legacy')
        answer_all(legacy, self.ns, [2, 2, 2, 2])
        ScenarioImplementation.objects.filter(user=legacy).update(
            version_confidence='legacy_unknown', scenario_version=None,
        )
        runs = runs_by_username(load_runs())
        self.assertTrue(runs['current'].is_current_version)
        self.assertFalse(runs['legacy'].is_current_version)

    def test_help_seconds_removed_from_time(self):
        student = make_student('helped')
        answer_all(student, self.ns, [2, 2, 2, 2])
        ended = timezone.now()
        HelpRequest.objects.create(
            user=student, scenario=self.ns.scenario, activity=self.ns.primaries[0],
            phase=self.ns.phases[0], started_at=ended - timedelta(seconds=10),
            ended_at=ended, duration_seconds=10,
        )
        run = runs_by_username(load_runs())['helped']
        first = next(event for event in run.answers if event.activity_id == self.ns.primaries[0].id)
        self.assertEqual(first.active_seconds, 20)
        self.assertEqual(run.help_by_position[1], {'seconds': 10, 'requests': 1})
