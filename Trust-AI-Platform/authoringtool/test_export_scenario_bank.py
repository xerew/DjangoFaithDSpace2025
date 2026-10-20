"""export_scenario_bank writes each public scenario as a graph (activities,
routes, evaluation branching) plus aggregate results of real students' runs,
for the scenario walk-through of the simulated-students study. It never
writes data about individual students."""
import io
import json
import os
import shutil
import tempfile

from django.contrib.auth.models import Group, User
from django.core.management import call_command
from django.test import TestCase, override_settings

from authoringtool.models import (
    Activity, ActivityType, Answer, EvQuestionBranching, NextQuestionLogic, Phase,
    QuestionBunch, Scenario, ScenarioImplementation, UserAnswer,
)


class ExportScenarioBankTests(TestCase):
    """Pendulum: P1 (page) -> Q1. Q1 right -> Q2, wrong -> P2 (remedial page) -> Q2.
    Q2 evaluates Q1+Q2: both right -> H, one -> M, none -> L. H, M, L end the scenario."""

    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)
        settings_override = override_settings(MEDIA_ROOT=self.out_dir)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        self.output = os.path.join(self.out_dir, 'scenario_bank.jsonl')

        self.owner = User.objects.create_user('walk_owner')
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='public', language='English',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(name='Explore', scenario=self.scenario,
                                          created_by=self.owner, updated_by=self.owner)
        self.p1 = self.page('<p>Watch the pendulum swing.</p>')
        self.q1, (self.q1_right, self.q1_wrong, self.q1_other) = self.question('Which is faster?')
        self.p2 = self.page('<p>A shorter pendulum swings faster.</p>')
        self.q2, (self.q2_right, self.q2_wrong, _) = self.question('Does mass matter?', evaluatable=True)
        self.high = self.page('<p>Well done.</p>', score_limit=1.0)
        self.mid = self.page('<p>Almost.</p>', score_limit=0.5)
        self.low = self.page('<p>Let us try again.</p>')
        self.scenario.start_activity = self.p1
        self.scenario.save()

        self.route(self.p1, None, self.q1)
        self.route(self.q1, self.q1_right, self.q2)
        self.route(self.q1, self.q1_wrong, self.p2)
        self.route(self.q1, self.q1_other, self.p2)
        self.route(self.p2, None, self.q2)
        QuestionBunch.objects.create(activity_ids=[self.q1.id, self.q2.id], activity_primary=self.q2)
        EvQuestionBranching.objects.create(
            activity=self.q2, next_question_on_high=self.high,
            next_question_on_mid=self.mid, next_question_on_low=self.low,
        )

        high_run = [(self.p1, None), (self.q1, self.q1_right), (self.q2, self.q2_right), (self.high, None)]
        mid_run = [(self.p1, None), (self.q1, self.q1_wrong), (self.p2, None), (self.q2, self.q2_right), (self.mid, None)]
        low_run = [(self.p1, None), (self.q1, self.q1_wrong), (self.p2, None), (self.q2, self.q2_wrong), (self.low, None)]
        for i in range(5):
            self.play(f'high_{i}', high_run)
        for i in range(4):
            self.play(f'mid_{i}', mid_run)
        for i in range(2):
            self.play(f'low_{i}', low_run)
        self.play('stopper', [(self.p1, None), (self.q1, self.q1_right)])
        # A legacy student (no implementation) who started over: only the first run counts.
        self.play('restarter', high_run + low_run)
        # A student with two implementations: only the first counts.
        twice = User.objects.create_user('twice')
        for status, run in (('completed', high_run), ('active', low_run)):
            implementation = ScenarioImplementation.objects.create(
                user=twice, scenario=self.scenario, status=status, version_confidence='legacy_unknown',
            )
            self.play(twice, run, implementation)
        teacher = User.objects.create_user('walk_teacher')
        teacher.groups.add(Group.objects.create(name='teachers'))
        self.play(teacher, low_run)

    # ── builders ──────────────────────────────────────────────────────────
    def activity_type(self, name):
        return ActivityType.objects.get_or_create(
            name=name, defaults={'created_by': self.owner, 'updated_by': self.owner},
        )[0]

    def page(self, text, score_limit=0.0):
        return Activity.objects.create(
            name=text[:20], text=text, phase=self.phase, scenario=self.scenario,
            activity_type=self.activity_type('Explanation'), score_limit=score_limit,
            created_by=self.owner, updated_by=self.owner,
        )

    def question(self, text, evaluatable=False):
        activity = Activity.objects.create(
            name=text[:20], text=f'<p>{text}</p>', phase=self.phase, scenario=self.scenario,
            activity_type=self.activity_type('Question'), is_evaluatable=evaluatable,
            created_by=self.owner, updated_by=self.owner,
        )
        options = tuple(
            Answer.objects.create(activity=activity, text=label, is_correct=(i == 0), answer_weight=int(i == 0),
                                  created_by=self.owner, updated_by=self.owner)
            for i, label in enumerate(['A. Short one', 'Heavy one', 'Same'])
        )
        return activity, options

    def route(self, activity, answer, next_activity):
        NextQuestionLogic.objects.create(activity=activity, answer=answer, next_activity=next_activity)

    def play(self, student, steps, implementation=None):
        if isinstance(student, str):
            student = User.objects.create_user(student)
        for activity, answer in steps:
            UserAnswer.objects.create(user=student, activity=activity, answer=answer,
                                      implementation=implementation, timing=5)

    def export(self, **options):
        call_command('export_scenario_bank', output=self.output, min_students=5, stdout=io.StringIO(), **options)
        with open(self.output, encoding='utf-8') as handle:
            return [json.loads(line) for line in handle]

    # ── tests ─────────────────────────────────────────────────────────────
    def test_exports_the_scenario_graph_as_plato_routes_it(self):
        [record] = self.export()
        self.assertEqual(record['scenario_id'], self.scenario.id)
        self.assertEqual(record['start'], self.p1.id)
        nodes = record['activities']
        self.assertEqual(nodes[str(self.p1.id)]['kind'], 'page')
        self.assertEqual(nodes[str(self.p1.id)]['next'], self.q1.id)
        self.assertEqual(nodes[str(self.p1.id)]['text'], 'Watch the pendulum swing.')
        self.assertIsNone(nodes[str(self.high.id)]['next'])

        q1 = nodes[str(self.q1.id)]
        self.assertEqual(q1['kind'], 'question')
        self.assertIsNone(q1['evaluation'])
        self.assertEqual(
            [(o['id'], o['text'], o['is_correct'], o['weight'], o['next']) for o in q1['options']],
            [(self.q1_right.id, 'Short one', True, 1, self.q2.id),
             (self.q1_wrong.id, 'Heavy one', False, 0, self.p2.id),
             (self.q1_other.id, 'Same', False, 0, self.p2.id)],
        )
        self.assertEqual(nodes[str(self.q2.id)]['evaluation'], {
            'bunch': sorted([self.q1.id, self.q2.id]),
            'high': {'to': self.high.id, 'limit': 1.0},
            'mid': {'to': self.mid.id, 'limit': 0.5},
            'low': {'to': self.low.id},
        })

    def test_real_runs_are_summarised_per_scenario(self):
        [record] = self.export()
        real = record['real']
        # 5 high + 4 mid + 2 low + stopper + restarter + twice; the teacher is not a student.
        self.assertEqual((real['students'], real['runs_used']), (14, 14))
        self.assertEqual(real['completed'], 13)
        self.assertEqual(real['follows_routes'], 1.0)
        q1, q2, p2 = str(self.q1.id), str(self.q2.id), str(self.p2.id)
        self.assertEqual(real['transitions'][q1], {q2: 7, p2: 6, 'STOP': 1})
        self.assertEqual(real['transitions'][str(self.high.id)], {'END': 7})
        self.assertEqual(real['branching'][q2], {'high': 7, 'mid': 4, 'low': 2})
        self.assertEqual(real['score_histogram'], {'2': 7, '1': 4, '0': 2})
        self.assertEqual(real['path_length_histogram'], {'4': 7, '5': 6})

    def test_only_paths_taken_by_enough_students_are_listed(self):
        [record] = self.export()
        self.assertEqual(record['real']['paths'], [
            {'path': [self.p1.id, self.q1.id, self.q2.id, self.high.id], 'n': 7},
        ])

    def test_item_statistics_use_each_students_first_run(self):
        [record] = self.export()
        q1 = record['real']['items'][str(self.q1.id)]
        self.assertEqual(q1['n'], 14)
        self.assertAlmostEqual(q1['p_correct'], 8 / 14)
        self.assertAlmostEqual(q1['option_shares'][str(self.q1_wrong.id)], 6 / 14)
        # Students who got Q1 right also got Q2 right more often.
        self.assertGreater(q1['point_biserial'], 0)

    def test_runs_that_do_not_follow_the_current_routes_are_counted_but_not_summarised(self):
        # E.g. students who used the scenario before a teacher changed its routes.
        self.play('skipper', [(self.p1, None), (self.q2, self.q2_right), (self.mid, None)])
        [record] = self.export()
        real = record['real']
        self.assertEqual((real['students'], real['runs_used']), (15, 14))
        self.assertAlmostEqual(real['follows_routes'], 14 / 15)
        self.assertEqual(real['completed'], 13)
        self.assertNotIn(str(self.q2.id), real['transitions'][str(self.p1.id)])
        self.assertEqual(real['branching'][str(self.q2.id)], {'high': 7, 'mid': 4, 'low': 2})

    def test_the_minimum_applies_to_runs_on_the_current_routes(self):
        for i in range(2):
            self.play(f'skipper_{i}', [(self.p1, None), (self.q2, self.q2_right), (self.mid, None)])
        call_command('export_scenario_bank', output=self.output, min_students=15, stdout=io.StringIO())
        with open(self.output, encoding='utf-8') as handle:
            self.assertEqual(handle.read().strip(), '')

    def test_scenarios_with_too_few_students_or_not_public_are_left_out(self):
        self.assertEqual(len(self.export()), 1)
        call_command('export_scenario_bank', output=self.output, min_students=15, stdout=io.StringIO())
        with open(self.output, encoding='utf-8') as handle:
            self.assertEqual(handle.read().strip(), '')
        self.scenario.visibility_status = 'private'
        self.scenario.save()
        self.assertEqual(self.export(), [])

    def test_no_student_data_in_the_export(self):
        self.export()
        with open(self.output, encoding='utf-8') as handle:
            content = handle.read()
        for username in ('high_0', 'stopper', 'restarter', 'twice', 'walk_teacher'):
            self.assertNotIn(username, content)
        self.assertNotIn('user', content)
        with open(self.output + '.manifest.json', encoding='utf-8') as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest['scenarios'], 1)
        self.assertEqual(manifest['min_path_students'], 5)
