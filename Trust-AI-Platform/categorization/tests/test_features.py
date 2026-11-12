import math
from datetime import datetime, timedelta

from django.test import SimpleTestCase

from categorization.dataset import AnswerEvent, Run
from categorization.features import (
    build_norms,
    checkpoint_table,
    final_table,
    input_columns,
    run_inputs,
)

T0 = datetime(2026, 1, 1, 9, 0)


def event(activity_id, position, weight=2, seconds=30, minute=0, is_last=True):
    return AnswerEvent(
        activity_id=activity_id, phase_position=position, created_on=T0 + timedelta(minutes=minute),
        active_seconds=seconds, chosen_weight=weight, is_last=is_last,
    )


def make_run(answers, phase_scores, implementation_id=1, user_id=1, language='English',
             help_by_position=None, label='High', scenario_id=10):
    phase_count = len(phase_scores)
    return Run(
        implementation_id=implementation_id, user_id=user_id, username=f'u{user_id}',
        scenario_id=scenario_id, scenario_name='Pendulum', language=language, group_key=scenario_id,
        is_current_version=True, phase_count=phase_count, phase_scores=phase_scores,
        scored_by_position={pos: (1, 1) for pos in range(1, phase_count + 1)},
        answers=answers, help_by_position=help_by_position or {},
        final_percentage=90.0, label=label, finished_rule='route_end',
    )


def four_phase_run(**kwargs):
    answers = [event(101, 1, minute=0), event(102, 2, minute=1),
               event(103, 3, minute=2), event(104, 4, minute=3)]
    return make_run(answers, [(2, 2), (2, 2), (2, 2), (2, 2)], **kwargs)


class FeatureTests(SimpleTestCase):
    def test_checkpoint_sees_nothing_later(self):
        run = four_phase_run()
        norms = build_norms([run])
        before = run_inputs(run, norms, upto=1)
        run.answers += [event(102, 2, weight=0, seconds=500, minute=5, is_last=False),
                        event(103, 3, weight=1, seconds=1, minute=6, is_last=False)]
        run.help_by_position = {2: {'seconds': 99, 'requests': 3}, 3: {'seconds': 5, 'requests': 1}}
        run.phase_scores = [(2, 2), (0, 2), (1, 2), (0, 2)]
        after = run_inputs(run, norms, upto=1)
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            same = before[key] == after[key] or (
                isinstance(before[key], float) and math.isnan(before[key]) and math.isnan(after[key])
            )
            self.assertTrue(same, key)

    def test_final_table_has_no_final_percentage_input(self):
        runs = [four_phase_run()]
        columns = input_columns(final_table(runs, build_norms(runs)))
        self.assertNotIn('final_percentage', columns)
        self.assertNotIn('pct_so_far', columns)
        self.assertNotIn('checkpoint', columns)
        self.assertIn('p1_pct', columns)
        self.assertIn('is_current_version', columns)
        self.assertIn('language', columns)

    def test_baseline_label_uses_reached_phases_only(self):
        run = make_run([event(101, 1), event(102, 2), event(103, 3)],
                       [(2, 2), (0.8, 2), (0, 2), (0, 2)])
        table = checkpoint_table([run], build_norms([run]))
        row = table[table.checkpoint == 2].iloc[0]
        self.assertAlmostEqual(row.pct_so_far, 70.0)
        self.assertEqual(row.baseline_label, 'Moderate')
        # Checkpoints run from 1 to phase_count - 1, for phases answered.
        self.assertEqual(sorted(table.checkpoint), [1, 2, 3])

    def test_fast_share(self):
        runs = [
            make_run([event(101, 1, seconds=100)], [(2, 2)], implementation_id=1, user_id=1),
            make_run([event(101, 1, seconds=100)], [(2, 2)], implementation_id=2, user_id=2),
            make_run([event(102, 1, seconds=100)], [(2, 2)], implementation_id=3, user_id=3),
        ]
        target = make_run([event(101, 1, seconds=5), event(102, 1, seconds=5, minute=1),
                           event(101, 1, seconds=100, minute=2, is_last=False),
                           event(103, 1, seconds=100, minute=3), event(104, 1, seconds=100, minute=4)],
                          [(2, 2)], implementation_id=4, user_id=4)
        norms = build_norms(runs + [target])
        # Medians: 101 -> 100 (100, 100, 5), 102 -> 52.5, 103 -> 100, 104 -> 100.
        # Last answers 101 (5 < 10) and 102 (5 < 5.25) are very fast: 2 of 4.
        self.assertAlmostEqual(run_inputs(target, norms, upto=None)['fast_share'], 0.5)

    def test_retries_counts_changed_answers(self):
        run = make_run([event(101, 1, weight=0, is_last=False), event(101, 1, minute=1),
                        event(102, 1, weight=0, minute=2, is_last=False),
                        event(102, 1, weight=1, minute=3, is_last=False), event(102, 1, minute=4)],
                       [(4, 4)])
        self.assertEqual(run_inputs(run, build_norms([run]), upto=None)['retries'], 3)

    def test_blank_language_is_unspecified(self):
        run = four_phase_run(language='')
        self.assertEqual(run_inputs(run, build_norms([run]), upto=None)['language'], 'Unspecified')

    def test_answers_without_activity_are_ignored(self):
        run = four_phase_run()
        run.answers.append(event(None, None, minute=9, is_last=False))
        inputs = run_inputs(run, build_norms([run]), upto=None)
        self.assertEqual(inputs['retries'], 0)
