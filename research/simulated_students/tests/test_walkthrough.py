import json
import os
import shutil
import tempfile
import unittest

from simulated_students import prompts, walkthrough
from simulated_students.tests.scenario_fixture import H, L, M, P1, P2, Q1, Q2, page, scenario

RIGHT = {'Short one', 'No'}
WRONG = {'Heavy one', 'Yes'}


class FakeChat:
    """Picks the offered option whose text is in `picks` (by persona when a dict)."""

    def __init__(self, picks=RIGHT, vision_models=(), broken=False):
        self.picks = picks
        self.vision_models = set(vision_models)
        self.broken = broken
        self.calls = []

    def has_vision(self, model):
        return model in self.vision_models

    def chat(self, model, messages, schema, temperature, seed, think, num_ctx):
        self.calls.append({'model': model, 'messages': [dict(m) for m in messages], 'seed': seed})
        if self.broken:
            return 'not json', 0.01
        picks = self.picks
        if isinstance(picks, dict):
            picks = next(p for persona, p in picks.items() if prompts.PERSONAS[persona] in messages[0]['content'])
        offered = [line.split('. ', 1) for line in messages[-1]['content'].splitlines()
                   if len(line) > 3 and line[1:3] == '. ']
        letter = next((letter for letter, text in offered if text in picks), offered[0][0])
        return json.dumps({'choice': letter, 'confidence': 70, 'reason': 'I think so'}), 0.01


def walk(client, data=None, persona='mid', model='m', **options):
    return walkthrough.walk(client, data or scenario(), model, persona, replicate=0,
                            temperature=0.8, base_dir='.', **options)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.nodes = walkthrough.nodes_of(scenario())

    def route(self, activity_id, option_id, answered):
        return walkthrough.route(self.nodes, self.nodes[activity_id], option_id, answered)

    def test_pages_and_plain_questions_follow_their_routes(self):
        self.assertEqual(self.route(P1, None, []), (Q1, None))
        self.assertEqual(self.route(Q1, 21, [(Q1, 21)]), (Q2, None))
        self.assertEqual(self.route(Q1, 23, [(Q1, 23)]), (P2, None))
        self.assertEqual(self.route(H, None, []), (None, None))

    def test_evaluation_uses_the_average_weight_of_the_bunch(self):
        self.assertEqual(self.route(Q2, 41, [(Q1, 21), (Q2, 41)]), (H, 'high'))
        self.assertEqual(self.route(Q2, 41, [(Q1, 22), (Q2, 41)]), (M, 'mid'))
        self.assertEqual(self.route(Q2, 42, [(Q1, 22), (Q2, 42)]), (L, 'low'))

    def test_incomplete_evaluation_branching_is_an_error(self):
        self.nodes[Q2]['evaluation']['mid']['to'] = None
        with self.assertRaises(walkthrough.RouteError):
            self.route(Q2, 41, [(Q1, 21), (Q2, 41)])


class WalkTests(unittest.TestCase):
    def test_a_student_goes_through_the_scenario_along_the_real_routes(self):
        row = walk(FakeChat(RIGHT))
        self.assertEqual([step['activity_id'] for step in row['steps']], [P1, Q1, Q2, H])
        self.assertEqual(row['end_reason'], 'end')
        self.assertEqual((row['total_score'], row['questions'], row['correct']), (2, 2, 2))
        self.assertEqual(row['steps'][1]['choice_option_id'], 21)

        row = walk(FakeChat(WRONG))
        self.assertEqual([step['activity_id'] for step in row['steps']], [P1, Q1, P2, Q2, L])
        self.assertEqual(row['total_score'], 0)

    def test_the_student_remembers_the_pages_and_its_earlier_answers(self):
        client = FakeChat(WRONG)
        walk(client)
        first, second = client.calls
        self.assertIn('Watch the pendulum swing.', first['messages'][-1]['content'])
        self.assertIn('Which is faster?', first['messages'][-1]['content'])
        history = [m['content'] for m in second['messages']]
        self.assertIn('Which is faster?', history[1])
        self.assertEqual(second['messages'][2]['role'], 'assistant')
        self.assertIn('A shorter pendulum swings faster.', history[-1])
        self.assertNotIn('Watch the pendulum swing.', history[-1])  # already read before Q1

    def test_unusable_replies_end_the_run_after_one_retry(self):
        client = FakeChat(broken=True)
        row = walk(client)
        self.assertEqual(row['end_reason'], 'invalid_reply')
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([step['activity_id'] for step in row['steps']], [P1])

    def test_a_network_error_is_retried_once(self):
        class Flaky(FakeChat):
            failures = 1

            def chat(self, *args):
                if self.failures:
                    self.failures -= 1
                    raise TimeoutError('read timed out')
                return super().chat(*args)

        row = walk(Flaky(RIGHT))
        self.assertEqual(row['end_reason'], 'end')
        self.assertEqual([step['activity_id'] for step in row['steps']], [P1, Q1, Q2, H])

        class Down(FakeChat):
            def chat(self, *args):
                raise TimeoutError('read timed out')

        row = walk(Down())
        self.assertEqual(row['end_reason'], 'error')
        self.assertIn('TimeoutError', row['error'])

    def test_route_errors_end_the_run(self):
        data = scenario()
        data['activities'][str(Q2)]['evaluation']['high']['to'] = None
        row = walk(FakeChat(RIGHT), data)
        self.assertEqual(row['end_reason'], 'route_error')
        self.assertEqual([step['activity_id'] for step in row['steps']], [P1, Q1, Q2])

    def test_endless_loops_stop(self):
        data = scenario()
        data['activities'] = {'1': page(1, 'A', 2), '2': page(2, 'B', 1)}
        data['start'] = 1
        row = walk(FakeChat(), data, max_steps=10)
        self.assertEqual(row['end_reason'], 'max_steps')
        self.assertEqual(len(row['steps']), 10)

    def test_the_option_order_is_the_same_for_every_model(self):
        first, second = walk(FakeChat(), model='a'), walk(FakeChat(), model='b')
        self.assertEqual([s.get('offered_order') for s in first['steps']],
                         [s.get('offered_order') for s in second['steps']])

    def test_images_are_sent_only_to_vision_models(self):
        data = scenario()
        data['activities'][str(Q1)]['images'] = ['images/2_1.png']
        data['activities'][str(Q1)]['text'] = 'Which is faster? [Image 1]'
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        os.makedirs(os.path.join(base, 'images'))
        with open(os.path.join(base, 'images', '2_1.png'), 'wb') as image:
            image.write(b'png')

        client = FakeChat(vision_models={'eyes'})
        row = walkthrough.walk(client, data, 'eyes', 'mid', 0, 0.8, base, vision=True)
        self.assertTrue(row['used_images'])
        self.assertEqual(len(client.calls[0]['messages'][-1]['images']), 1)

        client = FakeChat()
        row = walkthrough.walk(client, data, 'blind', 'mid', 0, 0.8, base, vision=False)
        self.assertFalse(row['used_images'])
        self.assertNotIn('images', client.calls[0]['messages'][-1])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.bank = os.path.join(self.dir, 'scenario_bank.jsonl')
        with open(self.bank, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(scenario()) + '\n')
        self.out = os.path.join(self.dir, 'runs', 'walk.jsonl')

    def run_walk(self, client):
        return walkthrough.run(client, self.bank, self.out, models=['m'], personas=['low', 'high'],
                               replicates=2, temperature=0.8, workers=2)

    def results(self):
        with open(self.out, encoding='utf-8') as handle:
            return [json.loads(line) for line in handle]

    def test_one_row_per_simulated_student_and_resumable(self):
        self.run_walk(FakeChat({'low': WRONG, 'high': RIGHT}))
        rows = self.results()
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['persona']: r['total_score'] for r in rows}, {'low': 0, 'high': 2})
        self.assertEqual(rows[0]['prompt_version'], walkthrough.PROMPT_VERSION)

        client = FakeChat()
        self.run_walk(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(self.results()), 4)


if __name__ == '__main__':
    unittest.main()
