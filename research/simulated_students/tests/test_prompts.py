import json
import unittest

from simulated_students import prompts

ITEM = {
    'item_id': 7, 'language': 'Greek', 'age_min': 12, 'age_max': 15,
    'text': 'Which pendulum is faster? [Image 1]',
    'options': [
        {'id': 101, 'text': 'Short one', 'is_correct': True, 'image': None},
        {'id': 102, 'text': 'Heavy one', 'is_correct': False, 'image': 'images/7_option102.png'},
        {'id': 103, 'text': 'Same', 'is_correct': False, 'image': None},
    ],
    'images': ['images/7_1.png'],
    'preceding_material': 'The period depends on the length.',
}


class PromptTests(unittest.TestCase):
    def test_option_order_is_shuffled_reproducibly(self):
        first = prompts.option_order(ITEM, seed=3)
        self.assertEqual(sorted(first), [0, 1, 2])
        self.assertEqual(first, prompts.option_order(ITEM, seed=3))
        orders = {tuple(prompts.option_order(ITEM, seed=s)) for s in range(30)}
        self.assertGreater(len(orders), 1)

    def test_prompt_maps_letters_to_the_shuffled_options(self):
        text, letters = prompts.build_prompt(ITEM, 'low', 'none', order=[2, 0, 1])
        self.assertEqual(letters, {'A': 103, 'B': 101, 'C': 102})
        self.assertIn('A. Same', text)
        self.assertIn('B. Short one', text)
        self.assertIn('Greek', text)
        self.assertIn('12', text)
        self.assertIn(prompts.PERSONAS['low'], text)
        self.assertNotIn('The period depends on the length.', text)

    def test_material_context_adds_what_students_read_before(self):
        text, _ = prompts.build_prompt(ITEM, 'high', 'material', order=[0, 1, 2])
        self.assertIn('The period depends on the length.', text)

    def test_option_images_are_announced_in_order(self):
        text, _ = prompts.build_prompt(ITEM, 'mid', 'none', order=[0, 1, 2])
        self.assertIn('Image 2 shows option B', text)
        self.assertEqual(prompts.image_paths(ITEM), ['images/7_1.png', 'images/7_option102.png'])

    def test_parse_reply(self):
        letters = {'A': 103, 'B': 101, 'C': 102}
        reply = json.dumps({'choice': 'b', 'confidence': 70, 'reason': 'Shorter swings faster.'})
        self.assertEqual(prompts.parse_reply(reply, letters), (101, 'B', 70, 'Shorter swings faster.'))
        self.assertIsNone(prompts.parse_reply('not json', letters))
        self.assertIsNone(prompts.parse_reply(json.dumps({'choice': 'Z'}), letters))

    def test_parse_reply_recovers_json_broken_by_a_typographic_quote(self):
        # Seen with gemma3:4b: the reason string closed with ” instead of ".
        letters = {'A': 103, 'B': 101, 'C': 102}
        reply = '{"choice": "C", "confidence": 60, "reason": "Push harder, more charge goes through.”}'
        self.assertEqual(prompts.parse_reply(reply, letters),
                         (102, 'C', 60, 'Push harder, more charge goes through.'))
        self.assertIsNone(prompts.parse_reply('{"confidence": 60, "reason": "no choice”}', letters))

    def test_schema_only_allows_the_offered_letters(self):
        schema = prompts.reply_schema(3)
        self.assertEqual(schema['properties']['choice']['enum'], ['A', 'B', 'C'])


if __name__ == '__main__':
    unittest.main()
