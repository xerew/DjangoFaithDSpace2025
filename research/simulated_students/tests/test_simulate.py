import json
import os
import shutil
import tempfile
import unittest

from simulated_students import simulate

TEXT_ITEM = {
    'item_id': 1, 'language': 'English', 'age_min': 12, 'age_max': 15, 'text': 'Q1?',
    'options': [
        {'id': 11, 'text': 'right', 'is_correct': True, 'image': None},
        {'id': 12, 'text': 'wrong', 'is_correct': False, 'image': None},
    ],
    'images': [], 'preceding_material': '',
}
IMAGE_ITEM = dict(TEXT_ITEM, item_id=2, text='Q2? [Image 1]', images=['images/2_1.png'])


class FakeClient:
    """Answers with the option whose text is 'right' unless told otherwise."""

    def __init__(self, vision_models=(), broken=False):
        self.vision_models = set(vision_models)
        self.broken = broken
        self.calls = []

    def has_vision(self, model):
        return model in self.vision_models

    def generate(self, model, prompt, schema, images, temperature, seed, think):
        self.calls.append({'model': model, 'images': images, 'seed': seed})
        if self.broken:
            return 'not json', 0.01
        letter = next(
            line.split('.')[0] for line in prompt.splitlines() if line.endswith('. right')
        )
        return json.dumps({'choice': letter, 'confidence': 80, 'reason': 'because'}), 0.01


class SimulateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        os.makedirs(os.path.join(self.dir, 'images'))
        with open(os.path.join(self.dir, 'images', '2_1.png'), 'wb') as image:
            image.write(b'png')
        self.items = os.path.join(self.dir, 'bank.jsonl')
        with open(self.items, 'w', encoding='utf-8') as handle:
            for item in (TEXT_ITEM, IMAGE_ITEM):
                handle.write(json.dumps(item) + '\n')
        self.out = os.path.join(self.dir, 'runs', 'run.jsonl')

    def run_sim(self, client, models=('text-model',), replicates=2):
        return simulate.run(
            client=client, items_path=self.items, out_path=self.out, models=list(models),
            personas=['low', 'high'], contexts=['none'], replicates=replicates,
            temperature=0.8, workers=2,
        )

    def results(self):
        with open(self.out, encoding='utf-8') as handle:
            return [json.loads(line) for line in handle]

    def test_records_the_choice_mapped_back_to_the_option(self):
        self.run_sim(FakeClient())
        rows = self.results()
        self.assertEqual(len(rows), 4)  # 1 text item x 2 personas x 2 replicates
        self.assertTrue(all(row['correct'] and row['choice_option_id'] == 11 for row in rows))
        self.assertEqual(rows[0]['prompt_version'], simulate.prompts.PROMPT_VERSION)

    def test_image_items_only_go_to_vision_models(self):
        client = FakeClient(vision_models={'vision-model'})
        summary = self.run_sim(client, models=('text-model', 'vision-model'), replicates=1)
        rows = self.results()
        self.assertEqual({(r['model'], r['item_id']) for r in rows},
                         {('text-model', 1), ('vision-model', 1), ('vision-model', 2)})
        sent = [c['images'] for c in client.calls if c['model'] == 'vision-model' and c['images']]
        self.assertEqual(len(sent), 2)  # the image item, for both personas
        self.assertEqual(summary['skipped_image_items']['text-model'], 1)

    def test_resume_does_not_repeat_finished_work(self):
        self.run_sim(FakeClient())
        client = FakeClient()
        self.run_sim(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(self.results()), 4)

    def test_unparseable_replies_are_recorded_not_fatal(self):
        self.run_sim(FakeClient(broken=True), replicates=1)
        rows = self.results()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['error'] and row['choice_option_id'] is None for row in rows))

    def test_replicates_use_different_seeds(self):
        client = FakeClient()
        self.run_sim(client)
        self.assertEqual(len({c['seed'] for c in client.calls}), 4)



class SampleTests(unittest.TestCase):
    def test_sample_spreads_over_languages_and_is_reproducible(self):
        items = [dict(TEXT_ITEM, item_id=i, language=lang)
                 for i, lang in enumerate(['Greek'] * 20 + ['English'] * 10 + ['Estonian'] * 2)]
        picked = simulate.sample_items(items, 8, seed=1)
        self.assertEqual(len(picked), 8)
        self.assertEqual({item['language'] for item in picked}, {'Greek', 'English', 'Estonian'})
        self.assertEqual([i['item_id'] for i in picked],
                         [i['item_id'] for i in simulate.sample_items(items, 8, seed=1)])
        self.assertEqual(len(simulate.sample_items(items, 100, seed=1)), 32)


if __name__ == '__main__':
    unittest.main()
