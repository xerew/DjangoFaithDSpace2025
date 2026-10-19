"""Plain-text Ollama calls used to build the context behind AI proposals."""
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import SimpleTestCase

from authoringtool.tasks import ollama_text


def envelope(**body):
    response = MagicMock(status_code=200)
    response.json.return_value = body
    response.raise_for_status.return_value = None
    return response


class OllamaTextTests(SimpleTestCase):
    def call(self, body, **kwargs):
        with patch('authoringtool.tasks.requests.post', return_value=envelope(**body)) as post:
            text = ollama_text('Summarise this.', **kwargs)
        return text, post.call_args.kwargs['json']

    def test_proposal_model_is_asked_without_thinking(self):
        text, payload = self.call({'response': ' A summary. '})

        self.assertEqual(text, 'A summary.')
        self.assertEqual(payload['model'], settings.LLM_PROPOSAL_MODEL)
        self.assertIs(payload['think'], False)
        self.assertIs(payload['stream'], False)

    def test_text_left_in_thinking_is_used(self):
        text, _ = self.call({'response': '', 'thinking': 'From thinking.'})

        self.assertEqual(text, 'From thinking.')

    def test_ollama_error_is_raised_not_returned_as_empty_text(self):
        with self.assertRaisesRegex(RuntimeError, 'not found'):
            self.call({'error': "model 'qwen3.6:35b' not found"})

    def test_other_models_are_not_sent_a_thinking_flag(self):
        _, payload = self.call(
            {'response': 'A pendulum.'},
            model=settings.LLM_IMAGE_DESCRIPTION_MODEL, think=None, images=['abc'],
        )

        self.assertNotIn('think', payload)
        self.assertEqual(payload['images'], ['abc'])

    def test_context_building_has_no_direct_generate_calls_left(self):
        source = Path(__file__).with_name('tasks.py').read_text(encoding='utf-8')
        start = source.index('def generate_llm_context_for_scenario')
        end = source.index('# 5️⃣ FLAG-DRIVEN PROPOSALS', start)

        self.assertEqual(re.findall(r'api/generate', source[start:end]), [])
