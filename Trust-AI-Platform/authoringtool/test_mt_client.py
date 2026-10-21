import json
from unittest.mock import MagicMock, patch

import requests
from django.conf import settings
from django.test import SimpleTestCase

from authoringtool.machine_translation.client import (
    TranslationOutputInvalid,
    TranslationServiceUnavailable,
    TranslationTimeout,
    translate_fields,
)

POST = 'authoringtool.machine_translation.client.requests.post'
FIELDS = {'name': 'Final check', 'text': '<p>Pick one.</p>'}


def reply(content, status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = content
    response.text = json.dumps(content)
    return response


class TranslateFieldsTests(SimpleTestCase):
    def call(self):
        return translate_fields(FIELDS, 'English', 'Greek', 'Ohm\'s Law: use V = IR.')

    def test_sends_model_schema_and_timeout(self):
        with patch(POST, return_value=reply({'message': {'content': json.dumps(FIELDS)}})) as post:
            self.call()
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], settings.LLM_MACHINE_TRANSLATION_MODEL)
        self.assertEqual(sorted(payload['format']['required']), sorted(FIELDS))
        self.assertFalse(payload['format']['additionalProperties'])
        self.assertFalse(payload['stream'])
        self.assertEqual(post.call_args.kwargs['timeout'], settings.MACHINE_TRANSLATION_TIMEOUT)
        self.assertTrue(post.call_args.args[0].endswith('/api/chat'))
        system = payload['messages'][0]['content']
        self.assertIn('English', system)
        self.assertIn('Greek', system)

    def test_returns_parsed_fields(self):
        translated = {'name': 'Τελικός έλεγχος', 'text': '<p>Διάλεξε.</p>'}
        with patch(POST, return_value=reply({'message': {'content': json.dumps(translated)}})):
            self.assertEqual(self.call(), translated)

    def test_connection_error_raises_unavailable(self):
        with patch(POST, side_effect=requests.ConnectionError('refused')):
            with self.assertRaises(TranslationServiceUnavailable):
                self.call()

    def test_server_error_raises_unavailable(self):
        with patch(POST, return_value=reply({'error': 'boom'}, status=502)):
            with self.assertRaises(TranslationServiceUnavailable):
                self.call()

    def test_timeout_raises_translation_timeout(self):
        with patch(POST, side_effect=requests.Timeout('slow')):
            with self.assertRaises(TranslationTimeout):
                self.call()

    def test_non_json_raises_output_invalid(self):
        with patch(POST, return_value=reply({'message': {'content': 'Sure! Here it is'}})):
            with self.assertRaises(TranslationOutputInvalid):
                self.call()

    def test_non_string_values_raise_output_invalid(self):
        with patch(POST, return_value=reply({'message': {'content': json.dumps({'name': 3, 'text': 'x'})}})):
            with self.assertRaises(TranslationOutputInvalid):
                self.call()

    def test_model_missing_raises_unavailable(self):
        with patch(POST, return_value=reply({'error': "model 'gemma4:31b' not found"}, status=404)):
            with self.assertRaises(TranslationServiceUnavailable):
                self.call()
