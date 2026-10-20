import unittest
from unittest import mock

from simulated_students.ollama import OllamaClient


class ChatTests(unittest.TestCase):
    def test_chat_sends_the_conversation_and_returns_the_reply(self):
        response = mock.Mock()
        response.json.return_value = {'message': {'role': 'assistant', 'content': ' {"choice": "A"} '}}
        messages = [{'role': 'system', 'content': 'You are a student.'},
                    {'role': 'user', 'content': 'Q?', 'images': ['aW1n']}]
        with mock.patch('simulated_students.ollama.requests.post', return_value=response) as post:
            text, seconds = OllamaClient('http://ollama:11434/').chat(
                'gemma3:4b', messages, {'type': 'object'}, 0.8, 7, None, 8192,
            )
        self.assertEqual(text, '{"choice": "A"}')
        self.assertGreaterEqual(seconds, 0)
        url, = post.call_args.args
        payload = post.call_args.kwargs['json']
        self.assertEqual(url, 'http://ollama:11434/api/chat')
        self.assertEqual(payload['messages'], messages)
        self.assertEqual(payload['format'], {'type': 'object'})
        self.assertEqual(payload['options'], {'temperature': 0.8, 'seed': 7, 'num_ctx': 8192})
        self.assertFalse(payload['stream'])
        self.assertNotIn('think', payload)


if __name__ == '__main__':
    unittest.main()
