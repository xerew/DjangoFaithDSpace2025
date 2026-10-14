import json
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(RASA_WEBHOOK_URL='http://rasa.internal/webhooks/rest/webhook')
class RasaChatProxyTests(TestCase):
    def setUp(self):
        self.student = User.objects.create_user('proxy_student', password='pass')
        self.victim = User.objects.create_user('proxy_victim', password='pass')
        self.client.force_login(self.student)
        self.url = reverse('rasa_chat')

    def post(self, payload):
        return self.client.post(
            self.url, data=json.dumps(payload), content_type='application/json',
        )

    def rasa_replies(self, replies):
        response = MagicMock(status_code=200)
        response.json.return_value = replies
        return patch('studentview.views.requests.post', return_value=response)

    def forwarded(self, mock_post):
        return mock_post.call_args.kwargs['json']

    def test_forwards_message_with_session_user_not_client_user(self):
        with self.rasa_replies([{'text': 'hi'}]) as mock_post:
            response = self.post({
                'message': 'hello',
                'sender': 'abc-123',
                'metadata': {
                    'user_id': str(self.victim.id),
                    'userId': str(self.victim.id),
                    'scenario_id': '7',
                    'scenario_lang': 'Greek',
                    'pendulum_data': {'mass': 1},
                },
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{'text': 'hi'}])
        payload = self.forwarded(mock_post)
        self.assertEqual(
            mock_post.call_args.args[0],
            'http://rasa.internal/webhooks/rest/webhook',
        )
        self.assertEqual(payload['message'], 'hello')
        self.assertEqual(payload['metadata']['user_id'], str(self.student.id))
        self.assertNotIn('userId', payload['metadata'])
        self.assertEqual(payload['metadata']['scenario_id'], '7')
        self.assertEqual(payload['metadata']['scenario_lang'], 'Greek')
        self.assertEqual(payload['metadata']['pendulum_data'], {'mass': 1})

    def test_conversation_id_is_bound_to_session_user(self):
        with self.rasa_replies([]) as mock_post:
            self.post({'message': 'hello', 'sender': f'user-{self.victim.id}-x'})
        sender = self.forwarded(mock_post)['sender']
        self.assertTrue(sender.startswith(f'user-{self.student.id}-'))

    def test_rejects_non_numeric_scenario_id(self):
        with self.rasa_replies([]) as mock_post:
            response = self.post({
                'message': 'hello', 'metadata': {'scenario_id': '1 OR 1=1'},
            })
        self.assertEqual(response.status_code, 400)
        mock_post.assert_not_called()

    def test_rejects_missing_message(self):
        with self.rasa_replies([]) as mock_post:
            response = self.post({'sender': 'abc'})
        self.assertEqual(response.status_code, 400)
        mock_post.assert_not_called()

    def test_rejects_invalid_json(self):
        response = self.client.post(
            self.url, data='not json', content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_only_accepts_post(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_reports_bad_gateway_when_rasa_is_down(self):
        with patch(
            'studentview.views.requests.post',
            side_effect=requests.ConnectionError('down'),
        ):
            response = self.post({'message': 'hello'})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), [])

    def test_anonymous_users_are_not_forwarded(self):
        self.client.logout()
        with self.rasa_replies([]) as mock_post:
            response = self.post({'message': 'hello'})
        self.assertNotEqual(response.status_code, 200)
        mock_post.assert_not_called()
