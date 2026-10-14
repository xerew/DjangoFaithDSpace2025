"""Students cannot restart a scenario; only teachers and staff may reset.

Run inside the actions image:
    docker compose run --rm --no-deps -v ./RasaFaith/RasaFaith:/app -w /app \
        --entrypoint python rasa_actions -m unittest discover -s actions_tests -t .
"""
import unittest
from unittest.mock import patch

from rasa_sdk.executor import CollectingDispatcher

from actions import actions


class FakeCursor:
    def __init__(self, can_reset):
        self.can_reset = can_reset
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))

    def fetchone(self):
        return (self.can_reset,)

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


class FakeTracker:
    def __init__(self, user_id='42', scenario_id='7'):
        self.slots = {'user_id': user_id, 'scenario_id': scenario_id, 'locale': 'English'}
        self.latest_message = {'metadata': {}}

    def get_slot(self, name):
        return self.slots.get(name)


def run_action(action, can_reset):
    cursor = FakeCursor(can_reset)
    dispatcher = CollectingDispatcher()
    with patch.object(actions, 'get_database_connection', return_value=FakeConnection(cursor)), \
            patch.object(actions, 'release_connection'):
        action.run(dispatcher, FakeTracker(), domain={})
    texts = [message.get('text') or '' for message in dispatcher.messages]
    return cursor, texts


class RestartRuleTests(unittest.TestCase):
    def test_student_restart_keeps_answers(self):
        cursor, texts = run_action(actions.DeleteDatabaseData(), can_reset=False)
        self.assertFalse(
            [s for s in cursor.statements if s.startswith('DELETE')],
            'a student must not be able to delete their answers',
        )
        self.assertIn('Restarting the scenario is not available', ' '.join(texts))

    def test_teacher_reset_deletes_answers(self):
        cursor, _ = run_action(actions.DeleteDatabaseData(), can_reset=True)
        self.assertTrue([s for s in cursor.statements if s.startswith('DELETE')])

    def test_student_is_not_offered_a_restart_confirmation(self):
        _, texts = run_action(actions.ActionConfirm(), can_reset=False)
        self.assertIn('Restarting the scenario is not available', ' '.join(texts))


if __name__ == '__main__':
    unittest.main()
