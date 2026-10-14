"""Answers must belong to the current question, and missing timers are
stored as unknown (NULL) rather than as 0 seconds."""
import unittest
from datetime import datetime
from unittest.mock import patch

from rasa_sdk.executor import CollectingDispatcher

from actions import actions


class ScriptedCursor:
    """Answers each query by the first matching fragment of its SQL."""

    def __init__(self, replies):
        self.replies = replies
        self.statements = []
        self.last = None

    def execute(self, sql, params=None):
        flat = " ".join(sql.split())
        self.statements.append((flat, params))
        self.last = flat

    def fetchone(self):
        for fragment, reply in self.replies.items():
            if fragment in self.last:
                return reply
        return None

    def fetchall(self):
        return []

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
    def __init__(self, answer_id, asked_time=None):
        self.slots = {
            'user_id': '42', 'scenario_id': '7', 'locale': 'English',
            'last_question_id': 100, 'question_asked_time': asked_time,
        }
        self.latest_message = {'metadata': {}}
        self.answer_id = answer_id

    def get_slot(self, name):
        return self.slots.get(name)

    def get_latest_entity_values(self, name):
        return iter([self.answer_id] if name == 'answer_id' else [])


def run_handle_answer(answer_belongs, asked_time=None):
    cursor = ScriptedCursor({
        'authoringtool_scenariorevisiondraft': (False,),
        'FROM authoringtool_scenarioimplementation': (11, 5, 'active', 'exact'),
        'FROM authoringtool_answer WHERE id = %s AND activity_id = %s': (1,) if answer_belongs else None,
    })
    dispatcher = CollectingDispatcher()
    with patch.object(actions, 'get_database_connection', return_value=FakeConnection(cursor)), \
            patch.object(actions, 'release_connection'):
        try:
            actions.HandleAnswerAction().run(dispatcher, FakeTracker(7, asked_time), domain={})
        except Exception:
            pass  # later steps need more scripted rows; the insert is what matters
    inserts = [p for s, p in cursor.statements if s.startswith('INSERT INTO authoringtool_useranswer')]
    return inserts, dispatcher.messages


class AnswerChecksTests(unittest.TestCase):
    def test_answer_from_another_question_is_rejected(self):
        inserts, messages = run_handle_answer(answer_belongs=False)
        self.assertEqual(inserts, [])
        self.assertTrue(any('option' in (m.get('text') or '') for m in messages))

    def test_answer_of_the_current_question_is_saved(self):
        inserts, _ = run_handle_answer(answer_belongs=True, asked_time='2026-10-14 10:00:00')
        self.assertEqual(len(inserts), 1)

    def test_missing_timer_is_stored_as_unknown(self):
        inserts, _ = run_handle_answer(answer_belongs=True, asked_time=None)
        self.assertIsNone(inserts[0][3], 'timing must be NULL, not 0')

    def test_seconds_between(self):
        start = '2026-10-14 10:00:00'
        self.assertEqual(
            actions._seconds_between(start, datetime(2026, 10, 14, 10, 0, 42)), 42,
        )
        self.assertIsNone(actions._seconds_between(None, datetime.now()))
        self.assertIsNone(actions._seconds_between('garbage', datetime.now()))


if __name__ == '__main__':
    unittest.main()
