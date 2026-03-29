from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction, AllSlotsReset
import psycopg2
from psycopg2 import pool as psycopg2_pool
from datetime import datetime
from pygoogletranslation import Translator
import pycountry
from langdetect import detect
from bs4 import BeautifulSoup
import time
import os


# ---------------------------------------------------------------------------
# DB connection pool — created once at module load, reused across all actions
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.environ.get("DB_HOST", "db"),
    "database": os.environ.get("DB_NAME", "djangofaith"),
    "user":     os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "port":     os.environ.get("DB_PORT", "5432"),
}

_pool = None

def _get_pool():
    global _pool
    if _pool is None or _pool.closed:
        _pool = psycopg2_pool.ThreadedConnectionPool(minconn=2, maxconn=10, **DB_CONFIG)
    return _pool

def get_database_connection():
    return _get_pool().getconn()

def release_connection(conn):
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_greek(locale: str) -> bool:
    if not locale:
        return False
    # Canonical value is 'Greek'; keep legacy variants for backwards compat
    return locale.strip().lower() in ('greek', 'ελληνικά', 'ελληνικα', 'gr', 'el')

def _end_message(dispatcher, locale):
    if _is_greek(locale):
        dispatcher.utter_message(text="Αυτό είναι το τέλος του σεναρίου!")
    else:
        dispatcher.utter_message(text="This is the end of the scenario!")


# ---------------------------------------------------------------------------
class ActionReceiveUserId(Action):

    def name(self):
        return "action_receive_user_id"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        user_id = tracker.latest_message.get('metadata', {}).get('user_id', '')
        conn = get_database_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM auth_user WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            if result:
                dispatcher.utter_message(text=f"Received user ID: {user_id}")
            else:
                dispatcher.utter_message(text=f"User {user_id} not found.")
        finally:
            cursor.close()
            release_connection(conn)
        return [SlotSet("user_id", user_id)]


class ActionSetUserAndScenario(Action):
    def name(self):
        return "action_set_user_and_scenario"

    def run(self, dispatcher, tracker, domain):
        metadata = tracker.latest_message.get('metadata', {})
        return [
            SlotSet("user_id",     metadata.get("user_id", None)),
            SlotSet("scenario_id", metadata.get("scenario_id", None)),
        ]


class AskQuestionAction(Action):

    def name(self):
        return "action_ask_question"

    def run(self, dispatcher, tracker, domain):
        question_id  = tracker.get_slot("next_question_id")
        user_locale  = tracker.get_slot("locale")
        user_id      = tracker.get_slot("user_id")
        scenario_id  = tracker.get_slot("scenario_id")
        metadata     = tracker.latest_message.get('metadata', {})

        # Resolve IDs from metadata if slots are empty
        if not user_id:
            user_id = metadata.get("user_id") or metadata.get("userId")
        if not scenario_id:
            scenario_id = metadata.get("scenario_id") or metadata.get("scenarioId") or 1
        if not user_locale:
            user_locale = metadata.get("scenario_lang", "")

        if not user_id:
            dispatcher.utter_message(text="Could not identify user. Please reload and try again.")
            return []

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_database_connection()
        try:
            cursor = conn.cursor()

            # Determine starting activity if no question_id yet
            if not question_id:
                cursor.execute(
                    "SELECT last_activity_id FROM authoringtool_userscenarioscore "
                    "WHERE user_id = %s AND scenario_id = %s",
                    (user_id, scenario_id)
                )
                result = cursor.fetchone()
                if not result or not result[0]:
                    cursor.execute(
                        "SELECT id FROM authoringtool_activity "
                        "WHERE scenario_id = %s ORDER BY id ASC LIMIT 1",
                        (scenario_id,)
                    )
                    spec = cursor.fetchone()
                    if not spec:
                        dispatcher.utter_message(text="No activities found for this scenario.")
                        return []
                    question_id = spec[0]
                else:
                    question_id = result[0]

            # Fetch activity content
            cursor.execute(
                "SELECT text, scenario_id, activity_type_id "
                "FROM authoringtool_activity WHERE id = %s",
                (question_id,)
            )
            result = cursor.fetchone()
            if not result:
                dispatcher.utter_message(text="Activity not found.")
                return []

            question_text_html, scenario_id, question_type_id = result

            # Strip HTML for the chat bubble (keep meaningful text)
            soup = BeautifulSoup(question_text_html, 'html.parser')
            for img in soup.find_all('img'):
                img.decompose()
            for tag in soup.find_all(['p', 'div']):
                tag.unwrap()
            question_text = soup.get_text().strip()
            if len(question_text) > 300:
                question_text = question_text[:300] + '…'

            # Non-question activities (Explanation / Experiment)
            # Only fire the activity_id so the main screen renders the content.
            # The bot shows a brief prompt instead of repeating the full text.
            if question_type_id != 4:
                dispatcher.utter_message(
                    json_message={'activity_id': question_id}
                )
                if _is_greek(user_locale):
                    buttons = [{"title": "Ναι, συνέχισε!", "payload": "/confirm_read"}]
                    dispatcher.utter_message(text="📖 Διάβασε τη δραστηριότητα και πάτα συνέχεια όταν είσαι έτοιμος/η.", buttons=buttons)
                else:
                    buttons = [{"title": "Yes, continue!", "payload": "/confirm_read"}]
                    dispatcher.utter_message(text="📖 Read the activity and continue when you're ready.", buttons=buttons)
                return [
                    SlotSet("current_question_id", question_id),
                    SlotSet("question_asked_time", current_time),
                    SlotSet("scenario_id", scenario_id),
                    SlotSet("locale", user_locale),
                ]

            # Question activity — fetch answers
            cursor.execute(
                "SELECT id, text FROM authoringtool_answer "
                "WHERE activity_id = %s ORDER BY id ASC",
                (question_id,)
            )
            answers = cursor.fetchall()
            buttons = [
                {
                    "title":       ans[1],
                    "payload":     f'/provide_answer{{"answer_id": "{ans[0]}", "question_id": "{question_id}"}}',
                    "activity_id": question_id,
                }
                for ans in answers
            ]
            dispatcher.utter_message(text=question_text, buttons=buttons)
            return [
                SlotSet("scenario_id", scenario_id),
                SlotSet("last_question_id", question_id),
                SlotSet("question_asked_time", current_time),
            ]
        finally:
            cursor.close()
            release_connection(conn)


class HandleAnswerAction(Action):

    def name(self):
        return "action_handle_answer"

    def run(self, dispatcher, tracker, domain):
        user_locale = tracker.get_slot("locale") or \
                      tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        scenario_id = tracker.get_slot("scenario_id")
        user_id     = tracker.get_slot("user_id") or \
                      tracker.latest_message.get('metadata', {}).get('user_id', '')

        if not user_id:
            dispatcher.utter_message(text="Could not identify user.")
            return []

        question_id = tracker.get_slot("last_question_id")
        answer_id   = next(tracker.get_latest_entity_values("answer_id"), None)

        # Timing — guard against missing slot
        answer_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        question_time = tracker.get_slot("question_asked_time")
        if question_time:
            fmt = "%Y-%m-%d %H:%M:%S"
            seconds_taken = (
                datetime.strptime(answer_time, fmt) - datetime.strptime(question_time, fmt)
            ).total_seconds()
        else:
            seconds_taken = 0

        conn = get_database_connection()
        try:
            cursor = conn.cursor()

            # Resolve question_id from DB if slot is missing
            if not question_id:
                cursor.execute(
                    "SELECT last_activity_id FROM authoringtool_userscenarioscore "
                    "WHERE user_id = %s AND scenario_id = %s",
                    (user_id, scenario_id)
                )
                row = cursor.fetchone()
                question_id = row[0] if row else None
                if not question_id:
                    dispatcher.utter_message(text="Could not find current activity.")
                    return []

            # Record user answer
            cursor.execute(
                "INSERT INTO authoringtool_useranswer "
                "(user_id, activity_id, answer_id, timing, created_on) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, question_id, answer_id, seconds_taken, datetime.now())
            )
            conn.commit()

            # Fetch answer weight and correctness
            cursor.execute(
                "SELECT answer_weight, is_correct FROM authoringtool_answer WHERE id = %s",
                (answer_id,)
            )
            ans_result = cursor.fetchone()
            if not ans_result:
                dispatcher.utter_message(text="Could not find that answer.")
                return []
            score_for_current_answer, is_answer_correct = ans_result

            # Update activity correct/incorrect counters
            if is_answer_correct:
                cursor.execute(
                    "UPDATE authoringtool_activity SET correct_count = correct_count + 1 WHERE id = %s",
                    (question_id,)
                )
            else:
                cursor.execute(
                    "UPDATE authoringtool_activity SET incorrect_count = incorrect_count + 1 WHERE id = %s",
                    (question_id,)
                )

            # Update or insert user scenario score (last_activity_id set to next below)
            cursor.execute(
                "SELECT user_score FROM authoringtool_userscenarioscore "
                "WHERE user_id = %s AND scenario_id = %s",
                (user_id, scenario_id)
            )
            score_row = cursor.fetchone()
            if score_row:
                new_score = score_row[0] + score_for_current_answer
                cursor.execute(
                    "UPDATE authoringtool_userscenarioscore "
                    "SET user_score = %s "
                    "WHERE user_id = %s AND scenario_id = %s",
                    (new_score, user_id, scenario_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO authoringtool_userscenarioscore "
                    "(user_id, last_activity_id, scenario_id, user_score) "
                    "VALUES (%s, %s, %s, %s)",
                    (user_id, question_id, scenario_id, score_for_current_answer)
                )
            conn.commit()

            # Determine next activity
            cursor.execute(
                "SELECT is_evaluatable FROM authoringtool_activity WHERE id = %s",
                (question_id,)
            )
            ev_row = cursor.fetchone()
            if not ev_row:
                dispatcher.utter_message(text="Activity configuration error.")
                return []

            is_evaluatable = ev_row[0]

            if is_evaluatable:
                # Sum scores for this evaluation bunch
                cursor.execute(
                    """
                    SELECT SUM(a.answer_weight), COUNT(DISTINCT ua.activity_id)
                    FROM authoringtool_useranswer ua
                    JOIN authoringtool_answer a ON ua.answer_id = a.id
                    WHERE ua.user_id = %s
                      AND ua.activity_id IN (
                          SELECT unnest(activity_ids)
                          FROM authoringtool_questionbunch
                          WHERE activity_primary_id = %s
                      )
                    """,
                    (user_id, question_id)
                )
                bunch_row = cursor.fetchone()
                total_score   = bunch_row[0] or 0
                count_acts    = bunch_row[1] or 1

                cursor.execute(
                    "SELECT next_question_on_high_id, next_question_on_mid_id, next_question_on_low_id "
                    "FROM authoringtool_evquestionbranching WHERE activity_id = %s",
                    (question_id,)
                )
                branch_row = cursor.fetchone()
                if not branch_row:
                    dispatcher.utter_message(text="Branching configuration missing.")
                    return []

                high_dest, mid_dest, low_dest = branch_row

                cursor.execute("SELECT score_limit FROM authoringtool_activity WHERE id = %s", (high_dest,))
                high_limit = cursor.fetchone()[0]
                cursor.execute("SELECT score_limit FROM authoringtool_activity WHERE id = %s", (mid_dest,))
                mid_limit  = cursor.fetchone()[0]

                avg_score = total_score / count_acts
                if avg_score >= high_limit:
                    next_question_id = high_dest
                elif avg_score >= mid_limit:
                    next_question_id = mid_dest
                else:
                    next_question_id = low_dest

                if next_question_id:
                    # Persist next activity so reconnect resumes here, not at current
                    cursor.execute(
                        "UPDATE authoringtool_userscenarioscore SET last_activity_id = %s "
                        "WHERE user_id = %s AND scenario_id = %s",
                        (next_question_id, user_id, scenario_id)
                    )
                    conn.commit()
                    return [SlotSet("next_question_id", next_question_id)]
                else:
                    _end_message(dispatcher, user_locale)
                    return [AllSlotsReset(), FollowupAction("action_end_scenario")]

            else:
                cursor.execute(
                    "SELECT next_activity_id FROM authoringtool_nextquestionlogic "
                    "WHERE activity_id = %s AND answer_id = %s",
                    (question_id, answer_id)
                )
                next_row = cursor.fetchone()
                if not next_row:
                    _end_message(dispatcher, user_locale)
                    return [AllSlotsReset(), FollowupAction("action_end_scenario")]

                # Persist next activity so reconnect resumes here, not at current
                cursor.execute(
                    "UPDATE authoringtool_userscenarioscore SET last_activity_id = %s "
                    "WHERE user_id = %s AND scenario_id = %s",
                    (next_row[0], user_id, scenario_id)
                )
                conn.commit()
                return [SlotSet("next_question_id", next_row[0])]

        finally:
            cursor.close()
            release_connection(conn)


class ProvideHintAction(Action):

    def name(self):
        return "action_provide_hint"

    def run(self, dispatcher, tracker, domain):
        question_id = tracker.get_slot("last_question_id")
        user_locale = tracker.get_slot("locale")
        translator  = Translator()

        conn = get_database_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT hint_text, hint_img_url, hint_video_url FROM hints WHERE question_id = %s",
                (question_id,)
            )
            hint_data = cursor.fetchone()

            if hint_data:
                hint_text, hint_img_url, hint_vid_url = hint_data
                if hint_text:
                    if user_locale and user_locale != 'en':
                        hint_text = translator.translate(hint_text, src='en', dest=user_locale).text
                    dispatcher.utter_message(text=hint_text)
                if hint_img_url:
                    dispatcher.utter_message(image=hint_img_url)
                if hint_vid_url:
                    dispatcher.utter_message(custom={"video": hint_vid_url})
            else:
                do_text = "Looks like you can do it yourself!"
                if user_locale and user_locale != 'en':
                    do_text = translator.translate(do_text, src='en', dest=user_locale).text
                dispatcher.utter_message(text=do_text)
        finally:
            cursor.close()
            release_connection(conn)

        return [FollowupAction("action_ask_question")]


class DeleteDatabaseData(Action):

    def name(self):
        return "action_delete_db_data"

    def run(self, dispatcher, tracker, domain):
        user_id = tracker.get_slot("user_id") or \
                  tracker.latest_message.get('metadata', {}).get('user_id', '')
        scenario_id = tracker.get_slot("scenario_id") or \
                      tracker.latest_message.get('metadata', {}).get('scenario_id', '')

        if not user_id:
            dispatcher.utter_message(text="Could not identify user to delete data for.")
            return []

        conn = get_database_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM authoringtool_useranswer
                WHERE activity_id IN (
                    SELECT id FROM authoringtool_activity WHERE scenario_id = %s
                ) AND user_id = %s
                """,
                (scenario_id, user_id)
            )
            cursor.execute(
                "DELETE FROM authoringtool_userscenarioscore WHERE user_id = %s AND scenario_id = %s",
                (user_id, scenario_id)
            )
            cursor.execute(
                """
                DELETE FROM authoringtool_phetlabsessions
                WHERE activity_id IN (
                    SELECT id FROM authoringtool_activity WHERE scenario_id = %s
                ) AND user_id = %s
                """,
                (scenario_id, user_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print("Delete error:", e)
        finally:
            cursor.close()
            release_connection(conn)

        locale = tracker.get_slot("locale") or \
                 tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        if _is_greek(locale):
            dispatcher.utter_message(
                text="Έγινε! Η πρόοδός σου έχει διαγραφεί. "
                     "Πάτησε το κουμπί **Start** για να ξεκινήσεις το σενάριο ξανά."
            )
        else:
            dispatcher.utter_message(
                text="All set! Your progress has been cleared. "
                     "Press the **Start** button to begin the scenario again."
            )
        return [SlotSet("user_id", user_id), SlotSet("last_question_id", None)]


class ActionConfirm(Action):

    def name(self):
        return "action_confirm"

    def run(self, dispatcher, tracker, domain):
        locale = tracker.get_slot("locale") or \
                 tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        if _is_greek(locale):
            dispatcher.utter_message(
                text="Αυτό θα διαγράψει όλες τις απαντήσεις σου για αυτό το σενάριο. "
                     "Είσαι έτοιμος/η να ξεκινήσεις από την αρχή;",
                buttons=[
                    {"title": "✅ Ναι, επανεκκίνηση", "payload": "/affirm"},
                    {"title": "❌ Άκυρο",              "payload": "/deny"},
                ]
            )
        else:
            dispatcher.utter_message(
                text="This will clear all your answers for this scenario. "
                     "Ready to start fresh?",
                buttons=[
                    {"title": "✅ Yes, restart", "payload": "/affirm"},
                    {"title": "❌ Cancel",        "payload": "/deny"},
                ]
            )
        return [SlotSet("last_question_id", None), SlotSet("next_question_id", None)]


class ActionDenyRestart(Action):

    def name(self):
        return "action_deny_restart"

    def run(self, dispatcher, tracker, domain):
        locale = tracker.get_slot("locale") or \
                 tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        if _is_greek(locale):
            dispatcher.utter_message(text="Εντάξει! Η πρόοδός σου παραμένει ανέπαφη.")
        else:
            dispatcher.utter_message(text="No problem! Your progress has been kept.")
        return []


class ActionConfirmRead(Action):
    def name(self):
        return "action_confirm_read"

    def run(self, dispatcher, tracker, domain):
        question_id = tracker.get_slot("current_question_id")
        scenario_id = tracker.get_slot("scenario_id") or \
                      tracker.latest_message.get('metadata', {}).get('scenario_id', '')
        user_id     = tracker.get_slot("user_id") or \
                      tracker.latest_message.get('metadata', {}).get('user_id', '')
        user_locale = tracker.get_slot("locale") or \
                      tracker.latest_message.get('metadata', {}).get('scenario_lang', '')

        # Timing — guard against missing slot
        answer_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        question_time = tracker.get_slot("question_asked_time")
        if question_time:
            fmt = "%Y-%m-%d %H:%M:%S"
            seconds_taken = (
                datetime.strptime(answer_time, fmt) - datetime.strptime(question_time, fmt)
            ).total_seconds()
        else:
            seconds_taken = 0

        conn = get_database_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO authoringtool_useranswer (user_id, activity_id, timing, created_on) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, question_id, seconds_taken, datetime.now())
            )
            conn.commit()

            cursor.execute(
                "SELECT next_activity_id FROM authoringtool_nextquestionlogic WHERE activity_id = %s",
                (question_id,)
            )
            result = cursor.fetchone()
            if not result:
                _end_message(dispatcher, user_locale)
                return [AllSlotsReset(), FollowupAction("action_end_scenario")]

            next_question_id = result[0]

            # Persist next activity so reconnect resumes here, not at this explanation
            cursor.execute(
                "SELECT id FROM authoringtool_userscenarioscore "
                "WHERE user_id = %s AND scenario_id = %s",
                (user_id, scenario_id)
            )
            if cursor.fetchone():
                cursor.execute(
                    "UPDATE authoringtool_userscenarioscore SET last_activity_id = %s "
                    "WHERE user_id = %s AND scenario_id = %s",
                    (next_question_id, user_id, scenario_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO authoringtool_userscenarioscore "
                    "(user_id, scenario_id, last_activity_id, user_score) VALUES (%s, %s, %s, 0)",
                    (user_id, scenario_id, next_question_id)
                )
            conn.commit()

            # Check for Pendulum Lab
            cursor.execute(
                """
                SELECT sim.name FROM authoringtool_activity q
                JOIN authoringtool_simulation sim ON q.simulation_id = sim.id
                WHERE q.id = %s
                """,
                (question_id,)
            )
            sim_row = cursor.fetchone()
            if sim_row and sim_row[0] == 'Pendulum Lab':
                getPhetPendulumData(tracker)

            return [
                SlotSet("next_question_id", next_question_id),
                FollowupAction("action_ask_question"),
                SlotSet("scenario_id", scenario_id),
            ]
        finally:
            cursor.close()
            release_connection(conn)


class ActionEndScenario(Action):
    def name(self):
        return "action_end_scenario"

    def run(self, dispatcher, tracker, domain):
        user_locale = tracker.get_slot("locale") or \
                      tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        if _is_greek(user_locale):
            dispatcher.utter_message(text="Ευχαριστούμε για την συμμετοχή σου!")
        else:
            dispatcher.utter_message(text="Thank you for participating!")
        return [AllSlotsReset()]


class ActionRequestUserInput(Action):
    def name(self):
        return "action_request_user_input"

    def run(self, dispatcher, tracker, domain):
        metadata = tracker.latest_message.get('metadata', {})
        locale = (
            tracker.get_slot("locale")
            or metadata.get("scenario_lang", "")
        )
        if _is_greek(locale):
            dispatcher.utter_message(
                text="Γεια σου! Πες μου κάτι για να συνεχίσουμε — για παράδειγμα πώς σε λένε;"
            )
        else:
            dispatcher.utter_message(
                text="Hello! Tell me something to continue — for example, what's your name?"
            )
        return [SlotSet("locale", locale)]


class ActionDetectLanguage(Action):
    def name(self):
        return "action_detect_language"

    def run(self, dispatcher, tracker, domain):
        metadata = tracker.latest_message.get('metadata', {})
        # Prefer metadata locale (set by the parent page); fall back to langdetect
        locale = (
            tracker.get_slot("locale")
            or metadata.get("scenario_lang", "")
        )
        if not locale:
            try:
                user_text = tracker.latest_message.get("text", "")
                detected = detect(user_text) if user_text else "en"
                locale = "el" if detected == "el" else "en"
            except Exception:
                locale = "en"

        if _is_greek(locale):
            dispatcher.utter_message(
                text=f"Εντόπισα ότι μιλάς Ελληνικά. Είναι σωστό;",
                buttons=[
                    {"title": "Ναι", "payload": "/affirm"},
                    {"title": "Όχι", "payload": "/deny"},
                ]
            )
        else:
            dispatcher.utter_message(
                text=f"I detected English as your language. Is that correct?",
                buttons=[
                    {"title": "Yes", "payload": "/affirm"},
                    {"title": "No",  "payload": "/deny"},
                ]
            )
        return [SlotSet("locale", locale)]


class ActionHandleLanguageConfirmation(Action):
    def name(self):
        return "action_handle_language_confirmation"

    def run(self, dispatcher, tracker, domain):
        locale = tracker.get_slot("locale") or ""
        if _is_greek(locale):
            dispatcher.utter_message(
                text="Τέλεια! Όταν είσαι έτοιμος, πες μου «Ξεκίνα».",
                buttons=[{"title": "Ξεκίνα!", "payload": "/ask_me"}]
            )
        else:
            dispatcher.utter_message(
                text="Great! Whenever you're ready, say \"Let's go\".",
                buttons=[{"title": "Let's go!", "payload": "/ask_me"}]
            )
        return [SlotSet("locale", locale)]


def getPhetPendulumData(tracker):
    pendulum_data = tracker.latest_message.get('metadata', {}).get('pendulum_data', {})
    print(f"[Pendulum] Data received: {pendulum_data}")
    return pendulum_data
