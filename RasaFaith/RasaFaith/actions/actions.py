from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction, AllSlotsReset
import psycopg2
from datetime import datetime
from pygoogletranslation import Translator
import pycountry
from langdetect import detect
from bs4 import BeautifulSoup
import time



def get_database_connection():
    return psycopg2.connect(
        host="localhost",
        database="djangofaith",
        user="postgres",
        password="root",
        port="5432"
    )

class ActionReceiveUserId(Action):

    def name(self):
        return "action_receive_user_id"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):

        # Extract the user ID from the latest user message
        user_id = tracker.latest_message.get('metadata', {}).get('user_id', '')
        # somethiong = tracker.latest_message.get

        connection = get_database_connection()
        cursor = connection.cursor()

        # Checking if user exists
        cursor.execute("""
            SELECT *
            FROM auth_user
            WHERE user_id = %s
        """, (user_id,))

        result = cursor.fetchone()
        if result:
            # Send a response back to the user (optional)
            dispatcher.utter_message(text=f"Received user ID: {user_id}")
        else:
            cursor.execute("""
                INSERT INTO users(user_id, user_type, created_on)
                VALUES %s, 2, NOW()""", (user_id,))
            dispatcher.utter_message(text=f"Created user ID: {user_id}")
        return [SlotSet("user_id", user_id)]

# class ActionRequestUserInput(Action):
#     def name(self):
#         return "action_request_user_input"

#     def run(self, dispatcher, tracker, domain):
#         dispatcher.utter_message(text="Hey, write something in your language in the chat.")

#         return []

# class ActionDetectLanguage(Action):

#     def name(self):
#         return "action_detect_language"

#     def run(self, dispatcher, tracker, domain):
#         # Get the user's last message
#         user_message = tracker.latest_message.get('text')

#         if not user_message:
#             dispatcher.utter_message(text="I didn't receive any message to detect the language from.")
#             return [SlotSet("locale", 'en')]

#         try:
#             # Detect the language
#             detected_language = detect(user_message)
#             dispatcher.utter_message(text=f"Detected language: {detected_language}")
#             return [SlotSet("locale", detected_language)]
#         except Exception as e:
#             dispatcher.utter_message(text="There was an error detecting the language.")
#             print(f"Error in language detection: {e}")
#             return [SlotSet("locale", 'en')]

# class AskScenarionAction(Action): #CHOOSING SCENARIO LOGIC 30/12/23 HAPPY NEW YEAR!

#     def name(self):
#         return "action_ask_scenario"
    
#     def run(self, dispatcher, tracker, domain):

#         user_locale = tracker.get_slot("locale")
#         connection = get_database_connection()
#         cursor = connection.cursor()
#         translator = Translator()

#         cursor.execute("""
#             SELECT id, name FROM authoringtool_scenario
#         """)
        
#         scenarios = cursor.fetchall()
#         print(f"SCENARIOS DB: ", scenarios)
#         scenarios_text = 'Available Scenarios:'
#         if user_locale != 'en':
#             scenarios_text = translator.translate(scenarios_text, src='en', dest=user_locale).text
#         # dispatcher.utter_message(text=scenarios_text)
#         scenario_buttons = []

#         for scenario_id, scenario_name in scenarios:
#             if user_locale != 'en':
#                 translated_text = translator.translate(scenario_name, src='en', dest=user_locale).text
#                 scenario_buttons.append({
#                     "title": translated_text,
#                     "payload": f'/scenario_selection{{"scenario_id": "{scenario_id}"}}'
#                 })
#             else:
#                 scenario_buttons.append({
#                     "title": scenario_name,
#                     "payload": f'/scenario_selection{{"scenario_id": "{scenario_id}"}}'
#                 })

#         dispatcher.utter_message(text=scenarios_text, buttons=scenario_buttons)


#         return []

class ActionSetUserAndScenario(Action):
    def name(self):
        return "action_set_user_and_scenario"

    def run(self, dispatcher, tracker, domain):
        # Extract metadata
        metadata = tracker.latest_message.get('metadata', {})
        
        # Set slots from metadata
        return [
            SlotSet("userId", metadata.get("userId", None)),
            SlotSet("scenarioId", metadata.get("scenarioId", None))
        ]
    
# class HandleScenarioSelection(Action):

#     def name(self):
#         return "handle_scenario_selection"
    
#     def run(self, dispatcher, tracker, domain):
        
#         # user_locale = tracker.get_slot("locale")

#         print(f'TI DEIXNEI EDO GTXS: ', tracker.latest_message.get('metadata', {}).get('scenario_id', ''))

#         scenario_id = tracker.latest_message.get('metadata', {}).get('scenario_id', '')

#         print(f'RE TO SCENARIO: ', scenario_id)

#         # Get the answer_id entity from the latest message
#         # scenario_id = next(tracker.get_latest_entity_values("scenario_id"), None)

#         dispatcher.utter_message("Set")

#         return [SlotSet("scenario_id", scenario_id)]

class AskQuestionAction(Action):

    def name(self):
        return "action_ask_question"

    def run(self, dispatcher, tracker, domain):
        question_id = tracker.get_slot("next_question_id")
        user_locale = tracker.get_slot("locale")
        user_id = tracker.get_slot("user_id")
        scenario_id = tracker.get_slot("scenario_id")


        metadata = tracker.latest_message.get('metadata', {})
        print('META: ', metadata)
        if not user_id:
            user_id = metadata.get("userId", None)
        if not scenario_id:
            scenario_id = metadata.get("scenarioId", None)

        print(f"Scenario_ID:", scenario_id)
        print(f"Question:", question_id)
        print(f'USER_ID: ', user_id)

        latest_message = tracker.latest_message
        print(f'RE TO LATEST: ', latest_message)
        print("Metadata:", latest_message.get("metadata"))
        if not scenario_id:
            scenario_id = tracker.latest_message.get('metadata', {}).get('scenario_id', '')
            if not scenario_id:
                scenario_id = 1
        print(f"SCENARIOOOOOOO_ID:", scenario_id)
        
        if not user_locale:
            user_locale = tracker.latest_message.get('metadata', {}).get('scenario_lang', '')

        
        if not user_id:
            user_id = tracker.latest_message.get('metadata', {}).get('user_id', '')
            print(f'USER_ID SENDER: ', user_id)
            if not user_id:
                user_id = 7

        # Importing translator
        translator = Translator()

        # If it's the user's first interaction, default to the first question ID
        if not question_id:
            connection = get_database_connection()
            cursor = connection.cursor()

            cursor.execute("""
                SELECT last_activity_id from authoringtool_userscenarioscore
                Where user_id = %s AND scenario_id = %s
                           """, (user_id, scenario_id))
            print("doulepse nice")
            result = cursor.fetchone()
            print(f"RESULT", result)
            if not result or not result[0] or result is None:
                cursor.execute("""
                    SELECT id from authoringtool_activity
                    WHERE scenario_id = %s
                    ORDER BY id ASC
                    LIMIT 1;
                            """, (scenario_id,))
                spec_re = cursor.fetchone()
                question_id = spec_re[0]
            else:
                question_id = result[0]
            
        connection = get_database_connection()
        cursor = connection.cursor()

        # Fetching question_text, scenario_id, and question_type_id based on question_id
        cursor.execute("""
            SELECT text, scenario_id, activity_type_id
            FROM authoringtool_activity 
            WHERE id = %s
        """, (question_id,))
        # plain_text removed for enriched "text"
        
        result = cursor.fetchone()
        question_text_html = result[0]
        d_question_text = BeautifulSoup(question_text_html, 'html.parser')
        for img in d_question_text.find_all('img'):
            img.decompose()
        for p in d_question_text.find_all('p'):
            p.unwrap()
        for div in d_question_text.find_all('div'):
            div.unwrap()
        question_text = str(d_question_text) # clean HTML
        question_text = d_question_text.get_text()
        if len(question_text) > 100:
            question_text = question_text[:100] + '...'
        scenario_id = result[1]
        question_type_id = result[2]

        # Fetch all associated image URLs, video URLs, and additional texts for the current question from the question_data table
        # cursor.execute("""
        #    SELECT question_img_url, question_vid_url, question_additional_text
        #    FROM question_data 
        #    WHERE question_id = %s
        # """, (question_id,))

        # media_data = cursor.fetchall()

        # for img_url, vid_url, additional_text in media_data:
        #    if img_url:  # Checking if the image URL exists
        #        dispatcher.utter_message(image=img_url)
        #    if vid_url:  # Checking if the video URL exists
        #        dispatcher.utter_message(custom={"video": vid_url})
        #   if additional_text:  # Checking if there's additional text
        #        if user_locale != 'en':
        #            additional_text = translator.translate(additional_text, src='en', dest=user_locale).text
        #        dispatcher.utter_message(text=additional_text)

        #PEIRAZW PRAGMATA EDO ITAN TO #020
        #if user_locale != 'en':
        #    question_text = translator.translate(question_text, src='en', dest=user_locale).text
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        
        if question_type_id != 4: #4 is Question 2024
            # If question_type_id is 2 or 3 (exp or expl), fetch the next question id
            # Actual question being asked
            #020

            dispatcher.utter_message(text=question_text, json_message = {'activity_id': question_id}) #translated part
            #020

            #########################################################################################################
            if user_locale.lower() == 'greek' or user_locale.lower() == "ελληνικά":
                buttons = [
                    {"title": "Ναι, συνέχισε!", "payload": "/confirm_read"},
                ]
            else:
                buttons = [
                    {"title": "Yes, continue!", "payload": "/confirm_read"},
                ]
            # Yes, continue!
            # Ναι, συνέχισε!
            if user_locale.lower() == 'greek' or user_locale.lower() == "ελληνικά":
                dispatcher.utter_message(text="Να συνεχίσω με την επόμενη δραστηριότητα;", buttons=buttons)
            else:
                dispatcher.utter_message(text="Should I continue with the next activity?", buttons=buttons)
            # Should I continue with the next activity?
            # Να συνεχίσω με την επόμενη δραστηριότητα;

            return [SlotSet("current_question_id", question_id), SlotSet("question_asked_time", current_time), SlotSet("scenario_id", scenario_id), SlotSet("user_locale", user_locale)]

            #########################################################################################################
            # cursor.execute("""
            #     SELECT next_activity_id
            #     FROM authoringtool_nextquestionlogic
            #     WHERE activity_id = %s
            # """, (question_id,))

            # result = cursor.fetchone()
            # if result == None:
            #     dispatcher.utter_message(text="This is the end of the scenario!")
            #     next_question_id = None
            #     return [SlotSet("next_question_id", next_question_id), SlotSet("question_asked_time", current_time), SlotSet("scenario_id", scenario_id)]
            # else:
            #     result_1 = result[0]
            #     print(f'RESULT EINAI EDO: ', result, result_1)
                
            #     next_question_id = result[0]
            #     return [SlotSet("next_question_id", next_question_id), FollowupAction("action_ask_question"), SlotSet("question_asked_time", current_time), SlotSet("scenario_id", scenario_id)]
            #########################################################################################################

            # Trigger the action to ask the next question immediately
            # return [SlotSet("next_question_id", next_question_id), FollowupAction("action_ask_question"), SlotSet("question_asked_time", current_time), SlotSet("scenario_id", scenario_id)]

        # If question_type_id is not 2, continue with the existing logic to show answer buttons
        cursor.execute("""
            SELECT id, text 
            FROM authoringtool_answer
            WHERE activity_id = %s
            ORDER BY id ASC
        """, (question_id,))
        
        answers = cursor.fetchall()
        buttons = []
        # Translating
        for answer_id, answer_text in answers:
            # if user_locale != 'en':
            #     translated_text = translator.translate(answer_text, src='en', dest=user_locale).text
            #     buttons.append({
            #         "title": translated_text,
            #         "payload": f'/provide_answer{{"answer_id": "{answer_id}", "question_id": "{question_id}"}}',
            #         "activity_id": question_id
            #     })
            # else:
            buttons = [
                {"title": answer[1], "payload": f'/provide_answer{{"answer_id": "{answer[0]}", "question_id": "{question_id}"}}', "activity_id": question_id}
                for answer in answers
            ]

        dispatcher.utter_message(text=question_text, buttons=buttons)

        # Set the scenario_id as a slot
        return [SlotSet("scenario_id", scenario_id), SlotSet("last_question_id", question_id), SlotSet("question_asked_time", current_time)]

class HandleAnswerAction(Action):

    def name(self):
        return "action_handle_answer"

    def run(self, dispatcher, tracker, domain):

        # Importing translator
        translator = Translator()
        user_locale = tracker.get_slot("locale")
        scenario_id = tracker.get_slot("scenario_id")

        user_id = tracker.get_slot("user_id")
        print(f'USER_ID ON HANDLE: ', user_id)
        if not user_locale:
            user_locale = tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        if not user_id:
            user_id = tracker.latest_message.get('metadata', {}).get('user_id', '')
            if not user_id:
                user_id = 7
        question_id = tracker.get_slot("last_question_id")
        if not question_id:
            connection = get_database_connection()
            cursor = connection.cursor()

            cursor.execute("""
                SELECT last_activity_id from authoringtool_userscenarioscore
                Where user_id = %s AND scenario_id = %s
                           """, (user_id, scenario_id))
            print("doulepse nice")
            result = cursor.fetchall()
            question_id = result[0]
            if not result:
                question_id = 2
        # Get the answer_id entity from the latest message
        answer_id = next(tracker.get_latest_entity_values("answer_id"), None)
        # scenario_id = tracker.get_slot("scenario_id")
        # Timing
        answer_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        question_time = tracker.get_slot("question_asked_time")
        # Calculate time difference
        fmt = "%Y-%m-%d %H:%M:%S"
        diff = datetime.strptime(answer_time, fmt) - datetime.strptime(question_time, fmt)
        seconds_taken = diff.total_seconds()

        connection = get_database_connection()
        cursor = connection.cursor()
        print(f"User2 ID: {user_id}, Question ID: {question_id}, Answer ID: {answer_id}")

        # Update user_answers
        cursor.execute("""
            INSERT INTO authoringtool_useranswer (user_id, activity_id, answer_id, timing, created_on)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, question_id, answer_id, seconds_taken, datetime.now()))
        
        print(f'THE ANSWER HAS BEEN COMMITED in authoringtool_useranswer')

        connection.commit()

        # Fetch feedback based on the provided answer
        # cursor.execute("""
        #    SELECT feedback_text
        #    FROM answer_feedback
        #    WHERE answer_id = %s
        #    """, (answer_id,))
        #result = cursor.fetchone()
        #if result:
        #    feedback_text = result[0]
        #    #translating feedback
        #    if user_locale != 'en':
        #        feedback_text = translator.translate(feedback_text, src='en', dest= user_locale).text
        #    dispatcher.utter_message(text=feedback_text)
        #else:
        #    sorry_text = "Sorry, we couldn't find feedback for this answer."
        #    if user_locale != 'en':
        #        sorry_text = translator.translate(sorry_text, src='en', dest=user_locale).text
        #    dispatcher.utter_message(text=sorry_text)

        cursor.execute("SELECT answer_weight, is_correct FROM authoringtool_answer where id = %s", (answer_id,))
        result = cursor.fetchone()
        if result:
            score_for_current_answer = result[0]
            is_answer_correct = result[1]
        else:
            dispatcher.utter_message(text="Sorry, we couldn't find a weight for this answer.")

        cursor.execute("SELECT user_score FROM authoringtool_userscenarioscore WHERE user_id = %s AND scenario_id = %s", (user_id, scenario_id))
        result = cursor.fetchone()

        print(f"Score for current answer: {score_for_current_answer}")
        # Updating question count
        if is_answer_correct:
            cursor.execute("UPDATE authoringtool_activity SET correct_count = correct_count + 1 WHERE id = %s", (question_id,))
        else:
            cursor.execute("UPDATE authoringtool_activity SET incorrect_count = incorrect_count + 1 WHERE id = %s", (question_id,))

        connection.commit()

        if result:
            # Update the score
            current_score = result[0]
            print(f"Current score of user: {current_score}")
            new_score = current_score + score_for_current_answer
            print(f"New score: {new_score}")
            cursor.execute("UPDATE authoringtool_userscenarioscore SET user_score = %s, last_activity_id = %s WHERE (user_id = %s AND scenario_id = %s)", (new_score, question_id, user_id, scenario_id))
        else:
            # Insert a new score entry for the user
            cursor.execute("INSERT INTO authoringtool_userscenarioscore (user_id, last_activity_id, scenario_id, user_score) VALUES (%s, %s, %s, %s)", (user_id, question_id, scenario_id, score_for_current_answer))

        connection.commit()

        # After the score has been updated or inserted for the current question:
        # 1. Check if the current question is evaluatable.
        cursor.execute("SELECT is_evaluatable FROM authoringtool_activity WHERE id = %s", (question_id,))
        result = cursor.fetchone()
        
        if not result:
            dispatcher.utter_message(text="Sorry, we encountered an error.")
            return []

        is_evaluatable = result[0]
        print(f"Is evaluatable: {is_evaluatable}")

        if is_evaluatable:
        # 2. Sum up the scores of the bunch of questions linked to the current evaluatable question.
            cursor.execute("""
                SELECT SUM(a.answer_weight), COUNT(DISTINCT ua.activity_id)
                FROM authoringtool_useranswer ua
                JOIN authoringtool_answer a ON ua.answer_id = a.id
                WHERE ua.user_id = %s AND ua.activity_id IN (SELECT unnest(activity_ids) FROM authoringtool_questionbunch WHERE activity_primary_id = %s)
                """, (user_id, question_id))
    
            result = cursor.fetchone()
            total_score_for_bunch = result[0] if result else 0
            count_activity_ids = result[1] if result else 1
            print(f"total_score_for_bunch of user: {total_score_for_bunch} & count_activity_ids: {count_activity_ids}")
            

        # 3. Determine next question based on the score
            cursor.execute("""
                SELECT next_question_on_high_id, next_question_on_mid_id, next_question_on_low_id
                FROM authoringtool_evquestionbranching
                WHERE activity_id = %s
            """, (question_id,))
            result = cursor.fetchone()
            print(f"result of ids in next_ev: {result}")
            
            if not result:
                dispatcher.utter_message(text="Sorry, we couldn't find branching information.")
                return []

            high_dest, mid_dest, low_dest = result
            print(f"high_dest, mid_dest, low_dest: {high_dest}, {mid_dest}, {low_dest}")
            cursor.execute("SELECT score_limit FROM authoringtool_activity WHERE id = %s", (high_dest,))
            high_limit = cursor.fetchone()[0]

            cursor.execute("SELECT score_limit FROM authoringtool_activity WHERE id = %s", (mid_dest,))
            mid_limit = cursor.fetchone()[0]
            
            total_score_devided = total_score_for_bunch / count_activity_ids
            # Changed total score for the same devided the count of the answered activities 11/07/24
            if total_score_devided >= high_limit:
                next_question_id = high_dest
            elif total_score_devided >= mid_limit:
                next_question_id = mid_dest
            else:
                next_question_id = low_dest

        else:
            # If the question is not evaluatable, fetch the next question based on user's answer.
            cursor.execute("""
                SELECT next_activity_id
                FROM authoringtool_nextquestionlogic
                WHERE activity_id = %s AND answer_id = %s
            """, (question_id, answer_id))
            
            result = cursor.fetchone()
            if result == None:
                if user_locale.lower() == 'greek' or user_locale.lower() == "ελληνικά":
                    dispatcher.utter_message(text="Αυτό είναι το τέλος του σεναρίου!")
                else:
                    dispatcher.utter_message(text="This is the end of the scenario!")
                # This is the end of the scenario!
                # Αυτό είναι το τέλος του σεναρίου!
                next_question_id = None
                # return [SlotSet("next_question_id", next_question_id)] 10/05/24
                return [AllSlotsReset(), FollowupAction("action_end_scenario")]
            print(f'RESULT EINAI: ',result, 'KAIIIIII ', result[0])

            next_question_id = result[0] if result else None

        # Ending action
        if next_question_id:
            return [SlotSet("next_question_id", next_question_id)]

class ProvideHintAction(Action):

    def name(self):
        return "action_provide_hint"

    def run(self, dispatcher, tracker, domain):
        question_id = tracker.get_slot("last_question_id")
        user_locale = tracker.get_slot("locale")
        translator = Translator()

        connection = get_database_connection()
        cursor = connection.cursor()

        # Fetch the hint related to the current question from the hints table
        cursor.execute("""
            SELECT hint_text, hint_img_url, hint_video_url
            FROM hints
            WHERE question_id = %s
        """, (question_id,))

        hint_data = cursor.fetchone()

        # Check if there's a hint for the current question
        if hint_data:
            hint_text, hint_img_url, hint_vid_url = hint_data
            if hint_text:
                if user_locale != 'en':
                    hint_text = translator.translate(hint_text, src='en', dest=user_locale).text
                dispatcher.utter_message(text=hint_text)
            if hint_img_url:
                dispatcher.utter_message(image=hint_img_url)
            if hint_vid_url:
                dispatcher.utter_message(custom={"video": hint_vid_url})
        else:
            do_text = "Looks like you can do it yourself!"
            if user_locale != 'en':
                do_text = translator.translate(do_text, src='en', dest=user_locale).text
            dispatcher.utter_message(text=do_text)

        # Ask the question again.
        return [FollowupAction("action_ask_question")]
    
class DeleteDatabaseData(Action):

    def name(self):
        return "action_delete_db_data"
    
    def run(self, dispatcher, tracker, domain):
        user_id = tracker.get_slot("user_id")
        if not user_id:
            user_id = tracker.latest_message.get('metadata', {}).get('user_id', '')
            if not user_id:
                user_id = 7
        print(user_id)
        scenario_id = tracker.get_slot("scenario_id")
        if not scenario_id:
            scenario_id = tracker.latest_message.get('metadata', {}).get('scenario_id', '')
        user_locale = tracker.get_slot("locale")
        translator = Translator()

        try:
            connection = get_database_connection()
            cursor = connection.cursor()

            cursor.execute("""
                DELETE FROM authoringtool_useranswer
                WHERE activity_id IN (
                    SELECT authoringtool_activity.id
                    FROM authoringtool_activity
                    WHERE authoringtool_activity.scenario_id = %s
                )
                AND user_id = %s
            """, (scenario_id, user_id,))
            print("first ex")

            cursor.execute("""
                DELETE FROM authoringtool_userscenarioscore
                WHERE user_id = %s AND scenario_id = %s
            """, (user_id, scenario_id,))
            print("sec ex")

            cursor.execute("""
                DELETE FROM authoringtool_phetlabsessions
                WHERE activity_id IN (
                    SELECT authoringtool_activity.id
                    FROM authoringtool_activity
                    WHERE authoringtool_activity.scenario_id = %s
                )
                AND user_id = %s
            """, (scenario_id, user_id,))
            print("thrd ex")

            connection.commit()
            print("committed")

        except Exception as e:
            print("An error occurred:", e)
            connection.rollback()

        finally:
            cursor.close()
            connection.close()

        dispatched_text = "Data deleted."
        # if user_locale != 'en':
        #     dispatched_text = translator.translate(dispatched_text, src='en', dest=user_locale).text
        dispatcher.utter_message(text=dispatched_text)

        return [SlotSet("user_id", user_id), SlotSet("last_question_id", None)]

class ActionConfirm(Action):
    
    def name(self):
        return "action_confirm"

    def run(self, dispatcher,tracker, domain):

        user_locale = tracker.get_slot("locale")
        question_id = tracker.get_slot("last_question_id")
        print(tracker.current_slot_values())
        
        print(f"Question ID: ", question_id)
        translator = Translator()

        aff_text = "Are you sure you want to proceed?"
        # if user_locale != 'en':
        #      aff_text = translator.translate(aff_text, src='en', dest=user_locale).text
        dispatcher.utter_message(text=aff_text)

        return [SlotSet("last_question_id", None)], SlotSet("next_question_id", None)
    

class ActionConfirmRead(Action):
    def name(self):
        return "action_confirm_read"

    def run(self, dispatcher, tracker, domain):
        # Fetch next question ID from the database or context
        question_id = tracker.get_slot("current_question_id")
        scenario_id = tracker.get_slot("scenario_id")
        if not scenario_id:
            scenario_id = tracker.latest_message.get('metadata', {}).get('scenario_id', '')
        user_id = tracker.get_slot("user_id")
        if not user_id:
            user_id = tracker.latest_message.get('metadata', {}).get('user_id', '')
        user_locale = tracker.get_slot("user_locale")
        if not user_locale:
            user_locale = tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        
        
        # Timing
        answer_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        question_time = tracker.get_slot("question_asked_time")
        # Calculate time difference
        fmt = "%Y-%m-%d %H:%M:%S"
        diff = datetime.strptime(answer_time, fmt) - datetime.strptime(question_time, fmt)
        seconds_taken = diff.total_seconds()
        
        connection = get_database_connection()
        cursor = connection.cursor()
        
        # Update user_answers
        cursor.execute("""
            INSERT INTO authoringtool_useranswer (user_id, activity_id, timing, created_on)
            VALUES (%s, %s, %s, %s)
        """, (user_id, question_id, seconds_taken, datetime.now()))

        connection.commit()

        cursor.execute("""
                SELECT next_activity_id
                FROM authoringtool_nextquestionlogic
                WHERE activity_id = %s
            """, (question_id,))

        result = cursor.fetchone()
        if result == None:
            if user_locale.lower() == 'greek' or user_locale.lower() == "ελληνικά":
                dispatcher.utter_message(text="Αυτό είναι το τέλος του σεναρίου!")
            else:
                dispatcher.utter_message(text="This is the end of the scenario!")
            # This is the end of the scenario!
            # Αυτό είναι το τέλος του σεναρίου!
            next_question_id = None
            return [AllSlotsReset(), FollowupAction("action_end_scenario")]
            # return [SlotSet("next_question_id", next_question_id), SlotSet("scenario_id", scenario_id)]
        else:
            result_1 = result[0]
            print(f'RESULT EINAI EDO: ', result, result_1)
            
            next_question_id = result[0]
            query = """
            SELECT sim.name FROM authoringtool_activity q
            JOIN authoringtool_simulation sim ON q.simulation_id = sim.id
            WHERE q.id = %s
            """

            # Execute the query
            cursor.execute(query, (question_id,))
            # Fetch all the results
            result_exp = cursor.fetchone()
            print(f'EINAI MIPWS EXP?', result_exp)
            if result_exp:
                if result_exp[0] == 'Pendulum Lab':
                    getPhetPendulumData(tracker)
        
            return [SlotSet("next_question_id", next_question_id), FollowupAction("action_ask_question"), SlotSet("scenario_id", scenario_id)]
        

class ActionEndScenario(Action):
    def name(self):
        return "action_end_scenario"

    def run(self, dispatcher, tracker, domain):
        # Custom message to wrap things up
        user_locale = tracker.get_slot("user_locale")
        if not user_locale:
            user_locale = tracker.latest_message.get('metadata', {}).get('scenario_lang', '')
        if user_locale.lower() == 'greek' or user_locale.lower() == "ελληνικά":
            dispatcher.utter_message(text="Ευχαριστούμε για την συμμετοχή σου!")
        else:
            dispatcher.utter_message(text="Thank you for participating!")
        # Thank you for participating!
        # Ευχαριστούμε για την συμμετοχή σου!
        # Possibly offer buttons here for other options like restarting or seeing results
        return [AllSlotsReset()]

def getPhetPendulumData(tracker):
    # Define the default function
    def handle_null_parameters():
        # Logic to handle null parameters
        print("Some parameters are null. Handling default behavior.")
        # For example, you can set default values or skip the insert
        return

    metadata = tracker.latest_message.get('metadata', {})
    print(f'TA METADATA MESA STO FUNC: ', metadata)

    pendulum_data = metadata.get('pendulum_data', None)
    if not pendulum_data:
        handle_null_parameters()
        return

    # Getting Slots
    question_id = tracker.get_slot("current_question_id")
    user_id = tracker.get_slot("user_id")
    if not user_id:
        user_id = metadata.get("user_id", None)
    if not question_id:
        handle_null_parameters()
        return

    # Initializing lab's pendulum
    mass_1 = None
    mass_2 = None
    length_1 = None
    length_2 = None
    angle_1 = None
    angle_2 = None
    gravity = None
    friction = None

    # Current timestamp
    timestamp = datetime.now()

    # Check if Pendulum 1 exists
    pendulum_1 = metadata['pendulum_data'].get('Pendulum 1', None)
    if pendulum_1:
        # Extracting data for Pendulum 1
        try:
            mass_1 = metadata['pendulum_data']['Pendulum 1']['mass']
            length_1 = metadata['pendulum_data']['Pendulum 1']['length']
            angle_1 = metadata['pendulum_data']['Pendulum 1']['angle']
        except KeyError as e:
            handle_null_parameters()
            return

        # Check if Pendulum 2 exists
        pendulum_2 = metadata['pendulum_data'].get('Pendulum 2', None)
        if pendulum_2:
            try:
                # Extracting data for Pendulum 2
                mass_2 = metadata['pendulum_data']['Pendulum 2']['mass']
                length_2 = metadata['pendulum_data']['Pendulum 2']['length']
                angle_2 = metadata['pendulum_data']['Pendulum 2']['angle']
            except KeyError as e:
                handle_null_parameters()
                return

        # Rest of data - Same for Pendulum 1 & 2
        try:
            gravity = metadata['pendulum_data']['Pendulum 1']['gravity']
            friction = metadata['pendulum_data']['Pendulum 1']['friction']
        except KeyError as e:
            handle_null_parameters()
            return

    connection = None
    cursor = None
    try:
        # Connection to DB
        connection = get_database_connection()
        cursor = connection.cursor()

        # Insert data into the lab_sessions table
        insert_query = """
        INSERT INTO authoringtool_phetlabsessions (
            name, gravity, friction, timestamp, activity_id, user_id, angle_1, angle_2, 
            mass_1, mass_2, length_1, length_2
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            'PhetPendulum', gravity, friction, timestamp, question_id, user_id, angle_1, angle_2,
            mass_1, mass_2, length_1, length_2
        ))

        # Commit the transaction
        connection.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while connecting to PostgreSQL: {error}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            print("PostgreSQL connection is closed")
