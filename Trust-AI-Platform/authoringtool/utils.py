from .models import UserAnswer
from django.db.models import Max, Min
import re

def get_last_answers(scenario_id):
    # Fetch the last answers for each user and activity based on the created_on timestamp
    last_answers = UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id) \
        .values('user_id', 'activity_id') \
        .annotate(last_answer_id=Max('id'))  # Get the last answer ID for each user and activity

    # Use the last answer IDs to retrieve the corresponding UserAnswer objects
    return UserAnswer.objects.filter(id__in=[entry['last_answer_id'] for entry in last_answers])

# def get_first_answers(scenario_id):
#     # Fetch the earliest answer for each user and activity based on the created_on timestamp
#     first_answers = (
#         UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id)
#         .values('user_id', 'activity_id')
#         .annotate(first_answer_id=Min('id'))  # Get the first answer ID for each user and activity
#     )
def get_first_answers(scenario_id):
    # Fetch the earliest answer for each user and activity
    first_answers = (
        UserAnswer.objects
        .filter(activity__phase__scenario_id=scenario_id)
        .values('user_id', 'activity_id')
        .annotate(first_answer_id=Min('id'))  # use ID as a proxy for created_on
    )

    # Retrieve those UserAnswer objects
    return UserAnswer.objects.filter(
        id__in=[entry['first_answer_id'] for entry in first_answers]
    )

    # Use the first answer IDs to retrieve the corresponding UserAnswer objects
    # return UserAnswer.objects.filter(id__in=[entry['first_answer_id'] for entry in first_answers])