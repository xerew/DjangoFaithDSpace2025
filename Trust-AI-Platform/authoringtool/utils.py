from .models import UserAnswer
from django.db.models import Max, Min, OuterRef, Subquery
import re

def get_last_answers(scenario_id):
    # Correlated subquery: for each (user, activity) row, pick the one with the highest id.
    # This avoids pulling all IDs into Python memory and lets PostgreSQL resolve it in one query.
    latest_id = (
        UserAnswer.objects
        .filter(
            user_id=OuterRef('user_id'),
            activity_id=OuterRef('activity_id'),
            activity__phase__scenario_id=scenario_id,
        )
        .order_by('-id')
        .values('id')[:1]
    )
    return (
        UserAnswer.objects
        .filter(activity__phase__scenario_id=scenario_id)
        .filter(id=Subquery(latest_id))
        .select_related('answer', 'activity', 'user')
    )

# def get_first_answers(scenario_id):
#     # Fetch the earliest answer for each user and activity based on the created_on timestamp
#     first_answers = (
#         UserAnswer.objects.filter(activity__phase__scenario_id=scenario_id)
#         .values('user_id', 'activity_id')
#         .annotate(first_answer_id=Min('id'))  # Get the first answer ID for each user and activity
#     )
def get_first_answers(scenario_id):
    # Correlated subquery: for each (user, activity) row, pick the one with the lowest id.
    # This avoids pulling all IDs into Python memory and lets the database resolve it in one query.
    first_subquery = (
        UserAnswer.objects
        .filter(activity=OuterRef('activity'), user=OuterRef('user'))
        .order_by('id')
        .values('id')[:1]
    )
    return UserAnswer.objects.filter(
        activity__phase__scenario_id=scenario_id,
        id=Subquery(first_subquery)
    )