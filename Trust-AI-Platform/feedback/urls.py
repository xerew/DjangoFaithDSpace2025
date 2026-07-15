from django.urls import path
from . import views

urlpatterns = [
    path('submit/<int:form_id>/<int:scenario_id>/', views.submit_feedback, name='feedback_submit'),
]
