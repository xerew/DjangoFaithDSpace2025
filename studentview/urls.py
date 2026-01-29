from django.urls import path
from . import views
from authoringtool import views as authoring_views

urlpatterns = [
    path('scenario/<int:scenario_id>/', views.scenario_viewer, name='studentView'),
    path('scenario/<int:scenario_id>/questions/', views.scenario_form, name='scenario_questions'),
    path('scenario/<int:scenario_id>/questions/submit/', views.submit_answers, name='submit_answers'),
    path('chatbot/', views.chatbot_interaction, name='chatbot_interaction'),
    path('get_activity/<int:activity_id>/', views.get_activity, name='get_activity'),
    path('chat/', views.chat_interface, name='chat_interface'),
    path('scenarios', views.scenarios_view, name='studentScenarios'),
    path('pendulum-lab/', views.pendulum_lab_view, name='pendulum_lab'),
    path('ltiintegration/<int:experiment_id>/', authoring_views.lms_lti_launch, name='lti_integration_in_student'),
    path('save_iteration/', views.save_iteration_remlab, name='save_iteration')
]
