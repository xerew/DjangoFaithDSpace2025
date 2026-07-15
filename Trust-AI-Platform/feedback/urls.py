from django.urls import path
from . import views

urlpatterns = [
    path('submit/<int:form_id>/<int:scenario_id>/', views.submit_feedback, name='feedback_submit'),
    path('manage/', views.feedback_form_list, name='feedback_form_list'),
    path('manage/create/', views.feedback_form_create, name='feedback_form_create'),
    path('manage/<int:form_id>/edit/', views.feedback_form_edit, name='feedback_form_edit'),
    path('manage/<int:form_id>/delete/', views.feedback_form_delete, name='feedback_form_delete'),
    path('manage/<int:form_id>/responses/', views.feedback_form_responses, name='feedback_form_responses'),
    path('manage/response/<int:response_id>/delete/', views.feedback_response_delete, name='feedback_response_delete'),
    path('manage/<int:form_id>/export/csv/', views.feedback_form_export_csv, name='feedback_form_export_csv'),
    path('manage/<int:form_id>/export/xlsx/', views.feedback_form_export_xlsx, name='feedback_form_export_xlsx'),
]
