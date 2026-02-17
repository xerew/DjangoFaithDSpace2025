from django.urls import path
from . import views

urlpatterns = [
    path('groups/', views.list_groups, name='list_groups'),
    path('groups/create/', views.create_user_group, name='create_group'),
    path('groups/<int:group_id>/view/', views.view_group, name='view_group'),
    path('groups/<int:group_id>/download/', views.download_credentials, name='download_credentials'),
    path('groups/<int:group_id>/edit/', views.edit_group, name='edit_group'),
    path('groups/<int:group_id>/delete/', views.delete_group, name='delete_group'),
    path('groups/student-groups/', views.list_student_groups, name='list_student_groups'),
    path('export/multilingual-answers/', views.export_multilingual_answers_csv, name='export_multilingual_answers_csv'),
]
