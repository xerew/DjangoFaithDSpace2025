from django.urls import path

from . import views

urlpatterns = [
    path('', views.message_threads, name='message_threads'),
    path('send/', views.send_message, name='send_message'),
    path('unread_status/', views.unread_status, name='unread_status'),
    path('<int:user_id>/', views.thread, name='thread'),
]
