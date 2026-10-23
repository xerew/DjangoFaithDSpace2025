from django.urls import path

from . import views

urlpatterns = [
    path('switch/', views.switch_language, name='switch_language'),
]
