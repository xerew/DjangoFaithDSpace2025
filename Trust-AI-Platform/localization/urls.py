from django.urls import path

from . import views

urlpatterns = [
    path('switch/', views.switch_language, name='switch_language'),
    path('review/', views.review_index, name='review_index'),
    path('review/<str:code>/', views.review_translations, name='review_translations'),
    path('review/<str:code>/save/', views.save_translation, name='save_translation'),
]
