from django.contrib.auth import views as auth_views
from django.urls import path
from . import views
from .views import CustomPasswordResetView
from .forms import CustomSetPasswordForm
from django.views.generic import TemplateView

urlpatterns = [
    #path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/login/'), name='logout'),
    path('register/', views.registerAccount, name='register'),
    # path('password-reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/', CustomPasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    # path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path(
        'reset/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(form_class=CustomSetPasswordForm),
        name='password_reset_confirm',
    ),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
    path('documentation/', views.documentation_view, name='documentation_and_tutorials'),
    path('tos/', views.tos_view, name='tos'),
]