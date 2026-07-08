from django.contrib.auth import views as auth_views
from django.urls import path
from . import views
from . import admin_views
from . import admin_lab_views
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
    path('profile/', views.profile_view, name='profile'),
    path('admin/', admin_views.admin_dashboard, name='admin_dashboard'),
    path('admin/edit_user/<int:user_id>/', admin_views.admin_edit_user, name='admin_edit_user'),
    path('admin/delete_user/<int:user_id>/', admin_views.admin_delete_user, name='admin_delete_user'),
    path('admin/toggle_user/<int:user_id>/', admin_views.admin_toggle_user, name='admin_toggle_user'),
    path('admin/create_role/', admin_views.admin_create_role, name='admin_create_role'),
    path('admin/rename_role/<int:role_id>/', admin_views.admin_rename_role, name='admin_rename_role'),
    path('admin/delete_role/<int:role_id>/', admin_views.admin_delete_role, name='admin_delete_role'),
    path('admin/impersonate/<int:user_id>/', admin_views.admin_impersonate, name='admin_impersonate'),
    path('admin/impersonate_exit/', admin_views.admin_impersonate_exit, name='admin_impersonate_exit'),
    path('admin/simulations/create/', admin_lab_views.admin_create_simulation, name='admin_create_simulation'),
    path('admin/simulations/<int:sim_id>/edit/', admin_lab_views.admin_edit_simulation, name='admin_edit_simulation'),
    path('admin/simulations/<int:sim_id>/delete/', admin_lab_views.admin_delete_simulation, name='admin_delete_simulation'),
    path('admin/remote_labs/create/', admin_lab_views.admin_create_remote_lab, name='admin_create_remote_lab'),
    path('admin/remote_labs/<int:lab_id>/edit/', admin_lab_views.admin_edit_remote_lab, name='admin_edit_remote_lab'),
    path('admin/remote_labs/<int:lab_id>/delete/', admin_lab_views.admin_delete_remote_lab, name='admin_delete_remote_lab'),
    path('admin/vr_labs/create/', admin_lab_views.admin_create_vr_lab, name='admin_create_vr_lab'),
    path('admin/vr_labs/<int:vr_id>/edit/', admin_lab_views.admin_edit_vr_lab, name='admin_edit_vr_lab'),
    path('admin/vr_labs/<int:vr_id>/delete/', admin_lab_views.admin_delete_vr_lab, name='admin_delete_vr_lab'),
]