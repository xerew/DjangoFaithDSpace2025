from django.urls import path
from . import views

urlpatterns = [
    path('', views.list_organizations, name='list_organizations'),
    path('create_organization/', views.create_organization, name='create_organization'),
    path('organization/<int:org_id>/', views.organization_detail, name='organization_detail'),
    path('organization/<int:org_id>/make_admin/<int:user_id>/', views.make_admin, name='make_admin'),
    path('organization/<int:org_id>/delete/', views.delete_organization, name='delete_organization'),
    path('add_member/<int:org_id>/', views.add_member_to_org, name='add_member_to_org'),
    path('add_member_confirm/<int:org_id>/<int:user_id>/', views.add_member_to_org_confirm, name='add_member_to_org_confirm'),
    path('promote_admin/<int:org_id>/<int:user_id>/', views.promote_admin, name='promote_admin'),
    path('demote_admin/<int:org_id>/<int:user_id>/', views.demote_admin, name='demote_admin'),
    path('remove_member/<int:org_id>/<int:user_id>/', views.remove_member, name='remove_member'),
    path('edit_organization/<int:org_id>/', views.edit_organization, name='edit_organization'),
    path('organization/<int:org_id>/request_to_join/', views.request_to_join, name='request_to_join'),
    path('join_request/<int:request_id>/approve/', views.approve_join_request, name='approve_join_request'),
    path('join_request/<int:request_id>/reject/', views.reject_join_request, name='reject_join_request'),
    path('organization/<int:org_id>/announcements/create/', views.create_announcement, name='create_announcement'),
    path('organization/<int:org_id>/announcements/<int:announcement_id>/edit/', views.edit_announcement, name='edit_announcement'),
    path('organization/<int:org_id>/announcements/<int:announcement_id>/delete/', views.delete_announcement, name='delete_announcement'),
    path('organization/<int:org_id>/chat/', views.org_chat, name='org_chat'),
    path('organization/<int:org_id>/chat/send/', views.send_org_chat_message, name='send_org_chat_message'),
    path('organization/<int:org_id>/chat/poll/', views.org_chat_poll, name='org_chat_poll'),
]