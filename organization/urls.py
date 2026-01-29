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
]