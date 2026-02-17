from django.contrib import admin
from .models import UserGroup, UserGroupMembership

# Inline to manage memberships (students) directly from the UserGroup page
class UserGroupMembershipInline(admin.TabularInline):
    model = UserGroupMembership
    extra = 0  # No extra empty rows unless needed
    readonly_fields = ('password',)  # Show generated passwords (readonly)
    autocomplete_fields = ('user',)  # If you want to search users easily

# Admin for UserGroup
@admin.register(UserGroup)
class UserGroupAdmin(admin.ModelAdmin):
    list_display = ('name', 'prefix', 'number_of_users', 'created_by', 'created_on')  # Main columns shown in admin list
    search_fields = ('name', 'prefix', 'created_by__username')  # Searchable fields in admin
    list_filter = ('created_on', 'created_by')  # Filter sidebar
    inlines = [UserGroupMembershipInline]  # Add membership editor inside group
    filter_horizontal = ('assigned_scenarios',)  # Nice widget to assign scenarios

    # Optional: Make some fields read-only if you don't want them to change
    readonly_fields = ('created_on',)

    # Optional: Auto-assign 'created_by' to the current admin user when creating
    def save_model(self, request, obj, form, change):
        if not change:  # If it's a new object
            obj.created_by = request.user
        obj.save()

# Admin for UserGroupMembership (if you want to manage them separately too)
@admin.register(UserGroupMembership)
class UserGroupMembershipAdmin(admin.ModelAdmin):
    list_display = ('group', 'user', 'password')  # Show group, user, and password
    search_fields = ('group__name', 'user__username')  # Searchable
    list_filter = ('group',)  # Filter sidebar
    readonly_fields = ('password',)  # Don't allow password change here (optional)
