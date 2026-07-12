from django.contrib import admin
from django.utils.html import format_html
from .models import Organization, Announcement, OrgChatMessage


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'short_name', 'country', 'language', 'member_count', 'created_by', 'created_on')
    list_filter = ('country', 'language')
    search_fields = ('name', 'short_name', 'country')
    readonly_fields = ('created_on', 'updated_on', 'created_by', 'updated_by')
    filter_horizontal = ('admins', 'members')

    def get_queryset(self, request):
        from django.db.models import Count
        return super().get_queryset(request).annotate(_member_count=Count('members', distinct=True))

    def member_count(self, obj):
        return format_html('<strong>{}</strong>', obj._member_count)
    member_count.short_description = 'Members'
    member_count.admin_order_field = '_member_count'

    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(Announcement)
class AnnouncementAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'organization', 'created_by', 'created_on')
    list_filter = ('created_on',)
    search_fields = ('title', 'organization__name', 'organization__short_name')
    raw_id_fields = ('organization', 'created_by')
    readonly_fields = ('created_on', 'updated_on')
    date_hierarchy = 'created_on'


@admin.register(OrgChatMessage)
class OrgChatMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'organization', 'sender', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('organization__name', 'organization__short_name', 'sender__username', 'body')
    raw_id_fields = ('organization', 'sender')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
