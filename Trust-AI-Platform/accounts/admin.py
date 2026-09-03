from django.contrib import admin

from django.utils import timezone

from .models import BulkEmailCampaign, MaintenanceNotice

# Register your models here.


@admin.register(BulkEmailCampaign)
class BulkEmailCampaignAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'subject', 'target_type', 'status', 'recipient_count',
        'sent_count', 'failed_count', 'created_by', 'created_at',
    )
    list_filter = ('target_type', 'status', 'created_at')
    search_fields = ('subject', 'created_by__username', 'created_by__email')
    readonly_fields = (
        'created_at', 'started_at', 'completed_at', 'recipient_count',
        'sent_count', 'failed_count', 'error_summary', 'celery_task_id',
    )
    filter_horizontal = ('organizations', 'recipients')


@admin.register(MaintenanceNotice)
class MaintenanceNoticeAdmin(admin.ModelAdmin):
    list_display = (
        'reason_summary', 'starts_at', 'ends_at', 'notice_status',
        'is_enabled', 'created_by',
    )
    list_filter = ('is_enabled', 'starts_at', 'ends_at')
    search_fields = ('reason', 'created_by__username', 'created_by__email')
    readonly_fields = ('created_by', 'created_at', 'updated_at')
    date_hierarchy = 'starts_at'
    ordering = ('-starts_at',)
    fieldsets = (
        (None, {'fields': ('reason', 'is_enabled')}),
        ('Schedule', {'fields': ('starts_at', 'ends_at')}),
        ('Audit', {'fields': ('created_by', 'created_at', 'updated_at')}),
    )

    @admin.display(description='Reason')
    def reason_summary(self, obj):
        return obj.reason[:80]

    @admin.display(description='Status')
    def notice_status(self, obj):
        return obj.state(timezone.now()).title()

    def save_model(self, request, obj, form, change):
        if obj.created_by_id is None:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
