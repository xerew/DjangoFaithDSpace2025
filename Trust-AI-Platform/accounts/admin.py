from django.contrib import admin

from .models import BulkEmailCampaign

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
