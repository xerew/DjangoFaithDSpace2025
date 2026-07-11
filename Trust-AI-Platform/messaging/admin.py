from django.contrib import admin

from .models import Message


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'sender', 'recipient', 'created_at', 'read_at')
    list_filter = ('created_at',)
    search_fields = ('sender__username', 'recipient__username', 'body')
    raw_id_fields = ('sender', 'recipient')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
