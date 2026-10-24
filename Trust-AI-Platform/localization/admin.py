from django.contrib import admin

from .models import TranslationOverride, TranslationOverrideLog, TranslationReviewer


@admin.register(TranslationReviewer)
class TranslationReviewerAdmin(admin.ModelAdmin):
    list_display = ('user', 'language')
    list_filter = ('language',)
    search_fields = ('user__username',)
    raw_id_fields = ('user',)


@admin.register(TranslationOverride)
class TranslationOverrideAdmin(admin.ModelAdmin):
    list_display = ('language', 'msgid', 'msgctxt', 'plural_index', 'msgstr', 'updated_by', 'updated_at')
    list_filter = ('language',)
    search_fields = ('msgid', 'msgstr')
    readonly_fields = ('updated_by', 'updated_at')

    def save_model(self, request, obj, form, change):
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
        from .overrides import bump_overrides_version
        bump_overrides_version()

    def delete_model(self, request, obj):
        super().delete_model(request, obj)
        from .overrides import bump_overrides_version
        bump_overrides_version()


@admin.register(TranslationOverrideLog)
class TranslationOverrideLogAdmin(admin.ModelAdmin):
    list_display = ('at', 'language', 'msgid', 'old', 'new', 'user')
    list_filter = ('language',)
    search_fields = ('msgid', 'old', 'new')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
