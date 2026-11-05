from django.contrib import admin

from .models import TranslationOverride, TranslationOverrideLog, TranslationReviewer


@admin.register(TranslationReviewer)
class TranslationReviewerAdmin(admin.ModelAdmin):
    list_display = ('user', 'language')
    list_filter = ('language',)
    search_fields = ('user__username',)
    raw_id_fields = ('user',)


class ReadOnlyAdmin(admin.ModelAdmin):
    """Look, don't touch: corrections change on the reviewer page, which validates and logs them."""
    actions = None

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(TranslationOverride)
class TranslationOverrideAdmin(ReadOnlyAdmin):
    list_display = ('language', 'msgid', 'msgctxt', 'plural_index', 'msgstr', 'updated_by', 'updated_at')
    list_filter = ('language',)
    search_fields = ('msgid', 'msgstr')


@admin.register(TranslationOverrideLog)
class TranslationOverrideLogAdmin(ReadOnlyAdmin):
    list_display = ('at', 'language', 'msgid', 'old', 'new', 'user')
    list_filter = ('language',)
    search_fields = ('msgid', 'old', 'new')
