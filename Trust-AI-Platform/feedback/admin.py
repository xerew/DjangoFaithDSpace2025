from django.contrib import admin

from .models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse


class FeedbackQuestionInline(admin.TabularInline):
    model = FeedbackQuestion
    extra = 0


@admin.register(FeedbackForm)
class FeedbackFormAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'audience', 'is_active', 'assign_to_all', 'created_by', 'created_on')
    list_filter = ('audience', 'is_active', 'assign_to_all')
    search_fields = ('title',)
    filter_horizontal = ('included_scenarios', 'excluded_scenarios')
    readonly_fields = ('created_on', 'updated_on')
    inlines = [FeedbackQuestionInline]


@admin.register(FeedbackResponse)
class FeedbackResponseAdmin(admin.ModelAdmin):
    list_display = ('id', 'form', 'user', 'scenario', 'submitted_at')
    list_filter = ('form', 'submitted_at')
    search_fields = ('user__username', 'form__title', 'scenario__name')
    raw_id_fields = ('form', 'user', 'scenario')
    readonly_fields = ('submitted_at',)


@admin.register(FeedbackAnswer)
class FeedbackAnswerAdmin(admin.ModelAdmin):
    list_display = ('id', 'response', 'question', 'answer_text')
    search_fields = ('answer_text', 'question__text')
    raw_id_fields = ('response', 'question')
