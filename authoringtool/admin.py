from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import Scenario, Phase, Activity, ActivityType, Answer, AnswerFeedback, NextQuestionLogic, EvQuestionBranching, QuestionBunch, Simulation, SchoolDepartment, ExperimentLL, RemoteLabSession, VRARExperiment, UserScenarioScore, UserAnswer, PhetLabSessions, MultilingualQuestion, MultilingualAnswer, ActivityFlag, ActivityProposal,QValue, UserProposalReview
# Register your models here.

admin.site.register(Scenario)
admin.site.register(Phase)
admin.site.register(ActivityType)
admin.site.register(Activity)
admin.site.register(Simulation)
admin.site.unregister(User)
admin.site.register(Answer)
admin.site.register(AnswerFeedback)
admin.site.register(NextQuestionLogic)
admin.site.register(EvQuestionBranching)
admin.site.register(QuestionBunch)
admin.site.register(SchoolDepartment)
admin.site.register(ExperimentLL)
admin.site.register(RemoteLabSession)
admin.site.register(UserScenarioScore)
admin.site.register(UserAnswer)
admin.site.register(PhetLabSessions)
admin.site.register(ActivityFlag)
admin.site.register(ActivityProposal)
# admin.site.register(UserProposalReview)

@admin.register(VRARExperiment)
class VRARExperimentAdmin(admin.ModelAdmin):
    list_display = ('name', 'launch_url', 'qr_code_display')
    readonly_fields = ('qr_code_display',)

    def qr_code_display(self, obj):
        if obj.qr_code:
            return f'<img src="{obj.qr_code.url}" width="150" height="150" />'
        return "No QR Code"
    
    qr_code_display.allow_tags = True  # This is important to render the HTML code
    qr_code_display.short_description = 'QR Code'

class CustomUserAdmin(UserAdmin):
    list_display = ('id', 'username', 'email', 'first_name', 'last_name', 'is_staff', 'school_department')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email', 'school_department')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'password1', 'password2', 'school_department'),
        }),
    )

admin.site.register(User, CustomUserAdmin)

class MultilingualAnswerInline(admin.TabularInline):
    model = MultilingualAnswer
    extra = 0
    readonly_fields = ('created_on', 'updated_on', 'created_by', 'updated_by')
    fields = ('user', 'scenario', 'answer_text', 'created_on', 'updated_on', 'created_by', 'updated_by')

@admin.register(MultilingualQuestion)
class MultilingualQuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'question_text_en', 'is_required', 'order', 'created_on', 'updated_on')
    list_filter = ('is_required',)
    search_fields = ('question_text_en', 'question_text_pt', 'question_text_gr', 
                    'question_text_es', 'question_text_fr', 'question_text_de')
    ordering = ('order', 'created_on')
    inlines = [MultilingualAnswerInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('order', 'is_required')
        }),
        ('Question Text - English', {
            'fields': ('question_text_en',)
        }),
        ('Question Text - Portuguese', {
            'fields': ('question_text_pt',)
        }),
        ('Question Text - Greek', {
            'fields': ('question_text_gr',)
        }),
        ('Question Text - Spanish', {
            'fields': ('question_text_es',)
        }),
        ('Question Text - French', {
            'fields': ('question_text_fr',)
        }),
        ('Question Text - German', {
            'fields': ('question_text_de',)
        }),
    )
    
    def save_model(self, request, obj, form, change):
        if not change:  # If this is a new object
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(MultilingualAnswer)
class MultilingualAnswerAdmin(admin.ModelAdmin):
    list_display = ('id', 'question', 'user', 'scenario', 'created_on', 'updated_on')
    list_filter = ('scenario', 'user')
    search_fields = ('answer_text', 'question__question_text_en', 'user__username', 'scenario__name')
    readonly_fields = ('created_on', 'updated_on', 'created_by', 'updated_by')
    ordering = ('-created_on',)
    
    def save_model(self, request, obj, form, change):
        if not change:  # If this is a new object
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(QValue)
class QValueAdmin(admin.ModelAdmin):
    list_display = ('flag_type', 'category', 'action', 'q_value', 'updated_at')
    list_filter = ('flag_type', 'category', 'action')
    search_fields = ('flag_type',)
    ordering = ('-updated_at',)

@admin.register(UserProposalReview)
class UserProposalReviewAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'proposal',
        'user',
        'status',
        'reviewed_at',
    )
    list_filter = ('status', 'reviewed_at')
    search_fields = ('user__username', 'proposal__id')
    ordering = ('-reviewed_at',)