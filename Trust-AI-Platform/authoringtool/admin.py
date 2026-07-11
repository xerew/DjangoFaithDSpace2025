import csv
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.db.models import Count, Q
from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.utils.html import format_html
from django.urls import reverse
from .models import (
    Scenario, ScenarioHealthProxy, Phase, Activity, ActivityType,
    Answer, AnswerFeedback, NextQuestionLogic, EvQuestionBranching,
    QuestionBunch, Simulation, SchoolDepartment, ExperimentLL,
    RemoteLabSession, VRARExperiment, UserScenarioScore, UserAnswer,
    PhetLabSessions, MultilingualQuestion, MultilingualAnswer,
    ActivityFlag, ActivityProposal, ActivityProposalEditEvent, QValue, UserProposalReview, Language,
)
from usergroups.models import UserGroupMembership


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_csv_export(filename, headers, row_fn):
    """Return an admin action that exports the queryset as CSV."""
    def export(modeladmin, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}.csv"'
        writer = csv.writer(response)
        writer.writerow(headers)
        for obj in queryset.iterator():
            writer.writerow(row_fn(obj))
        return response
    export.short_description = f'Export selected as CSV ({filename})'
    export.__name__ = f'export_csv_{filename}'
    return export


# ─── Language ─────────────────────────────────────────────────────────────────

@admin.register(Language)
class LanguageAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)


# ─── Scenario ─────────────────────────────────────────────────────────────────

class PhaseInline(admin.TabularInline):
    model = Phase
    extra = 0
    fields = ('name', 'description', 'update_link')
    readonly_fields = ('update_link',)
    classes = ('collapse',)

    def update_link(self, obj):
        if obj.pk:
            url = reverse('admin:authoringtool_phase_change', args=[obj.pk])
            return format_html('<a href="{}">Update</a>', url)
        return '—'
    update_link.short_description = ''


@admin.register(Scenario)
class ScenarioAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'name', 'created_by', 'visibility_status', 'language',
        'implementation_count_col', 'is_personal', 'created_on',
    )
    list_filter = ('visibility_status', 'language', 'is_personal')
    search_fields = ('name', 'description', 'created_by__username')
    readonly_fields = ('created_on', 'updated_on', 'implementation_count_display')
    filter_horizontal = ('organizations',)
    raw_id_fields = ('created_by', 'updated_by', 'origin_scenario')
    date_hierarchy = 'created_on'
    inlines = [PhaseInline]

    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'description', 'learning_goals', 'subject_domains',
                       'language', 'age_of_students', 'suggested_learning_time'),
        }),
        ('Media', {
            'fields': ('image',),
            'classes': ('collapse',),
        }),
        ('Visibility & Access', {
            'fields': ('visibility_status', 'organizations', 'is_editable_by_org'),
        }),
        ('AI & Metrics', {
            'fields': ('implementation_count_display', 'ai_metrics_min_implementations'),
            'description': 'Controls when the "Scenario Metrics & AI" button appears for teachers viewing this scenario.',
        }),
        ('LLM Context', {
            'fields': ('llm_context',),
            'classes': ('collapse',),
        }),
        ('Origin', {
            'fields': ('is_personal', 'origin_scenario'),
            'classes': ('collapse',),
        }),
        ('Audit', {
            'fields': ('created_by', 'updated_by', 'created_on', 'updated_on'),
            'classes': ('collapse',),
        }),
    )

    def implementation_count_display(self, obj):
        if not obj.pk:
            return '—'
        count = UserScenarioScore.objects.filter(scenario=obj).values('user').distinct().count()
        threshold = obj.ai_metrics_min_implementations
        if count >= threshold:
            return format_html(
                '<strong style="color:#2e7d32">{} implementations</strong>'
                '&nbsp;<span style="color:#2e7d32;font-size:12px;">&#10003; Button is visible to teachers</span>',
                count,
            )
        needed = threshold - count
        return format_html(
            '<strong style="color:#e65100">{} implementations</strong>'
            '&nbsp;<span style="color:#e65100;font-size:12px;">'
            '&#10007; Need {} more to unlock the button (threshold: {})</span>',
            count, needed, threshold,
        )
    implementation_count_display.short_description = 'Current Implementations'

    def implementation_count_col(self, obj):
        count = UserScenarioScore.objects.filter(scenario=obj).values('user').distinct().count()
        color = '#2e7d32' if count >= obj.ai_metrics_min_implementations else '#e65100'
        return format_html('<span style="color:{};font-weight:600;">{}</span>', color, count)
    implementation_count_col.short_description = 'Implementations'


# ─── Scenario Health Check ────────────────────────────────────────────────────

@admin.register(ScenarioHealthProxy)
class ScenarioHealthAdmin(admin.ModelAdmin):
    change_list_template = 'admin/authoringtool/scenario_health.html'

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        base = Scenario.objects.select_related('created_by')
        admin_base = 'admin:authoringtool_scenario_change'

        # Critical: fewer than 5 phases (maximum allowed is 5)
        no_phases_qs = (
            base.annotate(pc=Count('phases', distinct=True))
            .filter(pc__lt=5)
        )
        no_phases = [
            {
                'name': s.name,
                'admin_url': reverse(admin_base, args=[s.pk]),
                'visibility_status': s.visibility_status,
                'get_visibility_status_display': s.get_visibility_status_display(),
                'phase_count': s.pc,
                'created_by': s.created_by,
                'created_on': s.created_on,
            }
            for s in no_phases_qs
        ]

        # Critical: has phases but no activities
        no_activities_qs = (
            base.annotate(
                pc=Count('phases', distinct=True),
                ac=Count('phases__activities', distinct=True),
            )
            .filter(pc__gt=0, ac=0)
        )
        no_activities = [
            {
                'name': s.name,
                'admin_url': reverse(admin_base, args=[s.pk]),
                'visibility_status': s.visibility_status,
                'get_visibility_status_display': s.get_visibility_status_display(),
                'phase_count': s.pc,
                'created_by': s.created_by,
            }
            for s in no_activities_qs
        ]

        # Warning: no evaluatable activities
        no_ev_qs = (
            base.annotate(
                activity_count=Count('phases__activities', distinct=True),
                ev_count=Count(
                    'phases__activities',
                    filter=Q(phases__activities__is_evaluatable=True),
                    distinct=True,
                ),
            )
            .filter(activity_count__gt=0, ev_count=0)
        )
        no_ev_activities = [
            {
                'name': s.name,
                'admin_url': reverse(admin_base, args=[s.pk]),
                'visibility_status': s.visibility_status,
                'get_visibility_status_display': s.get_visibility_status_display(),
                'activity_count': s.activity_count,
                'created_by': s.created_by,
            }
            for s in no_ev_qs
        ]

        # Warning: not assigned to any group
        no_groups_qs = (
            base.annotate(gc=Count('assigned_groups', distinct=True))
            .filter(gc=0)
        )
        no_groups = [
            {
                'name': s.name,
                'admin_url': reverse(admin_base, args=[s.pk]),
                'visibility_status': s.visibility_status,
                'get_visibility_status_display': s.get_visibility_status_display(),
                'impl_count': UserScenarioScore.objects.filter(scenario=s).values('user').distinct().count(),
                'created_by': s.created_by,
            }
            for s in no_groups_qs
        ]

        total = Scenario.objects.count()
        # Scenarios that appear in any issue bucket
        issue_ids = set(
            [s['name'] for s in no_phases]
            + [s['name'] for s in no_activities]
            + [s['name'] for s in no_ev_activities]
            + [s['name'] for s in no_groups]
        )
        healthy_count = total - len(set(
            list(no_phases_qs.values_list('pk', flat=True))
            + list(no_activities_qs.values_list('pk', flat=True))
            + list(no_ev_qs.values_list('pk', flat=True))
            + list(no_groups_qs.values_list('pk', flat=True))
        ))
        warning_count = len(set(
            list(no_ev_qs.values_list('pk', flat=True))
            + list(no_groups_qs.values_list('pk', flat=True))
        ))
        critical_count = len(set(
            list(no_phases_qs.values_list('pk', flat=True))
            + list(no_activities_qs.values_list('pk', flat=True))
        ))

        context = {
            **(extra_context or {}),
            **self.admin_site.each_context(request),
            'title': 'Scenario Health Check',
            'total': total,
            'healthy_count': healthy_count,
            'warning_count': warning_count,
            'critical_count': critical_count,
            'no_phases': no_phases,
            'no_activities': no_activities,
            'no_ev_activities': no_ev_activities,
            'no_groups': no_groups,
        }
        return TemplateResponse(request, self.change_list_template, context)


# ─── Phase ────────────────────────────────────────────────────────────────────

class ActivityInline(admin.TabularInline):
    model = Activity
    extra = 0
    fields = ('name', 'activity_type_label', 'is_evaluatable', 'is_primary_ev')
    readonly_fields = ('activity_type_label',)
    classes = ('collapse',)
    show_change_link = False

    def activity_type_label(self, obj):
        return obj.activity_type.name if obj.activity_type_id else '—'
    activity_type_label.short_description = 'Activity Type'


@admin.register(Phase)
class PhaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'scenario_link', 'created_by', 'created_on')
    list_filter = ('scenario',)
    search_fields = ('name', 'scenario__name', 'created_by__username')
    readonly_fields = ('created_on', 'updated_on')
    raw_id_fields = ('scenario', 'created_by', 'updated_by')
    list_select_related = ('scenario',)
    inlines = [ActivityInline]

    def scenario_link(self, obj):
        url = reverse('admin:authoringtool_scenario_change', args=[obj.scenario_id])
        return format_html('<a href="{}">{}</a>', url, obj.scenario.name)
    scenario_link.short_description = 'Scenario'
    scenario_link.admin_order_field = 'scenario__name'


# ─── Activity ─────────────────────────────────────────────────────────────────

class AnswerInline(admin.TabularInline):
    model = Answer
    extra = 0
    fields = ('text', 'is_correct', 'answer_weight', 'update_link')
    readonly_fields = ('update_link',)

    def update_link(self, obj):
        if obj.pk:
            url = reverse('admin:authoringtool_answer_change', args=[obj.pk])
            return format_html('<a href="{}">Update</a>', url)
        return '—'
    update_link.short_description = ''


@admin.register(Activity)
class ActivityAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'scenario_name', 'phase_name', 'activity_type', 'is_evaluatable', 'created_on')
    list_filter = ('activity_type', 'is_evaluatable', 'is_primary_ev')
    search_fields = ('name', 'scenario__name', 'phase__name')
    readonly_fields = ('created_on', 'updated_on', 'activity_type')
    raw_id_fields = ('scenario', 'phase', 'created_by', 'updated_by', 'simulation', 'experiment_ll', 'vr_ar_experiment')
    list_select_related = ('scenario', 'phase', 'activity_type')
    inlines = [AnswerInline]

    def scenario_name(self, obj):
        return obj.scenario.name if obj.scenario_id else '—'
    scenario_name.short_description = 'Scenario'
    scenario_name.admin_order_field = 'scenario__name'

    def phase_name(self, obj):
        return obj.phase.name if obj.phase_id else '—'
    phase_name.short_description = 'Phase'
    phase_name.admin_order_field = 'phase__name'


# ─── ActivityType ─────────────────────────────────────────────────────────────

@admin.register(ActivityType)
class ActivityTypeAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_on')
    search_fields = ('name',)


# ─── Answer ───────────────────────────────────────────────────────────────────

class AnswerFeedbackInline(admin.TabularInline):
    model = AnswerFeedback
    extra = 0
    fields = ('text',)


@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    list_display = ('id', 'short_text', 'activity_link', 'is_correct', 'answer_weight', 'created_on')
    list_filter = ('is_correct',)
    search_fields = ('text', 'activity__name', 'activity__scenario__name')
    raw_id_fields = ('activity', 'created_by', 'updated_by')
    list_select_related = ('activity',)
    inlines = [AnswerFeedbackInline]

    def short_text(self, obj):
        return obj.text[:60] + '…' if len(obj.text) > 60 else obj.text
    short_text.short_description = 'Text'

    def activity_link(self, obj):
        if obj.activity_id:
            url = reverse('admin:authoringtool_activity_change', args=[obj.activity_id])
            return format_html('<a href="{}">{}</a>', url, obj.activity.name)
        return '—'
    activity_link.short_description = 'Activity'
    activity_link.admin_order_field = 'activity__name'


# ─── AnswerFeedback ───────────────────────────────────────────────────────────

@admin.register(AnswerFeedback)
class AnswerFeedbackAdmin(admin.ModelAdmin):
    list_display = ('id', 'answer', 'short_text', 'created_on')
    search_fields = ('text', 'answer__text')
    raw_id_fields = ('answer', 'created_by', 'updated_by')

    def short_text(self, obj):
        if not obj.text:
            return ''
        return obj.text[:60] + '…' if len(obj.text) > 60 else obj.text
    short_text.short_description = 'Text'


# ─── NextQuestionLogic ────────────────────────────────────────────────────────

@admin.register(NextQuestionLogic)
class NextQuestionLogicAdmin(admin.ModelAdmin):
    list_display = ('id', 'activity', 'answer', 'next_activity')
    search_fields = ('activity__name', 'next_activity__name')
    raw_id_fields = ('activity', 'answer', 'next_activity')
    list_select_related = ('activity', 'answer', 'next_activity')


# ─── EvQuestionBranching ──────────────────────────────────────────────────────

@admin.register(EvQuestionBranching)
class EvQuestionBranchingAdmin(admin.ModelAdmin):
    list_display = ('activity', 'next_question_on_high', 'next_question_on_mid', 'next_question_on_low')
    search_fields = ('activity__name',)
    raw_id_fields = ('activity', 'next_question_on_high', 'next_question_on_mid', 'next_question_on_low')


# ─── QuestionBunch ────────────────────────────────────────────────────────────

@admin.register(QuestionBunch)
class QuestionBunchAdmin(admin.ModelAdmin):
    list_display = ('id', 'activity_primary', 'activity_ids')
    raw_id_fields = ('activity_primary',)


# ─── Simulation ───────────────────────────────────────────────────────────────

@admin.register(Simulation)
class SimulationAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'iframe_url', 'width', 'height', 'allow_fullscreen')
    search_fields = ('name', 'iframe_url')
    list_filter = ('allow_fullscreen',)


# ─── SchoolDepartment ─────────────────────────────────────────────────────────

@admin.register(SchoolDepartment)
class SchoolDepartmentAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)


# ─── ExperimentLL ─────────────────────────────────────────────────────────────

@admin.register(ExperimentLL)
class ExperimentLLAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'launch_url', 'consumer_key')
    search_fields = ('name', 'launch_url')


# ─── VRARExperiment ───────────────────────────────────────────────────────────

@admin.register(VRARExperiment)
class VRARExperimentAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'launch_url', 'qr_code_display')
    readonly_fields = ('qr_code_display',)
    search_fields = ('name',)

    def qr_code_display(self, obj):
        if obj.qr_code:
            return format_html('<img src="{}" width="150" height="150" />', obj.qr_code.url)
        return 'No QR Code'
    qr_code_display.short_description = 'QR Code'


# ─── RemoteLabSession ─────────────────────────────────────────────────────────

_export_remlab = make_csv_export(
    'remote_lab_sessions',
    ['ID', 'User', 'Scenario', 'Activity', 'Lab', 'Start', 'End', 'Iteration', 'Angle', 'Mass', 'Pre Duration', 'Exec Duration'],
    lambda o: [o.id, o.user, o.scenario, o.activity, o.lab_name, o.start, o.end, o.iteration, o.angle, o.mass, o.pre_duration, o.exec_duration],
)


@admin.register(RemoteLabSession)
class RemoteLabSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'scenario', 'activity', 'lab_name', 'start', 'end', 'iteration')
    list_filter = ('lab_name', 'scenario')
    search_fields = ('user__username', 'lab_name', 'scenario__name')
    raw_id_fields = ('activity', 'phase', 'scenario', 'user')
    list_select_related = ('user', 'scenario', 'activity')
    date_hierarchy = 'start'
    actions = [_export_remlab]


# ─── UserScenarioScore ────────────────────────────────────────────────────────

_export_scores = make_csv_export(
    'user_scenario_scores',
    ['ID', 'User', 'Scenario', 'Score', 'Last Activity', 'Time (s)'],
    lambda o: [o.id, o.user, o.scenario, o.user_score, o.last_activity, o.timeDoingScenario],
)


@admin.register(UserScenarioScore)
class UserScenarioScoreAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'scenario_link', 'user_score', 'last_activity', 'timeDoingScenario')
    list_filter = ('scenario',)
    search_fields = ('user__username', 'scenario__name')
    raw_id_fields = ('user', 'scenario', 'last_activity')
    list_select_related = ('user', 'scenario')
    actions = [_export_scores]

    def scenario_link(self, obj):
        url = reverse('admin:authoringtool_scenario_change', args=[obj.scenario_id])
        return format_html('<a href="{}">{}</a>', url, obj.scenario.name)
    scenario_link.short_description = 'Scenario'
    scenario_link.admin_order_field = 'scenario__name'


# ─── UserAnswer ───────────────────────────────────────────────────────────────

_export_answers = make_csv_export(
    'user_answers',
    ['ID', 'User', 'Activity', 'Scenario', 'Answer', 'Is Correct', 'Timing (s)', 'Created On'],
    lambda o: [
        o.id, o.user, o.activity,
        o.activity.scenario if o.activity_id else '',
        o.answer, o.answer.is_correct if o.answer_id else '',
        o.timing, o.created_on,
    ],
)


@admin.register(UserAnswer)
class UserAnswerAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'activity', 'answer', 'timing', 'created_on')
    list_filter = ('activity__scenario',)
    search_fields = ('user__username', 'activity__name')
    raw_id_fields = ('user', 'activity', 'answer')
    list_select_related = ('user', 'activity', 'answer')
    date_hierarchy = 'created_on'
    actions = [_export_answers]


# ─── PhetLabSessions ──────────────────────────────────────────────────────────

@admin.register(PhetLabSessions)
class PhetLabSessionsAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'user', 'activity', 'gravity', 'friction', 'timestamp')
    list_filter = ('name',)
    search_fields = ('name', 'user__username')
    raw_id_fields = ('user', 'activity')
    list_select_related = ('user', 'activity')
    date_hierarchy = 'timestamp'


# ─── ActivityFlag ─────────────────────────────────────────────────────────────

@admin.register(ActivityFlag)
class ActivityFlagAdmin(admin.ModelAdmin):
    list_display = ('id', 'activity', 'scenario', 'category', 'flag_type', 'is_at_risk', 'auto_flagged', 'flagged_on')
    list_filter = ('category', 'is_at_risk', 'auto_flagged', 'scenario')
    search_fields = ('flag_type', 'flag_reason', 'activity__name', 'scenario__name')
    raw_id_fields = ('activity', 'scenario', 'phase')
    list_select_related = ('activity', 'scenario')
    readonly_fields = ('flagged_on',)
    date_hierarchy = 'flagged_on'


# ─── ActivityProposal ─────────────────────────────────────────────────────────

@admin.register(ActivityProposal)
class ActivityProposalAdmin(admin.ModelAdmin):
    list_display = ('id', 'proposal_type', 'activity', 'scenario', 'status', 'reviewer', 'created_at', 'reviewed_at')
    list_filter = ('status', 'proposal_type')
    search_fields = ('activity__name', 'scenario__name', 'suggested_action')
    raw_id_fields = ('scenario', 'phase', 'activity', 'reviewer')
    list_select_related = ('activity', 'scenario', 'reviewer')
    readonly_fields = ('created_at',)
    filter_horizontal = ('flag', 'categories_in_risk')
    date_hierarchy = 'created_at'


# ─── MultilingualQuestion ─────────────────────────────────────────────────────

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
        ('Basic Information', {'fields': ('order', 'is_required')}),
        ('Question Text - English', {'fields': ('question_text_en',)}),
        ('Question Text - Portuguese', {'fields': ('question_text_pt',)}),
        ('Question Text - Greek', {'fields': ('question_text_gr',)}),
        ('Question Text - Spanish', {'fields': ('question_text_es',)}),
        ('Question Text - French', {'fields': ('question_text_fr',)}),
        ('Question Text - German', {'fields': ('question_text_de',)}),
    )

    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)


# ─── MultilingualAnswer ───────────────────────────────────────────────────────

@admin.register(MultilingualAnswer)
class MultilingualAnswerAdmin(admin.ModelAdmin):
    list_display = ('id', 'question', 'user', 'scenario', 'created_on', 'updated_on')
    list_filter = ('scenario', 'user')
    search_fields = ('answer_text', 'question__question_text_en', 'user__username', 'scenario__name')
    readonly_fields = ('created_on', 'updated_on', 'created_by', 'updated_by')
    ordering = ('-created_on',)

    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)


# ─── QValue ───────────────────────────────────────────────────────────────────

@admin.register(QValue)
class QValueAdmin(admin.ModelAdmin):
    list_display = ('flag_type', 'category', 'action', 'q_value', 'updated_at')
    list_filter = ('flag_type', 'category', 'action')
    search_fields = ('flag_type',)
    ordering = ('-updated_at',)


# ─── UserProposalReview ───────────────────────────────────────────────────────

@admin.register(UserProposalReview)
class UserProposalReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'proposal', 'user', 'status', 'was_edited', 'edit_count', 'reviewed_at')
    list_filter = ('status', 'reviewed_at')
    search_fields = ('user__username', 'proposal__id')
    ordering = ('-reviewed_at',)


# ─── ActivityProposalEditEvent ─────────────────────────────────────────────────

@admin.register(ActivityProposalEditEvent)
class ActivityProposalEditEventAdmin(admin.ModelAdmin):
    list_display = ('id', 'review', 'edit_number', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('review__user__username', 'review__proposal__id')
    raw_id_fields = ('review',)
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'


# ─── User (Custom) ────────────────────────────────────────────────────────────

class RoleFilter(admin.SimpleListFilter):
    title = 'Role'
    parameter_name = 'role'

    def lookups(self, request, model_admin):
        return [
            ('teacher', 'Teachers'),
            ('student', 'Students'),
            ('staff', 'Staff / Superusers'),
        ]

    def queryset(self, request, queryset):
        if self.value() == 'teacher':
            return queryset.filter(groups__name='teachers')
        if self.value() == 'student':
            return queryset.exclude(
                Q(groups__name='teachers') | Q(is_staff=True) | Q(is_superuser=True)
            )
        if self.value() == 'staff':
            return queryset.filter(Q(is_staff=True) | Q(is_superuser=True))
        return queryset


class UserGroupMembershipInline(admin.TabularInline):
    model = UserGroupMembership
    extra = 0
    fields = ('group_link', 'initial_password')
    readonly_fields = ('group_link', 'initial_password')
    can_delete = False
    verbose_name = 'Student Group'
    verbose_name_plural = 'Student Groups'

    def has_add_permission(self, request, obj=None):
        return False

    def group_link(self, obj):
        url = reverse('admin:usergroups_usergroup_change', args=[obj.group_id])
        return format_html('<a href="{}">{}</a>', url, obj.group.name)
    group_link.short_description = 'Group'


class CustomUserAdmin(UserAdmin):
    list_display = (
        'id', 'username', 'email', 'first_name', 'last_name',
        'role_badge', 'is_active', 'school_department', 'last_login_display', 'date_joined',
    )
    list_filter = (RoleFilter, 'is_active', 'is_staff', 'school_department')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    ordering = ('-date_joined',)
    inlines = [UserGroupMembershipInline]
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

    def role_badge(self, obj):
        if obj.is_superuser:
            color, label = '#c62828', 'Superuser'
        elif obj.is_staff:
            color, label = '#1565c0', 'Staff'
        elif obj.groups.filter(name='teachers').exists():
            color, label = '#2e7d32', 'Teacher'
        else:
            color, label = '#e65100', 'Student'
        return format_html(
            '<span style="background:{};color:#fff;padding:2px 8px;border-radius:4px;'
            'font-size:11px;font-weight:600;">{}</span>',
            color, label,
        )
    role_badge.short_description = 'Role'

    def last_login_display(self, obj):
        if obj.last_login:
            return format_html(
                '<span title="{}">{}</span>',
                obj.last_login.strftime('%Y-%m-%d %H:%M'),
                obj.last_login.strftime('%d %b %Y'),
            )
        return format_html('<span style="color:#aaa;">Never</span>')
    last_login_display.short_description = 'Last Login'
    last_login_display.admin_order_field = 'last_login'


admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)
