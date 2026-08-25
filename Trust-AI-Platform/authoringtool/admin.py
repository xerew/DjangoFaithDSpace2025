import csv
from django import forms
from django.conf import settings
from django.contrib import admin, messages
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.core import signing
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Count, Prefetch, Q
from django.http import HttpResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.utils.html import format_html, format_html_join
from django.utils import timezone
from django.urls import path, reverse
from .models import (
    Scenario, ScenarioFamily, ScenarioFamilyReviewProxy,
    ScenarioHealthProxy, ScenarioVersion,
    ScenarioRevisionDraft,
    EvidenceCompatibilityCluster, ScenarioVersionCompatibility, Phase,
    Activity, ActivityConcept, ActivityMatchingProxy, ActivityRevision,
    ActivityType,
    Answer, AnswerFeedback, NextQuestionLogic, EvQuestionBranching,
    QuestionBunch, Simulation, SchoolDepartment, ExperimentLL,
    RemoteLabSession, VRARExperiment, ScenarioImplementation,
    UserScenarioScore, UserAnswer,
    PhetLabSessions, MultilingualQuestion, MultilingualAnswer,
    ActivityFlag, ActivityProposal, ActivityProposalEditEvent,
    BanditPolicyConfiguration, ProposalGenerationRun,
    ProposalStructuralFailure, QValue, UserProposalReview, Language,
    ScenarioSimilarityProfile, ScenarioFamilyCandidate,
    ScenarioFamilyMatchDecision,
)
from .scenario_matching import (
    apply_candidate_decision,
    create_manual_family_candidate,
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

class ManualScenarioAssociationForm(forms.Form):
    RELATIONSHIP_CHOICES = (
        ('translation', 'Official translation'),
        ('adaptation', 'Adaptation / revised copy'),
        ('related_topic', 'Related topic only'),
        ('unrelated', 'Not related'),
    )

    target_scenario = forms.ModelChoiceField(
        queryset=Scenario.objects.none(),
        label='Scenario whose family should remain',
        help_text=(
            'For a translation or adaptation, this scenario’s family and '
            'canonical scenario will be kept.'
        ),
        widget=forms.Select(attrs={
            'class': 'scenario-association-select',
        }),
    )
    source_scenario = forms.ModelChoiceField(
        queryset=Scenario.objects.none(),
        label='Scenario to associate',
        help_text=(
            'For a translation or adaptation, this scenario and its existing '
            'family variants will move into the family above.'
        ),
        widget=forms.Select(attrs={
            'class': 'scenario-association-select',
        }),
    )
    relationship = forms.ChoiceField(
        choices=RELATIONSHIP_CHOICES,
        help_text=(
            'A shared topic alone should be marked “Related topic only,” not '
            'as a translation or adaptation.'
        ),
    )
    review_notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 4}),
        help_text='Reason for this administrator decision.',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scenario_queryset = (
            Scenario.objects
            .select_related('family', 'current_version')
            .order_by('name', 'id')
        )
        self.fields['target_scenario'].queryset = scenario_queryset
        self.fields['source_scenario'].queryset = scenario_queryset

    def clean(self):
        cleaned = super().clean()
        target = cleaned.get('target_scenario')
        source = cleaned.get('source_scenario')
        if not target or not source:
            return cleaned
        if target.pk == source.pk:
            raise ValidationError(
                'Choose two different scenarios. Revisions belong to one '
                'scenario and are created through its revision workflow.'
            )
        if (
            target.family_id
            and source.family_id
            and target.family_id == source.family_id
        ):
            raise ValidationError(
                'These scenarios already belong to the same family. Use the '
                'Scenario admin to correct a variant classification.'
            )
        return cleaned


@admin.register(Language)
class LanguageAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)


# ─── Scenario ─────────────────────────────────────────────────────────────────

@admin.register(ScenarioFamily)
class ScenarioFamilyAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'title',
        'canonical_scenario',
        'scenario_count',
        'compatibility_cluster_count',
        'implementation_count',
        'created_by',
        'updated_on',
    )
    search_fields = (
        'title',
        'description',
        'canonical_scenario__name',
        'scenarios__name',
    )
    filter_horizontal = ('subjects',)
    raw_id_fields = ('canonical_scenario', 'created_by')
    readonly_fields = ('created_on', 'updated_on')
    fields = (
        'title',
        'description',
        'canonical_scenario',
        'subjects',
        'created_by',
        'created_on',
        'updated_on',
    )

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related('canonical_scenario', 'created_by')
            .annotate(
                _scenario_count=Count('scenarios', distinct=True),
                _compatibility_cluster_count=Count(
                    'evidence_clusters',
                    distinct=True,
                ),
            )
        )

    @admin.display(description='Variants', ordering='_scenario_count')
    def scenario_count(self, obj):
        return obj._scenario_count

    @admin.display(
        description='Evidence pools',
        ordering='_compatibility_cluster_count',
    )
    def compatibility_cluster_count(self, obj):
        return obj._compatibility_cluster_count

    @admin.display(description='Implementations')
    def implementation_count(self, obj):
        return (
            ScenarioImplementation.objects
            .filter(scenario__family=obj)
            .exclude(user__groups__name='teachers')
            .count()
        )


@admin.register(ScenarioFamilyReviewProxy)
class ScenarioFamilyReviewAdmin(admin.ModelAdmin):
    """One admin-only overview of families, variants, and review work."""

    change_list_template = (
        'admin/authoringtool/scenariofamilyreviewproxy/change_list.html'
    )
    list_display = (
        'family_identity',
        'variant_revision_overview',
        'implementation_overview',
        'review_queue',
    )
    search_fields = (
        'title',
        'canonical_scenario__name',
        'scenarios__name',
        'scenarios__language',
    )
    list_per_page = 25
    association_token_salt = 'authoringtool.manual-scenario-association'

    def get_urls(self):
        return [
            path(
                'associate/',
                self.admin_site.admin_view(
                    self.associate_scenarios_view
                ),
                name=(
                    'authoringtool_scenariofamilyreviewproxy_'
                    'associate'
                ),
            ),
        ] + super().get_urls()

    def can_associate_scenarios(self, request):
        return (
            request.user.is_active
            and request.user.is_staff
            and request.user.has_perms([
                'authoringtool.change_scenario',
                'authoringtool.change_scenariofamily',
                'authoringtool.change_scenariofamilycandidate',
            ])
        )

    def changelist_view(self, request, extra_context=None):
        extra_context = {
            **(extra_context or {}),
            'can_associate_scenarios': self.can_associate_scenarios(
                request
            ),
        }
        return super().changelist_view(
            request,
            extra_context=extra_context,
        )

    def _association_preview(self, target, source, relationship):
        target_family = target.ensure_family()
        source_family = source.ensure_family()
        target_version = target.ensure_current_version()
        source_version = source.ensure_current_version()
        target_variants = list(
            target_family.scenarios
            .select_related('current_version')
            .order_by('variant_type', 'language', 'name')
        )
        source_variants = list(
            source_family.scenarios
            .select_related('current_version')
            .order_by('variant_type', 'language', 'name')
        )
        target_implementations = (
            ScenarioImplementation.objects
            .filter(scenario__family=target_family)
            .exclude(user__groups__name='teachers')
            .count()
        )
        source_implementations = (
            ScenarioImplementation.objects
            .filter(scenario__family=source_family)
            .exclude(user__groups__name='teachers')
            .count()
        )
        warnings = []
        same_language = (
            (target.language or '').strip().casefold()
            == (source.language or '').strip().casefold()
        )
        if relationship == 'translation' and same_language:
            warnings.append(
                'Both selected scenarios use the same language. Verify that '
                '“Official translation” is the intended classification.'
            )
        if (
            relationship == 'translation'
            and target_version.structure_fingerprint
            != source_version.structure_fingerprint
        ):
            warnings.append(
                'The current revisions have different graph structures. If '
                'the teaching flow changed materially, classify the scenario '
                'as an adaptation/revised copy instead.'
            )
        if (
            relationship == 'adaptation'
            and target_version.is_exactly_compatible_with(source_version)
        ):
            warnings.append(
                'The current revisions have identical structure and content '
                'fingerprints. Verify whether this is actually a translation '
                'or an unchanged duplicate.'
            )
        will_merge = relationship in {'translation', 'adaptation'}
        return {
            'target': target,
            'source': source,
            'target_family': target_family,
            'source_family': source_family,
            'target_version': target_version,
            'source_version': source_version,
            'target_variants': target_variants,
            'source_variants': source_variants,
            'target_implementations': target_implementations,
            'source_implementations': source_implementations,
            'relationship': relationship,
            'relationship_label': dict(
                ManualScenarioAssociationForm.RELATIONSHIP_CHOICES
            )[relationship],
            'will_merge': will_merge,
            'warnings': warnings,
        }

    def _association_confirmation_token(self, preview):
        return signing.dumps(
            {
                'target_id': preview['target'].id,
                'source_id': preview['source'].id,
                'target_version_id': preview['target_version'].id,
                'source_version_id': preview['source_version'].id,
                'relationship': preview['relationship'],
            },
            salt=self.association_token_salt,
            compress=True,
        )

    def _validate_association_confirmation(
        self,
        token,
        target,
        source,
        relationship,
    ):
        try:
            payload = signing.loads(
                token,
                salt=self.association_token_salt,
                max_age=1800,
            )
        except signing.BadSignature:
            raise ValidationError(
                'The preview expired or was changed. Preview the association '
                'again before confirming.'
            )
        target.refresh_from_db(fields=['current_version'])
        source.refresh_from_db(fields=['current_version'])
        expected = {
            'target_id': target.id,
            'source_id': source.id,
            'target_version_id': target.current_version_id,
            'source_version_id': source.current_version_id,
            'relationship': relationship,
        }
        if payload != expected:
            raise ValidationError(
                'The selected scenarios or their current revisions changed '
                'after the preview. Preview the association again.'
            )

    def associate_scenarios_view(self, request):
        if not self.can_associate_scenarios(request):
            raise PermissionDenied

        form = ManualScenarioAssociationForm(
            request.POST or None
        )
        preview = None
        confirmation_token = ''
        if request.method == 'POST' and form.is_valid():
            target = form.cleaned_data['target_scenario']
            source = form.cleaned_data['source_scenario']
            relationship = form.cleaned_data['relationship']
            preview = self._association_preview(
                target,
                source,
                relationship,
            )
            if request.POST.get('action') == 'confirm':
                try:
                    self._validate_association_confirmation(
                        request.POST.get('confirmation_token', ''),
                        target,
                        source,
                        relationship,
                    )
                    candidate = create_manual_family_candidate(
                        target,
                        source,
                    )
                    candidate, _ = apply_candidate_decision(
                        candidate,
                        relationship,
                        request.user,
                        notes=form.cleaned_data['review_notes'],
                        target_family=preview['target_family'],
                    )
                except ValidationError as exc:
                    form.add_error(None, '; '.join(exc.messages))
                    preview = None
                else:
                    self.message_user(
                        request,
                        (
                            'Scenario association recorded as '
                            f'{candidate.get_decision_display()}.'
                        ),
                        level=messages.SUCCESS,
                    )
                    return HttpResponseRedirect(reverse(
                        'admin:authoringtool_scenariofamilycandidate_change',
                        args=[candidate.id],
                    ))
            else:
                confirmation_token = (
                    self._association_confirmation_token(preview)
                )

        context = {
            **self.admin_site.each_context(request),
            'opts': self.model._meta,
            'title': 'Associate scenarios manually',
            'form': form,
            'preview': preview,
            'confirmation_token': confirmation_token,
            'scenario_versions_url': reverse(
                'admin:authoringtool_scenarioversion_changelist'
            ),
        }
        request.current_app = self.admin_site.name
        return TemplateResponse(
            request,
            (
                'admin/authoringtool/scenariofamilyreviewproxy/'
                'associate_scenarios.html'
            ),
            context,
        )

    def get_queryset(self, request):
        scenario_queryset = (
            Scenario.objects
            .select_related('current_version')
            .prefetch_related('versions')
            .annotate(
                _student_implementation_count=Count(
                    'implementations',
                    filter=~Q(
                        implementations__user__groups__name='teachers'
                    ),
                    distinct=True,
                )
            )
            .order_by('variant_type', 'language', 'name')
        )
        return (
            super()
            .get_queryset(request)
            .select_related('canonical_scenario')
            .prefetch_related(
                Prefetch(
                    'scenarios',
                    queryset=scenario_queryset,
                    to_attr='_family_review_scenarios',
                ),
                'subjects',
            )
        )

    def has_module_permission(self, request):
        return (
            request.user.is_staff
            and request.user.has_perm(
                'authoringtool.view_scenariofamilyreviewproxy'
            )
        )

    def has_view_permission(self, request, obj=None):
        return self.has_module_permission(request)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.display(description='Scenario family')
    def family_identity(self, obj):
        family_url = reverse(
            'admin:authoringtool_scenariofamily_change',
            args=[obj.id],
        )
        canonical = obj.canonical_scenario
        subjects = ', '.join(
            subject.name for subject in obj.subjects.all()
        ) or 'No subject assigned'
        return format_html(
            '<strong><a href="{}">{}</a></strong>'
            '<div class="family-review-meta">'
            'Canonical: {}<br>Subjects: {}</div>',
            family_url,
            obj.title,
            canonical.name if canonical else 'Not selected',
            subjects,
        )

    @admin.display(description='Variants and immutable revisions')
    def variant_revision_overview(self, obj):
        variants = getattr(obj, '_family_review_scenarios', [])
        if not variants:
            return 'No scenarios in this family.'
        rows = []
        for scenario in variants:
            scenario_url = reverse(
                'admin:authoringtool_scenario_change',
                args=[scenario.id],
            )
            versions = list(scenario.versions.all())
            revision_summary = ', '.join(
                f'v{version.version_number} '
                f'({version.get_revision_status_display()})'
                for version in sorted(
                    versions,
                    key=lambda item: item.version_number,
                    reverse=True,
                )
            )
            rows.append((
                scenario_url,
                scenario.name,
                scenario.language or 'Unspecified',
                scenario.get_variant_type_display(),
                (
                    f'v{scenario.current_version.version_number}'
                    if scenario.current_version
                    else 'No current revision'
                ),
                revision_summary or 'No revisions',
            ))
        return format_html(
            '<div style="display:grid;gap:10px;">{}</div>',
            format_html_join(
                '',
                (
                    '<div class="family-review-variant">'
                    '<strong><a href="{}">{}</a></strong>'
                    '<div class="family-review-meta">{} · {} · {}</div>'
                    '<div class="family-review-revisions">{}</div>'
                    '</div>'
                ),
                rows,
            ),
        )

    @admin.display(description='Student implementations')
    def implementation_overview(self, obj):
        variants = getattr(obj, '_family_review_scenarios', [])
        total = sum(
            scenario._student_implementation_count
            for scenario in variants
        )
        lines = [
            (
                scenario.name,
                scenario._student_implementation_count,
            )
            for scenario in variants
        ]
        return format_html(
            '<strong>{} total</strong><ul style="margin:6px 0 0 16px;">'
            '{}</ul>',
            total,
            format_html_join(
                '',
                '<li>{}: {}</li>',
                lines,
            ),
        )

    @admin.display(description='Admin review queue')
    def review_queue(self, obj):
        queue = (
            ScenarioFamilyCandidate.objects
            .filter(is_current=True, decision__in=['pending', 'deferred'])
            .filter(
                Q(scenario_a__family=obj)
                | Q(scenario_b__family=obj)
            )
        )
        pending_count = queue.count()
        llm_completed = queue.filter(llm_status='completed').count()
        llm_failed = queue.filter(llm_status='failed').count()
        candidate_rows = []
        for candidate in queue.select_related(
            'scenario_a',
            'scenario_b',
        ).order_by('-similarity_score')[:5]:
            candidate_rows.append((
                reverse(
                    'admin:authoringtool_scenariofamilycandidate_change',
                    args=[candidate.id],
                ),
                candidate.scenario_a.name,
                candidate.scenario_b.name,
                candidate.get_suggested_relationship_display(),
                (
                    candidate.get_llm_suggested_relationship_display()
                    if candidate.llm_status == 'completed'
                    else candidate.get_llm_status_display()
                ),
            ))
        inbox_url = (
            reverse(
                'admin:authoringtool_scenariofamilycandidate_changelist'
            )
            + '?is_current__exact=1&q='
            + str(obj.title)
        )
        return format_html(
            '<a class="button" href="{}">Open candidate inbox</a>'
            '<div style="margin-top:8px;">{} pending/review later<br>'
            '{} with LLM review{}</div>'
            '<div style="display:grid;gap:6px;margin-top:9px;">{}</div>',
            inbox_url,
            pending_count,
            llm_completed,
            (
                format_html(
                    '<br><span class="family-review-error">'
                    '{} LLM failed</span>',
                    llm_failed,
                )
                if llm_failed
                else ''
            ),
            format_html_join(
                '',
                (
                    '<a class="family-review-candidate" href="{}">{} ↔ {}<br>'
                    '<small>Matcher: {} · Ollama: {}</small></a>'
                ),
                candidate_rows,
            ),
        )


@admin.register(ScenarioVersion)
class ScenarioVersionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'scenario',
        'version_number',
        'revision_status',
        'is_current',
        'compatibility_status',
        'compatibility_cluster',
        'short_structure_fingerprint',
        'short_content_fingerprint',
        'created_by',
        'created_at',
        'published_at',
    )
    list_filter = (
        'revision_status',
        'is_current',
        'compatibility__status',
        'compatibility__decision_source',
        'created_at',
    )
    search_fields = (
        'scenario__name',
        'structure_fingerprint',
        'content_fingerprint',
        'change_summary',
    )
    readonly_fields = (
        'scenario',
        'version_number',
        'structure_fingerprint',
        'content_fingerprint',
        'snapshot',
        'previous_version',
        'created_by',
        'created_at',
        'change_summary',
        'is_current',
        'revision_status',
        'published_by',
        'published_at',
    )
    date_hierarchy = 'created_at'

    @admin.display(description='Structure')
    def short_structure_fingerprint(self, obj):
        return obj.structure_fingerprint[:12]

    @admin.display(description='Content')
    def short_content_fingerprint(self, obj):
        return obj.content_fingerprint[:12]

    @admin.display(description='Compatibility')
    def compatibility_status(self, obj):
        try:
            return obj.compatibility.get_status_display()
        except ScenarioVersionCompatibility.DoesNotExist:
            return 'Not classified'

    @admin.display(description='Evidence pool')
    def compatibility_cluster(self, obj):
        try:
            return obj.compatibility.cluster
        except ScenarioVersionCompatibility.DoesNotExist:
            return '—'

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(ScenarioRevisionDraft)
class ScenarioRevisionDraftAdmin(admin.ModelAdmin):
    list_display = (
        'scenario',
        'base_version',
        'created_by',
        'created_at',
        'updated_at',
        'short_structure_fingerprint',
        'short_content_fingerprint',
    )
    search_fields = (
        'scenario__name',
        'created_by__username',
        'structure_fingerprint',
        'content_fingerprint',
    )
    readonly_fields = (
        'scenario',
        'base_version',
        'snapshot',
        'structure_fingerprint',
        'content_fingerprint',
        'created_by',
        'created_at',
        'updated_at',
    )
    list_select_related = (
        'scenario',
        'base_version',
        'created_by',
    )

    @admin.display(description='Structure')
    def short_structure_fingerprint(self, obj):
        return obj.structure_fingerprint[:12]

    @admin.display(description='Content')
    def short_content_fingerprint(self, obj):
        return obj.content_fingerprint[:12]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(EvidenceCompatibilityCluster)
class EvidenceCompatibilityClusterAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'family',
        'is_automatic',
        'compatible_member_count',
        'review_member_count',
        'short_structure_fingerprint',
        'updated_at',
    )
    list_filter = ('is_automatic', 'family')
    search_fields = (
        'name',
        'family__title',
        'structure_fingerprint',
        'memberships__scenario_version__scenario__name',
    )
    raw_id_fields = ('family', 'created_by')
    readonly_fields = (
        'cluster_key',
        'structure_fingerprint',
        'is_automatic',
        'created_at',
        'updated_at',
    )

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related('family', 'created_by')
            .annotate(
                _compatible_member_count=Count(
                    'memberships',
                    filter=Q(memberships__status='compatible'),
                    distinct=True,
                ),
                _review_member_count=Count(
                    'memberships',
                    filter=Q(memberships__status='needs_review'),
                    distinct=True,
                ),
            )
        )

    @admin.display(description='Compatible')
    def compatible_member_count(self, obj):
        return obj._compatible_member_count

    @admin.display(description='Needs review')
    def review_member_count(self, obj):
        return obj._review_member_count

    @admin.display(description='Structure')
    def short_structure_fingerprint(self, obj):
        return (
            obj.structure_fingerprint[:12]
            if obj.structure_fingerprint
            else 'Manual'
        )

    def save_model(self, request, obj, form, change):
        if not change:
            obj.is_automatic = False
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(ScenarioVersionCompatibility)
class ScenarioVersionCompatibilityAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'scenario_name',
        'version_number',
        'language',
        'variant_type',
        'cluster',
        'status',
        'decision_source',
        'reviewed_by',
        'updated_at',
    )
    list_filter = (
        'status',
        'decision_source',
        'cluster__family',
        'scenario_version__scenario__variant_type',
        'scenario_version__scenario__language',
    )
    search_fields = (
        'scenario_version__scenario__name',
        'cluster__name',
        'cluster__family__title',
        'reason',
    )
    raw_id_fields = (
        'scenario_version',
        'cluster',
        'reviewed_by',
    )
    readonly_fields = (
        'decision_source',
        'reviewed_by',
        'reviewed_at',
        'created_at',
        'updated_at',
    )
    actions = (
        'approve_for_family_evidence',
        'mark_needs_review',
        'exclude_from_family_evidence',
    )

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related(
                'scenario_version__scenario',
                'cluster__family',
                'reviewed_by',
            )
        )

    def save_model(self, request, obj, form, change):
        obj.decision_source = 'admin'
        obj.reviewed_by = request.user
        obj.reviewed_at = timezone.now()
        obj.full_clean()
        super().save_model(request, obj, form, change)

    def _set_status(self, request, queryset, status):
        queryset.update(
            status=status,
            decision_source='admin',
            reviewed_by=request.user,
            reviewed_at=timezone.now(),
        )

    @admin.action(description='Approve selected for family evidence')
    def approve_for_family_evidence(self, request, queryset):
        self._set_status(request, queryset, 'compatible')

    @admin.action(description='Mark selected as needing review')
    def mark_needs_review(self, request, queryset):
        self._set_status(request, queryset, 'needs_review')

    @admin.action(description='Exclude selected from family evidence')
    def exclude_from_family_evidence(self, request, queryset):
        self._set_status(request, queryset, 'excluded')

    @admin.display(description='Scenario')
    def scenario_name(self, obj):
        return obj.scenario_version.scenario.name

    @admin.display(description='Version')
    def version_number(self, obj):
        return obj.scenario_version.version_number

    @admin.display(description='Language')
    def language(self, obj):
        return obj.scenario_version.scenario.language

    @admin.display(description='Variant')
    def variant_type(self, obj):
        return obj.scenario_version.scenario.get_variant_type_display()


@admin.register(ScenarioSimilarityProfile)
class ScenarioSimilarityProfileAdmin(admin.ModelAdmin):
    list_display = (
        'scenario',
        'scenario_version',
        'language',
        'structure_summary',
        'embedding_status',
        'stale_status',
        'generated_at',
    )
    list_filter = (
        'embedding_model',
        'feature_schema',
        'generated_at',
    )
    search_fields = (
        'scenario__name',
        'scenario__language',
        'embedding_model',
        'content_digest',
    )
    readonly_fields = (
        'scenario',
        'scenario_version',
        'content_digest',
        'feature_schema',
        'features',
        'embedding',
        'embedding_model',
        'embedding_error',
        'generated_at',
    )
    date_hierarchy = 'generated_at'

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related('scenario', 'scenario_version')
        )

    @admin.display(description='Language')
    def language(self, obj):
        return obj.scenario.language or 'Unspecified'

    @admin.display(description='Structure')
    def structure_summary(self, obj):
        features = obj.features or {}
        return (
            f"{features.get('phase_count', 0)} phases / "
            f"{features.get('activity_count', 0)} activities"
        )

    @admin.display(description='Embedding')
    def embedding_status(self, obj):
        if obj.embedding:
            return format_html(
                '<span style="color:#166534;font-weight:600;">Ready ({})</span>',
                obj.embedding_model,
            )
        if obj.embedding_error:
            return format_html(
                '<span style="color:#991b1b;" title="{}">Fallback used</span>',
                obj.embedding_error,
            )
        return 'Disabled'

    @admin.display(description='Current profile', boolean=True)
    def stale_status(self, obj):
        return not obj.is_stale

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(ScenarioFamilyCandidate)
class ScenarioFamilyCandidateAdmin(admin.ModelAdmin):
    change_form_template = (
        'admin/authoringtool/scenariofamilycandidate/change_form.html'
    )
    change_list_template = (
        'admin/authoringtool/scenariofamilycandidate/change_list.html'
    )
    list_display = (
        'scenario_pair',
        'score_display',
        'suggested_relationship',
        'llm_summary',
        'decision',
        'languages',
        'family_summary',
        'is_current',
        'updated_at',
    )
    list_filter = (
        'decision',
        'suggested_relationship',
        'is_current',
        'detection_method',
        'llm_status',
        'llm_suggested_relationship',
        'scenario_a__language',
        'scenario_b__language',
    )
    search_fields = (
        'scenario_a__name',
        'scenario_b__name',
        'scenario_a__family__title',
        'scenario_b__family__title',
        'review_notes',
        'llm_reasoning',
    )
    readonly_fields = (
        'scenario_comparison',
        'similarity_score',
        'family_score',
        'topic_score',
        'score_breakdown',
        'reasons_display',
        'family_impact_preview',
        'suggested_relationship',
        'llm_review_display',
        'decision',
        'scenario_a',
        'scenario_b',
        'scenario_a_version',
        'scenario_b_version',
        'reviewed_by',
        'reviewed_at',
        'is_current',
        'detection_method',
        'created_at',
        'updated_at',
        'decision_history',
    )
    raw_id_fields = ('target_family',)
    fields = (
        'scenario_comparison',
        ('similarity_score', 'family_score', 'topic_score'),
        'score_breakdown',
        'reasons_display',
        'llm_review_display',
        'family_impact_preview',
        ('suggested_relationship', 'decision'),
        'target_family',
        'review_notes',
        ('reviewed_by', 'reviewed_at'),
        'decision_history',
        (
            'scenario_a',
            'scenario_a_version',
            'scenario_b',
            'scenario_b_version',
        ),
        ('is_current', 'detection_method'),
        ('created_at', 'updated_at'),
    )
    actions = (
        'classify_as_translation',
        'classify_as_adaptation',
        'classify_as_related_topic',
        'classify_as_unrelated',
        'defer_review',
        'queue_selected_for_llm_review',
    )
    date_hierarchy = 'updated_at'

    def get_urls(self):
        return [
            path(
                'scan/',
                self.admin_site.admin_view(self.scan_candidates_view),
                name='authoringtool_scenariofamilycandidate_scan',
            ),
            path(
                'llm-review-pending/',
                self.admin_site.admin_view(
                    self.queue_pending_llm_reviews_view
                ),
                name=(
                    'authoringtool_scenariofamilycandidate_'
                    'llm_review_pending'
                ),
            ),
        ] + super().get_urls()

    def queue_pending_llm_reviews_view(self, request):
        if not self.has_change_permission(request):
            raise PermissionDenied
        if request.method != 'POST':
            return HttpResponseRedirect(
                reverse(
                    'admin:authoringtool_scenariofamilycandidate_changelist'
                )
            )
        force = request.POST.get('force_llm') == '1'
        queue = ScenarioFamilyCandidate.objects.filter(
            is_current=True,
            decision__in=['pending', 'deferred'],
        ).exclude(llm_status='pending')
        if not force:
            queue = queue.exclude(llm_status='completed')
        limit = max(
            1,
            int(getattr(
                settings,
                'SCENARIO_FAMILY_REVIEW_LLM_BATCH_LIMIT',
                25,
            )),
        )
        candidate_ids = list(
            queue.order_by('-similarity_score')
            .values_list('id', flat=True)[:limit]
        )
        self._queue_llm_candidate_ids(request, candidate_ids)
        return HttpResponseRedirect(
            reverse('admin:authoringtool_scenariofamilycandidate_changelist')
        )

    def scan_candidates_view(self, request):
        if not self.has_change_permission(request):
            raise PermissionDenied
        if request.method != 'POST':
            return HttpResponseRedirect(
                reverse(
                    'admin:authoringtool_scenariofamilycandidate_changelist'
                )
            )
        from .tasks import scan_scenario_family_candidates_task

        task = scan_scenario_family_candidates_task.delay(
            force_profiles=request.POST.get('force_profiles') == '1',
        )
        self.message_user(
            request,
            (
                'Scenario discovery scan started. Refresh this page after the '
                f'Celery task completes. Task ID: {task.id}'
            ),
            level=messages.INFO,
        )
        return HttpResponseRedirect(
            reverse('admin:authoringtool_scenariofamilycandidate_changelist')
        )

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related(
                'scenario_a__family',
                'scenario_b__family',
                'scenario_a_version',
                'scenario_b_version',
                'target_family',
                'reviewed_by',
            )
            .prefetch_related('decision_events__decided_by')
        )

    def _queue_llm_candidate_ids(self, request, candidate_ids):
        candidate_ids = list(dict.fromkeys(candidate_ids))
        if not candidate_ids:
            self.message_user(
                request,
                'There are no eligible candidates to send to Ollama.',
                level=messages.WARNING,
            )
            return None
        ScenarioFamilyCandidate.objects.filter(
            id__in=candidate_ids,
        ).update(
            llm_status='pending',
            llm_error='',
        )
        from .tasks import (
            review_scenario_family_candidates_with_llm_task,
        )

        task = review_scenario_family_candidates_with_llm_task.delay(
            candidate_ids
        )
        self.message_user(
            request,
            (
                f'{len(candidate_ids)} candidate(s) queued for a non-binding '
                f'Ollama review. Task ID: {task.id}'
            ),
            level=messages.INFO,
        )
        return task

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.display(description='Candidate scenarios')
    def scenario_pair(self, obj):
        left_url = reverse(
            'admin:authoringtool_scenario_change',
            args=[obj.scenario_a_id],
        )
        right_url = reverse(
            'admin:authoringtool_scenario_change',
            args=[obj.scenario_b_id],
        )
        return format_html(
            '<a href="{}">{}</a><br><span class="candidate-pair-arrow">↔</span> '
            '<a href="{}">{}</a>',
            left_url,
            obj.scenario_a.name,
            right_url,
            obj.scenario_b.name,
        )

    @admin.display(description='Score', ordering='similarity_score')
    def score_display(self, obj):
        score = float(obj.similarity_score)
        score_class = 'candidate-score-high' if score >= 0.75 else (
            'candidate-score-medium'
            if score >= 0.55
            else 'candidate-score-low'
        )
        return format_html(
            '<strong class="{}">{}</strong>',
            score_class,
            f'{score:.0%}',
        )

    @admin.display(description='Ollama second opinion')
    def llm_summary(self, obj):
        if obj.llm_status == 'completed':
            confidence = (
                f'{float(obj.llm_confidence):.0%}'
                if obj.llm_confidence is not None
                else 'No confidence'
            )
            agrees = (
                obj.llm_suggested_relationship
                == obj.suggested_relationship
            )
            return format_html(
                '<strong class="llm-status {}">{}</strong><br>'
                '<span class="llm-status-detail">{} · {}</span>',
                'llm-status-agrees' if agrees else 'llm-status-differs',
                obj.get_llm_suggested_relationship_display(),
                confidence,
                'agrees' if agrees else 'differs',
            )
        if obj.llm_status == 'failed':
            return format_html(
                '<span class="llm-status-failed" title="{}">Failed</span>',
                obj.llm_error,
            )
        return obj.get_llm_status_display()

    @admin.display(description='Non-binding Ollama review')
    def llm_review_display(self, obj):
        if obj.llm_status == 'not_requested':
            return format_html(
                '<div class="help">No LLM second opinion has been requested. '
                'The deterministic matcher remains available above.</div>'
            )
        if obj.llm_status == 'pending':
            return format_html(
                '<div class="llm-review-panel llm-review-pending">'
                'Ollama review is queued or running. Refresh this page later.'
                '</div>'
            )
        if obj.llm_status == 'failed':
            return format_html(
                '<div class="llm-review-panel llm-review-failed">'
                '<b>Review failed:</b> {}<br>'
                '<span class="help">You can request it again after checking '
                'the Ollama tunnel and Celery worker.</span></div>',
                obj.llm_error or 'Unknown Ollama error',
            )

        evidence = (obj.llm_details or {}).get('evidence') or []
        warnings = (obj.llm_details or {}).get('warnings') or []
        agrees = (
            obj.llm_suggested_relationship
            == obj.suggested_relationship
        )
        confidence = (
            f'{float(obj.llm_confidence):.0%}'
            if obj.llm_confidence is not None
            else 'Not supplied'
        )
        reviewed_at = (
            timezone.localtime(obj.llm_reviewed_at).strftime(
                '%Y-%m-%d %H:%M'
            )
            if obj.llm_reviewed_at
            else 'Unknown'
        )
        return format_html(
            '<div class="llm-review-panel">'
            '<h3 style="margin-top:0;">{} ({})</h3>'
            '<p><b>{}</b> the deterministic suggestion of <b>{}</b>.</p>'
            '<p>{}</p>'
            '<h4>Evidence cited by Ollama</h4><ul>{}</ul>'
            '<h4>Warnings / uncertainty</h4><ul>{}</ul>'
            '<p class="help">Model: {} · Reviewed: {}. This output never '
            'changes family membership; an administrator must use one of the '
            'classification buttons below.</p></div>',
            obj.get_llm_suggested_relationship_display(),
            confidence,
            'Agrees with' if agrees else 'Differs from',
            obj.get_suggested_relationship_display(),
            obj.llm_reasoning,
            format_html_join(
                '',
                '<li>{}</li>',
                ((item,) for item in evidence),
            ) if evidence else format_html('<li>None supplied.</li>'),
            format_html_join(
                '',
                '<li>{}</li>',
                ((item,) for item in warnings),
            ) if warnings else format_html('<li>None supplied.</li>'),
            obj.llm_model or 'Unspecified',
            reviewed_at,
        )

    @admin.display(description='Languages')
    def languages(self, obj):
        return (
            f'{obj.scenario_a.language or "Unspecified"} ↔ '
            f'{obj.scenario_b.language or "Unspecified"}'
        )

    @admin.display(description='Families')
    def family_summary(self, obj):
        if obj.scenario_a.family_id == obj.scenario_b.family_id:
            return obj.scenario_a.family.title
        return (
            f'{obj.scenario_a.family.title} ↔ '
            f'{obj.scenario_b.family.title}'
        )

    def _version_structure_summary(self, version):
        phases = (
            ((version.snapshot or {}).get('structure') or {}).get('phases')
            or []
        )
        activity_count = sum(
            len(phase.get('activities') or []) for phase in phases
        )
        answer_count = sum(
            len(activity.get('answers') or [])
            for phase in phases
            for activity in (phase.get('activities') or [])
        )
        activity_types = [
            activity.get('activity_type') or 'Unspecified'
            for phase in phases
            for activity in (phase.get('activities') or [])
        ]
        type_summary = ', '.join(activity_types[:8])
        if len(activity_types) > 8:
            type_summary += f' … (+{len(activity_types) - 8})'
        return {
            'phases': len(phases),
            'activities': activity_count,
            'answers': answer_count,
            'types': type_summary or 'No activities',
        }

    @admin.display(description='Side-by-side scenario comparison')
    def scenario_comparison(self, obj):
        left = self._version_structure_summary(obj.scenario_a_version)
        right = self._version_structure_summary(obj.scenario_b_version)
        left_url = reverse(
            'admin:authoringtool_scenario_change',
            args=[obj.scenario_a_id],
        )
        right_url = reverse(
            'admin:authoringtool_scenario_change',
            args=[obj.scenario_b_id],
        )
        return format_html(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
            '<div style="border:1px solid #dbe3ef;border-radius:8px;padding:14px;">'
            '<h3 style="margin-top:0;"><a href="{}">{}</a></h3>'
            '<p><b>Language:</b> {}<br><b>Family:</b> {}<br>'
            '<b>Variant:</b> {}<br><b>Version:</b> {}</p>'
            '<p><b>Structure:</b> {} phases, {} activities, {} answers</p>'
            '<p><b>Activity sequence:</b> {}</p>'
            '</div>'
            '<div style="border:1px solid #dbe3ef;border-radius:8px;padding:14px;">'
            '<h3 style="margin-top:0;"><a href="{}">{}</a></h3>'
            '<p><b>Language:</b> {}<br><b>Family:</b> {}<br>'
            '<b>Variant:</b> {}<br><b>Version:</b> {}</p>'
            '<p><b>Structure:</b> {} phases, {} activities, {} answers</p>'
            '<p><b>Activity sequence:</b> {}</p>'
            '</div></div>',
            left_url,
            obj.scenario_a.name,
            obj.scenario_a.language or 'Unspecified',
            obj.scenario_a.family.title,
            obj.scenario_a.get_variant_type_display(),
            obj.scenario_a_version.version_number,
            left['phases'],
            left['activities'],
            left['answers'],
            left['types'],
            right_url,
            obj.scenario_b.name,
            obj.scenario_b.language or 'Unspecified',
            obj.scenario_b.family.title,
            obj.scenario_b.get_variant_type_display(),
            obj.scenario_b_version.version_number,
            right['phases'],
            right['activities'],
            right['answers'],
            right['types'],
        )

    @admin.display(description='Explainable score components')
    def score_breakdown(self, obj):
        labels = {
            'structure': 'Learning-flow structure',
            'semantic': 'Semantic content',
            'embedding': 'Multilingual embedding',
            'lexical': 'Lexical overlap',
            'metadata': 'Subject metadata',
            'lineage': 'Activity lineage',
            'origin': 'Copy/origin history',
        }
        rows = [
            (
                labels.get(key, key.replace('_', ' ').title()),
                f'{float(value):.0%}',
            )
            for key, value in (obj.component_scores or {}).items()
        ]
        return format_html(
            '<table style="min-width:520px;"><thead><tr>'
            '<th style="text-align:left;">Signal</th><th>Score</th>'
            '<th style="width:260px;">Strength</th></tr></thead><tbody>{}</tbody>'
            '</table>',
            format_html_join(
                '',
                (
                    '<tr><td>{}</td><td style="text-align:center;">{}</td>'
                    '<td><div style="height:8px;background:#e5e7eb;border-radius:4px;">'
                    '<div style="height:8px;width:{};background:#2563eb;'
                    'border-radius:4px;"></div></div></td></tr>'
                ),
                ((label, percent, percent) for label, percent in rows),
            ),
        )

    @admin.display(description='Why this pair was suggested')
    def reasons_display(self, obj):
        return format_html(
            '<ul style="margin:0;padding-left:20px;">{}</ul>',
            format_html_join(
                '',
                '<li>{}</li>',
                ((reason,) for reason in (obj.reasons or [])),
            ),
        )

    @admin.display(description='Family-linking impact preview')
    def family_impact_preview(self, obj):
        left_family = obj.scenario_a.family
        right_family = obj.scenario_b.family
        if left_family_id := getattr(left_family, 'id', None):
            left_count = left_family.scenarios.count()
        else:
            left_count = 0
        if right_family_id := getattr(right_family, 'id', None):
            right_count = right_family.scenarios.count()
        else:
            right_count = 0
        if left_family_id == right_family_id:
            return format_html(
                '<div class="help">These scenarios already belong to the same '
                'family. A decision only updates the non-canonical variant '
                'classification and rebuilds its conservative evidence status.</div>'
            )
        target = obj.target_family or left_family
        source_count = (
            right_count if target.id == left_family_id else left_count
        )
        return format_html(
            '<div style="padding:12px;border-left:4px solid #d97706;'
            'background:#fffbeb;">'
            '<b>Logical merge only:</b> {} scenario variant{} will move into '
            '<b>{}</b>. Student implementations remain attached to their '
            'original scenarios. Compatibility is recalculated separately; '
            'adaptations require review and are not automatically pooled.'
            '</div>',
            source_count,
            '' if source_count == 1 else 's',
            target.title,
        )

    @admin.display(description='Decision history')
    def decision_history(self, obj):
        events = list(obj.decision_events.all())
        if not events:
            return 'No decisions recorded.'
        return format_html(
            '<ol style="margin:0;padding-left:20px;">{}</ol>',
            format_html_join(
                '',
                '<li><b>{}</b> — {} by {}{}</li>',
                (
                    (
                        event.get_decision_display(),
                        timezone.localtime(event.decided_at).strftime(
                            '%Y-%m-%d %H:%M'
                        ),
                        event.decided_by or 'Deleted administrator',
                        f': {event.notes}' if event.notes else '',
                    )
                    for event in events
                ),
            ),
        )

    def _apply_decision(self, request, candidate, decision):
        try:
            apply_candidate_decision(
                candidate,
                decision,
                request.user,
                notes=candidate.review_notes,
                target_family=candidate.target_family,
            )
        except ValidationError as exc:
            self.message_user(
                request,
                '; '.join(exc.messages),
                level=messages.ERROR,
            )
            return False
        return True

    def _apply_bulk_decision(self, request, queryset, decision):
        success_count = 0
        for candidate in queryset.order_by('-similarity_score'):
            success_count += int(
                self._apply_decision(request, candidate, decision)
            )
        if success_count:
            self.message_user(
                request,
                f'{success_count} candidate decision(s) recorded.',
                level=messages.SUCCESS,
            )

    @admin.action(description='Classify as same-family translation')
    def classify_as_translation(self, request, queryset):
        self._apply_bulk_decision(request, queryset, 'translation')

    @admin.action(description='Classify as same-family adaptation')
    def classify_as_adaptation(self, request, queryset):
        self._apply_bulk_decision(request, queryset, 'adaptation')

    @admin.action(description='Classify as related topic only')
    def classify_as_related_topic(self, request, queryset):
        self._apply_bulk_decision(request, queryset, 'related_topic')

    @admin.action(description='Classify as not related')
    def classify_as_unrelated(self, request, queryset):
        self._apply_bulk_decision(request, queryset, 'unrelated')

    @admin.action(description='Review selected later')
    def defer_review(self, request, queryset):
        self._apply_bulk_decision(request, queryset, 'deferred')

    @admin.action(description='Ask Ollama to review selected candidates')
    def queue_selected_for_llm_review(self, request, queryset):
        candidate_ids = list(
            queryset.filter(is_current=True)
            .exclude(llm_status='pending')
            .values_list('id', flat=True)
        )
        self._queue_llm_candidate_ids(request, candidate_ids)

    def response_change(self, request, obj):
        if '_request_llm_review' in request.POST:
            self._queue_llm_candidate_ids(request, [obj.id])
            return HttpResponseRedirect(request.path)
        button_decisions = {
            '_classify_translation': 'translation',
            '_classify_adaptation': 'adaptation',
            '_classify_related_topic': 'related_topic',
            '_classify_unrelated': 'unrelated',
            '_defer_review': 'deferred',
        }
        for button_name, decision in button_decisions.items():
            if button_name not in request.POST:
                continue
            if self._apply_decision(request, obj, decision):
                self.message_user(
                    request,
                    (
                        f'Decision recorded: '
                        f'{dict(ScenarioFamilyCandidate.DECISION_CHOICES)[decision]}.'
                    ),
                    level=messages.SUCCESS,
                )
            return HttpResponseRedirect(request.path)
        return super().response_change(request, obj)


@admin.register(ScenarioFamilyMatchDecision)
class ScenarioFamilyMatchDecisionAdmin(admin.ModelAdmin):
    list_display = (
        'candidate',
        'decision',
        'decided_by',
        'decided_at',
    )
    list_filter = ('decision', 'decided_at')
    search_fields = (
        'candidate__scenario_a__name',
        'candidate__scenario_b__name',
        'notes',
        'decided_by__username',
    )
    readonly_fields = (
        'candidate',
        'decision',
        'notes',
        'decided_by',
        'decided_at',
        'details',
    )
    date_hierarchy = 'decided_at'

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related(
                'candidate__scenario_a',
                'candidate__scenario_b',
                'decided_by',
            )
        )

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


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
        'family', 'variant_type', 'current_version', 'start_activity',
        'implementation_count_col', 'is_personal', 'created_on',
    )
    list_filter = (
        'visibility_status',
        'language',
        'variant_type',
        'is_personal',
    )
    search_fields = (
        'name',
        'description',
        'family__title',
        'created_by__username',
    )
    readonly_fields = (
        'created_on',
        'updated_on',
        'current_version',
        'implementation_count_display',
    )
    filter_horizontal = ('organizations', 'subjects')
    raw_id_fields = (
        'created_by',
        'updated_by',
        'origin_scenario',
        'family',
        'start_activity',
    )
    date_hierarchy = 'created_on'
    inlines = [PhaseInline]
    actions = ('scan_selected_for_family_matches',)

    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'description', 'learning_goals', 'subject_domains',
                       'subjects',
                       'language', 'age_of_students', 'suggested_learning_time',
                       'start_activity'),
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
        ('Family & Origin', {
            'fields': (
                'family',
                'variant_type',
                'current_version',
                'is_personal',
                'origin_scenario',
            ),
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
        obj.ensure_current_version()
        local_count = obj.eligible_implementation_count()
        count = obj.compatible_implementation_count()
        threshold = obj.ai_metrics_min_implementations
        if count >= threshold:
            return format_html(
                '<strong style="color:#2e7d32">{} compatible implementations</strong>'
                '&nbsp;<span style="color:#607d8b;font-size:12px;">'
                '({} local)</span>'
                '&nbsp;<span style="color:#2e7d32;font-size:12px;">&#10003; Button is visible to teachers</span>',
                count, local_count,
            )
        needed = threshold - count
        return format_html(
            '<strong style="color:#e65100">{} compatible implementations</strong>'
            '&nbsp;<span style="color:#607d8b;font-size:12px;">'
            '({} local)</span>'
            '&nbsp;<span style="color:#e65100;font-size:12px;">'
            '&#10007; Need {} more to unlock the button (threshold: {})</span>',
            count, local_count, needed, threshold,
        )
    implementation_count_display.short_description = 'Compatible Evidence'

    def implementation_count_col(self, obj):
        obj.ensure_current_version()
        count = obj.compatible_implementation_count()
        color = '#2e7d32' if count >= obj.ai_metrics_min_implementations else '#e65100'
        return format_html('<span style="color:{};font-weight:600;">{}</span>', color, count)
    implementation_count_col.short_description = 'Implementations'

    @admin.action(description='Scan selected for possible family matches')
    def scan_selected_for_family_matches(self, request, queryset):
        from .tasks import scan_scenario_family_candidates_task

        scenario_ids = list(queryset.values_list('id', flat=True))
        task = scan_scenario_family_candidates_task.delay(
            scenario_ids=scenario_ids,
        )
        self.message_user(
            request,
            (
                f'Scenario discovery started for {len(scenario_ids)} selected '
                f'scenario(s). Task ID: {task.id}'
            ),
            level=messages.INFO,
        )


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

class ActivityConceptMemberInline(admin.TabularInline):
    model = Activity
    fk_name = 'concept'
    fields = (
        'name',
        'scenario',
        'phase',
        'activity_type',
        'lineage_key',
    )
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ActivityConcept)
class ActivityConceptAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'title',
        'family',
        'activity_count',
        'revision_count',
        'updated_at',
    )
    list_filter = ('family',)
    search_fields = (
        'title',
        'description',
        'family__title',
        'activities__name',
    )
    raw_id_fields = ('family', 'created_by')
    readonly_fields = ('created_at', 'updated_at')
    inlines = (ActivityConceptMemberInline,)

    def get_queryset(self, request):
        return (
            super().get_queryset(request)
            .select_related('family')
            .annotate(
                _activity_count=Count('activities', distinct=True),
                _revision_count=Count('revisions', distinct=True),
            )
        )

    @admin.display(description='Activities', ordering='_activity_count')
    def activity_count(self, obj):
        return obj._activity_count

    @admin.display(description='Revisions', ordering='_revision_count')
    def revision_count(self, obj):
        return obj._revision_count


@admin.register(ActivityRevision)
class ActivityRevisionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'concept',
        'scenario_name',
        'version_number',
        'activity',
        'short_structure_fingerprint',
        'short_content_fingerprint',
        'created_at',
    )
    list_filter = (
        'scenario_version__scenario__family',
        'scenario_version__scenario__language',
        'scenario_version__version_number',
    )
    search_fields = (
        'concept__title',
        'scenario_version__scenario__name',
        'lineage_key',
        'structure_fingerprint',
        'content_fingerprint',
    )
    readonly_fields = (
        'activity',
        'concept',
        'scenario_version',
        'lineage_key',
        'revision_number',
        'structure_fingerprint',
        'content_fingerprint',
        'snapshot',
        'created_at',
    )
    list_select_related = (
        'concept',
        'scenario_version__scenario',
        'activity',
    )

    @admin.display(description='Scenario')
    def scenario_name(self, obj):
        return obj.scenario_version.scenario.name

    @admin.display(description='Version')
    def version_number(self, obj):
        return obj.scenario_version.version_number

    @admin.display(description='Structure')
    def short_structure_fingerprint(self, obj):
        return obj.structure_fingerprint[:12]

    @admin.display(description='Content')
    def short_content_fingerprint(self, obj):
        return obj.content_fingerprint[:12]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(ActivityMatchingProxy)
class ActivityMatchingAdmin(admin.ModelAdmin):
    """Dedicated manual mapping surface for cross-variant activities."""

    list_display = (
        'id',
        'name',
        'scenario_name',
        'family_name',
        'language',
        'activity_type',
        'concept',
        'lineage_key',
    )
    list_editable = ('concept',)
    list_filter = (
        'scenario__family',
        'scenario__language',
        'scenario__variant_type',
        'activity_type',
        'concept',
    )
    search_fields = (
        'name',
        'scenario__name',
        'scenario__family__title',
        'concept__title',
        'lineage_key',
    )
    autocomplete_fields = ('concept',)
    list_select_related = (
        'scenario__family',
        'activity_type',
        'concept',
    )
    ordering = (
        'scenario__family__title',
        'scenario__name',
        'phase__created_on',
        'created_on',
        'id',
    )

    @admin.display(description='Scenario', ordering='scenario__name')
    def scenario_name(self, obj):
        return obj.scenario.name

    @admin.display(description='Family', ordering='scenario__family__title')
    def family_name(self, obj):
        return obj.scenario.family

    @admin.display(description='Language', ordering='scenario__language')
    def language(self, obj):
        return obj.scenario.language or 'Unspecified'

    def save_model(self, request, obj, form, change):
        obj.full_clean()
        super().save_model(request, obj, form, change)

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


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
    list_display = (
        'id',
        'name',
        'scenario_name',
        'phase_name',
        'activity_type',
        'concept',
        'is_evaluatable',
        'created_on',
    )
    list_filter = (
        'activity_type',
        'concept',
        'is_evaluatable',
        'is_primary_ev',
    )
    search_fields = (
        'name',
        'scenario__name',
        'phase__name',
        'concept__title',
    )
    readonly_fields = (
        'created_on',
        'updated_on',
        'activity_type',
        'lineage_key',
    )
    raw_id_fields = ('scenario', 'phase', 'created_by', 'updated_by', 'simulation', 'experiment_ll', 'vr_ar_experiment')
    autocomplete_fields = ('concept',)
    list_select_related = ('scenario', 'phase', 'activity_type', 'concept')
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

_export_implementations = make_csv_export(
    'scenario_implementations',
    [
        'ID',
        'User',
        'Scenario',
        'Family',
        'Language',
        'Variant',
        'Scenario Version',
        'Version Confidence',
        'Status',
        'Data Quality',
        'Started At',
        'Completed At',
        'Last Activity',
    ],
    lambda o: [
        o.id,
        o.user,
        o.scenario,
        o.scenario.family,
        o.scenario.language,
        o.scenario.variant_type,
        o.scenario_version,
        o.version_confidence,
        o.status,
        o.data_quality_status,
        o.started_at,
        o.completed_at,
        o.last_activity,
    ],
)


@admin.register(ScenarioImplementation)
class ScenarioImplementationAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'scenario',
        'scenario_version',
        'status',
        'version_confidence',
        'data_quality_status',
        'started_at',
        'completed_at',
        'last_activity',
    )
    list_filter = (
        'status',
        'version_confidence',
        'data_quality_status',
        'scenario__family',
        'scenario__language',
        'scenario',
    )
    search_fields = (
        'user__username',
        'user__email',
        'scenario__name',
        'scenario__family__title',
    )
    raw_id_fields = (
        'user',
        'scenario',
        'scenario_version',
        'last_activity',
    )
    readonly_fields = ('started_at', 'completed_at')
    list_select_related = (
        'user',
        'scenario__family',
        'scenario_version',
        'last_activity',
    )
    date_hierarchy = 'started_at'
    actions = (
        _export_implementations,
        'mark_clean',
        'mark_suspect',
        'exclude_from_evidence',
    )

    def _set_quality(self, queryset, status):
        implementation_ids = list(queryset.values_list('id', flat=True))
        queryset.update(data_quality_status=status)
        UserScenarioScore.objects.filter(
            implementation_id__in=implementation_ids
        ).update(data_quality_status=status)

    @admin.action(description='Mark selected implementations as clean')
    def mark_clean(self, request, queryset):
        self._set_quality(queryset, 'clean')

    @admin.action(description='Mark selected implementations as suspect')
    def mark_suspect(self, request, queryset):
        self._set_quality(queryset, 'suspect')

    @admin.action(description='Exclude selected implementations from evidence')
    def exclude_from_evidence(self, request, queryset):
        self._set_quality(queryset, 'excluded')


_export_scores = make_csv_export(
    'user_scenario_scores',
    [
        'ID',
        'Implementation ID',
        'User',
        'Scenario',
        'Family',
        'Language',
        'Variant',
        'Scenario Version',
        'Version Confidence',
        'Data Quality',
        'Score',
        'Last Activity',
        'Time (s)',
    ],
    lambda o: [
        o.id,
        o.implementation_id,
        o.user,
        o.scenario,
        o.scenario.family,
        o.scenario.language,
        o.scenario.variant_type,
        o.scenario_version,
        o.version_confidence,
        o.data_quality_status,
        o.user_score,
        o.last_activity,
        o.timeDoingScenario,
    ],
)


@admin.register(UserScenarioScore)
class UserScenarioScoreAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'implementation',
        'user',
        'scenario_link',
        'scenario_version',
        'version_confidence',
        'data_quality_status',
        'user_score',
        'last_activity',
        'timeDoingScenario',
    )
    list_filter = (
        'scenario',
        'version_confidence',
        'data_quality_status',
    )
    search_fields = ('user__username', 'scenario__name')
    raw_id_fields = (
        'implementation',
        'user',
        'scenario',
        'scenario_version',
        'last_activity',
    )
    list_select_related = (
        'implementation',
        'user',
        'scenario',
        'scenario_version',
    )
    actions = [
        _export_scores,
        'mark_clean',
        'mark_suspect',
        'exclude_from_evidence',
    ]

    @admin.action(description='Mark selected implementations as clean')
    def mark_clean(self, request, queryset):
        implementation_ids = list(
            queryset.exclude(implementation__isnull=True)
            .values_list('implementation_id', flat=True)
        )
        queryset.update(data_quality_status='clean')
        ScenarioImplementation.objects.filter(
            id__in=implementation_ids
        ).update(data_quality_status='clean')

    @admin.action(description='Mark selected implementations as suspect')
    def mark_suspect(self, request, queryset):
        implementation_ids = list(
            queryset.exclude(implementation__isnull=True)
            .values_list('implementation_id', flat=True)
        )
        queryset.update(data_quality_status='suspect')
        ScenarioImplementation.objects.filter(
            id__in=implementation_ids
        ).update(data_quality_status='suspect')

    @admin.action(description='Exclude selected implementations from evidence')
    def exclude_from_evidence(self, request, queryset):
        implementation_ids = list(
            queryset.exclude(implementation__isnull=True)
            .values_list('implementation_id', flat=True)
        )
        queryset.update(data_quality_status='excluded')
        ScenarioImplementation.objects.filter(
            id__in=implementation_ids
        ).update(data_quality_status='excluded')

    def scenario_link(self, obj):
        url = reverse('admin:authoringtool_scenario_change', args=[obj.scenario_id])
        return format_html('<a href="{}">{}</a>', url, obj.scenario.name)
    scenario_link.short_description = 'Scenario'
    scenario_link.admin_order_field = 'scenario__name'


# ─── UserAnswer ───────────────────────────────────────────────────────────────

_export_answers = make_csv_export(
    'user_answers',
    [
        'ID',
        'Implementation ID',
        'User',
        'Activity',
        'Activity Concept',
        'Activity Revision',
        'Scenario',
        'Family',
        'Language',
        'Scenario Version',
        'Version Confidence',
        'Answer',
        'Is Correct',
        'Timing (s)',
        'Created On',
    ],
    lambda o: [
        o.id,
        o.implementation_id,
        o.user,
        o.activity,
        o.activity.concept if o.activity_id else '',
        o.activity_revision,
        o.activity.scenario if o.activity_id else '',
        o.activity.scenario.family if o.activity_id else '',
        o.activity.scenario.language if o.activity_id else '',
        o.scenario_version,
        o.version_confidence,
        o.answer,
        o.answer.is_correct if o.answer_id else '',
        o.timing,
        o.created_on,
    ],
)


@admin.register(UserAnswer)
class UserAnswerAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'implementation',
        'user',
        'activity',
        'activity_revision',
        'scenario_version',
        'version_confidence',
        'answer',
        'timing',
        'created_on',
    )
    list_filter = ('activity__scenario', 'version_confidence')
    search_fields = ('user__username', 'activity__name')
    raw_id_fields = (
        'implementation',
        'user',
        'activity',
        'activity_revision',
        'answer',
        'scenario_version',
    )
    list_select_related = (
        'implementation',
        'user',
        'activity',
        'activity_revision',
        'answer',
        'scenario_version',
    )
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
    list_display = (
        'id', 'activity', 'scenario', 'category', 'flag_type',
        'evidence_scope', 'is_at_risk', 'auto_flagged', 'flagged_on',
    )
    list_filter = (
        'category', 'evidence_scope', 'is_at_risk', 'auto_flagged',
        'scenario',
    )
    search_fields = ('flag_type', 'flag_reason', 'activity__name', 'scenario__name')
    raw_id_fields = ('activity', 'scenario', 'phase')
    list_select_related = ('activity', 'scenario')
    readonly_fields = (
        'evidence_signature',
        'evidence_sources',
        'flagged_on',
    )
    date_hierarchy = 'flagged_on'


# ─── ProposalGenerationRun ─────────────────────────────────────────────────────

@admin.register(ProposalGenerationRun)
class ProposalGenerationRunAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'scenario',
        'scenario_version',
        'evidence_scope',
        'created_by',
        'created_at',
        'is_current',
    )
    list_filter = ('is_current', 'evidence_scope', 'created_at')
    search_fields = ('scenario__name',)
    raw_id_fields = ('scenario', 'scenario_version', 'created_by')
    readonly_fields = (
        'evidence_version_ids',
        'evidence_summary',
        'created_at',
    )
    date_hierarchy = 'created_at'


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
    list_display = (
        'flag_type', 'category', 'action', 'q_value', 'reward_count',
        'positive_reward_count', 'negative_reward_count', 'mean_reward',
        'updated_at',
    )
    list_filter = ('flag_type', 'category', 'action')
    search_fields = ('flag_type',)
    ordering = ('-updated_at',)

    @admin.display(description='Mean reward', ordering='reward_sum')
    def mean_reward(self, obj):
        if not obj.reward_count:
            return '—'
        return f"{obj.reward_sum / obj.reward_count:.3f}"


# ─── UserProposalReview ───────────────────────────────────────────────────────

@admin.register(BanditPolicyConfiguration)
class BanditPolicyConfigurationAdmin(admin.ModelAdmin):
    list_display = (
        'name',
        'policy',
        'is_active',
        'minimum_context_rewards',
        'cold_start_weights',
        'updated_at',
    )
    list_filter = ('policy', 'is_active')
    fieldsets = (
        ('Active policy', {
            'fields': (
                'name',
                'is_active',
                'policy',
                'minimum_context_rewards',
            ),
        }),
        ('Cold-start action weights', {
            'fields': ('create_weight', 'skip_weight', 'revise_weight'),
            'description': (
                'Used until each flag type/category context has sufficient '
                'teacher rewards.'
            ),
        }),
        ('Thompson Sampling', {
            'fields': ('thompson_prior_alpha', 'thompson_prior_beta'),
            'classes': ('collapse',),
        }),
        ('UCB', {
            'fields': ('ucb_exploration_strength',),
            'classes': ('collapse',),
        }),
    )

    @admin.display(description='Create / Skip / Revise')
    def cold_start_weights(self, obj):
        return (
            f"{obj.create_weight:.0%} / "
            f"{obj.skip_weight:.0%} / "
            f"{obj.revise_weight:.0%}"
        )

    def save_model(self, request, obj, form, change):
        if obj.is_active:
            BanditPolicyConfiguration.objects.exclude(pk=obj.pk).update(
                is_active=False
            )
        super().save_model(request, obj, form, change)


@admin.register(ProposalStructuralFailure)
class ProposalStructuralFailureAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'scenario',
        'activity',
        'selected_action',
        'stage',
        'resolved',
        'created_at',
    )
    list_filter = ('stage', 'selected_action', 'resolved', 'created_at')
    search_fields = (
        'scenario__name',
        'activity__name',
        'errors',
        'raw_output',
    )
    raw_id_fields = ('scenario', 'generation_run', 'proposal', 'activity')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'


@admin.register(UserProposalReview)
class UserProposalReviewAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'proposal', 'user', 'status', 'feedback_type',
        'was_edited', 'edit_count', 'reviewed_at',
    )
    list_filter = ('status', 'feedback_type', 'reviewed_at')
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
