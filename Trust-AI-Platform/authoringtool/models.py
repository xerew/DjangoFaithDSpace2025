from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import IntegerRangeField, ArrayField
from organization.models import Organization
import qrcode
from io import BytesIO
from django.core.files import File
from django.utils import timezone
from django.db import transaction
from django.db.models.signals import m2m_changed, pre_save, post_save, post_delete
from django.dispatch import receiver
from django.core.exceptions import ValidationError
import hashlib
import json
import uuid

class Language(models.Model):
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Subject(models.Model):
    CATEGORY_CHOICES = [
        ('STEM', 'STEM'),
        ('Humanities', 'Humanities'),
        ('Social Sciences', 'Social Sciences'),
        ('Arts', 'Arts'),
        ('Other', 'Other'),
    ]
    name = models.CharField(max_length=100, unique=True)
    icon = models.CharField(max_length=60, default='bi-book')
    color = models.CharField(max_length=20, default='#1a56db')
    category = models.CharField(max_length=30, choices=CATEGORY_CHOICES, default='STEM')
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order', 'name']
        verbose_name = 'Subject'
        verbose_name_plural = 'Subjects'

    def __str__(self):
        return self.name


class ScenarioFamily(models.Model):
    """A shared lesson identity across translations and teacher adaptations."""

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    canonical_scenario = models.ForeignKey(
        'Scenario',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='canonical_for_families',
    )
    subjects = models.ManyToManyField(
        Subject,
        blank=True,
        related_name='scenario_families',
    )
    created_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='created_scenario_families',
    )
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Scenario Family'
        verbose_name_plural = 'Scenario Families'
        ordering = ['title', 'id']

    def __str__(self):
        return self.title

    def clean(self):
        super().clean()
        if (
            self.canonical_scenario_id
            and self.pk
            and self.canonical_scenario.family_id != self.pk
        ):
            raise ValidationError({
                'canonical_scenario': (
                    'The canonical scenario must belong to this family.'
                )
            })


class ScenarioFamilyReviewProxy(ScenarioFamily):
    """Admin-only dashboard proxy for reviewing family membership."""

    class Meta:
        proxy = True
        verbose_name = 'Scenario family review dashboard'
        verbose_name_plural = 'Scenario family review dashboard'


class Scenario(models.Model):
    VISIBILITY_CHOICES = [
        ('private', 'Private (In-Progress)'),
        ('org', 'Organization Users Only'),
        ('public', 'Public'),
    ]
    VARIANT_TYPE_CHOICES = [
        ('canonical', 'Canonical'),
        ('translation', 'Official translation'),
        ('adaptation', 'Teacher adaptation'),
    ]
    name = models.CharField(max_length=255, unique=True)
    learning_goals = models.TextField(blank=True)
    description = models.TextField(blank=True)
    age_of_students = IntegerRangeField(blank=True, null=True)
    subject_domains = models.CharField(max_length=255, blank=True)
    language = models.CharField(max_length=255, blank=True)
    suggested_learning_time = models.IntegerField(null=True)
    image = models.ImageField(upload_to='images', null=True, blank=True)
    llm_context = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='created_scenarios')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='updated_scenarios')
    updated_on = models.DateTimeField(auto_now=True)
    is_personal = models.BooleanField(default=False, help_text="True if created by user from proposals")
    origin_scenario = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='personal_clones',
        help_text="Original scenario this was cloned from"
    )
    family = models.ForeignKey(
        ScenarioFamily,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='scenarios',
        help_text=(
            'Shared lesson identity across translations and teacher '
            'adaptations.'
        ),
    )
    variant_type = models.CharField(
        max_length=20,
        choices=VARIANT_TYPE_CHOICES,
        default='canonical',
        help_text='How this scenario relates to the rest of its family.',
    )
    current_version = models.ForeignKey(
        'ScenarioVersion',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='current_for_scenarios',
        help_text='Current immutable evidence definition for this scenario.',
    )
    start_activity = models.ForeignKey(
        'Activity',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='starting_scenarios',
        help_text=(
            "Explicit entry point for this scenario. It must belong to the "
            "same scenario."
        ),
    )

    # New field to manage visibility status
    visibility_status = models.CharField(max_length=20, choices=VISIBILITY_CHOICES, default='private')

    # Add an optional link to the organization (if needed)
    organizations = models.ManyToManyField(Organization, blank=True, related_name='scenarios')  # Allow multiple organizations

    # Editable by organization's members
    is_editable_by_org = models.BooleanField(default=False, help_text="If checked, members of the selected organization(s) can edit this scenario.")

    subjects = models.ManyToManyField('Subject', blank=True, related_name='scenarios')

    # Minimum implementations before Scenario Metrics & AI button is shown
    ai_metrics_min_implementations = models.PositiveIntegerField(
        default=200,
        help_text="Minimum number of student implementations required before Metrics & AI / Proposals are shown without a warning."
    )

    class Meta:
        verbose_name = 'Scenario'
        verbose_name_plural = 'Scenarios'
        ordering = ['created_on']

    def __str__(self):
        return self.name

    def get_start_activity(self):
        """Return the explicit entry activity, with a rollout-safe fallback."""
        if self.start_activity_id:
            return self.start_activity
        return self.activities.order_by('id').first()

    def ensure_family(self):
        """Return this scenario's family, creating a canonical one if needed."""
        if self.family_id:
            return self.family
        if not self.pk:
            raise ValueError('A scenario must be saved before creating a family.')

        family = ScenarioFamily.objects.create(
            title=self.name,
            description=self.description or '',
            canonical_scenario=self,
            created_by_id=self.created_by_id,
        )
        family.subjects.set(self.subjects.all())
        Scenario.objects.filter(pk=self.pk, family__isnull=True).update(
            family=family,
            variant_type='canonical',
        )
        self.family = family
        self.variant_type = 'canonical'
        return family

    @staticmethod
    def _fingerprint_payload(payload):
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def build_version_snapshot(self):
        """Serialize the pedagogical definition used to classify evidence."""
        activities = list(
            self.activities
            .select_related(
                'phase',
                'activity_type',
                'simulation',
                'experiment_ll',
                'vr_ar_experiment',
            )
            .prefetch_related('answers')
            .order_by('phase__created_on', 'phase_id', 'created_on', 'id')
        )
        activity_keys = {
            activity.id: str(activity.lineage_key)
            for activity in activities
        }
        phase_ids = []
        for activity in activities:
            if activity.phase_id not in phase_ids:
                phase_ids.append(activity.phase_id)
        for phase_id in (
            self.phases.order_by('created_on', 'id')
            .values_list('id', flat=True)
        ):
            if phase_id not in phase_ids:
                phase_ids.append(phase_id)

        logic_by_activity = {}
        for logic in (
            NextQuestionLogic.objects
            .filter(activity__scenario=self)
            .select_related('answer', 'next_activity')
            .order_by('activity_id', 'answer_id', 'id')
        ):
            logic_by_activity.setdefault(logic.activity_id, []).append(logic)

        branching_by_activity = {
            branching.activity_id: branching
            for branching in (
                EvQuestionBranching.objects
                .filter(activity__scenario=self)
                .select_related(
                    'next_question_on_high',
                    'next_question_on_mid',
                    'next_question_on_low',
                )
            )
        }
        bunches_by_activity = {
            bunch.activity_primary_id: bunch
            for bunch in QuestionBunch.objects.filter(
                activity_primary__scenario=self
            )
        }

        structure_phases = []
        content_phases = []
        for phase_index, phase_id in enumerate(phase_ids):
            phase = (
                self.phases.filter(pk=phase_id).first()
                if phase_id is not None
                else None
            )
            phase_activities = [
                activity
                for activity in activities
                if activity.phase_id == phase_id
            ]
            structural_activities = []
            content_activities = []
            for activity_index, activity in enumerate(phase_activities):
                answers = list(activity.answers.all().order_by('id'))
                routes = logic_by_activity.get(activity.id, [])
                route_by_answer = {
                    route.answer_id: activity_keys.get(route.next_activity_id)
                    for route in routes
                    if route.answer_id
                }
                direct_routes = [
                    activity_keys.get(route.next_activity_id)
                    for route in routes
                    if not route.answer_id
                ]
                branching = branching_by_activity.get(activity.id)
                bunch = bunches_by_activity.get(activity.id)

                structural_activities.append({
                    'position': activity_index,
                    'lineage_key': str(activity.lineage_key),
                    'activity_type': (
                        activity.activity_type.name.strip().casefold()
                        if activity.activity_type
                        else ''
                    ),
                    'is_evaluatable': activity.is_evaluatable,
                    'is_primary_ev': activity.is_primary_ev,
                    'must_wait': activity.must_wait,
                    'score_limit': str(activity.score_limit),
                    'simulation': (
                        activity.simulation.name
                        if activity.simulation
                        else None
                    ),
                    'experiment': (
                        activity.experiment_ll.name
                        if activity.experiment_ll
                        else None
                    ),
                    'vr_ar_experiment': (
                        activity.vr_ar_experiment.name
                        if activity.vr_ar_experiment
                        else None
                    ),
                    'answers': [
                        {
                            'position': answer_index,
                            'is_correct': answer.is_correct,
                            'weight': answer.answer_weight,
                            'next_activity': route_by_answer.get(answer.id),
                        }
                        for answer_index, answer in enumerate(answers)
                    ],
                    'direct_routes': direct_routes,
                    'branching': {
                        'high': activity_keys.get(
                            branching.next_question_on_high_id
                        ),
                        'mid': activity_keys.get(
                            branching.next_question_on_mid_id
                        ),
                        'low': activity_keys.get(
                            branching.next_question_on_low_id
                        ),
                    } if branching else None,
                    'evaluation_bunch': [
                        activity_keys.get(activity_id)
                        for activity_id in (bunch.activity_ids if bunch else [])
                    ],
                })
                content_activities.append({
                    'lineage_key': str(activity.lineage_key),
                    'name': activity.name,
                    'text': activity.text,
                    'plain_text': activity.plain_text,
                    'helper': activity.helper,
                    'answers': [
                        {
                            'position': answer_index,
                            'text': answer.text,
                            'image': answer.image.name if answer.image else '',
                            'video': answer.vid_url or '',
                        }
                        for answer_index, answer in enumerate(answers)
                    ],
                })

            structure_phases.append({
                'position': phase_index,
                'activities': structural_activities,
            })
            content_phases.append({
                'position': phase_index,
                'name': phase.name if phase else '',
                'description': (phase.description or '') if phase else '',
                'activities': content_activities,
            })

        structure = {
            'schema': 1,
            'start_activity': activity_keys.get(self.start_activity_id),
            'phases': structure_phases,
        }
        content = {
            'schema': 1,
            'language': (self.language or '').strip(),
            'learning_goals': self.learning_goals or '',
            'description': self.description or '',
            'phases': content_phases,
        }
        return {
            'schema': 1,
            'structure': structure,
            'content': content,
        }

    def ensure_current_version(
        self,
        created_by=None,
        change_summary='',
        publish_draft=False,
    ):
        """Return a current version, creating one when evidence changed."""
        if not self.pk:
            raise ValueError('A scenario must be saved before it can be versioned.')

        with transaction.atomic():
            locked = (
                Scenario.objects
                .select_for_update()
                .get(pk=self.pk)
            )
            snapshot = locked.build_version_snapshot()
            structure_fingerprint = self._fingerprint_payload(
                snapshot['structure']
            )
            content_fingerprint = self._fingerprint_payload(
                snapshot['content']
            )
            current = locked.current_version
            if (
                current
                and not publish_draft
                and ScenarioRevisionDraft.objects.filter(
                    scenario=locked
                ).exists()
            ):
                # Live authoring records hold the working draft. Until an
                # explicit publish, analytics and implementations remain tied
                # to the last published immutable version.
                self.current_version = current
                return current
            if (
                current
                and current.structure_fingerprint == structure_fingerprint
                and current.content_fingerprint == content_fingerprint
            ):
                ScenarioVersionCompatibility.assign_automatic(current)
                ActivityRevision.capture_for_version(current)
                self.current_version = current
                return current

            latest_number = (
                locked.versions.aggregate(models.Max('version_number'))[
                    'version_number__max'
                ]
                or 0
            )
            ScenarioVersion.objects.filter(
                scenario=locked,
                is_current=True,
            ).update(is_current=False)
            if not change_summary:
                if current is None:
                    change_summary = 'Initial evidence version'
                elif current.structure_fingerprint != structure_fingerprint:
                    change_summary = 'Scenario structure changed'
                else:
                    change_summary = 'Scenario learning content changed'
            version = ScenarioVersion.objects.create(
                scenario=locked,
                version_number=latest_number + 1,
                structure_fingerprint=structure_fingerprint,
                content_fingerprint=content_fingerprint,
                snapshot=snapshot,
                previous_version=current,
                created_by=created_by or locked.updated_by,
                change_summary=change_summary,
                is_current=True,
                revision_status='published',
                published_by=created_by or locked.updated_by,
                published_at=timezone.now(),
            )
            Scenario.objects.filter(pk=locked.pk).update(
                current_version=version
            )
            ScenarioVersionCompatibility.assign_automatic(version)
            ActivityRevision.capture_for_version(version)
            self.current_version = version
            return version

    def refresh_version_if_initialized(self, created_by=None):
        """Refresh an established evidence boundary after an authoring edit."""
        if not self.current_version_id:
            return None
        draft = ScenarioRevisionDraft.objects.filter(
            scenario=self
        ).first()
        if draft:
            draft.refresh_from_scenario()
            return self.current_version
        if self.has_student_evidence():
            raise ValidationError(
                'Start a revision draft before editing a scenario that has '
                'student implementations.'
            )
        return self.ensure_current_version(created_by=created_by)

    def has_student_evidence(self):
        """Return whether any non-teacher implementation uses this scenario."""
        if ScenarioImplementation.objects.filter(
            scenario=self
        ).exclude(user__groups__name='teachers').exists():
            return True
        return UserScenarioScore.objects.filter(
            scenario=self
        ).exclude(user__groups__name='teachers').exists()

    def begin_revision_draft(self, created_by):
        """Open a protected working draft based on the published version."""
        current = self.ensure_current_version(created_by=created_by)
        draft, _ = ScenarioRevisionDraft.objects.get_or_create(
            scenario=self,
            defaults={
                'base_version': current,
                'created_by': created_by,
            },
        )
        draft.refresh_from_scenario()
        return draft

    def publish_revision_draft(self, published_by, change_summary=''):
        """Atomically publish live draft content as a new immutable version."""
        with transaction.atomic():
            draft = (
                ScenarioRevisionDraft.objects
                .select_for_update()
                .get(scenario=self)
            )
            if not change_summary:
                change_summary = (
                    f'Published revision draft based on '
                    f'v{draft.base_version.version_number}'
                )
            version = self.ensure_current_version(
                created_by=published_by,
                change_summary=change_summary,
                publish_draft=True,
            )
            draft.delete()
            self.current_version = version
            return version

    def eligible_implementation_scores(self):
        if not self.current_version_id:
            return UserScenarioScore.objects.none()
        return (
            UserScenarioScore.objects
            .filter(
                scenario=self,
                scenario_version_id=self.current_version_id,
                version_confidence='exact',
                data_quality_status__in=['unreviewed', 'clean'],
            )
            .exclude(user__groups__name='teachers')
        )

    def eligible_implementation_count(self):
        return (
            self.eligible_implementation_scores()
            .values('implementation_id')
            .distinct()
            .count()
        )

    def compatible_current_versions(self):
        """Current versions approved for pooled family-level evidence."""
        current = self.ensure_current_version()
        membership = ScenarioVersionCompatibility.assign_automatic(current)
        if membership.status != 'compatible':
            return ScenarioVersion.objects.filter(pk=current.pk)

        compatible_ids = (
            ScenarioVersionCompatibility.objects
            .filter(
                cluster=membership.cluster,
                status='compatible',
                scenario_version__is_current=True,
                scenario_version__scenario__current_version=models.F(
                    'scenario_version'
                ),
            )
            .values_list('scenario_version_id', flat=True)
        )
        return ScenarioVersion.objects.filter(
            models.Q(pk=current.pk) | models.Q(pk__in=compatible_ids)
        ).distinct()

    def compatible_implementation_scores(self):
        return UserScenarioScore.objects.filter(
            scenario_version__in=self.compatible_current_versions(),
            version_confidence='exact',
            data_quality_status__in=['unreviewed', 'clean'],
        ).exclude(user__groups__name='teachers')

    def compatible_implementation_count(self):
        """Count distinct student implementations across compatible variants."""
        return (
            self.compatible_implementation_scores()
            .values('implementation_id')
            .distinct()
            .count()
        )

    def clean(self):
        super().clean()
        if (
            self.start_activity_id
            and self.pk
            and self.start_activity.scenario_id != self.pk
        ):
            raise ValidationError({
                'start_activity': (
                    'The start activity must belong to this scenario.'
                )
            })


class ScenarioVersion(models.Model):
    REVISION_STATUS_CHOICES = [
        ('legacy', 'Legacy revision'),
        ('published', 'Published revision'),
    ]
    scenario = models.ForeignKey(
        Scenario,
        on_delete=models.CASCADE,
        related_name='versions',
    )
    version_number = models.PositiveIntegerField()
    structure_fingerprint = models.CharField(max_length=64, db_index=True)
    content_fingerprint = models.CharField(max_length=64, db_index=True)
    snapshot = models.JSONField(default=dict)
    previous_version = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='next_versions',
    )
    created_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='created_scenario_versions',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    change_summary = models.CharField(max_length=255, blank=True)
    is_current = models.BooleanField(default=True)
    revision_status = models.CharField(
        max_length=20,
        choices=REVISION_STATUS_CHOICES,
        default='published',
    )
    published_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='published_scenario_versions',
    )
    published_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Scenario Version'
        verbose_name_plural = 'Scenario Versions'
        ordering = ['scenario', '-version_number']
        constraints = [
            models.UniqueConstraint(
                fields=['scenario', 'version_number'],
                name='unique_scenario_version_number',
            ),
            models.UniqueConstraint(
                fields=['scenario'],
                condition=models.Q(is_current=True),
                name='unique_current_scenario_version',
            ),
        ]

    def __str__(self):
        return f'{self.scenario.name} v{self.version_number}'

    def is_exactly_compatible_with(self, other):
        if not other:
            return False
        return (
            self.structure_fingerprint == other.structure_fingerprint
            and self.content_fingerprint == other.content_fingerprint
        )

    def save(self, *args, **kwargs):
        if self.pk:
            original = ScenarioVersion.objects.filter(pk=self.pk).values(
                'scenario_id',
                'version_number',
                'structure_fingerprint',
                'content_fingerprint',
                'snapshot',
                'previous_version_id',
                'revision_status',
            ).first()
            immutable_fields = {
                'scenario_id': self.scenario_id,
                'version_number': self.version_number,
                'structure_fingerprint': self.structure_fingerprint,
                'content_fingerprint': self.content_fingerprint,
                'snapshot': self.snapshot,
                'previous_version_id': self.previous_version_id,
                'revision_status': self.revision_status,
            }
            if original and original != immutable_fields:
                raise ValidationError(
                    'Published scenario revisions are immutable.'
                )
        super().save(*args, **kwargs)


class ScenarioRevisionDraft(models.Model):
    """Protected working snapshot awaiting an explicit teacher publish."""

    scenario = models.OneToOneField(
        Scenario,
        on_delete=models.CASCADE,
        related_name='revision_draft',
    )
    base_version = models.ForeignKey(
        ScenarioVersion,
        on_delete=models.PROTECT,
        related_name='revision_drafts',
    )
    snapshot = models.JSONField(default=dict)
    structure_fingerprint = models.CharField(max_length=64, blank=True)
    content_fingerprint = models.CharField(max_length=64, blank=True)
    created_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='created_scenario_revision_drafts',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Scenario Revision Draft'
        verbose_name_plural = 'Scenario Revision Drafts'
        ordering = ['-updated_at']

    def __str__(self):
        return (
            f'Draft for {self.scenario} based on '
            f'v{self.base_version.version_number}'
        )

    def refresh_from_scenario(self):
        snapshot = self.scenario.build_version_snapshot()
        self.snapshot = snapshot
        self.structure_fingerprint = Scenario._fingerprint_payload(
            snapshot['structure']
        )
        self.content_fingerprint = Scenario._fingerprint_payload(
            snapshot['content']
        )
        self.save(
            update_fields=[
                'snapshot',
                'structure_fingerprint',
                'content_fingerprint',
                'updated_at',
            ]
        )


class EvidenceCompatibilityCluster(models.Model):
    """A reviewed pool of scenario versions that may share evidence."""

    family = models.ForeignKey(
        ScenarioFamily,
        on_delete=models.CASCADE,
        related_name='evidence_clusters',
    )
    name = models.CharField(max_length=255)
    cluster_key = models.CharField(max_length=100)
    structure_fingerprint = models.CharField(
        max_length=64,
        blank=True,
        db_index=True,
    )
    is_automatic = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='created_evidence_clusters',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Evidence Compatibility Cluster'
        verbose_name_plural = 'Evidence Compatibility Clusters'
        ordering = ['family', 'name', 'id']
        constraints = [
            models.UniqueConstraint(
                fields=['family', 'cluster_key'],
                name='unique_family_evidence_cluster_key',
            ),
        ]

    def __str__(self):
        return f'{self.family.title}: {self.name}'

    def save(self, *args, **kwargs):
        if not self.cluster_key:
            self.cluster_key = f'manual:{uuid.uuid4()}'
        super().save(*args, **kwargs)


class ScenarioVersionCompatibility(models.Model):
    STATUS_CHOICES = [
        ('compatible', 'Compatible'),
        ('needs_review', 'Needs review'),
        ('excluded', 'Excluded from family evidence'),
    ]
    DECISION_SOURCE_CHOICES = [
        ('automatic', 'Automatic structural match'),
        ('admin', 'Administrator decision'),
    ]

    scenario_version = models.OneToOneField(
        ScenarioVersion,
        on_delete=models.CASCADE,
        related_name='compatibility',
    )
    cluster = models.ForeignKey(
        EvidenceCompatibilityCluster,
        on_delete=models.CASCADE,
        related_name='memberships',
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='compatible',
    )
    decision_source = models.CharField(
        max_length=20,
        choices=DECISION_SOURCE_CHOICES,
        default='automatic',
    )
    reason = models.CharField(max_length=500, blank=True)
    reviewed_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='reviewed_scenario_compatibilities',
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Scenario Version Compatibility'
        verbose_name_plural = 'Scenario Version Compatibilities'
        ordering = [
            'cluster__family',
            'cluster',
            'scenario_version__scenario',
            '-scenario_version__version_number',
        ]

    def __str__(self):
        return (
            f'{self.scenario_version} -> {self.cluster.name} '
            f'({self.get_status_display()})'
        )

    def clean(self):
        super().clean()
        if (
            self.cluster_id
            and self.scenario_version_id
            and self.cluster.family_id
            != self.scenario_version.scenario.family_id
        ):
            raise ValidationError({
                'cluster': (
                    'The compatibility cluster must belong to the scenario '
                    'version family.'
                )
            })

    @classmethod
    def assign_automatic(cls, scenario_version):
        """Create, but never overwrite, a deterministic compatibility decision."""
        try:
            return scenario_version.compatibility
        except cls.DoesNotExist:
            pass

        scenario = scenario_version.scenario
        family = scenario.ensure_family()
        fingerprint = scenario_version.structure_fingerprint
        cluster, _ = EvidenceCompatibilityCluster.objects.get_or_create(
            family=family,
            cluster_key=f'auto:{fingerprint}',
            defaults={
                'name': f'Structure {fingerprint[:12]}',
                'structure_fingerprint': fingerprint,
                'is_automatic': True,
                'created_by_id': scenario_version.created_by_id,
            },
        )
        needs_review = scenario.variant_type == 'adaptation'
        return cls.objects.create(
            scenario_version=scenario_version,
            cluster=cluster,
            status='needs_review' if needs_review else 'compatible',
            decision_source='automatic',
            reason=(
                'Teacher adaptations require review before their evidence is '
                'pooled with another scenario.'
                if needs_review
                else 'Automatically matched by family and structural fingerprint.'
            ),
        )


class ScenarioSimilarityProfile(models.Model):
    """Current explainable matching features for a scenario version."""

    scenario = models.OneToOneField(
        Scenario,
        on_delete=models.CASCADE,
        related_name='similarity_profile',
    )
    scenario_version = models.ForeignKey(
        ScenarioVersion,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='similarity_profiles',
    )
    content_digest = models.CharField(max_length=64, db_index=True)
    feature_schema = models.PositiveSmallIntegerField(default=1)
    features = models.JSONField(default=dict)
    embedding = models.JSONField(default=list, blank=True)
    embedding_model = models.CharField(max_length=255, blank=True)
    embedding_error = models.CharField(max_length=500, blank=True)
    generated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Scenario Similarity Profile'
        verbose_name_plural = 'Scenario Similarity Profiles'
        ordering = ['scenario__name', 'scenario_id']

    def __str__(self):
        version = (
            f'v{self.scenario_version.version_number}'
            if self.scenario_version_id
            else 'unversioned'
        )
        return f'{self.scenario.name} ({version})'

    @property
    def is_stale(self):
        return (
            not self.scenario_version_id
            or self.scenario.current_version_id != self.scenario_version_id
        )


class ScenarioFamilyCandidate(models.Model):
    """An explainable, reviewable possible relationship between scenarios."""

    RELATIONSHIP_CHOICES = [
        ('translation', 'Same family — translation'),
        ('adaptation', 'Same family — adaptation'),
        ('related_topic', 'Related topic only'),
    ]
    DECISION_CHOICES = [
        ('pending', 'Pending review'),
        ('translation', 'Same family — translation'),
        ('adaptation', 'Same family — adaptation'),
        ('related_topic', 'Related topic only'),
        ('unrelated', 'Not related'),
        ('deferred', 'Review later'),
    ]

    LLM_RELATIONSHIP_CHOICES = [
        ('translation', 'Same family — translation'),
        ('adaptation', 'Same family — adaptation'),
        ('related_topic', 'Related topic only'),
        ('unrelated', 'Not related'),
    ]
    LLM_STATUS_CHOICES = [
        ('not_requested', 'Not requested'),
        ('pending', 'Queued or running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    scenario_a = models.ForeignKey(
        Scenario,
        on_delete=models.CASCADE,
        related_name='family_candidates_as_a',
    )
    scenario_b = models.ForeignKey(
        Scenario,
        on_delete=models.CASCADE,
        related_name='family_candidates_as_b',
    )
    scenario_a_version = models.ForeignKey(
        ScenarioVersion,
        on_delete=models.CASCADE,
        related_name='family_candidates_as_a',
    )
    scenario_b_version = models.ForeignKey(
        ScenarioVersion,
        on_delete=models.CASCADE,
        related_name='family_candidates_as_b',
    )
    similarity_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        default=0,
    )
    family_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        default=0,
    )
    topic_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        default=0,
    )
    component_scores = models.JSONField(default=dict)
    reasons = models.JSONField(default=list, blank=True)
    suggested_relationship = models.CharField(
        max_length=30,
        choices=RELATIONSHIP_CHOICES,
    )
    decision = models.CharField(
        max_length=30,
        choices=DECISION_CHOICES,
        default='pending',
        db_index=True,
    )
    target_family = models.ForeignKey(
        ScenarioFamily,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='targeted_family_candidates',
        help_text=(
            'Family that should remain when a same-family decision is '
            'approved. Leave blank to use the recommended family.'
        ),
    )
    review_notes = models.TextField(blank=True)
    reviewed_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='reviewed_scenario_family_candidates',
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    is_current = models.BooleanField(default=True, db_index=True)
    detection_method = models.CharField(
        max_length=100,
        default='explainable-hybrid-v1',
    )
    llm_status = models.CharField(
        max_length=20,
        choices=LLM_STATUS_CHOICES,
        default='not_requested',
        db_index=True,
    )
    llm_suggested_relationship = models.CharField(
        max_length=30,
        choices=LLM_RELATIONSHIP_CHOICES,
        blank=True,
    )
    llm_confidence = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
    )
    llm_reasoning = models.TextField(blank=True)
    llm_details = models.JSONField(default=dict, blank=True)
    llm_model = models.CharField(max_length=255, blank=True)
    llm_error = models.CharField(max_length=500, blank=True)
    llm_reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Scenario Family Candidate'
        verbose_name_plural = 'Scenario Family Candidates'
        ordering = ['decision', '-similarity_score', '-updated_at']
        constraints = [
            models.CheckConstraint(
                check=models.Q(scenario_a__lt=models.F('scenario_b')),
                name='scenario_candidate_ordered_pair',
            ),
            models.UniqueConstraint(
                fields=[
                    'scenario_a',
                    'scenario_b',
                    'scenario_a_version',
                    'scenario_b_version',
                ],
                name='unique_scenario_candidate_version_pair',
            ),
            models.UniqueConstraint(
                fields=['scenario_a', 'scenario_b'],
                condition=models.Q(is_current=True),
                name='unique_current_scenario_candidate_pair',
            ),
        ]
        indexes = [
            models.Index(
                fields=['is_current', 'decision', '-similarity_score'],
                name='scenario_match_review_idx',
            ),
        ]

    def __str__(self):
        return (
            f'{self.scenario_a.name} ↔ {self.scenario_b.name} '
            f'({float(self.similarity_score):.0%})'
        )

    def clean(self):
        super().clean()
        errors = {}
        if self.scenario_a_id and self.scenario_b_id:
            if self.scenario_a_id >= self.scenario_b_id:
                errors['scenario_b'] = (
                    'Candidate scenarios must be stored in ascending ID order.'
                )
        if (
            self.scenario_a_version_id
            and self.scenario_a_id
            and self.scenario_a_version.scenario_id != self.scenario_a_id
        ):
            errors['scenario_a_version'] = (
                'The first version must belong to the first scenario.'
            )
        if (
            self.scenario_b_version_id
            and self.scenario_b_id
            and self.scenario_b_version.scenario_id != self.scenario_b_id
        ):
            errors['scenario_b_version'] = (
                'The second version must belong to the second scenario.'
            )
        if (
            self.target_family_id
            and self.scenario_a_id
            and self.scenario_b_id
            and self.target_family_id
            not in {self.scenario_a.family_id, self.scenario_b.family_id}
        ):
            errors['target_family'] = (
                'The target must be one of the two scenario families.'
            )
        if errors:
            raise ValidationError(errors)


class ScenarioFamilyMatchDecision(models.Model):
    """Immutable audit event for a scenario-family candidate decision."""

    candidate = models.ForeignKey(
        ScenarioFamilyCandidate,
        on_delete=models.CASCADE,
        related_name='decision_events',
    )
    decision = models.CharField(
        max_length=30,
        choices=ScenarioFamilyCandidate.DECISION_CHOICES,
    )
    notes = models.TextField(blank=True)
    decided_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='scenario_family_match_decisions',
    )
    decided_at = models.DateTimeField(auto_now_add=True)
    details = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = 'Scenario Family Match Decision'
        verbose_name_plural = 'Scenario Family Match Decisions'
        ordering = ['-decided_at', '-id']

    def __str__(self):
        return (
            f'{self.candidate}: {self.get_decision_display()} '
            f'by {self.decided_by or "system"}'
        )


class ScenarioHealthProxy(Scenario):
    """Proxy used solely to power the Scenario Health Check admin page."""
    class Meta:
        proxy = True
        verbose_name = 'Scenario Health Check'
        verbose_name_plural = 'Scenario Health Check'


class Phase(models.Model):
    name = models.CharField(max_length=255, null=False)
    description = models.TextField(blank=True, null=True)
    image = models.ImageField(upload_to='images', null=True, blank=True)
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='phases')
    llm_context = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='created_phases')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='updated_phases')
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Phase'
        verbose_name_plural = 'Phases'
        ordering = ['created_on']

    def __str__(self):
        return self.name

class ActivityType(models.Model):
    name = models.CharField(max_length=255, null=False, blank=False)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='created_activity_types')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='updated_activity_types')
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Activity Type"
        verbose_name_plural = "Activity Types"
        ordering = ['created_on']

    def __str__(self):
        return self.name

class Simulation(models.Model):
    LANGUAGE_CHOICES = [
        ('', '— Select Language —'),
        ('English', 'English'),
        ('Greek', 'Greek'),
        ('Spanish', 'Spanish'),
        ('French', 'French'),
        ('German', 'German'),
        ('Italian', 'Italian'),
        ('Portuguese', 'Portuguese'),
        ('Dutch', 'Dutch'),
        ('Polish', 'Polish'),
        ('Romanian', 'Romanian'),
        ('Turkish', 'Turkish'),
        ('Arabic', 'Arabic'),
        ('Other', 'Other'),
    ]

    name = models.CharField(max_length=200)
    iframe_url = models.URLField()
    width = models.PositiveIntegerField(default=800)
    height = models.PositiveIntegerField(default=600)
    allow_fullscreen = models.BooleanField(default=True)
    language = models.CharField(max_length=100, blank=True, default='', choices=LANGUAGE_CHOICES)

    class Meta:
        verbose_name = "Simulation"
        verbose_name_plural = "Simulations"
        ordering = ['name']

    def __str__(self):
        return self.name

# LabsLand Integration
class ExperimentLL(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    launch_url = models.URLField()
    consumer_key = models.CharField(max_length=100)
    shared_secret = models.CharField(max_length=100)
    picture = models.ImageField(upload_to='experiment_pictures/', blank=True, null=True)

    class Meta:
        verbose_name = "LabsLand Experiment"
        verbose_name_plural = "LabsLand Experiments"
        ordering = ["id"]

    def __str__(self):
        return self.name

# VR/AR Integration QR CODE - 31/03
class VRARExperiment(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    launch_url = models.URLField()
    qr_code = models.ImageField(upload_to='qr_codes/', blank=True, null=True)  # Add QR code field
    picture = models.ImageField(upload_to='vr_ar_experiment_pictures/', blank=True, null=True)

    class Meta:
        verbose_name = "VR/AR Experiment"
        verbose_name_plural = "VR/AR Experiments"
        ordering = ['id']

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        # Generate QR code only if the URL is present and QR code does not already exist
        if self.launch_url and not self.qr_code:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(self.launch_url)
            qr.make(fit=True)

            img = qr.make_image(fill='black', back_color='white')

            # Save the QR code to a file
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            filename = f'{self.name}_qr.png'
            self.qr_code.save(filename, File(buffer), save=False)
        
        super().save(*args, **kwargs)


class ActivityConcept(models.Model):
    """Language-independent identity for comparable family activities."""

    family = models.ForeignKey(
        ScenarioFamily,
        on_delete=models.CASCADE,
        related_name='activity_concepts',
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='created_activity_concepts',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Activity Concept'
        verbose_name_plural = 'Activity Concepts'
        ordering = ['family', 'title', 'id']

    def __str__(self):
        return f'{self.family.title}: {self.title}'


class Activity(models.Model):
    name = models.CharField(max_length=255, null=False, blank=False, default='ActivityDefaultName')
    text = models.TextField(null=False, blank=False)
    plain_text = models.TextField(blank=True)
    correct_count = models.IntegerField(default=0)
    incorrect_count = models.IntegerField(default=0)
    is_evaluatable = models.BooleanField(default=False)
    is_primary_ev = models.BooleanField(default=False)
    must_wait = models.BooleanField(default=False)
    score_limit = models.FloatField(default=0.0)
    scenario = models.ForeignKey('Scenario', on_delete=models.CASCADE, null=True, related_name='activities')
    phase = models.ForeignKey('Phase', on_delete=models.CASCADE, null=True, related_name='activities')
    activity_type = models.ForeignKey('ActivityType', on_delete=models.SET_NULL, null=True, related_name='activities')
    helper = models.CharField(max_length=255, blank=True)
    simulation = models.ForeignKey(Simulation, on_delete=models.SET_NULL, null=True, blank=True)
    experiment_ll = models.ForeignKey(ExperimentLL, on_delete=models.SET_NULL, null=True, blank=True)  # LabsLand Integration
    vr_ar_experiment = models.ForeignKey('VRARExperiment', on_delete=models.SET_NULL, null=True, blank=True) # VR_AR
    llm_context = models.TextField(blank=True, null=True)
    related_act_llm_context = models.TextField(blank=True, null=True)
    llm_image_description = models.TextField(blank=True, null=True)
    short_llm_summary = models.TextField(blank=True, null=True)
    lineage_key = models.UUIDField(
        default=uuid.uuid4,
        db_index=True,
        editable=False,
        help_text=(
            'Stable activity identity shared by translations and unchanged '
            'copies.'
        ),
    )
    concept = models.ForeignKey(
        ActivityConcept,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='activities',
        help_text=(
            'Language-independent family concept used to compare equivalent '
            'activities across variants.'
        ),
    )
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='created_activities')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, related_name='updated_activities')
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Activity"
        verbose_name_plural = "Activities"
        ordering = ['created_on']

    def __str__(self):
        return f"{self.name} Scenario {self.scenario.id}"

    def clean(self):
        super().clean()
        if (
            self.concept_id
            and self.scenario_id
            and self.scenario.family_id
            and self.concept.family_id != self.scenario.family_id
        ):
            raise ValidationError({
                'concept': (
                    'The activity concept must belong to the scenario family.'
                )
            })


class ActivityMatchingProxy(Activity):
    """Proxy used for the dedicated activity-to-concept matching screen."""

    class Meta:
        proxy = True
        verbose_name = 'Activity Matching'
        verbose_name_plural = 'Activity Matching'


class ActivityRevision(models.Model):
    """Immutable activity definition within an immutable scenario version."""

    activity = models.ForeignKey(
        Activity,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='revisions',
    )
    concept = models.ForeignKey(
        ActivityConcept,
        on_delete=models.PROTECT,
        related_name='revisions',
    )
    scenario_version = models.ForeignKey(
        ScenarioVersion,
        on_delete=models.CASCADE,
        related_name='activity_revisions',
    )
    lineage_key = models.UUIDField(db_index=True)
    revision_number = models.PositiveIntegerField()
    structure_fingerprint = models.CharField(max_length=64, db_index=True)
    content_fingerprint = models.CharField(max_length=64, db_index=True)
    snapshot = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Activity Revision'
        verbose_name_plural = 'Activity Revisions'
        ordering = [
            'scenario_version__scenario',
            'scenario_version__version_number',
            'revision_number',
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['scenario_version', 'lineage_key'],
                name='unique_activity_revision_per_scenario_version',
            ),
        ]
        indexes = [
            models.Index(
                fields=['concept', 'scenario_version'],
                name='actrev_concept_version_idx',
            ),
        ]

    def __str__(self):
        return (
            f'{self.concept} @ {self.scenario_version} '
            f'(r{self.revision_number})'
        )

    def save(self, *args, **kwargs):
        if self.pk:
            raise ValidationError('Activity revisions are immutable.')
        super().save(*args, **kwargs)

    @classmethod
    def capture_for_version(cls, scenario_version):
        """Create missing immutable activity revisions from a version snapshot."""
        snapshot = scenario_version.snapshot or {}
        structure_by_lineage = {}
        content_by_lineage = {}
        for phase in snapshot.get('structure', {}).get('phases', []):
            for activity in phase.get('activities', []):
                lineage_key = activity.get('lineage_key')
                if lineage_key:
                    structure_by_lineage[lineage_key] = activity
        for phase in snapshot.get('content', {}).get('phases', []):
            for activity in phase.get('activities', []):
                lineage_key = activity.get('lineage_key')
                if lineage_key:
                    content_by_lineage[lineage_key] = activity

        activities = {
            str(activity.lineage_key): activity
            for activity in (
                scenario_version.scenario.activities
                .select_related('concept', 'scenario__family')
                .all()
            )
        }
        created = []
        for lineage_key, structural in structure_by_lineage.items():
            activity = activities.get(lineage_key)
            if not activity:
                continue
            concept = activity.concept
            if concept is None:
                family = activity.scenario.ensure_family()
                concept = ActivityConcept.objects.create(
                    family=family,
                    title=activity.name,
                    created_by_id=activity.created_by_id,
                )
                Activity.objects.filter(pk=activity.pk).update(
                    concept=concept
                )
                activity.concept = concept
            content = content_by_lineage.get(lineage_key, {})
            revision_snapshot = {
                'schema': 1,
                'structure': structural,
                'content': content,
            }
            revision, was_created = cls.objects.get_or_create(
                scenario_version=scenario_version,
                lineage_key=lineage_key,
                defaults={
                    'activity': activity,
                    'concept': concept,
                    'revision_number': scenario_version.version_number,
                    'structure_fingerprint': Scenario._fingerprint_payload(
                        structural
                    ),
                    'content_fingerprint': Scenario._fingerprint_payload(
                        content
                    ),
                    'snapshot': revision_snapshot,
                },
            )
            if was_created:
                created.append(revision)
        return created


class Answer(models.Model):
    activity = models.ForeignKey(Activity, on_delete=models.CASCADE, related_name='answers')
    text = models.TextField(null=False, blank=False)
    is_correct = models.BooleanField(default=False)
    answer_weight = models.IntegerField(default=0, blank=True)
    image = models.ImageField(upload_to='images', null=True, blank=True)
    vid_url = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='created_answers')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='updated_answers')
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Answer"
        verbose_name_plural = "Answers"
        ordering = ['created_on']
    
    def __str__(self):
        return self.text
    
class AnswerFeedback(models.Model):
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE, related_name='feedbacks')
    text = models.TextField(null=True, blank=True)
    image= models.ImageField(upload_to='images', null=True, blank=True)
    vid_url = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='created_answer_feedbacks')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='updated_answer_feedbacks')
    updated_on = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Feedback"
        verbose_name_plural = "Feedbacks"
        ordering = ['created_on']

    def __str__(self):
        return self.text

class NextQuestionLogic(models.Model):
    activity = models.ForeignKey(Activity, related_name='next_logic', on_delete=models.CASCADE)
    answer = models.ForeignKey(Answer, related_name='next_logic', on_delete=models.CASCADE, null=True, blank=True)
    next_activity = models.ForeignKey(Activity, related_name='previous_logic', on_delete=models.CASCADE, null=True)

    class Meta:
        unique_together = ('activity', 'answer')  # Enforces the unique constraint for question-answer pairs
        verbose_name = "Next Question"
        verbose_name_plural = "Next Questions"
        ordering = ["activity"]
    
    def __str__(self):
        return f"Activity {self.activity.id} to Next Activity {self.next_activity.id}"
    
class QuestionBunch(models.Model):
    activity_ids = ArrayField(models.IntegerField(), blank=False)
    activity_primary = models.ForeignKey(Activity, on_delete=models.CASCADE, related_name='question_bunches')

    class Meta:
        verbose_name = "Activity Bunch"
        verbose_name_plural = "Activity Bunches"
        ordering = ["activity_ids"]

    def __str__(self):
        return f"Bunch {self.pk}"

class EvQuestionBranching(models.Model):
    activity = models.OneToOneField(Activity, on_delete=models.CASCADE, primary_key=True, related_name='branching')
    next_question_on_high = models.ForeignKey(Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='next_high')
    next_question_on_high_feedback = models.TextField(blank=True)
    next_question_on_mid = models.ForeignKey(Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='next_mid')
    next_question_on_mid_feedback = models.TextField(blank=True)
    next_question_on_low = models.ForeignKey(Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='next_low')
    next_question_on_low_feedback = models.TextField(blank=True)

    class Meta:
        verbose_name = "Evaluating Question Branching"
        verbose_name_plural = "Evaluating Question Branching"
        ordering = ["activity"]

    def __str__(self):
        return f"Branching for Activity {self.activity}"


class ScenarioImplementation(models.Model):
    """One student attempt against one exact immutable scenario version."""

    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
        ('legacy', 'Legacy imported attempt'),
    ]
    VERSION_CONFIDENCE_CHOICES = [
        ('exact', 'Exact scenario version'),
        ('legacy_unknown', 'Legacy version unknown'),
    ]
    DATA_QUALITY_CHOICES = [
        ('unreviewed', 'Unreviewed'),
        ('clean', 'Clean'),
        ('suspect', 'Suspect'),
        ('excluded', 'Excluded'),
    ]

    user = models.ForeignKey(
        'auth.User',
        on_delete=models.CASCADE,
        related_name='scenario_implementations',
    )
    scenario = models.ForeignKey(
        Scenario,
        on_delete=models.CASCADE,
        related_name='implementations',
    )
    scenario_version = models.ForeignKey(
        ScenarioVersion,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='implementations',
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='active',
    )
    version_confidence = models.CharField(
        max_length=20,
        choices=VERSION_CONFIDENCE_CHOICES,
        default='exact',
    )
    data_quality_status = models.CharField(
        max_length=20,
        choices=DATA_QUALITY_CHOICES,
        default='unreviewed',
    )
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    last_activity = models.ForeignKey(
        Activity,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='implementation_progress',
    )

    class Meta:
        verbose_name = 'Scenario Implementation'
        verbose_name_plural = 'Scenario Implementations'
        ordering = ['-started_at', '-id']
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'scenario'],
                condition=models.Q(status='active'),
                name='unique_active_implementation_per_user_scenario',
            ),
            models.CheckConstraint(
                check=(
                    models.Q(version_confidence='legacy_unknown')
                    | models.Q(scenario_version__isnull=False)
                ),
                name='exact_implementation_requires_version',
            ),
        ]
        indexes = [
            models.Index(
                fields=['scenario', 'scenario_version', 'status'],
                name='impl_scenario_version_idx',
            ),
            models.Index(
                fields=['user', 'scenario', 'status'],
                name='impl_user_scenario_idx',
            ),
            models.Index(
                fields=['data_quality_status', 'version_confidence'],
                name='impl_quality_conf_idx',
            ),
        ]

    def __str__(self):
        return (
            f'{self.user} → {self.scenario} '
            f'({self.get_status_display()})'
        )

    def clean(self):
        super().clean()
        errors = {}
        if (
            self.scenario_version_id
            and self.scenario_version.scenario_id != self.scenario_id
        ):
            errors['scenario_version'] = (
                'The scenario version must belong to the implementation '
                'scenario.'
            )
        if (
            self.last_activity_id
            and self.last_activity.scenario_id != self.scenario_id
        ):
            errors['last_activity'] = (
                'The last activity must belong to the implementation scenario.'
            )
        if (
            self.version_confidence == 'exact'
            and not self.scenario_version_id
        ):
            errors['scenario_version'] = (
                'Exact implementations require a scenario version.'
            )
        if errors:
            raise ValidationError(errors)

    @classmethod
    def start_or_resume(cls, user, scenario):
        """Return the active exact attempt for the current scenario version."""
        version = scenario.ensure_current_version()
        with transaction.atomic():
            active = (
                cls.objects.select_for_update()
                .filter(user=user, scenario=scenario, status='active')
                .first()
            )
            if (
                active
                and active.version_confidence == 'exact'
                and active.scenario_version_id == version.id
            ):
                return active, False
            if active:
                active.status = 'abandoned'
                active.completed_at = timezone.now()
                active.save(update_fields=['status', 'completed_at'])
            return cls.objects.create(
                user=user,
                scenario=scenario,
                scenario_version=version,
                version_confidence='exact',
                data_quality_status='unreviewed',
                status='active',
            ), True

    def complete(self):
        if self.status == 'completed':
            return
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'completed_at'])


class UserScenarioScore(models.Model):
    VERSION_CONFIDENCE_CHOICES = [
        ('exact', 'Exact scenario version'),
        ('legacy_unknown', 'Legacy version unknown'),
    ]
    DATA_QUALITY_CHOICES = [
        ('unreviewed', 'Unreviewed'),
        ('clean', 'Clean'),
        ('suspect', 'Suspect'),
        ('excluded', 'Excluded'),
    ]
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE)
    implementation = models.OneToOneField(
        ScenarioImplementation,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='score',
    )
    scenario_version = models.ForeignKey(
        ScenarioVersion,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='implementation_scores',
    )
    version_confidence = models.CharField(
        max_length=20,
        choices=VERSION_CONFIDENCE_CHOICES,
        default='exact',
    )
    data_quality_status = models.CharField(
        max_length=20,
        choices=DATA_QUALITY_CHOICES,
        default='unreviewed',
    )
    user_score = models.IntegerField(default=0)
    last_activity = models.ForeignKey(Activity, on_delete=models.SET_NULL, null=True, blank=True)
    timeDoingScenario = models.IntegerField(default=0, blank=True, null=True)

    class Meta:
        verbose_name = "User & Scenario Score"
        verbose_name_plural = "User & Scenario Scores"
        ordering = ["user"]
        indexes = [
            models.Index(fields=['user', 'scenario'], name='uss_user_scenario_idx'),
            models.Index(
                fields=['scenario', 'scenario_version', 'version_confidence'],
                name='uss_version_evidence_idx',
            ),
        ]

    def save(self, *args, **kwargs):
        sync_implementation_quality = False
        if (
            not self.implementation_id
            and self.user_id
            and self.scenario_id
        ):
            if self.version_confidence == 'legacy_unknown':
                self.implementation = ScenarioImplementation.objects.create(
                    user_id=self.user_id,
                    scenario_id=self.scenario_id,
                    scenario_version_id=self.scenario_version_id,
                    status='legacy',
                    version_confidence='legacy_unknown',
                    data_quality_status=self.data_quality_status,
                    last_activity_id=self.last_activity_id,
                )
            else:
                implementation, _ = (
                    ScenarioImplementation.start_or_resume(
                        self.user,
                        self.scenario,
                    )
                )
                if not hasattr(implementation, 'score'):
                    self.implementation = implementation
        if self.implementation_id:
            self.user_id = self.implementation.user_id
            self.scenario_id = self.implementation.scenario_id
            self.scenario_version_id = (
                self.implementation.scenario_version_id
            )
            self.version_confidence = (
                self.implementation.version_confidence
            )
            if self.pk:
                sync_implementation_quality = (
                    self.data_quality_status
                    != self.implementation.data_quality_status
                )
            else:
                self.data_quality_status = (
                    self.implementation.data_quality_status
                )
        if (
            not self.scenario_version_id
            and self.version_confidence == 'exact'
            and self.scenario_id
        ):
            self.scenario_version = self.scenario.ensure_current_version()
        super().save(*args, **kwargs)
        if sync_implementation_quality:
            ScenarioImplementation.objects.filter(
                pk=self.implementation_id
            ).update(data_quality_status=self.data_quality_status)
        if (
            self.implementation_id
            and self.implementation.last_activity_id
            != self.last_activity_id
        ):
            ScenarioImplementation.objects.filter(
                pk=self.implementation_id
            ).update(last_activity_id=self.last_activity_id)

    def __str__(self):
        last = self.last_activity.name if self.last_activity else "None"
        return f"{self.user.username} - {self.scenario.name} Score: {self.user_score} Last Activity Answered: {last}"

class UserAnswer(models.Model):
    VERSION_CONFIDENCE_CHOICES = [
        ('exact', 'Exact scenario version'),
        ('legacy_unknown', 'Legacy version unknown'),
    ]
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    activity = models.ForeignKey(Activity, on_delete=models.CASCADE)
    answer = models.ForeignKey(Answer, on_delete=models.SET_NULL, null=True, blank=True)
    implementation = models.ForeignKey(
        ScenarioImplementation,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='answers',
    )
    activity_revision = models.ForeignKey(
        ActivityRevision,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='user_answers',
    )
    scenario_version = models.ForeignKey(
        ScenarioVersion,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='user_answers',
    )
    version_confidence = models.CharField(
        max_length=20,
        choices=VERSION_CONFIDENCE_CHOICES,
        default='exact',
    )
    timing = models.IntegerField(default=0, blank=True, null=True)
    created_on = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "User's Answer"
        verbose_name_plural = "User's Answers"
        ordering = ["user"]
        indexes = [
            # Speeds up get_last_answers() GROUP BY (user_id, activity_id)
            # and per-user per-activity filter lookups in analytics tasks
            models.Index(fields=['user', 'activity'], name='useranswer_user_activity_idx'),
            models.Index(
                fields=['scenario_version', 'version_confidence'],
                name='useranswer_version_idx',
            ),
            models.Index(
                fields=['implementation', 'activity_revision'],
                name='useranswer_impl_actrev_idx',
            ),
        ]

    def save(self, *args, **kwargs):
        if self.implementation_id:
            self.scenario_version_id = (
                self.implementation.scenario_version_id
            )
            self.version_confidence = (
                self.implementation.version_confidence
            )
        if (
            not self.scenario_version_id
            and self.version_confidence == 'exact'
            and self.activity_id
        ):
            self.scenario_version = (
                self.activity.scenario.ensure_current_version()
            )
        if (
            not self.implementation_id
            and self.user_id
            and self.activity_id
            and self.version_confidence == 'exact'
        ):
            self.implementation, _ = ScenarioImplementation.start_or_resume(
                self.user,
                self.activity.scenario,
            )
            self.scenario_version_id = (
                self.implementation.scenario_version_id
            )
        elif (
            not self.implementation_id
            and self.user_id
            and self.activity_id
            and self.version_confidence == 'legacy_unknown'
        ):
            scenario = self.activity.scenario
            self.implementation = (
                ScenarioImplementation.objects
                .filter(
                    user_id=self.user_id,
                    scenario=scenario,
                    version_confidence='legacy_unknown',
                )
                .order_by('-started_at', '-id')
                .first()
            )
            if self.implementation is None:
                self.implementation = ScenarioImplementation.objects.create(
                    user_id=self.user_id,
                    scenario=scenario,
                    status='legacy',
                    version_confidence='legacy_unknown',
                    data_quality_status='unreviewed',
                )
        if (
            not self.activity_revision_id
            and self.scenario_version_id
            and self.activity_id
        ):
            ActivityRevision.capture_for_version(self.scenario_version)
            self.activity_revision = ActivityRevision.objects.filter(
                scenario_version_id=self.scenario_version_id,
                lineage_key=self.activity.lineage_key,
            ).first()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} - {self.activity.name}"
        
class SchoolDepartment(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        verbose_name = "School Department"
        verbose_name_plural = "School Departments"
        ordering = ["id"]

    def __str__(self):
        return self.name
        
class PhetLabSessions(models.Model):
    name = models.CharField(max_length=255, null=False, blank=False, default='Phet Default')
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    activity = models.ForeignKey(Activity, on_delete=models.DO_NOTHING)
    mass_1 = models.FloatField(null=True, blank=True)
    mass_2 = models.FloatField(null=True, blank=True)
    length_1 = models.FloatField(null=True, blank=True)
    length_2 = models.FloatField(null=True, blank=True)
    angle_1 = models.FloatField(null=True, blank=True)
    angle_2 = models.FloatField(null=True, blank=True)
    gravity = models.FloatField()
    friction = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Phet Lab Session"
        verbose_name_plural = "Phet Lab Sessions"
        ordering = ["user"]
    
    def __str__(self):
        return f"{self.name} - {self.user.id} - Activity {self.activity}"

class RemoteLabSession(models.Model):
    activity = models.ForeignKey(Activity, on_delete=models.CASCADE, related_name='lab_sessions')  # Adjusted to refer to Activity instead of Question
    phase = models.ForeignKey(Phase, on_delete=models.CASCADE, related_name='lab_sessions')
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='lab_sessions')
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    start = models.DateTimeField()
    end = models.DateTimeField()
    lab_name = models.CharField(max_length=255)
    pre_duration = models.DurationField()
    exec_duration = models.DurationField()
    mass = models.CharField(max_length=255)
    angle = models.IntegerField()
    iteration = models.IntegerField()

    class Meta:
        verbose_name = "LabsLand Lab Session"
        verbose_name_plural = "LabsLand Lab Sessions"
        ordering = ["id"]

class MultilingualQuestion(models.Model):
    LANGUAGES = [
        ('en', 'English'),
        ('pt', 'Portuguese'),
        ('gr', 'Greek'),
        ('es', 'Spanish'),
        ('fr', 'French'),
        ('de', 'German'),
    ]
    
    # Remove the scenario foreign key since questions will be common
    question_text_en = models.TextField(verbose_name='Question (English)')
    question_text_pt = models.TextField(verbose_name='Question (Portuguese)', blank=True)
    question_text_gr = models.TextField(verbose_name='Question (Greek)', blank=True)
    question_text_es = models.TextField(verbose_name='Question (Spanish)', blank=True)
    question_text_fr = models.TextField(verbose_name='Question (French)', blank=True)
    question_text_de = models.TextField(verbose_name='Question (German)', blank=True)
    is_required = models.BooleanField(default=False)
    order = models.IntegerField(default=0)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='created_multilingual_questions')
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='updated_multilingual_questions')

    class Meta:
        verbose_name = "Multilingual Question"
        verbose_name_plural = "Multilingual Questions"
        ordering = ['order', 'created_on']

    def __str__(self):
        return f"Question: {self.question_text_en[:50]}..."

class MultilingualAnswer(models.Model):
    question = models.ForeignKey(MultilingualQuestion, on_delete=models.CASCADE, related_name='answers')
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='question_answers', null=True, blank=True)
    answer_text = models.TextField()
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='created_multilingual_answers')
    updated_by = models.ForeignKey('auth.User', on_delete=models.SET_DEFAULT, default=1, null=True, related_name='updated_multilingual_answers')

    class Meta:
        verbose_name = "Multilingual Answer"
        verbose_name_plural = "Multilingual Answers"
        ordering = ['-created_on']
        unique_together = ['question', 'user', 'scenario']

    def __str__(self):
        return f"Answer by {self.user.username} for question {self.question.id} in scenario {self.scenario.name}"

class ActivityFlag(models.Model):
    FLAG_CATEGORIES = [
        ('High', 'High'),
        ('Moderate', 'Moderate'),
        ('Low', 'Low')
    ]
    activity = models.ForeignKey(Activity, on_delete=models.CASCADE, related_name='flags')
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, null=True, blank=True)
    phase = models.ForeignKey(Phase, on_delete=models.CASCADE, null=True, blank=True)
    category = models.CharField(max_length=16, choices=FLAG_CATEGORIES)
    flag_type = models.CharField(max_length=128)        # e.g. "Dynamic correctness threshold"
    flag_reason = models.TextField()                    # e.g. "Moderate group 60% wrong > threshold"
    is_at_risk = models.BooleanField(default=True)
    value_at_risk = models.FloatField(null=True, blank=True)    # e.g. the actual %Wrong, AvgTime, etc
    threshold_used = models.FloatField(null=True, blank=True)   # e.g. mean+std threshold
    auto_flagged = models.BooleanField(default=True)            # True=automatic, False=manual
    evidence_scope = models.CharField(
        max_length=20,
        choices=[
            ('local', 'This scenario only'),
            ('compatible', 'Compatible family evidence'),
        ],
        default='local',
    )
    evidence_signature = models.CharField(max_length=64, blank=True)
    evidence_sources = models.JSONField(default=list, blank=True)
    flagged_on = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Activity Flag"
        verbose_name_plural = "Activity Flags"
        indexes = [
            models.Index(fields=["activity", "category"]),
            models.Index(fields=['activity'], name='actflag_activity_idx'),
        ]

    def __str__(self):
        return f"{self.activity} | {self.category} | {self.flag_type}"

class CategoryTag(models.Model):
    name = models.CharField(max_length=16, choices=[
        ('High', 'High'),
        ('Moderate', 'Moderate'),
        ('Low', 'Low'),
    ], unique=True)

    def __str__(self):
        return self.name

REJECTION_REASON_CHOICES = [
    ('wrong_action_create',  'Wrong action — should be Create'),
    ('wrong_action_revise',  'Wrong action — should be Revise'),
    ('wrong_action_skip',    'Wrong action — should be Skip'),
    ('poor_content',         'Poor content quality'),
    ('not_relevant',         'Not relevant to this scenario'),
    ('already_covered',      'Already covered elsewhere'),
    ('difficulty_mismatch',  'Difficulty mismatch'),
    ('structural_invalid',    'Malformed or structurally invalid proposal'),
]

# Rejection reason → (alternative action to nudge, positive reward magnitude)
_REASON_TO_POSITIVE_ACTION = {
    'wrong_action_create': ('create', 0.5),
    'wrong_action_revise': ('revise', 0.5),
    'wrong_action_skip':   ('skip',   0.5),
    'not_relevant':        ('skip',   0.3),
    'already_covered':     ('skip',   0.3),
}

def update_q_value(flag_type, category, action, reward, ALPHA=0.2):
    # Serialize updates to the same bandit arm so concurrent teacher reviews do
    # not overwrite either the learned value or its observation count.
    with transaction.atomic():
        qv, _ = QValue.objects.select_for_update().get_or_create(
            flag_type=flag_type,
            category=category,
            action=action,
            defaults={
                "q_value": 0.0,
                "reward_count": 0,
                "positive_reward_count": 0,
                "negative_reward_count": 0,
                "reward_sum": 0.0,
            }
        )
        old = qv.q_value
        qv.q_value += ALPHA * (reward - qv.q_value)
        qv.reward_count += 1
        qv.reward_sum += reward
        if reward > 0:
            qv.positive_reward_count += 1
        elif reward < 0:
            qv.negative_reward_count += 1
        qv.save(
            update_fields=[
                "q_value",
                "reward_count",
                "positive_reward_count",
                "negative_reward_count",
                "reward_sum",
                "updated_at",
            ]
        )
    print(
        f"[bandit] Q-value updated for ({flag_type}, {category}, {action}): "
        f"{old:.2f} -> {qv.q_value:.2f}"
    )


def rebuild_q_values_for_context(flag_type, category, ALPHA=0.2):
    """Rebuild one context from current pedagogical review decisions."""
    rewards_by_action = {
        "create": [],
        "revise": [],
        "skip": [],
    }
    reviews = (
        UserProposalReview.objects.filter(
            status__in=["accepted", "rejected"],
            feedback_type="pedagogical",
            proposal__flag__flag_type=flag_type,
            proposal__flag__category=category,
        )
        .select_related("proposal")
        .order_by("reviewed_at", "id")
        .distinct()
    )
    for review in reviews:
        proposal = review.proposal
        if not proposal.is_bandit_reward_eligible():
            continue
        rewards_by_action[proposal.proposal_type].append(
            1.0 if review.status == "accepted" else -1.0
        )
        if review.status == "rejected":
            for reason in review.rejection_reasons or []:
                nudge = _REASON_TO_POSITIVE_ACTION.get(reason)
                if nudge:
                    action, reward = nudge
                    rewards_by_action[action].append(reward)

    with transaction.atomic():
        for action, rewards in rewards_by_action.items():
            qv = QValue.objects.select_for_update().filter(
                flag_type=flag_type,
                category=category,
                action=action,
            ).first()
            if not rewards and not qv:
                continue
            if not qv:
                qv = QValue(
                    flag_type=flag_type,
                    category=category,
                    action=action,
                )

            q_value = 0.0
            for reward in rewards:
                q_value += ALPHA * (reward - q_value)
            qv.q_value = q_value
            qv.reward_count = len(rewards)
            qv.positive_reward_count = sum(
                1 for reward in rewards if reward > 0
            )
            qv.negative_reward_count = sum(
                1 for reward in rewards if reward < 0
            )
            qv.reward_sum = sum(rewards)
            qv.save()

class ProposalGenerationRun(models.Model):
    scenario = models.ForeignKey('Scenario', on_delete=models.CASCADE, related_name='proposal_generation_runs')
    scenario_version = models.ForeignKey(
        'ScenarioVersion',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='proposal_generation_runs',
    )
    evidence_scope = models.CharField(
        max_length=20,
        choices=[
            ('local', 'This scenario only'),
            ('compatible', 'Compatible family evidence'),
        ],
        default='local',
    )
    evidence_version_ids = models.JSONField(default=list, blank=True)
    evidence_summary = models.JSONField(default=dict, blank=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, related_name='triggered_proposal_generation_runs')
    created_at = models.DateTimeField(auto_now_add=True)
    is_current = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Proposal Generation Run"
        verbose_name_plural = "Proposal Generation Runs"
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['scenario'],
                condition=models.Q(is_current=True),
                name='unique_current_run_per_scenario',
            ),
        ]

    def __str__(self):
        status = 'current' if self.is_current else 'archived'
        return f"Run for '{self.scenario.name}' @ {self.created_at:%Y-%m-%d %H:%M} ({status})"

    @classmethod
    def start_new(
        cls,
        scenario,
        created_by,
        scenario_version=None,
        evidence_scope='local',
        evidence_version_ids=None,
        evidence_summary=None,
    ):
        """Archive the scenario's current run (if any) and start a new current one, atomically."""
        with transaction.atomic():
            scenario_version = (
                scenario_version
                or scenario.ensure_current_version(created_by=created_by)
            )
            cls.objects.filter(scenario=scenario, is_current=True).update(is_current=False)
            return cls.objects.create(
                scenario=scenario,
                scenario_version=scenario_version,
                created_by=created_by,
                is_current=True,
                evidence_scope=evidence_scope,
                evidence_version_ids=evidence_version_ids or [],
                evidence_summary=evidence_summary or {},
            )


class ActivityProposal(models.Model):
    STATUS_CHOICES = [
        ('new', 'New'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    ]
    PROPOSAL_TYPE_CHOICES = [
        ('create', 'Create New Activity'),
        ('revise', 'Revise Activity'),
        ('skip', 'Skip Activity'),
    ]

    scenario = models.ForeignKey('Scenario', on_delete=models.CASCADE, related_name='proposals', db_index=True)
    generation_run = models.ForeignKey(
        'ProposalGenerationRun', on_delete=models.CASCADE, null=True, blank=True, related_name='proposals',
    )
    phase = models.ForeignKey('Phase', on_delete=models.CASCADE, related_name='proposals')
    activity = models.ForeignKey('Activity', on_delete=models.CASCADE, related_name='proposals')
    flag = models.ManyToManyField('ActivityFlag', blank=True, related_name='proposals')
    categories_in_risk = models.ManyToManyField("CategoryTag", related_name='proposals')
    proposal_type = models.CharField(max_length=32, choices=PROPOSAL_TYPE_CHOICES)
    suggested_action = models.TextField()
    translated_action = models.TextField()
    json_action = models.TextField()
    json_translated_action = models.TextField()
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='new')
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewer = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, blank=True, help_text="Teacher who accepted/rejected")

    class Meta:
        verbose_name = "Activity Proposal"
        verbose_name_plural = "Activity Proposals"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_proposal_type_display()} for '{self.activity}' ({self.get_status_display()})"

    def is_bandit_reward_eligible(self):
        """Only current, compatible, provenance-valid runs may teach bandit."""
        run = self.generation_run
        if (
            run is None
            or not run.is_current
            or run.evidence_scope != 'compatible'
            or run.scenario_version_id is None
            or run.scenario.current_version_id != run.scenario_version_id
        ):
            return False
        from .evidence import get_evidence_context
        current_context = get_evidence_context(
            run.scenario,
            scope='compatible',
        )
        expected_source_signature = (
            run.evidence_summary or {}
        ).get('source_signature')
        return (
            bool(expected_source_signature)
            and expected_source_signature
            == current_context.get('source_signature')
            and sorted(run.evidence_version_ids or [])
            == sorted(current_context.get('version_ids') or [])
        )

    # def accept(self, reviewer):
    #     self.status = 'accepted'
    #     self.reviewed_at = timezone.now()
    #     self.reviewer = reviewer
    #     self.save()

    #     # Q-learning update
    #     if not self.flag.exists():
    #         print(f"No flags associated with proposal {self.id}. Q-value update skipped.")
    #     # for flag in self.flag.all():
    #     #     update_q_value(flag.flag_type, flag.category, self.proposal_type, reward=1)
    #     def update_after_commit():
    #         for flag in self.flag.all():
    #             update_q_value(flag.flag_type, flag.category, self.proposal_type, reward=1)
    #     transaction.on_commit(update_after_commit)

    # def reject(self, reviewer):
    #     self.status = 'rejected'
    #     self.reviewed_at = timezone.now()
    #     self.reviewer = reviewer
    #     self.save()

    #     # Q-learning update
    #     if not self.flag.exists():
    #         print(f"No flags associated with proposal {self.id}. Q-value update skipped.")
    #     # for flag in self.flag.all():
    #     #     update_q_value(flag.flag_type, flag.category, self.proposal_type, reward=-1)
    #     def update_after_commit():
    #         for flag in self.flag.all():
    #             update_q_value(flag.flag_type, flag.category, self.proposal_type, reward=-1)
    #     transaction.on_commit(update_after_commit)

class QValue(models.Model):
    FLAG_TYPES = [
        ("Dynamic correctness threshold", "Dynamic correctness threshold"),
        ("Extreme correctness pattern", "Extreme correctness pattern"),
        ("Engagement risk (too slow, question)", "Engagement risk (too slow, question)"),
        ("Engagement risk (too fast, question)", "Engagement risk (too fast, question)"),
        ("Engagement risk (too slow, non-question)", "Engagement risk (too slow, non-question)"),
        ("Engagement risk (too fast, non-question)", "Engagement risk (too fast, non-question)"),
        ("Timing discrepancy under extreme correctness", "Timing discrepancy under extreme correctness"),
        ("Systemic failure", "Systemic failure"),
    ]
    CATEGORIES = [
        ("High", "High"),
        ("Moderate", "Moderate"),
        ("Low", "Low"),
    ]
    ACTIONS = [
        ("create", "Create"),
        ("revise", "Revise"),
        ("skip", "Skip"),
    ]

    flag_type = models.CharField(max_length=128, choices=FLAG_TYPES)
    category = models.CharField(max_length=16, choices=CATEGORIES)
    action = models.CharField(max_length=16, choices=ACTIONS)
    q_value = models.FloatField(default=0.0)
    reward_count = models.PositiveIntegerField(default=0)
    positive_reward_count = models.PositiveIntegerField(default=0)
    negative_reward_count = models.PositiveIntegerField(default=0)
    reward_sum = models.FloatField(default=0.0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("flag_type", "category", "action")

    def __str__(self):
        return f"{self.flag_type} | {self.category} | {self.action} → {self.q_value:.2f}"
    
class BanditPolicyConfiguration(models.Model):
    POLICY_CHOICES = [
        ('thompson', 'Thompson Sampling'),
        ('ucb', 'Upper Confidence Bound (UCB)'),
    ]

    name = models.CharField(max_length=100, default='Default policy', unique=True)
    is_active = models.BooleanField(default=True)
    policy = models.CharField(
        max_length=20,
        choices=POLICY_CHOICES,
        default='thompson',
    )
    minimum_context_rewards = models.PositiveIntegerField(
        default=200,
        help_text=(
            "Use weighted cold-start exploration until this many rewards "
            "exist for the flag type/category context."
        ),
    )
    create_weight = models.FloatField(default=0.50)
    skip_weight = models.FloatField(default=0.30)
    revise_weight = models.FloatField(default=0.20)
    thompson_prior_alpha = models.FloatField(default=1.0)
    thompson_prior_beta = models.FloatField(default=1.0)
    ucb_exploration_strength = models.FloatField(default=1.41421356237)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Bandit Policy Configuration'
        verbose_name_plural = 'Bandit Policy Configuration'

    def __str__(self):
        status = 'active' if self.is_active else 'inactive'
        return f"{self.name}: {self.get_policy_display()} ({status})"

    @classmethod
    def get_active(cls):
        return cls.objects.filter(is_active=True).first() or cls()

    def clean(self):
        super().clean()
        weights = (
            self.create_weight,
            self.skip_weight,
            self.revise_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValidationError(
                'Cold-start weights must be non-negative and total above zero.'
            )
        if self.thompson_prior_alpha <= 0 or self.thompson_prior_beta <= 0:
            raise ValidationError(
                'Thompson prior alpha and beta must be greater than zero.'
            )
        if self.ucb_exploration_strength < 0:
            raise ValidationError(
                'UCB exploration strength cannot be negative.'
            )


class UserProposalReview(models.Model):
    proposal = models.ForeignKey('ActivityProposal', on_delete=models.CASCADE, related_name='user_reviews')
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    status = models.CharField(
        max_length=16,
        choices=[
            ('new', 'Pending'),
            ('accepted', 'Accepted'),
            ('rejected', 'Rejected')
        ],
        default='new'
    )
    reviewed_at = models.DateTimeField(auto_now=True)
    teacher_edited_json = models.JSONField(null=True, blank=True)
    rejection_reasons = models.JSONField(default=list, blank=True)
    feedback_type = models.CharField(
        max_length=16,
        choices=[
            ('pedagogical', 'Pedagogical'),
            ('structural', 'Structural / malformed'),
        ],
        default='pedagogical',
        help_text=(
            "Structural feedback is tracked separately and never updates "
            "the learning bandit."
        ),
    )
    was_edited = models.BooleanField(default=False)
    edit_count = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ('proposal', 'user')  # one review per user per proposal

    def accept(self):
        """Transition any status -> ACCEPTED. Q-update (+1) handled by post_save signal."""
        if self.status == 'accepted':
            return
        self.status = 'accepted'
        self.feedback_type = 'pedagogical'
        self.rejection_reasons = []
        self.save(
            update_fields=[
                'status',
                'reviewed_at',
                'feedback_type',
                'rejection_reasons',
            ]
        )

    def reject(self, reasons=None):
        """Transition any status -> REJECTED, recording reasons.
        Q-update (−1 on chosen action, positive nudges per reason) handled by post_save signal."""
        if self.status == 'rejected':
            return
        self.rejection_reasons = reasons or []
        self.feedback_type = (
            'structural'
            if 'structural_invalid' in self.rejection_reasons
            else 'pedagogical'
        )
        self.status = 'rejected'
        self.save(
            update_fields=[
                'status',
                'reviewed_at',
                'rejection_reasons',
                'feedback_type',
            ]
        )

    def reset_to_pending(self):
        """Undo a decision without turning it into the opposite decision."""
        if self.status == 'new':
            return
        self.status = 'new'
        self.rejection_reasons = []
        self.feedback_type = 'pedagogical'
        self.save(
            update_fields=[
                'status',
                'reviewed_at',
                'rejection_reasons',
                'feedback_type',
            ]
        )


class ProposalStructuralFailure(models.Model):
    STAGE_CHOICES = [
        ('generation', 'LLM generation'),
        ('translation', 'Translation'),
        ('acceptance', 'Teacher acceptance'),
        ('teacher_review', 'Teacher review'),
        ('application', 'Proposal application'),
        ('graph_integrity', 'Scenario graph integrity'),
    ]

    scenario = models.ForeignKey(
        'Scenario',
        on_delete=models.CASCADE,
        related_name='proposal_structural_failures',
    )
    generation_run = models.ForeignKey(
        'ProposalGenerationRun',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='structural_failures',
    )
    proposal = models.ForeignKey(
        'ActivityProposal',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='structural_failures',
    )
    activity = models.ForeignKey(
        'Activity',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='proposal_structural_failures',
    )
    selected_action = models.CharField(max_length=16, blank=True)
    stage = models.CharField(max_length=24, choices=STAGE_CHOICES)
    errors = models.JSONField(default=list)
    raw_output = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Proposal Structural Failure'
        verbose_name_plural = 'Proposal Structural Failures'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_stage_display()} failure for scenario {self.scenario_id}"


class ActivityProposalEditEvent(models.Model):
    review = models.ForeignKey(
        'UserProposalReview', on_delete=models.CASCADE, related_name='edit_events'
    )
    edit_number = models.PositiveIntegerField()
    edited_json = models.JSONField()
    changed_fields = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Activity Proposal Edit Event"
        verbose_name_plural = "Activity Proposal Edit Events"
        ordering = ['review', 'edit_number']
        constraints = [
            models.UniqueConstraint(
                fields=['review', 'edit_number'], name='unique_review_edit_number'
            )
        ]

    def __str__(self):
        return f"Edit #{self.edit_number} on review {self.review_id}"

# ─────────────────────────────────────────────────────────────────────────────
# Signals: ensure Q-value updates even if status set directly in admin/UI
# ─────────────────────────────────────────────────────────────────────────────

@receiver(post_save, sender=Activity)
def _initialize_scenario_start_activity(sender, instance, created, **kwargs):
    if created and instance.scenario_id:
        Scenario.objects.filter(
            pk=instance.scenario_id,
            start_activity__isnull=True,
        ).update(start_activity=instance)


@receiver(post_save, sender=Scenario)
def _initialize_scenario_family(sender, instance, created, **kwargs):
    if not created or instance.family_id:
        return
    instance.ensure_family()


@receiver(post_delete, sender=Scenario)
def _repair_scenario_family_after_delete(sender, instance, **kwargs):
    """Keep each surviving family anchored to one canonical scenario."""
    if not instance.family_id:
        return

    family = ScenarioFamily.objects.filter(pk=instance.family_id).first()
    if not family:
        return

    replacement = (
        family.scenarios.filter(variant_type='canonical')
        .order_by('created_on', 'id')
        .first()
        or family.scenarios.order_by('created_on', 'id').first()
    )
    if replacement is None:
        family.delete()
        return

    if family.canonical_scenario_id != replacement.id:
        family.scenarios.exclude(pk=replacement.pk).filter(
            variant_type='canonical'
        ).update(variant_type='adaptation')
        if replacement.variant_type != 'canonical':
            Scenario.objects.filter(pk=replacement.pk).update(
                variant_type='canonical'
            )
        family.canonical_scenario = replacement
        family.save(update_fields=['canonical_scenario', 'updated_on'])
        family.subjects.set(replacement.subjects.all())


@receiver(m2m_changed, sender=Scenario.subjects.through)
def _sync_canonical_scenario_family_subjects(
    sender,
    instance,
    action,
    **kwargs,
):
    if action not in {'post_add', 'post_remove', 'post_clear'}:
        return
    if (
        instance.family_id
        and instance.family.canonical_scenario_id == instance.id
    ):
        instance.family.subjects.set(instance.subjects.all())


@receiver(post_delete, sender=Activity)
def _replace_deleted_scenario_start_activity(sender, instance, **kwargs):
    if not instance.scenario_id:
        return
    scenario = Scenario.objects.filter(pk=instance.scenario_id).first()
    if not scenario or scenario.start_activity_id:
        return
    replacement_id = (
        Activity.objects.filter(scenario_id=instance.scenario_id)
        .order_by('id')
        .values_list('id', flat=True)
        .first()
    )
    if replacement_id:
        Scenario.objects.filter(pk=instance.scenario_id).update(
            start_activity_id=replacement_id
        )


@receiver(pre_save, sender=UserProposalReview)
def _cache_old_status(sender, instance, **kwargs):
    if not instance.pk:
        instance._old_status = None
        return
    try:
        instance._old_status = sender.objects.get(pk=instance.pk).status
    except sender.DoesNotExist:
        instance._old_status = None


@receiver(post_save, sender=UserProposalReview)
def _reward_on_review_status_change(sender, instance, created, **kwargs):
    # Rebuild contexts so undo-to-pending removes superseded rewards.
    # Covers: new→accepted, new→rejected, accepted→rejected, rejected→accepted.
    old = getattr(instance, '_old_status', None)
    new = instance.status
    if created and new == 'new':
        return
    if old == new or new not in ('new', 'accepted', 'rejected'):
        return

    prop = instance.proposal
    contexts = {
        (flag.flag_type, flag.category)
        for flag in prop.flag.all()
    }
    if not contexts:
        print(f"No flags for proposal {prop.id}; Q update skipped.")
        return

    def _on_commit():
        for flag_type, category in contexts:
            rebuild_q_values_for_context(flag_type, category)

    transaction.on_commit(_on_commit)

User.add_to_class('school_department', models.ForeignKey(SchoolDepartment, on_delete=models.SET_NULL, null=True, blank=True))
