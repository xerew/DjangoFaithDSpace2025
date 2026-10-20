from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.utils import timezone

COUNTRY_CHOICES = [
    ('', '— Select country —'),
    ('Afghanistan', 'Afghanistan'), ('Albania', 'Albania'), ('Algeria', 'Algeria'),
    ('Argentina', 'Argentina'), ('Armenia', 'Armenia'), ('Australia', 'Australia'),
    ('Austria', 'Austria'), ('Azerbaijan', 'Azerbaijan'), ('Bangladesh', 'Bangladesh'),
    ('Belarus', 'Belarus'), ('Belgium', 'Belgium'), ('Bolivia', 'Bolivia'),
    ('Bosnia and Herzegovina', 'Bosnia and Herzegovina'), ('Brazil', 'Brazil'),
    ('Bulgaria', 'Bulgaria'), ('Cambodia', 'Cambodia'), ('Canada', 'Canada'),
    ('Chile', 'Chile'), ('China', 'China'), ('Colombia', 'Colombia'),
    ('Croatia', 'Croatia'), ('Cyprus', 'Cyprus'), ('Czech Republic', 'Czech Republic'),
    ('Denmark', 'Denmark'), ('Ecuador', 'Ecuador'), ('Egypt', 'Egypt'),
    ('Estonia', 'Estonia'), ('Ethiopia', 'Ethiopia'), ('Finland', 'Finland'),
    ('France', 'France'), ('Georgia', 'Georgia'), ('Germany', 'Germany'),
    ('Ghana', 'Ghana'), ('Greece', 'Greece'), ('Guatemala', 'Guatemala'),
    ('Hungary', 'Hungary'), ('India', 'India'), ('Indonesia', 'Indonesia'),
    ('Iran', 'Iran'), ('Iraq', 'Iraq'), ('Ireland', 'Ireland'),
    ('Israel', 'Israel'), ('Italy', 'Italy'), ('Japan', 'Japan'),
    ('Jordan', 'Jordan'), ('Kazakhstan', 'Kazakhstan'), ('Kenya', 'Kenya'),
    ('Kosovo', 'Kosovo'), ('Latvia', 'Latvia'), ('Lebanon', 'Lebanon'),
    ('Lithuania', 'Lithuania'), ('Luxembourg', 'Luxembourg'), ('Malaysia', 'Malaysia'),
    ('Mexico', 'Mexico'), ('Moldova', 'Moldova'), ('Morocco', 'Morocco'),
    ('Netherlands', 'Netherlands'), ('New Zealand', 'New Zealand'), ('Nigeria', 'Nigeria'),
    ('North Macedonia', 'North Macedonia'), ('Norway', 'Norway'), ('Pakistan', 'Pakistan'),
    ('Palestine', 'Palestine'), ('Peru', 'Peru'), ('Philippines', 'Philippines'),
    ('Poland', 'Poland'), ('Portugal', 'Portugal'), ('Romania', 'Romania'),
    ('Russia', 'Russia'), ('Saudi Arabia', 'Saudi Arabia'), ('Serbia', 'Serbia'),
    ('Singapore', 'Singapore'), ('Slovakia', 'Slovakia'), ('Slovenia', 'Slovenia'),
    ('South Africa', 'South Africa'), ('South Korea', 'South Korea'), ('Spain', 'Spain'),
    ('Sri Lanka', 'Sri Lanka'), ('Sweden', 'Sweden'), ('Switzerland', 'Switzerland'),
    ('Syria', 'Syria'), ('Taiwan', 'Taiwan'), ('Thailand', 'Thailand'),
    ('Tunisia', 'Tunisia'), ('Turkey', 'Turkey'), ('Ukraine', 'Ukraine'),
    ('United Arab Emirates', 'United Arab Emirates'), ('United Kingdom', 'United Kingdom'),
    ('United States', 'United States'), ('Uruguay', 'Uruguay'), ('Uzbekistan', 'Uzbekistan'),
    ('Venezuela', 'Venezuela'), ('Vietnam', 'Vietnam'), ('Yemen', 'Yemen'),
    ('Zimbabwe', 'Zimbabwe'),
]


GENDER_CHOICES = [
    ('', '— Prefer not to say —'),
    ('male', 'Male'),
    ('female', 'Female'),
]


class UserProfile(models.Model):
    user        = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    country     = models.CharField(max_length=100, blank=True, choices=COUNTRY_CHOICES)
    institution = models.CharField(max_length=255, blank=True, help_text="School or university you work at")
    bio         = models.TextField(max_length=500, blank=True, help_text="A short bio (max 500 characters)")
    gender      = models.CharField(max_length=10, blank=True, choices=GENDER_CHOICES)
    picture     = models.ImageField(upload_to='profile_pictures', null=True, blank=True)

    def __str__(self):
        return f"Profile of {self.user.username}"


class MaintenanceNoticeQuerySet(models.QuerySet):
    def active(self, at=None):
        at = at or timezone.now()
        return self.filter(
            is_enabled=True,
            starts_at__lte=at,
            ends_at__gt=at,
        )


class MaintenanceNotice(models.Model):
    reason = models.TextField(
        help_text='Explain why the platform is undergoing maintenance.',
    )
    starts_at = models.DateTimeField(
        'start date and time',
        db_index=True,
    )
    ends_at = models.DateTimeField(
        'end date and time',
        db_index=True,
    )
    is_enabled = models.BooleanField(
        default=True,
        db_index=True,
        help_text='Disable this notice without deleting its schedule.',
    )
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='maintenance_notices_created',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = MaintenanceNoticeQuerySet.as_manager()

    class Meta:
        ordering = ['-starts_at', '-id']
        indexes = [
            models.Index(
                fields=['is_enabled', 'starts_at', 'ends_at'],
                name='maintenance_active_window_idx',
            ),
        ]

    def __str__(self):
        return f'Maintenance from {self.starts_at:%Y-%m-%d %H:%M}'

    def clean(self):
        super().clean()
        if self.starts_at and self.ends_at and self.ends_at <= self.starts_at:
            raise ValidationError({
                'ends_at': 'The end date and time must be after the start.',
            })

    def is_active(self, at=None):
        at = at or timezone.now()
        return self.is_enabled and self.starts_at <= at < self.ends_at

    def state(self, at=None):
        at = at or timezone.now()
        if not self.is_enabled:
            return 'disabled'
        if at < self.starts_at:
            return 'scheduled'
        if at >= self.ends_at:
            return 'ended'
        return 'active'


class BulkEmailCampaign(models.Model):
    TARGET_ALL_TEACHERS = 'all_teachers'
    TARGET_SELECTED_TEACHERS = 'selected_teachers'
    TARGET_ORGANIZATIONS = 'organizations'
    TARGET_TYPE_CHOICES = [
        (TARGET_ALL_TEACHERS, 'All teachers'),
        (TARGET_SELECTED_TEACHERS, 'Selected teachers'),
        (TARGET_ORGANIZATIONS, 'Selected organizations'),
    ]

    STATUS_QUEUED = 'queued'
    STATUS_SENDING = 'sending'
    STATUS_COMPLETED = 'completed'
    STATUS_PARTIAL = 'partial'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_QUEUED, 'Queued'),
        (STATUS_SENDING, 'Sending'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_PARTIAL, 'Completed with errors'),
        (STATUS_FAILED, 'Failed'),
    ]

    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_bulk_email_campaigns',
    )
    target_type = models.CharField(max_length=30, choices=TARGET_TYPE_CHOICES)
    organizations = models.ManyToManyField(
        'organization.Organization',
        blank=True,
        related_name='bulk_email_campaigns',
    )
    recipients = models.ManyToManyField(
        User,
        blank=True,
        related_name='bulk_email_campaigns',
    )
    subject = models.CharField(max_length=200)
    body_html = models.TextField()
    site_url = models.URLField(blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_QUEUED,
    )
    recipient_count = models.PositiveIntegerField(default=0)
    sent_count = models.PositiveIntegerField(default=0)
    failed_count = models.PositiveIntegerField(default=0)
    error_summary = models.TextField(blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at', '-id']
        indexes = [
            models.Index(fields=['status', 'created_at'], name='bulk_email_status_created_idx'),
        ]

    def __str__(self):
        return f'{self.subject} ({self.get_status_display()})'

    @property
    def target_summary(self):
        if self.target_type == self.TARGET_ALL_TEACHERS:
            return 'All teachers'
        if self.target_type == self.TARGET_SELECTED_TEACHERS:
            return f'{self.recipient_count} selected teacher(s)'
        organizations = list(self.organizations.all())
        names = [organization.short_name for organization in organizations[:3]]
        suffix = '…' if len(organizations) > 3 else ''
        return f"Organizations: {', '.join(names)}{suffix}"

    @property
    def waiting_count(self):
        """Recipients not yet sent to or failed (waiting or being sent now)."""
        return self.deliveries.filter(status__in=BulkEmailDelivery.PENDING_STATUSES).count()

    def refresh_counts(self):
        """Recompute counts and status from the campaign's delivery rows."""
        counts = dict(
            self.deliveries.values_list('status').annotate(n=models.Count('id'))
        )
        total = sum(counts.values())
        sent = counts.get(BulkEmailDelivery.STATUS_SENT, 0)
        failed = counts.get(BulkEmailDelivery.STATUS_FAILED, 0)
        pending = total - sent - failed

        if pending == total:
            self.status = self.STATUS_QUEUED
        elif pending:
            self.status = self.STATUS_SENDING
        elif not failed:
            self.status = self.STATUS_COMPLETED
        elif sent:
            self.status = self.STATUS_PARTIAL
        else:
            self.status = self.STATUS_FAILED

        self.sent_count = sent
        self.failed_count = failed
        if sent and self.started_at is None:
            self.started_at = timezone.now()
        if pending:
            self.completed_at = None
        elif self.completed_at is None:
            self.completed_at = timezone.now()
        self.save(update_fields=[
            'status', 'sent_count', 'failed_count', 'started_at', 'completed_at',
        ])


class BulkEmailDelivery(models.Model):
    """One campaign email to one recipient, sent by the rate-limited outbox."""

    STATUS_WAITING = 'waiting'
    STATUS_SENDING = 'sending'
    STATUS_SENT = 'sent'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_WAITING, 'Waiting'),
        (STATUS_SENDING, 'Sending'),
        (STATUS_SENT, 'Sent'),
        (STATUS_FAILED, 'Failed'),
    ]
    PENDING_STATUSES = (STATUS_WAITING, STATUS_SENDING)

    campaign = models.ForeignKey(
        BulkEmailCampaign,
        on_delete=models.CASCADE,
        related_name='deliveries',
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='bulk_email_deliveries',
    )
    # Copied when queued: the address and name the admin confirmed.
    email = models.EmailField()
    recipient_name = models.CharField(max_length=150, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_WAITING)
    error = models.TextField(blank=True)
    attempts = models.PositiveIntegerField(default=0)
    claimed_at = models.DateTimeField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    # True for rows rebuilt from the totals of a campaign sent before the outbox.
    reconstructed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['id']
        constraints = [
            models.UniqueConstraint(
                fields=['campaign', 'user'],
                condition=models.Q(user__isnull=False),
                name='bulk_email_delivery_unique_user',
            ),
        ]
        indexes = [
            models.Index(fields=['status', 'sent_at'], name='bulk_delivery_status_sent_idx'),
            models.Index(fields=['campaign', 'status'], name='bulk_delivery_campaign_idx'),
        ]

    def __str__(self):
        return f'{self.email} ({self.status})'


class EmailOutboxState(models.Model):
    """Singleton row: whether campaign sending is paused, and why."""

    paused_until = models.DateTimeField(null=True, blank=True)
    pause_reason = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def load(cls):
        state, _ = cls.objects.get_or_create(pk=1)
        return state

    def is_paused(self, now):
        return self.paused_until is not None and self.paused_until > now
