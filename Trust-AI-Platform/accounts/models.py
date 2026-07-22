from django.db import models
from django.contrib.auth.models import User

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
