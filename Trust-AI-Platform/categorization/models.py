"""Training runs of the admin-only XGBoost categorization models.

A run is started by an admin, trained on a Celery worker, and keeps
everything the dashboard shows in `results`, so earlier runs stay viewable.
"""

from datetime import timedelta

from django.conf import settings
from django.db import models, transaction
from django.utils import timezone


class TrainingAlreadyRunning(Exception):
    """Another training run is queued or running."""


class TrainingRun(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('finished', 'Finished'),
        ('failed', 'Failed'),
    ]
    ACTIVE = ('queued', 'running')
    # A run this old that never finished was lost (worker restart); it must
    # not keep the Train button disabled forever.
    STALE_AFTER = timedelta(hours=6)
    STALE_ERROR = 'Stopped without finishing (the worker may have restarted).'

    started_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='categorization_training_runs',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='queued')
    progress = models.CharField(max_length=255, blank=True)
    error = models.TextField(blank=True)
    versions = models.JSONField(default=dict, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    counts = models.JSONField(default=dict, blank=True)
    results = models.JSONField(default=dict, blank=True)
    final_model_path = models.CharField(max_length=500, blank=True)
    early_model_path = models.CharField(max_length=500, blank=True)

    class Meta:
        ordering = ['-created_at', '-id']

    def __str__(self):
        return f'Training run {self.pk} ({self.status})'

    @classmethod
    def expire_stale(cls):
        """Mark runs that were lost (queued or running too long) as failed."""
        cls.objects.filter(
            status__in=cls.ACTIVE, created_at__lt=timezone.now() - cls.STALE_AFTER,
        ).update(status='failed', error=cls.STALE_ERROR, finished_at=timezone.now())

    @classmethod
    def start(cls, user):
        """Queue a new run, unless one is already queued or running."""
        with transaction.atomic():
            cls.expire_stale()
            if cls.objects.select_for_update().filter(status__in=cls.ACTIVE).exists():
                raise TrainingAlreadyRunning()
            return cls.objects.create(started_by=user, status='queued')

    def set_progress(self, text):
        self.progress = text[:255]
        self.save(update_fields=['progress'])
