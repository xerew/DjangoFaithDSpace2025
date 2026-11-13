"""Background jobs of the admin-only Student categorization page.

TrainingRun: the XGBoost categorization models, trained on a Celery worker.
ChatbotEvaluation: cross-validation of the chatbot's intents, run by the
`rasa-eval` container. Both keep everything the page shows, so earlier
runs stay viewable.
"""

from datetime import timedelta

from django.conf import settings
from django.db import models, transaction
from django.utils import timezone


class TrainingAlreadyRunning(Exception):
    """Another training run is queued or running."""


class EvaluationAlreadyRunning(Exception):
    """Another chatbot evaluation is queued or running."""


STATUS_CHOICES = [
    ('queued', 'Queued'),
    ('running', 'Running'),
    ('finished', 'Finished'),
    ('failed', 'Failed'),
]


class OneAtATime:
    """One queued or running job of a kind; lost jobs expire.

    A job this old that never finished was lost (worker restart); it must
    not keep the start button disabled forever.
    """

    ACTIVE = ('queued', 'running')
    STALE_ERROR = 'Stopped without finishing (the worker may have restarted).'

    @classmethod
    def expire_stale(cls):
        """Mark jobs that were lost (queued or running too long) as failed."""
        cls.objects.filter(
            status__in=cls.ACTIVE, created_at__lt=timezone.now() - cls.STALE_AFTER,
        ).update(status='failed', error=cls.STALE_ERROR, finished_at=timezone.now())

    @classmethod
    def start(cls, user, **fields):
        """Queue a new job, unless one is already queued or running."""
        with transaction.atomic():
            cls.expire_stale()
            if cls.objects.select_for_update().filter(status__in=cls.ACTIVE).exists():
                raise cls.AlreadyRunning()
            return cls.objects.create(started_by=user, status='queued', **fields)

    def set_progress(self, text):
        self.progress = text[:255]
        self.save(update_fields=['progress'])


class TrainingRun(OneAtATime, models.Model):
    STATUS_CHOICES = STATUS_CHOICES
    AlreadyRunning = TrainingAlreadyRunning
    STALE_AFTER = timedelta(hours=6)

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



class ChatbotEvaluation(OneAtATime, models.Model):
    """`rasa test nlu --cross-validation` on the chatbot's training sentences.

    Queued here by an admin; the `rasa-eval` container claims it, runs Rasa
    on a copy of `data/nlu.yml` and `config.yml`, and writes Rasa's result
    files back into this row (see deploy/rasa-eval/worker.py, which writes
    to this table directly).
    """

    STATUS_CHOICES = STATUS_CHOICES
    FOLD_CHOICES = (3, 5)
    AlreadyRunning = EvaluationAlreadyRunning
    # A 3-fold run took about 30 minutes; 5 folds take longer.
    STALE_AFTER = timedelta(hours=3)

    started_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chatbot_evaluations',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='queued')
    progress = models.CharField(max_length=255, blank=True)
    error = models.TextField(blank=True)
    folds = models.PositiveSmallIntegerField(default=3)
    # The training sentences the result was measured on.
    nlu_text = models.TextField(blank=True)
    nlu_sha256 = models.CharField(max_length=64, blank=True)
    rasa_version = models.CharField(max_length=20, blank=True)
    fallback_threshold = models.FloatField(null=True, blank=True)
    # Rasa's intent_report.json, intent_errors.json and intent_successes.json.
    report = models.JSONField(default=dict, blank=True)
    errors = models.JSONField(default=list, blank=True)
    successes = models.JSONField(default=list, blank=True)
    # The end of Rasa's console output: per-fold mean (std) lines, or why it failed.
    log_tail = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at', '-id']

    def __str__(self):
        return f'Chatbot evaluation {self.pk} ({self.status})'
