from django.contrib.auth.models import User
from django.db import models


class FeedbackForm(models.Model):
    AUDIENCE_CHOICES = [('teacher', 'Teacher'), ('student', 'Student')]

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    audience = models.CharField(max_length=16, choices=AUDIENCE_CHOICES)
    is_active = models.BooleanField(default=True)
    assign_to_all = models.BooleanField(default=True)
    included_scenarios = models.ManyToManyField('authoringtool.Scenario', blank=True, related_name='included_feedback_forms')
    excluded_scenarios = models.ManyToManyField('authoringtool.Scenario', blank=True, related_name='excluded_feedback_forms')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='feedback_forms')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_on', '-pk']

    def __str__(self):
        return f"{self.title} ({self.get_audience_display()})"

    def applies_to(self, scenario):
        if not self.is_active:
            return False
        if self.assign_to_all:
            return not self.excluded_scenarios.filter(pk=scenario.pk).exists()
        return self.included_scenarios.filter(pk=scenario.pk).exists()


class FeedbackQuestion(models.Model):
    TYPE_CHOICES = [('choice', 'Multiple Choice'), ('text', 'Free Text')]

    form = models.ForeignKey(FeedbackForm, on_delete=models.CASCADE, related_name='questions')
    text = models.CharField(max_length=500)
    question_type = models.CharField(max_length=16, choices=TYPE_CHOICES)
    options = models.JSONField(default=list, blank=True)
    is_required = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order', 'pk']

    def __str__(self):
        return f"{self.text[:50]} ({self.form.title})"


class FeedbackResponse(models.Model):
    form = models.ForeignKey(FeedbackForm, on_delete=models.CASCADE, related_name='responses')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedback_responses')
    scenario = models.ForeignKey('authoringtool.Scenario', on_delete=models.CASCADE, related_name='feedback_responses')
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-submitted_at', '-pk']
        constraints = [
            models.UniqueConstraint(fields=['form', 'user', 'scenario'], name='unique_feedback_response'),
        ]

    def __str__(self):
        return f"{self.user.username} -> {self.form.title} ({self.scenario.name})"


class FeedbackAnswer(models.Model):
    response = models.ForeignKey(FeedbackResponse, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(FeedbackQuestion, on_delete=models.CASCADE, related_name='answers')
    answer_text = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['response', 'question'], name='unique_feedback_answer_per_question'),
        ]

    def __str__(self):
        return f"{self.question.text[:30]}: {self.answer_text[:30]}"
