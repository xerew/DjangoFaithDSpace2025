from django.conf import settings
from django.db import models


class TranslationReviewer(models.Model):
    """A user allowed to correct the interface translation of one language."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='translation_reviews')
    language = models.CharField(max_length=10)

    class Meta:
        unique_together = [('user', 'language')]
        ordering = ['language', 'user__username']

    def __str__(self):
        return f'{self.user} ({self.language})'


class TranslationOverride(models.Model):
    """A reviewer's correction, applied at runtime on top of the .po catalog."""

    language = models.CharField(max_length=10)
    msgctxt = models.TextField(blank=True, default='')
    msgid = models.TextField()
    plural_index = models.PositiveSmallIntegerField(default=0)
    msgstr = models.TextField()
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name='+')
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [('language', 'msgctxt', 'msgid', 'plural_index')]

    def __str__(self):
        return f'{self.language}: {self.msgid[:60]}'


class TranslationOverrideLog(models.Model):
    """Who changed which translation, from what, to what, and when."""

    override = models.ForeignKey(TranslationOverride, null=True, blank=True,
                                 on_delete=models.SET_NULL, related_name='log')
    language = models.CharField(max_length=10)
    msgctxt = models.TextField(blank=True, default='')
    msgid = models.TextField()
    plural_index = models.PositiveSmallIntegerField(default=0)
    old = models.TextField(blank=True, default='')
    new = models.TextField(blank=True, default='')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                             on_delete=models.SET_NULL, related_name='+')
    at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-at', '-id']

    def __str__(self):
        return f'{self.language}: {self.msgid[:60]} ({self.at:%Y-%m-%d %H:%M})'
