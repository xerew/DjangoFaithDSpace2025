from django import forms
from .models import Organization, Announcement

class OrganizationForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = ['name', 'short_name', 'description', 'country', 'language', 'picture']


class AnnouncementForm(forms.ModelForm):
    class Meta:
        model = Announcement
        fields = ['title', 'body']