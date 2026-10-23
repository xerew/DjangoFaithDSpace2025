from django import forms
from django.utils.translation import gettext_lazy as _
from .models import Organization, Announcement

class OrganizationForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = ['name', 'short_name', 'description', 'country', 'language', 'picture']


class AnnouncementForm(forms.ModelForm):
    class Meta:
        model = Announcement
        fields = ['title', 'body']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': _('Announcement title')}),
        }