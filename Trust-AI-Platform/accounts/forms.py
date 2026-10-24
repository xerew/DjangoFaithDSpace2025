from django import forms
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm
from django.utils import translation

from localization.languages import recipient_language
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password

class CustomSetPasswordForm(SetPasswordForm):
    def clean_new_password2(self):
        password2 = self.cleaned_data.get('new_password2')
        if password2:
            try:
                validate_password(password2, self.user)
            except ValidationError as e:
                self.add_error('new_password2', e)
        return password2


class LocalizedPasswordResetForm(PasswordResetForm):
    """Writes the reset email in the language of the account it goes to."""

    def send_mail(self, subject_template_name, email_template_name, context,
                  from_email, to_email, html_email_template_name=None):
        with translation.override(recipient_language(context['user'])):
            super().send_mail(subject_template_name, email_template_name, context,
                              from_email, to_email, html_email_template_name)
