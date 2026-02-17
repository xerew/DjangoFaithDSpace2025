from django import forms
from django.contrib.auth.forms import SetPasswordForm
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