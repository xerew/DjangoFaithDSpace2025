from django import forms
from .models import UserGroup, UserGroupMembership
from authoringtool.models import Scenario  # Assuming Scenario is in the authoringtool app
from django.contrib.auth.models import User

class UserGroupForm(forms.ModelForm):
    number_of_users = forms.IntegerField(min_value=1, label="Number of Users")
    prefix = forms.CharField(max_length=50, label="User Prefix")
    assigned_scenarios = forms.ModelMultipleChoiceField(
        queryset=Scenario.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple,
        label="Assign Scenarios"
    )

    class Meta:
        model = UserGroup
        fields = ['name', 'prefix', 'number_of_users', 'assigned_scenarios']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'prefix': forms.TextInput(attrs={'class': 'form-control'}),
            'number_of_users': forms.NumberInput(attrs={'class': 'form-control'}),
            'assigned_scenarios': forms.SelectMultiple(attrs={'class': 'form-control'}),
        }

    def clean_name(self):
        """Ensure that group name is unique."""
        name = self.cleaned_data['name']
        if UserGroup.objects.filter(name=name).exclude(id=self.instance.id).exists():
            raise forms.ValidationError("A group with this name already exists.")
        return name

    def clean_prefix(self):
        """Ensure that prefix is not empty."""
        prefix = self.cleaned_data['prefix']
        if not prefix:
            raise forms.ValidationError("Prefix cannot be empty.")
        return prefix

    def save(self, commit=True):
        group = super().save(commit=False)
        if commit:
            group.save()
            self.save_m2m()  # Save the assigned scenarios to the group
        return group

    # Override the __init__ method to add 'form-control' class to all fields
    def __init__(self, *args, **kwargs):
        super(UserGroupForm, self).__init__(*args, **kwargs)
        for field in self.fields:
            if not isinstance(self.fields[field].widget, forms.CheckboxSelectMultiple):
                self.fields[field].widget.attrs.update({'class': 'form-control'})