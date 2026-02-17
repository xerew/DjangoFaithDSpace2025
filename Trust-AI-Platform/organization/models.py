from django.db import models
from django.contrib.auth.models import User

class Organization(models.Model):
    name = models.CharField(max_length=255, unique=True)
    short_name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)
    country = models.CharField(max_length=100, null=True, blank=True)
    language = models.CharField(max_length=50, null=True, blank=True)
    picture = models.ImageField(upload_to='org_pictures', null=True, blank=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_organizations')
    created_on = models.DateTimeField(auto_now_add=True)
    admins = models.ManyToManyField(User, related_name='admin_of_organizations')  # Users can be admins of multiple orgs
    members = models.ManyToManyField(User, related_name='member_of_organizations', blank=True)  # Users can be members of multiple orgs
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='updated_organizations')
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Organization"
        verbose_name_plural = "Organizations"
        ordering = ['name']

    def __str__(self):
        return self.name
    
#User.add_to_class('organization', models.ForeignKey(Organization, on_delete=models.SET_NULL, null=True, blank=True, related_name='members'))
#User.add_to_class('is_org_admin', models.BooleanField(default=False))