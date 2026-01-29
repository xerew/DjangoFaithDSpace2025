from django.db import models
from django.contrib.auth.models import User
from authoringtool.models import Scenario
from django.utils.crypto import get_random_string

class UserGroup(models.Model):
    name = models.CharField(max_length=255)
    prefix = models.CharField(max_length=50)
    number_of_users = models.IntegerField()
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="created_groups")
    created_on = models.DateTimeField(auto_now_add=True)
    
    # Many-to-Many relationship with Scenario
    assigned_scenarios = models.ManyToManyField(Scenario, related_name="assigned_groups", blank=True)

    # Many-to-Many relationship with Users through UserGroupMembership
    members = models.ManyToManyField(User, through='UserGroupMembership', related_name='member_of_groups')

    def create_users(self):
        """Generate users for the group."""
        for i in range(1, self.number_of_users + 1):
            username = f"{self.prefix}{i}"
            password = self.generate_password()
            # Create the user and assign them to the group
            user = User.objects.create_user(username=username, password=password)
            UserGroupMembership.objects.create(group=self, user=user, password=password)  # Store password for Excel

    def generate_password(self):
        return get_random_string(8)  # Generates a random password of length 8

    def __str__(self):
        return self.name


class UserGroupMembership(models.Model):
    group = models.ForeignKey(UserGroup, on_delete=models.CASCADE)
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # Ensures a user can belong to one group only once
    password = models.CharField(max_length=128)  # Store password for Excel export

    def __str__(self):
        return f"{self.user.username} in {self.group.name}"

    class Meta:
        unique_together = ('group', 'user')  # Ensures no duplicate memberships
