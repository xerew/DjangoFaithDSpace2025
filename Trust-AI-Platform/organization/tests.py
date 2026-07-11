from django.contrib.auth.models import Group, User
from django.test import Client, TestCase
from django.urls import reverse

from .models import Organization


class OrganizationDetailMessagingLinksTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.alice = User.objects.create_user('alice_org', password='pass')
        self.bob = User.objects.create_user('bob_org', password='pass', first_name='Bob', last_name='B')
        self.alice.groups.add(teachers)
        self.bob.groups.add(teachers)
        self.org = Organization.objects.create(name='Test Org', short_name='TO', created_by=self.alice)
        self.org.members.add(self.alice, self.bob)
        self.client.login(username='alice_org', password='pass')

    def test_member_name_links_to_profile(self):
        r = self.client.get(reverse('organization_detail', args=[self.org.id]))
        self.assertContains(r, reverse('view_profile', args=[self.bob.id]))

    def test_message_button_links_to_thread(self):
        r = self.client.get(reverse('organization_detail', args=[self.org.id]))
        self.assertContains(r, reverse('thread', args=[self.bob.id]))

    def test_no_message_button_for_self(self):
        r = self.client.get(reverse('organization_detail', args=[self.org.id]))
        self.assertNotContains(r, reverse('thread', args=[self.alice.id]))
