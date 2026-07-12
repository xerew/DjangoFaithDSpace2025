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


from .models import Announcement


class AnnouncementModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('announce_owner', password='pass')
        self.org = Organization.objects.create(
            name='Announce Org', short_name='AO', created_by=self.user,
        )

    def test_create_announcement(self):
        a = Announcement.objects.create(
            organization=self.org, title='Welcome', body='<p>Hello <b>team</b></p>',
            plain_text='Hello team', created_by=self.user,
        )
        self.assertEqual(a.organization, self.org)
        self.assertIn(a, self.org.announcements.all())

    def test_announcements_ordered_newest_first(self):
        older = Announcement.objects.create(
            organization=self.org, title='Older', body='<p>a</p>', created_by=self.user,
        )
        newer = Announcement.objects.create(
            organization=self.org, title='Newer', body='<p>b</p>', created_by=self.user,
        )
        titles = list(self.org.announcements.values_list('title', flat=True))
        self.assertEqual(titles, ['Newer', 'Older'])

    def test_announcement_survives_creator_deletion(self):
        a = Announcement.objects.create(
            organization=self.org, title='Orphan-safe', body='<p>x</p>', created_by=self.user,
        )
        self.user.delete()
        a.refresh_from_db()
        self.assertIsNone(a.created_by)


class AnnouncementViewsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user('announce_admin', password='pass')
        self.member = User.objects.create_user('announce_member', password='pass')
        self.org = Organization.objects.create(
            name='Views Org', short_name='VO', created_by=self.admin,
        )
        self.org.admins.add(self.admin)
        self.org.members.add(self.admin, self.member)

    def test_admin_can_create_announcement(self):
        self.client.login(username='announce_admin', password='pass')
        r = self.client.post(
            reverse('create_announcement', args=[self.org.id]),
            {'title': 'New Policy', 'body': '<p>Please <b>read</b> this.</p>'},
        )
        self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))
        a = Announcement.objects.get(organization=self.org, title='New Policy')
        self.assertEqual(a.created_by, self.admin)
        self.assertEqual(a.plain_text, 'Please read this.')

    def test_member_cannot_create_announcement(self):
        self.client.login(username='announce_member', password='pass')
        r = self.client.post(
            reverse('create_announcement', args=[self.org.id]),
            {'title': 'Nope', 'body': '<p>x</p>'},
        )
        self.assertFalse(Announcement.objects.filter(title='Nope').exists())
        self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))

    def test_admin_can_edit_announcement(self):
        a = Announcement.objects.create(
            organization=self.org, title='Old Title', body='<p>old</p>', created_by=self.admin,
        )
        self.client.login(username='announce_admin', password='pass')
        r = self.client.post(
            reverse('edit_announcement', args=[self.org.id, a.id]),
            {'title': 'Updated Title', 'body': '<p>updated</p>'},
        )
        self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))
        a.refresh_from_db()
        self.assertEqual(a.title, 'Updated Title')
        self.assertEqual(a.plain_text, 'updated')

    def test_member_cannot_edit_announcement(self):
        a = Announcement.objects.create(
            organization=self.org, title='Untouched', body='<p>x</p>', created_by=self.admin,
        )
        self.client.login(username='announce_member', password='pass')
        self.client.post(
            reverse('edit_announcement', args=[self.org.id, a.id]),
            {'title': 'Hacked', 'body': '<p>x</p>'},
        )
        a.refresh_from_db()
        self.assertEqual(a.title, 'Untouched')

    def test_admin_can_delete_announcement(self):
        a = Announcement.objects.create(
            organization=self.org, title='Delete Me', body='<p>x</p>', created_by=self.admin,
        )
        self.client.login(username='announce_admin', password='pass')
        r = self.client.post(reverse('delete_announcement', args=[self.org.id, a.id]))
        self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))
        self.assertFalse(Announcement.objects.filter(id=a.id).exists())

    def test_member_cannot_delete_announcement(self):
        a = Announcement.objects.create(
            organization=self.org, title='Stays', body='<p>x</p>', created_by=self.admin,
        )
        self.client.login(username='announce_member', password='pass')
        self.client.post(reverse('delete_announcement', args=[self.org.id, a.id]))
        self.assertTrue(Announcement.objects.filter(id=a.id).exists())

    def test_delete_requires_post(self):
        a = Announcement.objects.create(
            organization=self.org, title='Get-safe', body='<p>x</p>', created_by=self.admin,
        )
        self.client.login(username='announce_admin', password='pass')
        r = self.client.get(reverse('delete_announcement', args=[self.org.id, a.id]))
        self.assertEqual(r.status_code, 405)
        self.assertTrue(Announcement.objects.filter(id=a.id).exists())
