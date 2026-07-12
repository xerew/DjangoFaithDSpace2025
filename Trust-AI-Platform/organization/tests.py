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


class AnnouncementCardTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user('card_admin', password='pass')
        self.member = User.objects.create_user('card_member', password='pass')
        self.org = Organization.objects.create(
            name='Card Org', short_name='CO', created_by=self.admin,
        )
        self.org.admins.add(self.admin)
        self.org.members.add(self.admin, self.member)
        self.announcement = Announcement.objects.create(
            organization=self.org, title='Kickoff Meeting',
            body='<p>Join us <b>Monday</b> at 10am.</p>',
            plain_text='Join us Monday at 10am.', created_by=self.admin,
        )

    def test_announcement_visible_to_admin_with_controls(self):
        self.client.login(username='card_admin', password='pass')
        r = self.client.get(reverse('organization_detail', args=[self.org.id]))
        self.assertContains(r, 'Kickoff Meeting')
        self.assertContains(r, reverse('edit_announcement', args=[self.org.id, self.announcement.id]))
        self.assertContains(r, reverse('delete_announcement', args=[self.org.id, self.announcement.id]))
        self.assertContains(r, reverse('create_announcement', args=[self.org.id]))

    def test_announcement_visible_to_member_without_controls(self):
        self.client.login(username='card_member', password='pass')
        r = self.client.get(reverse('organization_detail', args=[self.org.id]))
        self.assertContains(r, 'Kickoff Meeting')
        self.assertNotContains(r, reverse('edit_announcement', args=[self.org.id, self.announcement.id]))
        self.assertNotContains(r, reverse('delete_announcement', args=[self.org.id, self.announcement.id]))
        self.assertNotContains(r, reverse('create_announcement', args=[self.org.id]))

    def test_no_announcements_shows_empty_state(self):
        empty_org = Organization.objects.create(
            name='Empty Org', short_name='EO', created_by=self.admin,
        )
        empty_org.admins.add(self.admin)
        empty_org.members.add(self.admin)
        self.client.login(username='card_admin', password='pass')
        r = self.client.get(reverse('organization_detail', args=[empty_org.id]))
        self.assertContains(r, 'No announcements yet')


from .models import OrgChatMessage


class OrgChatMessageModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('chat_owner', password='pass')
        self.org = Organization.objects.create(
            name='Chat Org', short_name='CHO', created_by=self.user,
        )

    def test_create_message(self):
        m = OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='Hello team')
        self.assertEqual(m.organization, self.org)
        self.assertIn(m, self.org.chat_messages.all())

    def test_messages_ordered_oldest_first(self):
        OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='First')
        OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='Second')
        bodies = list(self.org.chat_messages.values_list('body', flat=True))
        self.assertEqual(bodies, ['First', 'Second'])

    def test_message_deleted_with_sender(self):
        m = OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='Bye')
        self.user.delete()
        self.assertFalse(OrgChatMessage.objects.filter(id=m.id).exists())


class OrgChatViewsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.member1 = User.objects.create_user('chat_member1', password='pass')
        self.member2 = User.objects.create_user('chat_member2', password='pass')
        self.outsider = User.objects.create_user('chat_outsider', password='pass')
        self.org = Organization.objects.create(
            name='Chat Views Org', short_name='CVO', created_by=self.member1,
        )
        self.org.admins.add(self.member1)
        self.org.members.add(self.member1, self.member2)

    def test_member_can_view_chat_room(self):
        self.client.login(username='chat_member1', password='pass')
        r = self.client.get(reverse('org_chat', args=[self.org.id]))
        self.assertEqual(r.status_code, 200)

    def test_non_member_redirected_from_chat_room(self):
        self.client.login(username='chat_outsider', password='pass')
        r = self.client.get(reverse('org_chat', args=[self.org.id]))
        self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))

    def test_member_can_send_message(self):
        self.client.login(username='chat_member1', password='pass')
        r = self.client.post(reverse('send_org_chat_message', args=[self.org.id]), {'body': 'Hi team'})
        data = r.json()
        self.assertTrue(data['success'])
        self.assertEqual(OrgChatMessage.objects.filter(organization=self.org, body='Hi team').count(), 1)

    def test_non_member_cannot_send_message(self):
        self.client.login(username='chat_outsider', password='pass')
        r = self.client.post(reverse('send_org_chat_message', args=[self.org.id]), {'body': 'Sneaky'})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(OrgChatMessage.objects.filter(body='Sneaky').exists())

    def test_send_rejects_empty_body(self):
        self.client.login(username='chat_member1', password='pass')
        r = self.client.post(reverse('send_org_chat_message', args=[self.org.id]), {'body': '   '})
        data = r.json()
        self.assertFalse(data['success'])
        self.assertEqual(OrgChatMessage.objects.count(), 0)

    def test_send_requires_post(self):
        self.client.login(username='chat_member1', password='pass')
        r = self.client.get(reverse('send_org_chat_message', args=[self.org.id]))
        self.assertEqual(r.status_code, 405)

    def test_poll_returns_only_messages_after_since_id(self):
        m1 = OrgChatMessage.objects.create(organization=self.org, sender=self.member1, body='First')
        OrgChatMessage.objects.create(organization=self.org, sender=self.member2, body='Second')
        self.client.login(username='chat_member1', password='pass')
        r = self.client.get(reverse('org_chat_poll', args=[self.org.id]), {'since_id': m1.id})
        data = r.json()
        self.assertEqual(len(data['messages']), 1)
        self.assertEqual(data['messages'][0]['body'], 'Second')

    def test_poll_blocks_non_member(self):
        self.client.login(username='chat_outsider', password='pass')
        r = self.client.get(reverse('org_chat_poll', args=[self.org.id]))
        self.assertEqual(r.status_code, 403)
