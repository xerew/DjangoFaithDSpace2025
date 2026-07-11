from django.contrib.auth.models import Group, User
from django.test import Client, TestCase
from django.urls import reverse

from .models import Message


class MessagingViewsTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.alice = User.objects.create_user('alice_msg', password='pass', first_name='Alice', last_name='A')
        self.bob = User.objects.create_user('bob_msg', password='pass', first_name='Bob', last_name='B')
        self.alice.groups.add(teachers)
        self.bob.groups.add(teachers)
        self.student = User.objects.create_user('stu_msg', password='pass')
        self.client.login(username='alice_msg', password='pass')

    def test_send_message_creates_message(self):
        r = self.client.post(reverse('send_message'), {'recipient_id': self.bob.id, 'body': 'Hi Bob'})
        data = r.json()
        self.assertTrue(data['success'])
        self.assertEqual(Message.objects.filter(sender=self.alice, recipient=self.bob).count(), 1)

    def test_send_message_rejects_empty_body(self):
        r = self.client.post(reverse('send_message'), {'recipient_id': self.bob.id, 'body': '   '})
        data = r.json()
        self.assertFalse(data['success'])
        self.assertEqual(Message.objects.count(), 0)

    def test_send_message_rejects_non_teacher_recipient(self):
        r = self.client.post(reverse('send_message'), {'recipient_id': self.student.id, 'body': 'Hi'})
        data = r.json()
        self.assertFalse(data['success'])
        self.assertEqual(Message.objects.count(), 0)

    def test_send_message_requires_post(self):
        r = self.client.get(reverse('send_message'))
        self.assertEqual(r.status_code, 405)

    def test_thread_marks_messages_as_read(self):
        Message.objects.create(sender=self.bob, recipient=self.alice, body='Hello Alice')
        r = self.client.get(reverse('thread', args=[self.bob.id]))
        self.assertEqual(r.status_code, 200)
        msg = Message.objects.get(sender=self.bob, recipient=self.alice)
        self.assertIsNotNone(msg.read_at)

    def test_thread_shows_messages_both_directions(self):
        Message.objects.create(sender=self.bob, recipient=self.alice, body='From Bob')
        Message.objects.create(sender=self.alice, recipient=self.bob, body='From Alice')
        r = self.client.get(reverse('thread', args=[self.bob.id]))
        self.assertContains(r, 'From Bob')
        self.assertContains(r, 'From Alice')

    def test_thread_blocks_non_teacher_target(self):
        r = self.client.get(reverse('thread', args=[self.student.id]))
        self.assertRedirects(r, reverse('message_threads'))

    def test_message_threads_lists_partner_with_unread_count(self):
        Message.objects.create(sender=self.bob, recipient=self.alice, body='Unread 1')
        Message.objects.create(sender=self.bob, recipient=self.alice, body='Unread 2')
        r = self.client.get(reverse('message_threads'))
        threads = r.context['threads']
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]['partner'], self.bob)
        self.assertEqual(threads[0]['unread'], 2)

    def test_message_threads_ordered_by_latest_message(self):
        teachers = Group.objects.get(name='teachers')
        carol = User.objects.create_user('carol_msg', password='pass')
        carol.groups.add(teachers)
        Message.objects.create(sender=self.bob, recipient=self.alice, body='Older')
        Message.objects.create(sender=carol, recipient=self.alice, body='Newer')
        r = self.client.get(reverse('message_threads'))
        threads = r.context['threads']
        self.assertEqual(threads[0]['partner'], carol)
        self.assertEqual(threads[1]['partner'], self.bob)

    def test_unread_status_returns_latest_and_count(self):
        Message.objects.create(sender=self.bob, recipient=self.alice, body='Ping')
        r = self.client.get(reverse('unread_status'))
        data = r.json()
        self.assertEqual(data['unread_count'], 1)
        self.assertEqual(data['latest']['sender_id'], self.bob.id)
        self.assertEqual(data['latest']['snippet'], 'Ping')

    def test_unread_status_empty_when_no_messages(self):
        r = self.client.get(reverse('unread_status'))
        data = r.json()
        self.assertEqual(data['unread_count'], 0)
        self.assertIsNone(data['latest'])

    def test_non_teacher_cannot_access_message_threads(self):
        self.client.logout()
        self.client.login(username='stu_msg', password='pass')
        r = self.client.get(reverse('message_threads'))
        self.assertEqual(r.status_code, 403)
