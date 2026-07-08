from django.test import TestCase, Client
from django.contrib.auth.models import User, Group
from django.urls import reverse


class AdminDashboardAccessTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.regular = User.objects.create_user('regular', password='pass')
        self.staff = User.objects.create_user('staffuser', password='pass', is_staff=True)
        self.superuser = User.objects.create_superuser('super', password='pass')

    def test_anonymous_redirected(self):
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.status_code, 302)

    def test_regular_user_forbidden(self):
        self.client.login(username='regular', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.status_code, 403)

    def test_staff_can_access(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.status_code, 200)

    def test_toggle_user_requires_post(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.get(reverse('admin_toggle_user', args=[self.regular.id]))
        self.assertEqual(r.status_code, 405)

    def test_toggle_user_flips_active(self):
        self.client.login(username='staffuser', password='pass')
        self.assertTrue(self.regular.is_active)
        r = self.client.post(reverse('admin_toggle_user', args=[self.regular.id]))
        self.assertEqual(r.status_code, 200)
        self.assertJSONEqual(r.content, {'success': True, 'is_active': False})
        self.regular.refresh_from_db()
        self.assertFalse(self.regular.is_active)

    def test_delete_user(self):
        self.client.login(username='staffuser', password='pass')
        uid = self.regular.id
        r = self.client.post(reverse('admin_delete_user', args=[uid]))
        self.assertJSONEqual(r.content, {'success': True})
        self.assertFalse(User.objects.filter(id=uid).exists())

    def test_cannot_delete_self(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.post(reverse('admin_delete_user', args=[self.staff.id]))
        data = r.json()
        self.assertFalse(data['success'])

    def test_create_role(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.post(reverse('admin_create_role'), {'name': 'TestRole'})
        data = r.json()
        self.assertTrue(data['success'])
        self.assertTrue(Group.objects.filter(name='TestRole').exists())

    def test_delete_role_with_members_blocked(self):
        self.client.login(username='staffuser', password='pass')
        g = Group.objects.create(name='Occupied')
        self.regular.groups.add(g)
        r = self.client.post(reverse('admin_delete_role', args=[g.id]))
        data = r.json()
        self.assertFalse(data['success'])
        self.assertTrue(Group.objects.filter(id=g.id).exists())
