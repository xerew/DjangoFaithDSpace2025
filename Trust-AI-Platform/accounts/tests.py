import json
from django.test import TestCase, Client
from django.contrib.auth.models import User, Group
from django.urls import reverse
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment


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


class AdminLabViewsTest(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('stafflab', password='x', is_staff=True)
        self.regular = User.objects.create_user('regularlab', password='x')
        self.client.force_login(self.staff)

    # Access control
    def test_regular_user_forbidden(self):
        self.client.force_login(self.regular)
        r = self.client.post('/accounts/admin/simulations/create/', {
            'name': 'X', 'iframe_url': 'https://x.com', 'width': '800', 'height': '600', 'allow_fullscreen': 'true'
        })
        self.assertEqual(r.status_code, 403)

    # Simulation CRUD
    def test_create_simulation(self):
        r = self.client.post('/accounts/admin/simulations/create/', {
            'name': 'PhET Pendulum', 'iframe_url': 'https://phet.colorado.edu/',
            'width': '800', 'height': '600', 'allow_fullscreen': 'true',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertTrue(Simulation.objects.filter(name='PhET Pendulum').exists())
        self.assertEqual(data['width'], 800)

    def test_create_simulation_missing_fields(self):
        r = self.client.post('/accounts/admin/simulations/create/', {'name': '', 'iframe_url': ''})
        data = json.loads(r.content)
        self.assertFalse(data['success'])

    def test_edit_simulation(self):
        sim = Simulation.objects.create(name='Old', iframe_url='https://old.com', width=800, height=600)
        r = self.client.post(f'/accounts/admin/simulations/{sim.id}/edit/', {
            'name': 'New', 'iframe_url': 'https://new.com', 'width': '1024', 'height': '768', 'allow_fullscreen': 'false',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        sim.refresh_from_db()
        self.assertEqual(sim.name, 'New')
        self.assertEqual(sim.width, 1024)
        self.assertFalse(sim.allow_fullscreen)

    def test_delete_simulation(self):
        sim = Simulation.objects.create(name='ToDelete', iframe_url='https://x.com', width=800, height=600)
        r = self.client.post(f'/accounts/admin/simulations/{sim.id}/delete/')
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertFalse(Simulation.objects.filter(id=sim.id).exists())

    # Remote Lab CRUD
    def test_create_remote_lab(self):
        r = self.client.post('/accounts/admin/remote_labs/create/', {
            'name': 'LabsLand Pendulum', 'launch_url': 'https://labsland.com/lti',
            'consumer_key': 'key123', 'shared_secret': 'secret456', 'description': 'A lab',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertNotIn('shared_secret', data)
        self.assertTrue(ExperimentLL.objects.filter(name='LabsLand Pendulum').exists())

    def test_edit_remote_lab_blank_secret_preserved(self):
        lab = ExperimentLL.objects.create(
            name='Lab', launch_url='https://x.com', consumer_key='ck', shared_secret='original_secret'
        )
        r = self.client.post(f'/accounts/admin/remote_labs/{lab.id}/edit/', {
            'name': 'Lab Updated', 'launch_url': 'https://x.com',
            'consumer_key': 'ck', 'shared_secret': '',  # blank — keep existing
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertNotIn('shared_secret', data)
        lab.refresh_from_db()
        self.assertEqual(lab.shared_secret, 'original_secret')  # not changed

    def test_delete_remote_lab(self):
        lab = ExperimentLL.objects.create(name='X', launch_url='https://x.com', consumer_key='ck', shared_secret='ss')
        r = self.client.post(f'/accounts/admin/remote_labs/{lab.id}/delete/')
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertFalse(ExperimentLL.objects.filter(id=lab.id).exists())

    # VR Lab CRUD
    def test_create_vr_lab(self):
        r = self.client.post('/accounts/admin/vr_labs/create/', {
            'name': 'Mars VR', 'launch_url': 'https://vr.example.com/mars', 'description': 'VR lab',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertIn('qr_code_url', data)   # key present (may be None in test env)
        self.assertTrue(VRARExperiment.objects.filter(name='Mars VR').exists())

    def test_edit_vr_lab(self):
        vr = VRARExperiment.objects.create(name='Old VR', launch_url='https://old.com', description='')
        r = self.client.post(f'/accounts/admin/vr_labs/{vr.id}/edit/', {
            'name': 'New VR', 'launch_url': 'https://new.com', 'description': 'Updated',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertEqual(data['name'], 'New VR')
        self.assertIn('qr_code_url', data)

    def test_delete_vr_lab(self):
        vr = VRARExperiment.objects.create(name='X', launch_url='https://x.com', description='')
        r = self.client.post(f'/accounts/admin/vr_labs/{vr.id}/delete/')
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertFalse(VRARExperiment.objects.filter(id=vr.id).exists())


class ViewProfileTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.alice = User.objects.create_user('alice_prof', password='pass', first_name='Alice', last_name='A')
        self.bob = User.objects.create_user('bob_prof', password='pass', first_name='Bob', last_name='B')
        self.alice.groups.add(teachers)
        self.bob.groups.add(teachers)
        self.student = User.objects.create_user('stu_prof', password='pass')
        self.client.login(username='alice_prof', password='pass')

    def test_view_other_teacher_profile(self):
        r = self.client.get(reverse('view_profile', args=[self.bob.id]))
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.context['is_own_profile'])
        self.assertEqual(r.context['profile_user'], self.bob)

    def test_view_own_profile_via_view_profile_redirects(self):
        r = self.client.get(reverse('view_profile', args=[self.alice.id]))
        self.assertRedirects(r, reverse('profile'))

    def test_view_student_profile_404s(self):
        r = self.client.get(reverse('view_profile', args=[self.student.id]))
        self.assertEqual(r.status_code, 404)

    def test_own_profile_page_marks_is_own_profile_true(self):
        r = self.client.get(reverse('profile'))
        self.assertTrue(r.context['is_own_profile'])
        self.assertEqual(r.context['profile_user'], self.alice)

    def test_student_cannot_view_other_profile(self):
        self.client.logout()
        self.client.login(username='stu_prof', password='pass')
        r = self.client.get(reverse('view_profile', args=[self.bob.id]))
        self.assertEqual(r.status_code, 403)

    def test_other_profile_shows_message_button_not_edit_form(self):
        r = self.client.get(reverse('view_profile', args=[self.bob.id]))
        self.assertContains(r, 'Send Bob a message')
        self.assertNotContains(r, 'id="infoForm"')


class SidebarMessagingIntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.teacher = User.objects.create_user('sidebar_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.student = User.objects.create_user('sidebar_student', password='pass')

    def test_teacher_sees_messages_nav_and_polling_script(self):
        self.client.login(username='sidebar_teacher', password='pass')
        r = self.client.get(reverse('profile'))
        self.assertContains(r, reverse('message_threads'))
        self.assertContains(r, 'sidebar-unread-badge')
        self.assertContains(r, 'new-message-toast-container')

    def test_student_does_not_see_messages_nav_or_polling_script(self):
        self.client.login(username='sidebar_student', password='pass')
        r = self.client.get(reverse('studentScenarios'))
        self.assertNotContains(r, 'sidebar-unread-badge')
        self.assertNotContains(r, 'new-message-toast-container')
