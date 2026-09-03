import json
from datetime import timedelta

from django.contrib import admin
from django.core import mail
from django.core.exceptions import ValidationError
from django.test import TestCase, Client
from django.test import override_settings
from django.contrib.auth.models import User, Group
from django.urls import reverse
from django.utils import timezone
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment
from accounts.models import BulkEmailCampaign, MaintenanceNotice, UserProfile
from organization.models import Organization
from templatetags.profile_tags import avatar_url as avatar_url_filter


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

    def test_staff_without_teacher_role_sees_platform_admin_menu(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertContains(r, 'Platform Administration', count=None)
        self.assertContains(r, reverse('admin_dashboard'))

    def test_regular_user_does_not_see_platform_admin_menu(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.regular.groups.add(teachers)
        self.client.login(username='regular', password='pass')
        r = self.client.get(reverse('profile'))
        self.assertEqual(r.status_code, 200)
        self.assertNotContains(r, 'Platform Administration')

    def test_dashboard_links_to_django_admin_and_maintenance(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertContains(r, 'Django Admin')
        self.assertContains(r, 'id="tab-maintenance-btn"')
        self.assertContains(r, reverse('admin:accounts_maintenancenotice_add'))

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


class MaintenanceNoticeTests(TestCase):
    def setUp(self):
        self.now = timezone.now().replace(microsecond=0)

    def create_notice(self, **overrides):
        values = {
            'reason': 'Database upgrade',
            'starts_at': self.now - timedelta(hours=1),
            'ends_at': self.now + timedelta(hours=1),
            'is_enabled': True,
        }
        values.update(overrides)
        return MaintenanceNotice.objects.create(**values)

    def test_active_queryset_respects_window_and_enabled_flag(self):
        active = self.create_notice()
        self.create_notice(
            reason='Future work',
            starts_at=self.now + timedelta(hours=1),
            ends_at=self.now + timedelta(hours=2),
        )
        self.create_notice(
            reason='Disabled work',
            is_enabled=False,
        )

        self.assertEqual(list(MaintenanceNotice.objects.active(self.now)), [active])

    def test_window_is_start_inclusive_and_end_exclusive(self):
        notice = self.create_notice(starts_at=self.now)
        self.assertTrue(notice.is_active(self.now))
        self.assertFalse(notice.is_active(notice.ends_at))

    def test_end_must_be_after_start(self):
        notice = MaintenanceNotice(
            reason='Invalid window',
            starts_at=self.now,
            ends_at=self.now,
        )
        with self.assertRaises(ValidationError):
            notice.full_clean()

    def test_registered_in_django_admin(self):
        self.assertIn(MaintenanceNotice, admin.site._registry)


class MaintenanceBannerTests(TestCase):
    def setUp(self):
        self.now = timezone.now().replace(microsecond=0)
        self.staff = User.objects.create_user(
            'maintenance_staff', password='pass', is_staff=True,
        )

    def create_notice(self, reason='Planned database maintenance', **overrides):
        values = {
            'reason': reason,
            'starts_at': self.now - timedelta(minutes=30),
            'ends_at': self.now + timedelta(minutes=30),
            'is_enabled': True,
            'created_by': self.staff,
        }
        values.update(overrides)
        return MaintenanceNotice.objects.create(**values)

    def test_active_banner_is_visible_to_anonymous_users(self):
        self.create_notice()
        response = self.client.get(reverse('login'))
        self.assertContains(response, 'Platform maintenance')
        self.assertContains(response, 'Planned database maintenance')

    def test_future_expired_and_disabled_notices_are_hidden(self):
        self.create_notice(
            reason='Future maintenance',
            starts_at=self.now + timedelta(hours=1),
            ends_at=self.now + timedelta(hours=2),
        )
        self.create_notice(
            reason='Expired maintenance',
            starts_at=self.now - timedelta(hours=2),
            ends_at=self.now - timedelta(hours=1),
        )
        self.create_notice(reason='Disabled maintenance', is_enabled=False)

        response = self.client.get(reverse('login'))
        self.assertNotContains(response, 'Future maintenance')
        self.assertNotContains(response, 'Expired maintenance')
        self.assertNotContains(response, 'Disabled maintenance')

    def test_banner_reason_is_escaped(self):
        self.create_notice(reason='<script>alert("unsafe")</script>')
        response = self.client.get(reverse('login'))
        self.assertNotContains(response, '<script>alert("unsafe")</script>')
        self.assertContains(response, '&lt;script&gt;', html=False)

    def test_dashboard_lists_notice_and_active_count(self):
        notice = self.create_notice()
        self.client.login(username='maintenance_staff', password='pass')
        response = self.client.get(reverse('admin_dashboard'))
        self.assertContains(response, notice.reason)
        self.assertEqual(response.context['active_maintenance_count'], 1)

    def test_django_admin_has_back_to_platform_link(self):
        superuser = User.objects.create_superuser('maintenance_super', password='pass')
        self.client.login(username='maintenance_super', password='pass')
        response = self.client.get(reverse('admin:index'))
        self.assertContains(response, 'Back to platform')
        self.assertContains(response, reverse('teacher_home'))


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


class AvatarUrlFilterTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('avatar_user', password='pass')

    def test_no_profile_returns_empty_string(self):
        self.assertEqual(avatar_url_filter(self.user), '')

    def test_gender_male_returns_default_static_path(self):
        UserProfile.objects.create(user=self.user, gender='male')
        self.assertEqual(avatar_url_filter(self.user), '/static/img/profile_d_man.webp')

    def test_gender_female_returns_default_static_path(self):
        UserProfile.objects.create(user=self.user, gender='female')
        self.assertEqual(avatar_url_filter(self.user), '/static/img/profile_d_woman.jpg')

    def test_blank_gender_returns_empty_string(self):
        UserProfile.objects.create(user=self.user, gender='')
        self.assertEqual(avatar_url_filter(self.user), '')

    def test_custom_picture_takes_priority_over_gender(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        tiny_gif = (
            b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
            b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
        )
        profile = UserProfile.objects.create(
            user=self.user, gender='female',
            picture=SimpleUploadedFile('test.gif', tiny_gif, content_type='image/gif'),
        )
        url = avatar_url_filter(self.user)
        self.assertTrue(url.startswith('/media/profile_pictures/test'))
        self.assertNotIn('profile_d_woman', url)


class RegisterAccountGenderTests(TestCase):
    # accounts/views.py:14 imports TEACHER_ACCESS_CODE_HASHED at module load time
    # (`from faithDev.settings import TEACHER_ACCESS_CODE_HASHED`), so patching
    # django.conf.settings at test time would NOT affect the check at views.py:118 —
    # that name is a plain module attribute, not looked up dynamically. Simpler and
    # robust: settings.py:18 hashes this literal plaintext default whenever the
    # TEACHER_ACCESS_CODE_HASHED env var isn't set (the normal case in this test
    # environment), so just submit the real default plaintext code directly.
    VALID_ACCESS_CODE = r"}{80s%3B\x/+"

    def setUp(self):
        self.client = Client()

    def _register(self, **overrides):
        data = {
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'testuser@example.com',
            'username': 'testuser_gender',
            'password': 'SuperSecret123!',
            'access_code': self.VALID_ACCESS_CODE,
            'gender': 'female',
        }
        data.update(overrides)
        return self.client.post(
            reverse('register'), data,
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    def test_register_creates_userprofile_with_gender(self):
        r = self._register()
        data = r.json()
        self.assertTrue(data['success'])
        user = User.objects.get(username='testuser_gender')
        self.assertEqual(user.profile.gender, 'female')

    def test_register_without_gender_defaults_to_blank(self):
        r = self._register(gender='', username='testuser_nogender')
        data = r.json()
        self.assertTrue(data['success'])
        user = User.objects.get(username='testuser_nogender')
        self.assertEqual(user.profile.gender, '')


class ProfileEditGenderAvatarTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.user = User.objects.create_user(
            'profile_edit_user', password='pass', first_name='Pat', last_name='Doe',
            email='pat@example.com',
        )
        self.user.groups.add(teachers)
        self.client.login(username='profile_edit_user', password='pass')

    def _update_info(self, **overrides):
        data = {
            'action': 'update_info',
            'first_name': 'Pat',
            'last_name': 'Doe',
            'email': 'pat@example.com',
            'country': '',
            'institution': '',
            'bio': '',
            'gender': 'male',
        }
        data.update(overrides)
        return self.client.post(
            reverse('profile'), data,
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    def test_update_info_sets_gender(self):
        r = self._update_info(gender='female')
        self.assertTrue(r.json()['success'])
        profile = UserProfile.objects.get(user=self.user)
        self.assertEqual(profile.gender, 'female')

    def test_update_info_uploads_picture(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        tiny_gif = (
            b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
            b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
        )
        upload = SimpleUploadedFile('avatar.gif', tiny_gif, content_type='image/gif')
        r = self._update_info(picture=upload)
        self.assertTrue(r.json()['success'])
        profile = UserProfile.objects.get(user=self.user)
        self.assertTrue(profile.picture.name)

    def test_update_info_without_new_picture_keeps_existing(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        tiny_gif = (
            b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
            b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
        )
        profile = UserProfile.objects.create(
            user=self.user,
            picture=SimpleUploadedFile('original.gif', tiny_gif, content_type='image/gif'),
        )
        original_name = profile.picture.name
        r = self._update_info()
        self.assertTrue(r.json()['success'])
        profile.refresh_from_db()
        self.assertEqual(profile.picture.name, original_name)

    def test_update_info_rejects_non_image_content_with_image_extension(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        fake_image = SimpleUploadedFile(
            'avatar.png', b'this is not actually image data, just plain text bytes',
            content_type='image/png',
        )
        r = self._update_info(picture=fake_image)
        data = r.json()
        self.assertFalse(data['success'])
        self.assertIn('picture', data['errors'])
        self.assertFalse(UserProfile.objects.filter(user=self.user, picture__gt='').exists())

    def test_update_info_rejects_disallowed_extension(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        bad_file = SimpleUploadedFile(
            'avatar.exe', b'MZ some fake executable bytes', content_type='application/octet-stream',
        )
        r = self._update_info(picture=bad_file)
        data = r.json()
        self.assertFalse(data['success'])
        self.assertIn('picture', data['errors'])
        self.assertFalse(UserProfile.objects.filter(user=self.user, picture__gt='').exists())

    def test_update_info_rejects_oversized_image(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        oversized = SimpleUploadedFile(
            'avatar.png', b'\x00' * (5 * 1024 * 1024 + 1), content_type='image/png',
        )
        r = self._update_info(picture=oversized)
        data = r.json()
        self.assertFalse(data['success'])
        self.assertIn('picture', data['errors'])
        self.assertFalse(UserProfile.objects.filter(user=self.user, picture__gt='').exists())

    def test_profile_page_renders_avatar_img_when_picture_set(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        tiny_gif = (
            b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
            b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
        )
        UserProfile.objects.create(
            user=self.user,
            picture=SimpleUploadedFile('portrait.gif', tiny_gif, content_type='image/gif'),
        )
        r = self.client.get(reverse('profile'))
        self.assertContains(r, '<img src="/media/profile_pictures/portrait')

    def test_profile_page_falls_back_to_icon_without_picture_or_gender(self):
        UserProfile.objects.create(user=self.user)
        r = self.client.get(reverse('profile'))
        self.assertContains(r, 'bi-person-circle')


class AvatarRenderSitesTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.staff = User.objects.create_user('avatar_sites_staff', password='pass', is_staff=True)
        self.staff.groups.add(teachers)
        UserProfile.objects.create(user=self.staff, gender='male')
        self.client.login(username='avatar_sites_staff', password='pass')

    def test_head_nav_renders_gender_default_avatar_img(self):
        r = self.client.get(reverse('profile'))
        self.assertContains(r, '/static/img/profile_d_man.webp')

    def test_admin_dashboard_renders_gender_default_avatar_img(self):
        r = self.client.get(reverse('admin_dashboard'))
        self.assertContains(r, '/static/img/profile_d_man.webp')

    def test_user_without_gender_or_picture_falls_back_to_initials_in_nav(self):
        plain_user = User.objects.create_user('avatar_plain', password='pass')
        plain_user.groups.add(Group.objects.get(name='teachers'))
        self.client.logout()
        self.client.login(username='avatar_plain', password='pass')
        r = self.client.get(reverse('profile'))
        self.assertNotContains(r, '/static/img/profile_d_')
        self.assertContains(r, 'nav-profile-avatar')


class AdminDashboardUserListTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user('list_staff', password='pass', is_staff=True)
        self.client.login(username='list_staff', password='pass')

    def test_no_role_user_not_listed(self):
        User.objects.create_user('list_student', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        usernames = [u.username for u in r.context['all_users']]
        self.assertNotIn('list_student', usernames)

    def test_grouped_user_listed(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        teacher = User.objects.create_user('list_teacher', password='pass')
        teacher.groups.add(teachers)
        r = self.client.get(reverse('admin_dashboard'))
        usernames = [u.username for u in r.context['all_users']]
        self.assertIn('list_teacher', usernames)

    def test_staff_without_group_listed(self):
        r = self.client.get(reverse('admin_dashboard'))
        usernames = [u.username for u in r.context['all_users']]
        self.assertIn('list_staff', usernames)

    def test_multi_group_user_listed_once(self):
        g1, _ = Group.objects.get_or_create(name='teachers')
        g2, _ = Group.objects.get_or_create(name='dspace_partners')
        multi = User.objects.create_user('list_multi', password='pass')
        multi.groups.add(g1, g2)
        r = self.client.get(reverse('admin_dashboard'))
        usernames = [u.username for u in r.context['all_users']]
        self.assertEqual(usernames.count('list_multi'), 1)

    def test_stats_still_count_all_accounts(self):
        User.objects.create_user('list_student2', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.context['stats']['total'], User.objects.count())
        self.assertEqual(r.context['stats']['no_role'], 1)

    def test_no_role_filter_option_removed(self):
        r = self.client.get(reverse('admin_dashboard'))
        self.assertNotContains(r, 'id="rfNone"')


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    DEFAULT_FROM_EMAIL='Trust AI Lab <noreply@trust-ai.test>',
    SITE_URL='https://platform.test',
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class AdminBulkEmailTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.teachers, _ = Group.objects.get_or_create(name='teachers')
        self.staff = User.objects.create_user(
            'email_staff', password='pass', email='staff@example.com', is_staff=True,
        )
        self.regular = User.objects.create_user(
            'email_regular', password='pass', email='regular@example.com',
        )
        self.teacher_one = User.objects.create_user(
            'email_teacher_one', password='pass', email='teacher@example.com',
            first_name='Alice',
        )
        self.teacher_duplicate = User.objects.create_user(
            'email_teacher_duplicate', password='pass', email='TEACHER@example.com',
        )
        self.teacher_two = User.objects.create_user(
            'email_teacher_two', password='pass', email='second@example.com',
            first_name='Bob',
        )
        self.inactive_teacher = User.objects.create_user(
            'email_teacher_inactive', password='pass', email='inactive@example.com',
            is_active=False,
        )
        self.no_email_teacher = User.objects.create_user(
            'email_teacher_blank', password='pass', email='',
        )
        for teacher in (
            self.teacher_one,
            self.teacher_duplicate,
            self.teacher_two,
            self.inactive_teacher,
            self.no_email_teacher,
        ):
            teacher.groups.add(self.teachers)

        self.org_one = Organization.objects.create(
            name='Email Organization One', short_name='EO1', created_by=self.staff,
        )
        self.org_two = Organization.objects.create(
            name='Email Organization Two', short_name='EO2', created_by=self.staff,
        )
        self.org_one.members.add(self.teacher_one, self.teacher_duplicate, self.regular)
        self.org_two.members.add(self.teacher_two, self.inactive_teacher)
        self.client.login(username='email_staff', password='pass')

    def _send(self, **overrides):
        data = {
            'target_type': 'all_teachers',
            'subject': 'Important platform update',
            'body_html': (
                '<p>Please read the <a href="/accounts/documentation/">guide</a>.</p>'
                '<img src="/media/tinymce/update.png">'
            ),
            'confirmed': 'true',
        }
        data.update(overrides)
        return self.client.post(reverse('admin_send_bulk_email'), data)

    def test_dashboard_has_email_composer_and_filters(self):
        response = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="tab-email-btn"')
        self.assertContains(response, 'class="email-tinymce-editor"')
        self.assertContains(response, 'Email Organization One')
        self.assertContains(response, 'email_teacher_one')
        self.assertEqual(response.context['email_teacher_count'], 2)

    def test_recipient_count_deduplicates_and_excludes_ineligible_users(self):
        response = self.client.post(
            reverse('admin_bulk_email_recipient_count'),
            {'target_type': 'all_teachers'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['recipient_count'], 2)

        response = self.client.post(
            reverse('admin_bulk_email_recipient_count'),
            {'target_type': 'organizations', 'organization_ids': [self.org_one.id]},
        )
        self.assertEqual(response.json()['recipient_count'], 1)
        self.assertEqual(response.json()['organization_count'], 1)

    def test_all_teacher_campaign_sends_individual_branded_html_messages(self):
        response = self._send()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        self.assertEqual(response.json()['recipient_count'], 2)

        campaign = BulkEmailCampaign.objects.get()
        self.assertEqual(campaign.status, BulkEmailCampaign.STATUS_COMPLETED)
        self.assertEqual(campaign.sent_count, 2)
        self.assertEqual(campaign.failed_count, 0)
        self.assertEqual(campaign.recipients.count(), 2)
        self.assertEqual(len(mail.outbox), 2)
        self.assertTrue(all(len(message.to) == 1 for message in mail.outbox))
        self.assertEqual(
            {message.to[0].casefold() for message in mail.outbox},
            {'teacher@example.com', 'second@example.com'},
        )

        html_alternative = mail.outbox[0].alternatives[0]
        html_message = (
            html_alternative.content
            if hasattr(html_alternative, 'content')
            else html_alternative[0]
        )
        self.assertIn('background-color:#1a56db', html_message)
        self.assertIn('Platform Announcement', html_message)
        self.assertIn('https://platform.test/accounts/documentation/', html_message)
        self.assertIn('https://platform.test/media/tinymce/update.png', html_message)

    def test_selected_teacher_campaign_only_targets_checked_teacher(self):
        response = self._send(
            target_type='selected_teachers',
            teacher_ids=[self.teacher_two.id],
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['second@example.com'])

    def test_organization_campaign_only_targets_teacher_members(self):
        response = self._send(
            target_type='organizations',
            organization_ids=[self.org_one.id],
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['teacher@example.com'])
        campaign = BulkEmailCampaign.objects.get()
        self.assertEqual(list(campaign.organizations.all()), [self.org_one])

    def test_confirmation_and_content_are_required(self):
        response = self._send(confirmed='false')
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()['success'])
        self.assertEqual(BulkEmailCampaign.objects.count(), 0)

        response = self._send(confirmed='true', body_html='')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(BulkEmailCampaign.objects.count(), 0)

    def test_empty_selection_and_invalid_subject_are_rejected(self):
        response = self._send(target_type='selected_teachers', teacher_ids=[])
        self.assertEqual(response.status_code, 400)
        self.assertIn('No active teachers', response.json()['error'])

        response = self._send(subject='Unsafe\nBcc: somebody@example.com')
        self.assertEqual(response.status_code, 400)
        self.assertIn('invalid characters', response.json()['error'])
        self.assertEqual(BulkEmailCampaign.objects.count(), 0)

    def test_regular_user_cannot_count_or_send_campaign(self):
        self.client.logout()
        self.client.login(username='email_regular', password='pass')
        count_response = self.client.post(
            reverse('admin_bulk_email_recipient_count'),
            {'target_type': 'all_teachers'},
        )
        send_response = self._send()
        self.assertEqual(count_response.status_code, 403)
        self.assertEqual(send_response.status_code, 403)
        self.assertEqual(BulkEmailCampaign.objects.count(), 0)
