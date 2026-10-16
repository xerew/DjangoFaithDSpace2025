"""Profile picture modal, clickable user names and organizations in the
Platform Administration user table."""
import io
import shutil
import tempfile

from django.contrib.auth.models import Group, User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.template import Context, Template
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from PIL import Image

from accounts.models import UserProfile
from authoringtool.models import Scenario
from organization.models import Organization

MEDIA_ROOT = tempfile.mkdtemp()


def png_upload(name='me.png'):
    buffer = io.BytesIO()
    Image.new('RGB', (4, 4), 'blue').save(buffer, format='PNG')
    return SimpleUploadedFile(name, buffer.getvalue(), content_type='image/png')


class ProfileBase(TestCase):
    def setUp(self):
        self.teachers = Group.objects.create(name='teachers')
        self.students = Group.objects.create(name='students')
        self.teacher = User.objects.create_user(
            'pf_teacher', password='pass', first_name='Tina', last_name='Teacher',
            email='tina@example.org',
        )
        self.other = User.objects.create_user(
            'pf_other', password='pass', first_name='Otto', last_name='Other',
        )
        for user in (self.teacher, self.other):
            user.groups.add(self.teachers)
        self.student = User.objects.create_user(
            'pf_student', password='pass', first_name='Stella', last_name='Student',
        )
        self.student.groups.add(self.students)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class ProfilePictureModalTests(ProfileBase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def upload(self, picture):
        return self.client.post(
            reverse('profile'),
            {'action': 'update_picture', 'picture': picture},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    def test_own_profile_avatar_opens_modal_with_change_button(self):
        self.client.force_login(self.teacher)

        response = self.client.get(reverse('profile'))

        self.assertContains(response, 'data-bs-target="#avatarModal"')
        self.assertContains(response, 'id="avatarModal"')
        self.assertContains(response, 'Change profile picture')

    def test_other_profile_avatar_enlarges_without_change_button(self):
        self.client.force_login(self.other)

        response = self.client.get(reverse('view_profile', args=[self.teacher.id]))

        self.assertContains(response, 'id="avatarModal"')
        self.assertNotContains(response, 'Change profile picture')

    def test_uploading_a_picture_saves_it_and_returns_its_url(self):
        self.client.force_login(self.teacher)

        response = self.upload(png_upload())

        self.assertTrue(response.json()['success'])
        profile = UserProfile.objects.get(user=self.teacher)
        self.assertTrue(profile.picture.name.endswith('.png'))
        self.assertEqual(response.json()['picture_url'], profile.picture.url)

    def test_picture_upload_keeps_other_profile_fields(self):
        UserProfile.objects.create(user=self.teacher, institution='NTUA', bio='Hi')
        self.client.force_login(self.teacher)

        self.upload(png_upload())

        profile = UserProfile.objects.get(user=self.teacher)
        self.teacher.refresh_from_db()
        self.assertEqual(profile.institution, 'NTUA')
        self.assertEqual(profile.bio, 'Hi')
        self.assertEqual(self.teacher.email, 'tina@example.org')

    def test_non_image_is_refused(self):
        self.client.force_login(self.teacher)

        response = self.upload(SimpleUploadedFile('notes.txt', b'hello'))

        self.assertFalse(response.json()['success'])
        self.assertIn('picture', response.json()['errors'])
        self.assertFalse(
            UserProfile.objects.filter(user=self.teacher).exclude(picture='').exists()
        )

    def test_upload_without_file_is_refused(self):
        self.client.force_login(self.teacher)

        response = self.client.post(
            reverse('profile'), {'action': 'update_picture'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertFalse(response.json()['success'])


class UserLinkTagTests(ProfileBase):
    def render(self, target, viewer):
        request = RequestFactory().get('/')
        request.user = viewer
        return Template('{% load profile_tags %}{% user_link target %}').render(
            Context({'target': target, 'request': request})
        )

    def test_teacher_name_links_to_profile_for_a_teacher(self):
        html = self.render(self.teacher, self.other)

        self.assertIn(f'href="{reverse("view_profile", args=[self.teacher.id])}"', html)
        self.assertIn('Tina Teacher', html)

    def test_own_name_links_to_own_profile(self):
        html = self.render(self.teacher, self.teacher)

        self.assertIn(f'href="{reverse("profile")}"', html)

    def test_student_name_is_not_a_link(self):
        html = self.render(self.student, self.teacher)

        self.assertNotIn('href=', html)
        self.assertIn('Stella Student', html)

    def test_viewer_who_cannot_open_profiles_sees_plain_name(self):
        html = self.render(self.teacher, self.student)

        self.assertNotIn('href=', html)

    def test_username_is_used_without_a_full_name_and_names_are_escaped(self):
        self.other.first_name = '<b>Bold</b>'
        self.other.last_name = ''
        self.other.save()

        html = self.render(self.other, self.teacher)

        self.assertIn('&lt;b&gt;Bold&lt;/b&gt;', html)
        self.assertNotIn('<b>', html)
        nameless = User.objects.create_user('nameless')
        nameless.groups.add(self.teachers)
        self.assertIn('nameless', self.render(nameless, self.teacher))

    def test_missing_user_renders_nothing(self):
        self.assertEqual(self.render(None, self.teacher).strip(), '')


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ScenarioPageUserLinkTests(ProfileBase):
    def test_creator_on_scenario_page_links_to_profile(self):
        scenario = Scenario.objects.create(
            name='Linked', visibility_status='public',
            created_by=self.teacher, updated_by=self.teacher,
        )
        self.client.force_login(self.other)

        response = self.client.get(reverse('viewScenario', args=[scenario.id]))

        self.assertContains(
            response, f'href="{reverse("view_profile", args=[self.teacher.id])}"',
        )


class AdminUserTableOrganizationTests(ProfileBase):
    def setUp(self):
        super().setUp()
        self.admin = User.objects.create_superuser('pf_admin', password='pass')
        member_org = Organization.objects.create(
            name='National Technical University of Athens', short_name='NTUA',
            created_by=self.admin,
        )
        member_org.members.add(self.teacher)
        admin_org = Organization.objects.create(
            name='Institute of Communication', short_name='ICCS',
            created_by=self.admin,
        )
        admin_org.admins.add(self.teacher)
        self.client.force_login(self.admin)

    def row(self, response, user):
        html = response.content.decode()
        start = html.index(f'data-user-id="{user.id}"')
        return html[start:html.index('</tr>', start)]

    def test_user_table_has_an_organization_column(self):
        response = self.client.get(reverse('admin_dashboard'))

        self.assertContains(response, '>Organization</th>')

    def test_member_and_admin_organizations_are_listed(self):
        response = self.client.get(reverse('admin_dashboard'))

        row = self.row(response, self.teacher)
        self.assertIn('NTUA', row)
        self.assertIn('ICCS', row)
        self.assertIn('title="National Technical University of Athens"', row)

    def test_user_without_organization_shows_a_dash(self):
        response = self.client.get(reverse('admin_dashboard'))

        self.assertIn('no-org', self.row(response, self.other))

    def test_organization_is_searchable(self):
        response = self.client.get(reverse('admin_dashboard'))

        self.assertIn('ntua', self.row(response, self.teacher).split('data-search="')[1].split('"')[0])
