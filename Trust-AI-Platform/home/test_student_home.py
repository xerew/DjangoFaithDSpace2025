"""Students reach Student View from the logo and from /home/, not a 403."""
from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse


class HomeForStudentsTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user('home_teacher', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))
        self.student = User.objects.create_user('home_student', password='pass')

    def logo_href(self, response):
        html = response.content.decode()
        start = html.index('class="logo')
        tag_start = html.rindex('<a ', 0, start)
        return html[tag_start:start]

    def test_student_opening_home_goes_to_student_view(self):
        self.client.force_login(self.student)

        response = self.client.get(reverse('teacher_home'))

        self.assertRedirects(response, reverse('studentScenarios'))

    def test_teacher_still_gets_the_home_page(self):
        self.client.force_login(self.teacher)

        response = self.client.get(reverse('teacher_home'))

        self.assertEqual(response.status_code, 200)

    def test_anonymous_visitor_is_sent_to_login(self):
        response = self.client.get(reverse('teacher_home'))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response['Location'])

    def test_logo_points_students_to_student_view(self):
        self.client.force_login(self.student)

        response = self.client.get(reverse('studentScenarios'))

        self.assertIn(f'href="{reverse("studentScenarios")}"', self.logo_href(response))

    def test_logo_points_teachers_home(self):
        self.client.force_login(self.teacher)

        response = self.client.get(reverse('studentScenarios'))

        self.assertIn(f'href="{reverse("teacher_home")}"', self.logo_href(response))
