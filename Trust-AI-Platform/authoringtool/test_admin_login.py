"""The admin login page must render on the Django version the server runs (4.0).

Our admin/base_site.html overrides Django's, so it may only use admin
templates that exist in that version.
"""

from django.test import TestCase


class AdminLoginPageTests(TestCase):
    def test_login_page_renders_for_people_not_logged_in(self):
        response = self.client.get('/admin/login/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="site-name"')
