"""Choosing which NTUA address the Ollama SSH tunnel uses, from Django admin.

The tunnel container and Django share a small folder: Django writes the
admin's choice to `target`, the tunnel writes what it is doing to `status`.
"""

import shutil
import tempfile
import time
from pathlib import Path

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from . import ollama_tunnel

HOSTS = ['147.102.17.66', '147.102.23.222']


def write_status(folder, updated=None, **values):
    lines = {
        'state': 'connected',
        'host': HOSTS[0],
        'mode': 'auto',
        'hosts': ','.join(HOSTS),
        'since': str(int(time.time()) - 120),
        'updated': str(int(updated if updated is not None else time.time())),
        'message': '',
    }
    lines.update(values)
    Path(folder, 'status').write_text(
        ''.join(f'{key}={value}\n' for key, value in lines.items()),
        encoding='utf-8',
    )


class ControlFolderMixin:
    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        override = override_settings(OLLAMA_TUNNEL_CONTROL_DIR=self.folder)
        override.enable()
        self.addCleanup(override.disable)


class ReadStatusTests(ControlFolderMixin, SimpleTestCase):
    def test_no_status_file_means_the_tunnel_is_not_reporting(self):
        status = ollama_tunnel.read_status()
        self.assertFalse(status['reporting'])
        self.assertEqual(status['hosts'], [])
        self.assertEqual(status['state'], '')

    def test_fresh_status_is_read(self):
        write_status(self.folder)
        status = ollama_tunnel.read_status()
        self.assertTrue(status['reporting'])
        self.assertEqual(status['state'], 'connected')
        self.assertEqual(status['host'], HOSTS[0])
        self.assertEqual(status['mode'], 'auto')
        self.assertEqual(status['hosts'], HOSTS)
        self.assertIsNotNone(status['since'])
        self.assertIsNotNone(status['updated'])

    def test_old_status_means_the_tunnel_stopped_reporting(self):
        write_status(self.folder, updated=time.time() - 600)
        status = ollama_tunnel.read_status()
        self.assertFalse(status['reporting'])
        # what it last said is still shown
        self.assertEqual(status['host'], HOSTS[0])
        self.assertEqual(status['hosts'], HOSTS)

    def test_unknown_lines_and_blank_values_are_ignored(self):
        Path(self.folder, 'status').write_text(
            'state=connecting\nnonsense\nhost=\nhosts=147.102.17.66, \n'
            f'updated={int(time.time())}\n',
            encoding='utf-8',
        )
        status = ollama_tunnel.read_status()
        self.assertEqual(status['state'], 'connecting')
        self.assertEqual(status['host'], '')
        self.assertEqual(status['hosts'], ['147.102.17.66'])
        self.assertIsNone(status['since'])


class TargetTests(ControlFolderMixin, SimpleTestCase):
    def test_automatic_is_the_default(self):
        target = ollama_tunnel.read_target()
        self.assertEqual(target['target'], ollama_tunnel.AUTO)
        self.assertEqual(target['changed_by'], '')
        self.assertIsNone(target['changed_at'])

    def test_choosing_an_address_writes_it_first_for_the_tunnel(self):
        write_status(self.folder)
        ollama_tunnel.set_target(HOSTS[1], 'admin1')
        lines = Path(self.folder, 'target').read_text(encoding='utf-8').splitlines()
        self.assertEqual(lines[0], HOSTS[1])
        target = ollama_tunnel.read_target()
        self.assertEqual(target['target'], HOSTS[1])
        self.assertEqual(target['changed_by'], 'admin1')
        self.assertIsNotNone(target['changed_at'])

    def test_back_to_automatic(self):
        write_status(self.folder)
        ollama_tunnel.set_target(HOSTS[1], 'admin1')
        ollama_tunnel.set_target(ollama_tunnel.AUTO, 'admin2')
        self.assertEqual(ollama_tunnel.read_target()['target'], ollama_tunnel.AUTO)

    def test_an_address_the_server_does_not_list_is_refused(self):
        write_status(self.folder)
        with self.assertRaises(ValueError):
            ollama_tunnel.set_target('10.0.0.1', 'admin1')
        self.assertFalse(Path(self.folder, 'target').exists())

    def test_a_fixed_address_needs_the_tunnel_to_have_listed_it(self):
        # no status file: the allowed addresses are unknown
        with self.assertRaises(ValueError):
            ollama_tunnel.set_target(HOSTS[0], 'admin1')

    def test_automatic_can_always_be_chosen(self):
        ollama_tunnel.set_target(ollama_tunnel.AUTO, 'admin1')
        self.assertEqual(ollama_tunnel.read_target()['target'], ollama_tunnel.AUTO)

    def test_no_temporary_files_are_left_behind(self):
        write_status(self.folder)
        ollama_tunnel.set_target(HOSTS[0], 'admin1')
        self.assertEqual(
            sorted(p.name for p in Path(self.folder).iterdir()),
            ['status', 'target'],
        )

    def test_line_breaks_in_the_username_cannot_add_lines(self):
        write_status(self.folder)
        ollama_tunnel.set_target(HOSTS[0], 'evil\n10.0.0.1')
        lines = Path(self.folder, 'target').read_text(encoding='utf-8').splitlines()
        self.assertEqual(lines[0], HOSTS[0])
        self.assertEqual(len(lines), 3)


class AdminPageTests(ControlFolderMixin, TestCase):
    url_name = 'ai_server_connection'

    def setUp(self):
        super().setUp()
        self.url = reverse(self.url_name)
        self.staff = User.objects.create_user(
            'staff1', password='pw', is_staff=True,
        )
        self.teacher = User.objects.create_user('teacher1', password='pw')

    def test_people_who_are_not_staff_are_sent_to_the_admin_login(self):
        self.client.force_login(self.teacher)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('admin:login'), response['Location'])

    def test_staff_see_the_connection(self):
        write_status(self.folder)
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Connected to 147.102.17.66')
        for host in HOSTS:
            self.assertContains(response, f'value="{host}"')
        self.assertContains(response, 'value="auto"')

    def test_page_says_when_the_tunnel_is_not_reporting(self):
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'The tunnel is not reporting')

    def test_page_shows_the_last_error_when_no_address_works(self):
        write_status(
            self.folder, state='waiting', host='',
            message='Host key verification failed.',
        )
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertContains(response, 'Host key verification failed.')

    def test_staff_can_pick_a_fixed_address(self):
        write_status(self.folder)
        self.client.force_login(self.staff)
        response = self.client.post(self.url, {'target': HOSTS[1]})
        self.assertRedirects(response, self.url, fetch_redirect_response=False)
        self.assertEqual(ollama_tunnel.read_target()['target'], HOSTS[1])
        self.assertEqual(ollama_tunnel.read_target()['changed_by'], 'staff1')

    def test_staff_can_go_back_to_automatic(self):
        write_status(self.folder)
        ollama_tunnel.set_target(HOSTS[1], 'staff1')
        self.client.force_login(self.staff)
        self.client.post(self.url, {'target': 'auto'})
        self.assertEqual(ollama_tunnel.read_target()['target'], ollama_tunnel.AUTO)

    def test_an_unlisted_address_is_refused(self):
        write_status(self.folder)
        self.client.force_login(self.staff)
        response = self.client.post(self.url, {'target': '10.0.0.1'}, follow=True)
        self.assertContains(response, 'not one of the addresses')
        self.assertFalse(Path(self.folder, 'target').exists())

    def test_teachers_cannot_change_it(self):
        write_status(self.folder)
        self.client.force_login(self.teacher)
        self.client.post(self.url, {'target': HOSTS[1]})
        self.assertFalse(Path(self.folder, 'target').exists())

    def test_it_is_listed_on_the_admin_index_for_staff(self):
        self.client.force_login(self.staff)
        response = self.client.get(reverse('admin:index'))
        self.assertContains(response, 'AI server connection')
