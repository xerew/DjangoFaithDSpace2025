import os
import shutil
import tempfile

from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import Scenario


class ServeRagPdfAccessTests(TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        settings_override = override_settings(RAG_PDFS_ROOT=self.root)
        settings_override.enable()
        self.addCleanup(settings_override.disable)

        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('rag_owner', password='pass')
        self.other_teacher = User.objects.create_user('rag_other', password='pass')
        self.student = User.objects.create_user('rag_student', password='pass')
        for teacher in (self.owner, self.other_teacher):
            teacher.groups.add(teachers)

        self.private_scenario = self.scenario_with_pdf('Private RAG', 'private')
        self.public_scenario = self.scenario_with_pdf('Public RAG', 'public')

    def scenario_with_pdf(self, name, visibility):
        scenario = Scenario.objects.create(
            name=name, visibility_status=visibility,
            created_by=self.owner, updated_by=self.owner,
        )
        folder = os.path.join(self.root, f'scenario_{scenario.id}')
        os.makedirs(folder)
        with open(os.path.join(folder, 'notes.pdf'), 'wb') as pdf:
            pdf.write(b'%PDF-1.4 test')
        return scenario

    def fetch(self, scenario):
        response = self.client.get(
            reverse('serve_rag_pdf', args=[scenario.id, 'notes.pdf'])
        )
        if response.status_code == 200:
            # Read rather than close(): closing fires request_finished, which
            # would close the test database connection.
            self.assertEqual(response.getvalue(), b'%PDF-1.4 test')
        return response

    def test_owner_can_read_private_pdf(self):
        self.client.force_login(self.owner)
        self.assertEqual(self.fetch(self.private_scenario).status_code, 200)

    def test_other_teacher_cannot_read_private_pdf(self):
        self.client.force_login(self.other_teacher)
        self.assertEqual(self.fetch(self.private_scenario).status_code, 403)

    def test_other_teacher_can_read_public_scenario_pdf(self):
        self.client.force_login(self.other_teacher)
        self.assertEqual(self.fetch(self.public_scenario).status_code, 200)

    def test_student_cannot_read_pdf(self):
        self.client.force_login(self.student)
        self.assertEqual(self.fetch(self.public_scenario).status_code, 403)
