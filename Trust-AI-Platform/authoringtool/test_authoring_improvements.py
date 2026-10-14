"""Org co-editing (M2), phase/activity ordering (M4) and per-teacher scenario names (M5)."""
import csv
import io
import os
import shutil
import tempfile

from django.contrib.auth.models import Group, User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Phase,
    Scenario,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import compute_student_performance_metrics
from organization.models import Organization

TINY_GIF = (
    b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01'
    b'\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class AuthoringBase(TestCase):
    def setUp(self):
        self.teachers = Group.objects.create(name='teachers')
        self.owner = self.teacher('auth_owner')
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.explanation = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )

    def teacher(self, username):
        user = User.objects.create_user(username, password='pass')
        user.groups.add(self.teachers)
        return user

    def phase(self, name, scenario=None):
        return Phase.objects.create(
            name=name, scenario=scenario or self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )

    def activity(self, name, phase):
        return Activity.objects.create(
            name=name, text=name, phase=phase, scenario=phase.scenario,
            activity_type=self.explanation,
            created_by=self.owner, updated_by=self.owner,
        )


class OrgCoEditingTests(AuthoringBase):
    def setUp(self):
        super().setUp()
        self.editor = self.teacher('auth_editor')
        self.outsider = self.teacher('auth_outsider')
        org = Organization.objects.create(name='School', short_name='sch', created_by=self.owner)
        org.members.add(self.owner, self.editor)
        self.scenario.visibility_status = 'org'
        self.scenario.is_editable_by_org = True
        self.scenario.save()
        self.scenario.organizations.add(org)
        self.explore = self.phase('Explore')
        self.second = self.phase('Explain')
        self.read = self.activity('Read', self.explore)

    def update_phase(self, user):
        self.client.force_login(user)
        return self.client.post(
            reverse('updatePhaseData', args=[self.scenario.id, self.explore.id]),
            {'name': 'Explore (edited)', 'description': 'x'},
        )

    def test_org_editor_can_edit_content(self):
        response = self.update_phase(self.editor)
        self.assertNotEqual(response.status_code, 403)
        self.explore.refresh_from_db()
        self.assertEqual(self.explore.name, 'Explore (edited)')

    def test_org_editor_can_delete_an_activity(self):
        self.client.force_login(self.editor)
        self.client.post(reverse(
            'deleteActivity', args=[self.scenario.id, self.explore.id, self.read.id],
        ))
        self.assertFalse(Activity.objects.filter(pk=self.read.pk).exists())

    def test_outsider_cannot_edit(self):
        self.assertEqual(self.update_phase(self.outsider).status_code, 403)

    def test_org_member_cannot_edit_when_editing_is_not_shared(self):
        self.scenario.is_editable_by_org = False
        self.scenario.save()
        self.assertEqual(self.update_phase(self.editor).status_code, 403)

    def test_org_editor_cannot_delete_the_scenario(self):
        self.client.force_login(self.editor)
        response = self.client.post(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        self.assertEqual(response.status_code, 403)

    def test_org_editor_edits_details_but_not_sharing(self):
        self.client.force_login(self.editor)
        self.client.post(reverse('updateScenarioData', args=[self.scenario.id]), {
            'name': 'Pendulum (edited)', 'visibility': 'public',
            'learning_goals': 'g', 'description': 'd', 'min_age': 12, 'max_age': 15,
            'language': 'English', 'suggested_learning_time': 30,
        })
        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.name, 'Pendulum (edited)')
        self.assertEqual(self.scenario.visibility_status, 'org')
        self.assertTrue(self.scenario.is_editable_by_org)
        self.assertTrue(self.scenario.organizations.exists())

    def test_view_pages_offer_editing_to_org_editors(self):
        self.client.force_login(self.editor)
        response = self.client.get(reverse('viewScenario', args=[self.scenario.id]))
        self.assertTrue(response.context['can_edit'])


class RagUploadPermissionTests(AuthoringBase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        settings_override = override_settings(RAG_PDFS_ROOT=self.root)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        self.scenario.visibility_status = 'public'
        self.scenario.save()
        self.phase('Explore')

    def upload(self, user, name='notes.pdf', content=b'%PDF-1.4 x'):
        self.client.force_login(user)
        return self.client.post(
            reverse('viewScenario', args=[self.scenario.id]),
            {'rag_pdf': SimpleUploadedFile(name, content, content_type='application/pdf')},
        )

    def folder(self):
        path = os.path.join(self.root, f'scenario_{self.scenario.id}')
        return os.listdir(path) if os.path.isdir(path) else []

    def test_viewer_of_public_scenario_cannot_upload(self):
        response = self.upload(self.teacher('auth_viewer'))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.folder(), [])

    def test_owner_uploads_pdf(self):
        self.upload(self.owner)
        self.assertEqual(self.folder(), ['notes.pdf'])

    def test_non_pdf_is_rejected(self):
        self.upload(self.owner, name='evil.html', content=b'<script>')
        self.assertEqual(self.folder(), [])


class ScenarioNameTests(AuthoringBase):
    def create(self, user, name):
        self.client.force_login(user)
        return self.client.post(reverse('createScenarioData'), {
            'name': name, 'learning_goals': 'g', 'description': 'd',
            'min_age': 12, 'max_age': 15, 'language': 'English',
            'suggested_learning_time': 30,
            'image_upload': SimpleUploadedFile('a.gif', TINY_GIF, content_type='image/gif'),
        })

    def test_two_teachers_can_use_the_same_name(self):
        other = self.teacher('auth_other_school')
        self.create(other, 'Pendulum')
        self.assertEqual(Scenario.objects.filter(name='Pendulum').count(), 2)

    def test_one_teacher_cannot_reuse_a_name(self):
        self.create(self.owner, 'Pendulum')
        self.assertEqual(Scenario.objects.filter(name='Pendulum').count(), 1)


class OrderingTests(AuthoringBase):
    def setUp(self):
        super().setUp()
        self.first = self.phase('Engage')
        self.second = self.phase('Explore')
        self.third = self.phase('Explain')
        self.client.force_login(self.owner)

    def names(self):
        return list(self.scenario.phases.values_list('name', flat=True))

    def test_new_phases_and_activities_go_last(self):
        self.assertEqual(self.names(), ['Engage', 'Explore', 'Explain'])
        a = self.activity('A', self.first)
        b = self.activity('B', self.first)
        self.assertLess(a.order, b.order)

    def test_move_phase_up_and_down(self):
        self.client.post(reverse('move_phase', args=[self.scenario.id, self.third.id, 'up']))
        self.assertEqual(self.names(), ['Engage', 'Explain', 'Explore'])
        self.client.post(reverse('move_phase', args=[self.scenario.id, self.first.id, 'down']))
        self.assertEqual(self.names(), ['Explain', 'Engage', 'Explore'])

    def test_move_activity_within_its_phase(self):
        a = self.activity('A', self.first)
        b = self.activity('B', self.first)
        self.client.post(reverse('move_activity', args=[self.scenario.id, self.first.id, b.id, 'up']))
        self.assertEqual(
            list(self.first.activities.values_list('name', flat=True)), ['B', 'A'],
        )

    def test_moving_requires_edit_rights_and_post(self):
        self.client.get(reverse('move_phase', args=[self.scenario.id, self.third.id, 'up']))
        self.assertEqual(self.names(), ['Engage', 'Explore', 'Explain'])
        self.client.force_login(self.teacher('auth_stranger'))
        response = self.client.post(
            reverse('move_phase', args=[self.scenario.id, self.third.id, 'up'])
        )
        self.assertEqual(response.status_code, 403)

    def test_analytics_follow_the_teachers_phase_order(self):
        self.client.post(reverse('move_phase', args=[self.scenario.id, self.third.id, 'up']))
        student = User.objects.create_user('auth_student')
        student.school_department = SchoolDepartment.objects.create(name='Dept')
        student.save()
        read = self.activity('Read', self.first)
        UserScenarioScore.objects.create(user=student, scenario=self.scenario, user_score=0)
        UserAnswer.objects.create(user=student, activity=read, timing=10)
        result = compute_student_performance_metrics(self.scenario.id, [], None, None)
        header = next(csv.reader(io.StringIO(result['csv_content'])))
        phase_columns = [h for h in header if h.endswith(' Categorization') and h != 'Final Categorization']
        self.assertEqual(
            phase_columns,
            ['Engage Categorization', 'Explain Categorization', 'Explore Categorization'],
        )

    def test_duplicate_keeps_the_order(self):
        self.client.post(reverse('move_phase', args=[self.scenario.id, self.third.id, 'up']))
        self.client.post(reverse('duplicate_scenario', args=[self.scenario.id]))
        copy = Scenario.objects.exclude(pk=self.scenario.pk).get()
        self.assertEqual(
            list(copy.phases.values_list('name', flat=True)),
            ['Engage', 'Explain', 'Explore'],
        )
