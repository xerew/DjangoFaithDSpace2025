import html

from django.contrib.auth.models import User
from django.core import mail
from django.test import TestCase
from django.urls import reverse

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.models import (
    Activity,
    ScenarioCollaborator,
    ScenarioTranslationJob,
)
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher
from organization.models import Organization


class ReviewerTests(TestCase):
    def setUp(self):
        self.owner = make_teacher('mt_rev_owner')
        self.creator = make_teacher('mt_rev_creator')
        self.admin = make_teacher('mt_rev_admin', admin=True)
        self.reviewer = make_teacher('mt_rev_reviewer')
        self.other = make_teacher('mt_rev_other')
        self.student = User.objects.create_user('mt_rev_student', 's@example.com', 'pw')
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.creator, 'Greek', 'English')
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(status='done')
        self.target = self.job.target_scenario

    def assign(self, user, as_user=None):
        self.client.force_login(as_user or self.admin)
        return self.client.post(
            reverse('assign_translation_reviewer', args=[self.target.id]),
            {'user_id': user.id},
        )

    def page(self, name, *args):
        response = self.client.get(reverse(name, args=args))
        return response.status_code, html.unescape(response.content.decode())

    def test_admin_assigns_teacher_reviewer(self):
        response = self.assign(self.reviewer)
        self.assertEqual(response.status_code, 302)
        collaborator = ScenarioCollaborator.objects.get(scenario=self.target, user=self.reviewer)
        self.assertEqual(collaborator.role, 'translation_reviewer')
        self.assertEqual(collaborator.added_by, self.admin)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].subject, 'You have been asked to review a translation')
        self.assertIn(reverse('viewScenario', args=[self.target.id]), mail.outbox[0].body)
        self.assertEqual(mail.outbox[0].to, [self.reviewer.email])

    def test_non_admin_cannot_assign(self):
        response = self.assign(self.reviewer, as_user=self.creator)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(ScenarioCollaborator.objects.exists())

    def test_student_cannot_be_assigned(self):
        response = self.assign(self.student)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(ScenarioCollaborator.objects.exists())

    def test_only_machine_translations_take_reviewers(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse('assign_translation_reviewer', args=[self.source.id]),
            {'user_id': self.reviewer.id},
        )
        self.assertEqual(response.status_code, 404)

    def test_reviewer_can_view_and_edit_private_translation(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        self.assertEqual(self.page('viewScenario', self.target.id)[0], 200)
        ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        self.client.post(
            reverse('updateActivityData', args=[self.target.id, ask.phase_id, ask.id]),
            {'activity_name': 'Διπλασιασμός', 'activity_text': '<p>Τι;</p>'},
        )
        ask.refresh_from_db()
        self.assertEqual(ask.name, 'Διπλασιασμός')

    def test_unassigned_teacher_cannot_view_private_translation(self):
        self.client.force_login(self.other)
        self.assertEqual(self.page('viewScenario', self.target.id)[0], 403)

    def test_reviewer_sees_translation_in_list(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        self.assertIn(f'sc-card-title">{self.target.name}<', self.page('scenarios')[1])

    def test_reviewer_cannot_change_visibility_or_delete(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        response = self.client.post(reverse('deleteScenario', args=[self.target.id]))
        self.assertEqual(response.status_code, 403)
        self.target.refresh_from_db()
        self.assertIsNone(self.target.archived_at)

    def test_creator_and_admin_can_remove_reviewer(self):
        for remover in (self.creator, self.admin):
            with self.subTest(remover=remover.username):
                self.assign(self.reviewer)
                self.client.force_login(remover)
                self.client.post(reverse(
                    'remove_scenario_collaborator', args=[self.target.id, self.reviewer.id],
                ))
                self.assertFalse(ScenarioCollaborator.objects.filter(user=self.reviewer).exists())

    def test_reviewer_cannot_remove_self_or_others(self):
        self.assign(self.reviewer)
        self.assign(self.other)
        self.client.force_login(self.reviewer)
        for user in (self.reviewer, self.other):
            response = self.client.post(reverse(
                'remove_scenario_collaborator', args=[self.target.id, user.id],
            ))
            self.assertEqual(response.status_code, 403)
        self.assertEqual(ScenarioCollaborator.objects.count(), 2)

    def test_reviewer_marks_verified(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        self.assertIn('Mark translation verified', self.page('viewScenario', self.target.id)[1])
        self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        self.job.refresh_from_db()
        self.assertEqual(self.job.verified_by, self.reviewer)
        self.assertIsNotNone(self.job.verified_at)
        self.client.force_login(self.creator)
        self.assertIn('Translation verified', self.page('scenarios')[1])

    def test_unrelated_teacher_cannot_verify(self):
        self.client.force_login(self.creator)
        response = self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        self.assertEqual(response.status_code, 403)
        self.job.refresh_from_db()
        self.assertIsNone(self.job.verified_by)

    def test_pending_translation_cannot_be_verified(self):
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(status='translating')
        self.client.force_login(self.admin)
        self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        self.job.refresh_from_db()
        self.assertIsNone(self.job.verified_by)

    def test_verification_kept_after_text_edit(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        self.client.post(
            reverse('updateActivityData', args=[self.target.id, ask.phase_id, ask.id]),
            {'activity_name': 'Νέο', 'activity_text': '<p>Νέο</p>'},
        )
        self.job.refresh_from_db()
        self.assertEqual(self.job.verified_by, self.reviewer)

    def test_admin_can_unverify(self):
        self.client.force_login(self.admin)
        self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        self.client.post(reverse('unverify_machine_translation', args=[self.target.id]))
        self.job.refresh_from_db()
        self.assertIsNone(self.job.verified_by)
        self.assertIsNone(self.job.verified_at)

    def test_non_admin_cannot_unverify(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        response = self.client.post(reverse('unverify_machine_translation', args=[self.target.id]))
        self.assertEqual(response.status_code, 403)
        self.job.refresh_from_db()
        self.assertEqual(self.job.verified_by, self.reviewer)

    def test_candidates_lists_teachers_suggested_first(self):
        org = Organization.objects.create(name='Lyceum', short_name='LY', language='greek')
        org.members.add(self.other)
        self.client.force_login(self.admin)
        response = self.client.get(
            reverse('translation_reviewer_candidates', args=[self.target.id]), {'q': 'mt_rev'},
        )
        users = response.json()['users']
        usernames = [user['username'] for user in users]
        self.assertEqual(usernames[0], 'mt_rev_other')
        self.assertTrue(users[0]['suggested'])
        self.assertNotIn('mt_rev_student', usernames)
        self.assertIn('mt_rev_reviewer', usernames)

    def test_candidates_admin_only(self):
        self.client.force_login(self.creator)
        response = self.client.get(reverse('translation_reviewer_candidates', args=[self.target.id]))
        self.assertEqual(response.status_code, 403)

    def test_reviewer_stays_after_verification(self):
        self.assign(self.reviewer)
        self.client.force_login(self.reviewer)
        self.client.post(reverse('verify_machine_translation', args=[self.target.id]))
        self.assertTrue(ScenarioCollaborator.objects.filter(user=self.reviewer).exists())
        self.assertEqual(self.page('viewScenario', self.target.id)[0], 200)

    def test_admin_sees_assign_button_on_card(self):
        self.client.force_login(self.admin)
        self.assertIn('Assign reviewer', self.page('scenarios')[1])
        self.client.force_login(self.creator)
        self.assertNotIn('Assign reviewer', self.page('scenarios')[1])
