"""Co-authors chosen by a scenario's creator from any organization."""
import json
from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import translation

from accounts.models import UserProfile
from authoringtool.co_authors import CANDIDATE_LIMIT
from authoringtool.machine_translation.collaborators import assign_reviewer, can_verify
from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.models import Phase, Scenario, ScenarioCollaborator, ScenarioTranslationJob
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher
from organization.models import Organization


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class CoAuthorBase(TestCase):
    def setUp(self):
        self.creator = make_teacher('ca_creator')
        self.teacher = make_teacher('ca_teacher')
        self.other = make_teacher('ca_other')
        self.admin = make_teacher('ca_admin', admin=True)
        self.student = User.objects.create_user('ca_student', 'student@example.com', 'pw')
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='private',
            created_by=self.creator, updated_by=self.creator,
        )

    def add_url(self, scenario=None):
        return reverse('add_co_author', args=[(scenario or self.scenario).id])

    def remove_url(self, user):
        return reverse('remove_co_author', args=[self.scenario.id, user.id])

    def add(self, actor, user):
        self.client.force_login(actor)
        return self.client.post(self.add_url(), {'user_id': user.id})

    def make_co_author(self, user=None):
        return ScenarioCollaborator.objects.create(
            scenario=self.scenario, user=user or self.teacher,
            role=ScenarioCollaborator.ROLE_CO_AUTHOR, added_by=self.creator,
        )


class AddingTests(CoAuthorBase):
    def test_creator_adds_a_teacher_from_another_organization(self):
        response = self.add(self.creator, self.teacher)
        self.assertEqual(response.status_code, 200)
        row = ScenarioCollaborator.objects.get(scenario=self.scenario, user=self.teacher)
        self.assertEqual(row.role, ScenarioCollaborator.ROLE_CO_AUTHOR)
        self.assertEqual(row.added_by, self.creator)
        self.assertEqual(
            [person['id'] for person in response.json()['co_authors']], [self.teacher.id],
        )

    def test_works_for_every_visibility(self):
        for visibility in ('private', 'org', 'public'):
            with self.subTest(visibility=visibility):
                ScenarioCollaborator.objects.all().delete()
                Scenario.objects.filter(pk=self.scenario.pk).update(visibility_status=visibility)
                self.assertEqual(self.add(self.creator, self.teacher).status_code, 200)
                self.assertTrue(self.scenario.collaborators.filter(user=self.teacher).exists())

    def test_admin_can_add(self):
        self.assertEqual(self.add(self.admin, self.teacher).status_code, 200)

    def test_other_teacher_cannot_add(self):
        self.assertEqual(self.add(self.other, self.teacher).status_code, 403)
        self.assertFalse(ScenarioCollaborator.objects.exists())

    def test_co_author_cannot_add_another_co_author(self):
        self.make_co_author()
        self.assertEqual(self.add(self.teacher, self.other).status_code, 403)
        self.assertFalse(self.scenario.collaborators.filter(user=self.other).exists())

    def test_only_active_teachers_can_be_added(self):
        inactive = make_teacher('ca_inactive')
        inactive.is_active = False
        inactive.save()
        for user in (self.student, inactive, self.creator):
            with self.subTest(user=user.username):
                self.assertEqual(self.add(self.creator, user).status_code, 400)
        self.assertFalse(ScenarioCollaborator.objects.exists())

    def test_get_is_not_allowed(self):
        self.client.force_login(self.creator)
        self.assertEqual(self.client.get(self.add_url()).status_code, 405)

    def test_adding_twice_keeps_one_row_and_one_email(self):
        self.add(self.creator, self.teacher)
        self.add(self.creator, self.teacher)
        self.assertEqual(self.scenario.collaborators.count(), 1)
        self.assertEqual(len(mail.outbox), 1)

    def test_a_translation_reviewer_stays_a_reviewer(self):
        ScenarioCollaborator.objects.create(
            scenario=self.scenario, user=self.teacher,
            role=ScenarioCollaborator.ROLE_TRANSLATION_REVIEWER,
        )
        self.add(self.creator, self.teacher)
        self.assertEqual(
            self.scenario.collaborators.get(user=self.teacher).role,
            ScenarioCollaborator.ROLE_TRANSLATION_REVIEWER,
        )


class RemovingTests(CoAuthorBase):
    def test_creator_removes_and_access_ends(self):
        self.make_co_author()
        self.client.force_login(self.creator)
        response = self.client.post(self.remove_url(self.teacher))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['co_authors'], [])
        self.client.force_login(self.teacher)
        self.assertEqual(
            self.client.get(reverse('updateScenario', args=[self.scenario.id])).status_code, 403,
        )

    def test_co_author_cannot_remove(self):
        self.make_co_author()
        self.make_co_author(self.other)
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.post(self.remove_url(self.other)).status_code, 403)
        self.assertEqual(self.scenario.collaborators.count(), 2)


class AccessTests(CoAuthorBase):
    def test_co_author_edits_a_private_scenario(self):
        phase = Phase.objects.create(
            name='Explore', scenario=self.scenario,
            created_by=self.creator, updated_by=self.creator,
        )
        self.make_co_author()
        self.client.force_login(self.teacher)
        self.client.post(
            reverse('updatePhaseData', args=[self.scenario.id, phase.id]),
            {'name': 'Explore (edited)', 'description': 'x'},
        )
        phase.refresh_from_db()
        self.assertEqual(phase.name, 'Explore (edited)')

    def test_co_author_cannot_delete(self):
        self.make_co_author()
        self.client.force_login(self.teacher)
        response = self.client.post(reverse('deleteScenario', kwargs={'id': self.scenario.id}))
        self.assertEqual(response.status_code, 403)


class CandidateTests(CoAuthorBase):
    def setUp(self):
        super().setUp()
        self.own_org = Organization.objects.create(name='Zeta School', short_name='zeta', created_by=self.creator)
        self.first_org = Organization.objects.create(name='Alpha College', short_name='alpha', created_by=self.creator)
        self.second_org = Organization.objects.create(name='Beta Institute', short_name='bi', created_by=self.creator)
        self.own_org.members.add(self.creator, self.teacher)
        self.first_org.members.add(self.teacher)
        self.second_org.members.add(self.admin)

    def candidates(self, query='', actor=None):
        self.client.force_login(actor or self.creator)
        return self.client.get(
            reverse('co_author_candidates', args=[self.scenario.id]), {'q': query},
        )

    def groups(self, query=''):
        return self.candidates(query).json()['groups']

    def usernames(self, query=''):
        return {person['username'] for group in self.groups(query) for person in group['users']}

    def test_groups_are_own_organization_then_a_to_z_then_none(self):
        self.assertEqual(
            [group['name'] for group in self.groups()],
            ['Zeta School', 'Alpha College', 'Beta Institute', None],
        )

    def test_your_organization_is_flagged(self):
        flags = {group['name']: group['yours'] for group in self.groups()}
        self.assertTrue(flags['Zeta School'])
        self.assertFalse(flags['Alpha College'])
        self.assertFalse(flags[None])

    def test_teacher_in_two_organizations_appears_under_both(self):
        by_name = {group['name']: group['users'] for group in self.groups()}
        for name in ('Zeta School', 'Alpha College'):
            self.assertIn('ca_teacher', [person['username'] for person in by_name[name]])

    def test_teacher_without_organization_is_under_none(self):
        by_name = {group['name']: group['users'] for group in self.groups()}
        self.assertIn('ca_other', [person['username'] for person in by_name[None]])

    def test_leaves_out_students_inactive_creator_and_existing_collaborators(self):
        inactive = make_teacher('ca_inactive')
        inactive.is_active = False
        inactive.save()
        self.make_co_author(self.other)
        found = self.usernames()
        for username in ('ca_student', 'ca_inactive', 'ca_creator', 'ca_other'):
            self.assertNotIn(username, found)
        self.assertIn('ca_teacher', found)

    def test_search_by_name_username_and_email(self):
        User.objects.filter(pk=self.teacher.pk).update(first_name='Maria', last_name='Papadaki')
        for query in ('Maria Papadaki', 'papad', 'ca_teacher', 'ca_teacher@example'):
            with self.subTest(query=query):
                self.assertEqual(self.usernames(query), {'ca_teacher'})

    def test_search_by_organization_name_or_short_name(self):
        for query in ('Alpha College', 'alpha'):
            with self.subTest(query=query):
                self.assertEqual(self.usernames(query), {'ca_teacher'})

    def test_search_ignores_profile_institution(self):
        UserProfile.objects.update_or_create(user=self.other, defaults={'institution': 'Hidden Academy'})
        self.assertEqual(self.usernames('Hidden Academy'), set())

    def test_rows_do_not_expose_email(self):
        self.assertNotContains(self.candidates(), 'example.com')

    def test_list_is_capped_and_says_so(self):
        teachers = Group.objects.get(name='teachers')
        for index in range(CANDIDATE_LIMIT + 1):
            user = User.objects.create_user(f'ca_many{index}', password='pw')
            user.groups.add(teachers)
        data = self.candidates().json()
        shown = {person['username'] for group in data['groups'] for person in group['users']}
        self.assertEqual(len(shown), CANDIDATE_LIMIT)
        self.assertTrue(data['truncated'])
        self.assertFalse(self.candidates('ca_teacher').json()['truncated'])

    def test_cap_keeps_the_creators_organization_first(self):
        teachers = Group.objects.get(name='teachers')
        for index in range(CANDIDATE_LIMIT + 1):
            user = User.objects.create_user(f'aaa_many{index}', password='pw')
            user.groups.add(teachers)
        self.assertEqual(self.groups()[0]['name'], 'Zeta School')

    def test_only_managers_can_list(self):
        self.make_co_author()
        for actor in (self.other, self.teacher, self.student):
            with self.subTest(actor=actor.username):
                self.assertEqual(self.candidates(actor=actor).status_code, 403)
        self.assertEqual(self.candidates(actor=self.admin).status_code, 200)


class EmailTests(CoAuthorBase):
    def test_new_co_author_gets_a_link(self):
        self.add(self.creator, self.teacher)
        self.assertEqual(len(mail.outbox), 1)
        message = mail.outbox[0]
        self.assertEqual(message.to, [self.teacher.email])
        self.assertIn('Pendulum', message.body)
        self.assertIn(reverse('viewScenario', args=[self.scenario.id]), message.body)

    def test_email_uses_the_recipients_language(self):
        UserProfile.objects.update_or_create(user=self.teacher, defaults={'language': 'el'})
        self.add(self.creator, self.teacher)
        with translation.override('el'):
            subject = translation.gettext('You were added as a co-author')
        self.assertEqual(mail.outbox[0].subject, subject)
        self.assertNotEqual(subject, 'You were added as a co-author')

    def test_email_failure_keeps_the_co_author(self):
        with patch('authoringtool.co_authors.send_mail', side_effect=OSError('smtp down')):
            with self.assertLogs('authoringtool.co_authors', level='ERROR'):
                response = self.add(self.creator, self.teacher)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.scenario.collaborators.filter(user=self.teacher).exists())

    def test_no_email_address_sends_nothing(self):
        User.objects.filter(pk=self.teacher.pk).update(email='')
        self.assertEqual(self.add(self.creator, self.teacher).status_code, 200)
        self.assertEqual(mail.outbox, [])


class ReviewerSeparationTests(TestCase):
    def setUp(self):
        self.creator = make_teacher('ca_rev_creator')
        self.admin = make_teacher('ca_rev_admin', admin=True)
        self.person = make_teacher('ca_rev_person')
        source = build_source_scenario(make_teacher('ca_rev_owner'))
        job = copy_for_translation(source, self.creator, 'Greek', 'English')
        ScenarioTranslationJob.objects.filter(pk=job.pk).update(status='done')
        # Fetched again so its translation job has the "done" status.
        self.target = Scenario.objects.get(pk=job.target_scenario_id)

    def co_author(self):
        return ScenarioCollaborator.objects.create(
            scenario=self.target, user=self.person, role=ScenarioCollaborator.ROLE_CO_AUTHOR,
        )

    def test_co_author_cannot_verify_a_translation(self):
        self.co_author()
        self.assertFalse(can_verify(self.person, self.target))

    def test_assigning_a_co_author_as_reviewer_lets_them_verify(self):
        self.co_author()
        assign_reviewer(self.target, self.person, self.admin)
        self.assertEqual(
            self.target.collaborators.get(user=self.person).role,
            ScenarioCollaborator.ROLE_TRANSLATION_REVIEWER,
        )
        self.assertTrue(can_verify(self.person, self.target))
        self.assertEqual(len(mail.outbox), 1)

    def test_scenario_page_lists_reviewers_not_co_authors(self):
        self.co_author()
        self.client.force_login(self.creator)
        response = self.client.get(reverse('viewScenario', args=[self.target.id]))
        self.assertEqual(list(response.context['collaborators']), [])


class ListingTests(CoAuthorBase):
    """A private scenario reaches its co-author everywhere its creator finds it."""

    def setUp(self):
        super().setUp()
        self.make_co_author()
        self.client.force_login(self.teacher)

    def scenario_ids(self, **params):
        response = self.client.get(reverse('scenarios'), params)
        return {scenario.id for scenario in response.context['myScenarios']}

    def test_dashboard_includes_it(self):
        response = self.client.get(reverse('index'))
        self.assertIn(self.scenario, response.context['scenarios'])

    def test_dashboard_leaves_it_out_for_others(self):
        self.client.force_login(self.other)
        response = self.client.get(reverse('index'))
        self.assertNotIn(self.scenario, response.context['scenarios'])

    def test_show_mine_filter_includes_it(self):
        self.assertIn(self.scenario.id, self.scenario_ids(show_mine='on'))

    def test_private_filter_includes_it(self):
        self.assertIn(self.scenario.id, self.scenario_ids(visibility='private'))

    def test_filters_leave_it_out_for_others(self):
        self.client.force_login(self.other)
        self.assertNotIn(self.scenario.id, self.scenario_ids(show_mine='on'))
        self.assertNotIn(self.scenario.id, self.scenario_ids(visibility='private'))

    def test_card_shows_the_co_author_label(self):
        response = self.client.get(reverse('scenarios'))
        self.assertContains(response, 'data-co-author-label')

    def test_card_has_no_label_for_the_creator(self):
        self.client.force_login(self.creator)
        response = self.client.get(reverse('scenarios'))
        self.assertNotContains(response, 'data-co-author-label')

    def test_new_group_lists_it_under_my_scenarios(self):
        response = self.client.get(reverse('create_group'))
        ids = [row['selected'].id for row in response.context['my_scenarios']]
        self.assertIn(self.scenario.id, ids)

    def test_student_scenario_page_lists_it_for_the_co_author(self):
        response = self.client.get(reverse('studentScenarios'))
        self.assertIn(self.scenario.id, [s.id for s in response.context['page_obj']])


class EditPageTests(CoAuthorBase):
    def page(self, actor):
        self.client.force_login(actor)
        return self.client.get(reverse('updateScenario', args=[self.scenario.id]))

    def test_creator_sees_the_section_with_current_co_authors(self):
        self.make_co_author()
        response = self.page(self.creator)
        self.assertContains(response, 'id="coAuthorSection"')
        people = response.context['co_authors']
        self.assertEqual([person['id'] for person in people], [self.teacher.id])

    def test_section_is_shown_for_every_visibility(self):
        for visibility in ('private', 'org', 'public'):
            with self.subTest(visibility=visibility):
                Scenario.objects.filter(pk=self.scenario.pk).update(visibility_status=visibility)
                self.assertContains(self.page(self.creator), 'id="coAuthorSection"')

    def test_co_author_does_not_see_the_section(self):
        self.make_co_author()
        self.assertNotContains(self.page(self.teacher), 'id="coAuthorSection"')

    def test_initial_list_is_safe_json(self):
        User.objects.filter(pk=self.teacher.pk).update(first_name='</script><b>x')
        self.make_co_author()
        html = self.page(self.creator).content.decode()
        self.assertNotIn('</script><b>x', html)
        start = html.index('id="co-author-data"')
        body = html[html.index('>', start) + 1:html.index('</script>', start)]
        self.assertEqual(json.loads(body)[0]['id'], self.teacher.id)
