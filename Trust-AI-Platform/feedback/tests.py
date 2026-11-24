import json

from django.contrib.auth.models import Group, User
from django.db import IntegrityError, transaction
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse

from authoringtool.models import Scenario
from feedback.models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse


class FeedbackFormAppliesToTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_admin', password='pass', is_staff=True)
        self.scenario_a = Scenario.objects.create(name='FB Scenario A', created_by=self.user, updated_by=self.user)
        self.scenario_b = Scenario.objects.create(name='FB Scenario B', created_by=self.user, updated_by=self.user)

    def test_assign_to_all_applies_everywhere(self):
        form = FeedbackForm.objects.create(title='All', audience='student', assign_to_all=True, created_by=self.user)
        self.assertTrue(form.applies_to(self.scenario_a))
        self.assertTrue(form.applies_to(self.scenario_b))

    def test_assign_to_all_respects_exclusions(self):
        form = FeedbackForm.objects.create(title='All minus B', audience='student', assign_to_all=True, created_by=self.user)
        form.excluded_scenarios.add(self.scenario_b)
        self.assertTrue(form.applies_to(self.scenario_a))
        self.assertFalse(form.applies_to(self.scenario_b))

    def test_explicit_inclusion_mode(self):
        form = FeedbackForm.objects.create(title='Only A', audience='student', assign_to_all=False, created_by=self.user)
        form.included_scenarios.add(self.scenario_a)
        self.assertTrue(form.applies_to(self.scenario_a))
        self.assertFalse(form.applies_to(self.scenario_b))

    def test_inactive_form_never_applies(self):
        form = FeedbackForm.objects.create(title='Off', audience='student', assign_to_all=True, is_active=False, created_by=self.user)
        self.assertFalse(form.applies_to(self.scenario_a))


class FeedbackResponseConstraintTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_responder', password='pass')
        self.scenario = Scenario.objects.create(name='FB Constraint Scenario', created_by=self.user, updated_by=self.user)
        self.form = FeedbackForm.objects.create(title='F', audience='student', created_by=self.user)

    def test_one_response_per_form_user_scenario(self):
        FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)

    def test_one_answer_per_question_per_response(self):
        question = FeedbackQuestion.objects.create(form=self.form, text='Q1', question_type='text')
        response = FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)
        FeedbackAnswer.objects.create(response=response, question=question, answer_text='a')
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                FeedbackAnswer.objects.create(response=response, question=question, answer_text='b')

    def test_questions_ordered_by_order_field(self):
        FeedbackQuestion.objects.create(form=self.form, text='Second', question_type='text', order=2)
        FeedbackQuestion.objects.create(form=self.form, text='First', question_type='text', order=1)
        texts = list(self.form.questions.values_list('text', flat=True))
        self.assertEqual(texts, ['First', 'Second'])


class FeedbackSubmitEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.teacher = User.objects.create_user('fb_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.student = User.objects.create_user('fb_student', password='pass')
        self.scenario = Scenario.objects.create(name='FB Submit Scenario', created_by=self.teacher, updated_by=self.teacher)

        self.student_form = FeedbackForm.objects.create(title='Student form', audience='student', created_by=self.teacher)
        self.q_choice = FeedbackQuestion.objects.create(
            form=self.student_form, text='Useful?', question_type='choice',
            options=['Yes', 'No'], is_required=True, order=1,
        )
        self.q_text = FeedbackQuestion.objects.create(
            form=self.student_form, text='Comments?', question_type='text',
            is_required=False, order=2,
        )
        self.teacher_form = FeedbackForm.objects.create(title='Teacher form', audience='teacher', created_by=self.teacher)
        self.tq = FeedbackQuestion.objects.create(
            form=self.teacher_form, text='Proposals good?', question_type='choice',
            options=['Yes', 'No'], is_required=True, order=1,
        )

        self.greek = Scenario.objects.create(name='FB Submit Greek', language='Greek',
                                             created_by=self.teacher, updated_by=self.teacher)
        self.q_choice.translations = {'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', 'Όχι']}}
        self.q_choice.save()

    def _submit(self, form, answers):
        url = reverse('feedback_submit', args=[form.id, self.scenario.id])
        return self.client.post(url, json.dumps({'answers': answers}), content_type='application/json')

    def _submit_to(self, scenario, form, answers):
        url = reverse('feedback_submit', args=[form.id, scenario.id])
        return self.client.post(url, json.dumps({'answers': answers}), content_type='application/json')

    def test_choice_by_position_stores_original_wording_and_language(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit_to(self.greek, self.student_form, {str(self.q_choice.id): 1})
        self.assertTrue(r.json()['success'])
        response = FeedbackResponse.objects.get(user=self.student, scenario=self.greek)
        self.assertEqual(response.language, 'el')
        self.assertEqual(response.answers.get(question=self.q_choice).answer_text, 'No')

    def test_numeric_option_text_is_not_mistaken_for_position(self):
        q = FeedbackQuestion.objects.create(form=self.student_form, text='Rate', question_type='choice',
                                            options=['1', '2', '3', '4', '5'], is_required=False, order=3)
        self.client.login(username='fb_student', password='pass')
        self._submit_to(self.scenario, self.student_form, {str(self.q_choice.id): 0, str(q.id): 0})
        self.assertEqual(FeedbackAnswer.objects.get(question=q).answer_text, '1')
        FeedbackResponse.objects.all().delete()
        self._submit_to(self.scenario, self.student_form, {str(self.q_choice.id): 0, str(q.id): '5'})
        self.assertEqual(FeedbackAnswer.objects.get(question=q).answer_text, '5')

    def test_legacy_text_submission_still_accepted(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit_to(self.greek, self.student_form, {str(self.q_choice.id): 'Yes'})
        self.assertTrue(r.json()['success'])
        self.assertEqual(FeedbackAnswer.objects.get(question=self.q_choice).answer_text, 'Yes')

    def test_bad_positions_rejected(self):
        self.client.login(username='fb_student', password='pass')
        for bad in (2, -1, True, 1.0, 'Ναι'):
            r = self._submit_to(self.scenario, self.student_form, {str(self.q_choice.id): bad})
            self.assertEqual(r.status_code, 400, bad)
        self.assertFalse(FeedbackResponse.objects.exists())

    def test_required_error_quotes_question_in_shown_language(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit_to(self.greek, self.student_form, {})
        self.assertIn('Χρήσιμο;', r.json()['error'])

    def test_student_can_submit_student_form(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes', str(self.q_text.id): 'Great tool'})
        self.assertTrue(r.json()['success'])
        response = FeedbackResponse.objects.get(form=self.student_form, user=self.student, scenario=self.scenario)
        self.assertEqual(response.answers.count(), 2)

    def test_teacher_blocked_from_student_form(self):
        self.client.login(username='fb_teacher', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(FeedbackResponse.objects.filter(form=self.student_form).exists())

    def test_student_blocked_from_teacher_form(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.teacher_form, {str(self.tq.id): 'Yes'})
        self.assertEqual(r.status_code, 403)

    def test_form_not_applicable_to_scenario_blocked(self):
        self.student_form.assign_to_all = False
        self.student_form.save()
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        self.assertEqual(r.status_code, 403)

    def test_missing_required_answer_rejected(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_text.id): 'only optional'})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(FeedbackResponse.objects.filter(form=self.student_form).exists())

    def test_choice_answer_must_be_valid_option(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Maybe'})
        self.assertEqual(r.status_code, 400)

    def test_duplicate_submission_returns_friendly_error(self):
        self.client.login(username='fb_student', password='pass')
        self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        r = self._submit(self.student_form, {str(self.q_choice.id): 'No'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('already', r.json()['error'].lower())
        self.assertEqual(FeedbackResponse.objects.filter(form=self.student_form).count(), 1)

    def test_get_method_not_allowed(self):
        self.client.login(username='fb_student', password='pass')
        url = reverse('feedback_submit', args=[self.student_form.id, self.scenario.id])
        r = self.client.get(url)
        self.assertEqual(r.status_code, 405)

    def test_non_dict_answers_payload_rejected(self):
        self.client.login(username='fb_student', password='pass')
        url = reverse('feedback_submit', args=[self.student_form.id, self.scenario.id])
        r = self.client.post(url, json.dumps({'answers': 'bogus'}), content_type='application/json')
        self.assertEqual(r.status_code, 400)
        self.assertFalse(FeedbackResponse.objects.exists())

    def test_non_string_answer_value_treated_as_missing(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 42})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(FeedbackResponse.objects.exists())

    def test_integrity_error_on_race_returns_friendly_error(self):
        from unittest.mock import patch
        self.client.login(username='fb_student', password='pass')
        with patch('feedback.views.FeedbackResponse.objects') as mock_manager:
            mock_manager.filter.return_value.exists.return_value = False
            from django.db import IntegrityError as IE
            mock_manager.create.side_effect = IE('duplicate')
            r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('already', r.json()['error'].lower())


class FeedbackUtilsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_utils', password='pass')
        self.scenario = Scenario.objects.create(name='FB Utils Scenario', created_by=self.user, updated_by=self.user)

    def test_get_applicable_form_returns_newest_applicable(self):
        from feedback.utils import get_applicable_form
        older = FeedbackForm.objects.create(title='Older', audience='student', created_by=self.user)
        newer = FeedbackForm.objects.create(title='Newer', audience='student', created_by=self.user)
        self.assertEqual(get_applicable_form(self.scenario, 'student'), newer)

    def test_get_applicable_form_skips_wrong_audience_and_inactive(self):
        from feedback.utils import get_applicable_form
        FeedbackForm.objects.create(title='Teacher only', audience='teacher', created_by=self.user)
        FeedbackForm.objects.create(title='Inactive', audience='student', is_active=False, created_by=self.user)
        self.assertIsNone(get_applicable_form(self.scenario, 'student'))

    def test_serialize_form_shape(self):
        from feedback.utils import serialize_form
        form = FeedbackForm.objects.create(title='S', description='D', audience='student', created_by=self.user)
        FeedbackQuestion.objects.create(form=form, text='Q', question_type='choice', options=['A', 'B'], order=1)
        data = serialize_form(form)
        self.assertEqual(data['title'], 'S')
        self.assertEqual(len(data['questions']), 1)
        self.assertEqual(data['questions'][0]['options'], ['A', 'B'])
        self.assertEqual(data['questions'][0]['type'], 'choice')


class FeedbackManagementViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user('fb_staff', password='pass', is_staff=True)
        self.plain = User.objects.create_user('fb_plain', password='pass')
        self.scenario = Scenario.objects.create(name='FB Manage Scenario', created_by=self.staff, updated_by=self.staff)

    def test_non_staff_forbidden(self):
        self.client.login(username='fb_plain', password='pass')
        r = self.client.get(reverse('feedback_form_list'))
        self.assertEqual(r.status_code, 403)

    def test_staff_sees_form_list_with_response_counts(self):
        form = FeedbackForm.objects.create(title='Counted', audience='student', created_by=self.staff)
        FeedbackResponse.objects.create(form=form, user=self.plain, scenario=self.scenario)
        self.client.login(username='fb_staff', password='pass')
        r = self.client.get(reverse('feedback_form_list'))
        self.assertContains(r, 'Counted')
        self.assertEqual(r.context['forms'][0].response_count, 1)

    def test_create_form_with_questions(self):
        self.client.login(username='fb_staff', password='pass')
        r = self.client.post(reverse('feedback_form_create'), {
            'title': 'New Form',
            'description': 'Desc',
            'audience': 'teacher',
            'is_active': 'on',
            'assign_to_all': 'on',
            'scenarios': [str(self.scenario.id)],
            'questions_json': json.dumps([
                {'text': 'Useful?', 'type': 'choice', 'options': ['Yes', 'No'], 'required': True},
                {'text': 'Comments', 'type': 'text', 'options': [], 'required': False},
            ]),
        })
        self.assertRedirects(r, reverse('feedback_form_list'))
        form = FeedbackForm.objects.get(title='New Form')
        self.assertEqual(form.questions.count(), 2)
        self.assertTrue(form.assign_to_all)
        self.assertEqual(form.excluded_scenarios.count(), 0)

    def test_create_assign_to_all_unchecked_scenario_becomes_exclusion(self):
        other = Scenario.objects.create(name='FB Other Scenario', created_by=self.staff, updated_by=self.staff)
        self.client.login(username='fb_staff', password='pass')
        self.client.post(reverse('feedback_form_create'), {
            'title': 'Partial', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
            'scenarios': [str(self.scenario.id)],  # `other` left unchecked -> excluded
            'questions_json': json.dumps([{'text': 'Q', 'type': 'text', 'options': [], 'required': True}]),
        })
        form = FeedbackForm.objects.get(title='Partial')
        self.assertTrue(form.applies_to(self.scenario))
        self.assertFalse(form.applies_to(other))

    def test_create_without_assign_to_all_checked_scenarios_are_inclusions(self):
        other = Scenario.objects.create(name='FB Incl Scenario', created_by=self.staff, updated_by=self.staff)
        self.client.login(username='fb_staff', password='pass')
        self.client.post(reverse('feedback_form_create'), {
            'title': 'Incl', 'audience': 'student', 'is_active': 'on',
            'scenarios': [str(self.scenario.id)],
            'questions_json': json.dumps([{'text': 'Q', 'type': 'text', 'options': [], 'required': True}]),
        })
        form = FeedbackForm.objects.get(title='Incl')
        self.assertFalse(form.assign_to_all)
        self.assertTrue(form.applies_to(self.scenario))
        self.assertFalse(form.applies_to(other))

    def _answered_form(self, title='Editable'):
        form = FeedbackForm.objects.create(title=title, audience='student', created_by=self.staff)
        q = FeedbackQuestion.objects.create(form=form, text='Useful?', question_type='choice',
                                            options=['Yes', 'No'], order=0)
        response = FeedbackResponse.objects.create(form=form, user=self.plain, scenario=self.scenario)
        answer = FeedbackAnswer.objects.create(response=response, question=q, answer_text='No')
        return form, q, answer

    def _q(self, q, **changes):
        data = {'id': q.id, 'text': q.text, 'type': q.question_type, 'options': list(q.options),
                'required': q.is_required, 'translations': {}}
        data.update(changes)
        return data

    def _post_edit(self, form, questions, form_translations=None, source_language='en'):
        self.client.login(username='fb_staff', password='pass')
        return self.client.post(reverse('feedback_form_edit', args=[form.id]), {
            'title': form.title, 'audience': form.audience, 'is_active': 'on', 'assign_to_all': 'on',
            'source_language': source_language,
            'questions_json': json.dumps(questions),
            'form_translations_json': json.dumps(form_translations or {}),
        })

    def test_edit_keeps_answers_when_translating(self):
        form, q, answer = self._answered_form()
        self._post_edit(form, [self._q(q, translations={'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', 'Όχι']}})],
                        form_translations={'el': {'title': 'Σχόλια', 'description': ''}})
        q.refresh_from_db()
        form.refresh_from_db()
        self.assertTrue(FeedbackAnswer.objects.filter(pk=answer.pk).exists())
        self.assertEqual(q.translations['el']['options'], ['Ναι', 'Όχι'])
        self.assertEqual(form.translations['el']['title'], 'Σχόλια')

    def test_edit_creates_new_and_deletes_missing_questions(self):
        form, q_keep, keep_answer = self._answered_form()
        q_keep.text = 'Keep'
        q_keep.save()
        q_drop = FeedbackQuestion.objects.create(form=form, text='Drop', question_type='text', order=1)
        drop_answer = FeedbackAnswer.objects.create(response=keep_answer.response, question=q_drop, answer_text='x')
        self._post_edit(form, [self._q(q_keep),
                               {'id': None, 'text': 'New', 'type': 'text', 'options': [], 'required': False}])
        self.assertFalse(FeedbackQuestion.objects.filter(pk=q_drop.pk).exists())
        self.assertFalse(FeedbackAnswer.objects.filter(pk=drop_answer.pk).exists())
        self.assertTrue(FeedbackAnswer.objects.filter(pk=keep_answer.pk).exists())
        self.assertEqual(list(form.questions.values_list('text', flat=True)), ['Keep', 'New'])

    def test_answered_question_options_and_type_are_locked(self):
        form, q, answer = self._answered_form()
        for change in ({'options': ['Yes', 'No', 'Maybe']}, {'type': 'text', 'options': []}):
            r = self._post_edit(form, [self._q(q, **change)])
            self.assertEqual(r.status_code, 200)
            self.assertContains(r, 'already has answers')
            q.refresh_from_db()
            self.assertEqual((q.question_type, q.options), ('choice', ['Yes', 'No']))
            self.assertTrue(FeedbackAnswer.objects.filter(pk=answer.pk).exists())

    def test_unanswered_question_options_can_change(self):
        form = FeedbackForm.objects.create(title='Fresh', audience='student', created_by=self.staff)
        q = FeedbackQuestion.objects.create(form=form, text='Useful?', question_type='choice',
                                            options=['Yes', 'No'], order=0)
        self._post_edit(form, [self._q(q, options=['Yes', 'No', 'Maybe'])])
        q.refresh_from_db()
        self.assertEqual(q.options, ['Yes', 'No', 'Maybe'])

    def test_foreign_question_id_rejected(self):
        form, _q, _a = self._answered_form()
        other, other_q, _ = self._answered_form(title='Other')
        r = self._post_edit(form, [self._q(other_q, text='Hijacked')])
        self.assertContains(r, 'Invalid question data.')
        other_q.refresh_from_db()
        self.assertEqual((other_q.text, other_q.form_id), ('Useful?', other.id))

    def test_translation_validation(self):
        form, q, _ = self._answered_form()
        r = self._post_edit(form, [self._q(q, translations={'it': {'text': 'Utile?', 'options': []}})])
        self.assertContains(r, 'Unknown translation language: it.')
        r = self._post_edit(form, [self._q(q, translations={'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', 'Όχι', 'Ίσως']}})])
        self.assertContains(r, 'The Greek translation of &quot;Useful?&quot; has more options than the original.')

    def test_whitespace_translations_are_saved_empty(self):
        form, q, _ = self._answered_form()
        self._post_edit(form, [self._q(q, translations={'el': {'text': '   ', 'options': [' ', 'Όχι']}})])
        q.refresh_from_db()
        self.assertEqual(q.translations, {'el': {'text': '', 'options': ['', 'Όχι']}})

    def test_source_language_saved_and_validated(self):
        self.client.login(username='fb_staff', password='pass')
        payload = {'title': 'Greek original', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
                   'source_language': 'el',
                   'questions_json': json.dumps([{'text': 'Χρήσιμο;', 'type': 'text', 'options': [], 'required': True}])}
        self.client.post(reverse('feedback_form_create'), payload)
        self.assertEqual(FeedbackForm.objects.get(title='Greek original').source_language, 'el')
        r = self.client.post(reverse('feedback_form_create'), dict(payload, title='Bad', source_language='xx'))
        self.assertContains(r, 'Invalid original language.')
        self.assertFalse(FeedbackForm.objects.filter(title='Bad').exists())

    def test_edit_page_lists_ids_translations_and_answer_counts(self):
        form, q, _ = self._answered_form()
        q.translations = {'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', 'Όχι']}}
        q.save()
        self.client.login(username='fb_staff', password='pass')
        r = self.client.get(reverse('feedback_form_edit', args=[form.id]))
        item = json.loads(r.context['questions_json'])[0]
        self.assertEqual((item['id'], item['answer_count']), (q.id, 1))
        self.assertEqual(item['translations'], q.translations)

    def test_edit_page_has_language_controls_and_new_banner(self):
        form, q, _ = self._answered_form()
        self.client.login(username='fb_staff', password='pass')
        content = self.client.get(reverse('feedback_form_edit', args=[form.id])).content.decode()
        self.assertIn('name="source_language"', content)
        self.assertIn('name="form_translations_json"', content)
        self.assertIn('data-lang="el"', content)
        self.assertIn('Removing a question erases its answers', content)
        self.assertNotIn('permanently erase the answers already collected', content)

    def test_validation_error_keeps_posted_translations(self):
        form, q, _ = self._answered_form()
        r = self._post_edit(form, [self._q(q, options=['Yes', 'No', 'Maybe'])],
                            form_translations={'el': {'title': 'Σχόλια', 'description': ''}}, source_language='en')
        self.assertContains(r, 'already has answers')
        self.assertEqual(json.loads(r.context['form_translations_json']), {'el': {'title': 'Σχόλια', 'description': ''}})
        self.assertEqual(r.context['source_language'], 'en')

    def test_delete_form(self):
        form = FeedbackForm.objects.create(title='Doomed', audience='student', created_by=self.staff)
        self.client.login(username='fb_staff', password='pass')
        r = self.client.post(reverse('feedback_form_delete', args=[form.id]))
        self.assertRedirects(r, reverse('feedback_form_list'))
        self.assertFalse(FeedbackForm.objects.filter(id=form.id).exists())

    def test_delete_requires_post(self):
        form = FeedbackForm.objects.create(title='Get-safe', audience='student', created_by=self.staff)
        self.client.login(username='fb_staff', password='pass')
        r = self.client.get(reverse('feedback_form_delete', args=[form.id]))
        self.assertEqual(r.status_code, 405)
        self.assertTrue(FeedbackForm.objects.filter(id=form.id).exists())

    def test_create_rejects_choice_question_without_options(self):
        self.client.login(username='fb_staff', password='pass')
        r = self.client.post(reverse('feedback_form_create'), {
            'title': 'Bad', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
            'questions_json': json.dumps([{'text': 'Q', 'type': 'choice', 'options': [], 'required': True}]),
        })
        self.assertEqual(r.status_code, 200)  # re-rendered with error, not redirected
        self.assertFalse(FeedbackForm.objects.filter(title='Bad').exists())

    def test_new_form_page_defaults_all_scenarios_checked(self):
        import re
        self.client.login(username='fb_staff', password='pass')
        r = self.client.get(reverse('feedback_form_create'))
        m = re.search(r'<input[^>]*id="sc%d"[^>]*>' % self.scenario.id, r.content.decode('utf-8'))
        self.assertIsNotNone(m)
        self.assertIn('checked', m.group(0))

    def test_error_rerender_preserves_scenario_and_audience_selection(self):
        import re
        self.client.login(username='fb_staff', password='pass')
        r = self.client.post(reverse('feedback_form_create'), {
            'title': 'Bad', 'audience': 'student',
            'scenarios': [str(self.scenario.id)],
            'questions_json': json.dumps([{'text': 'Q', 'type': 'choice', 'options': [], 'required': True}]),
        })
        self.assertEqual(r.status_code, 200)
        content = r.content.decode('utf-8')
        m = re.search(r'<input[^>]*id="sc%d"[^>]*>' % self.scenario.id, content)
        self.assertIsNotNone(m)
        self.assertIn('checked', m.group(0))
        m2 = re.search(r'<option value="student"[^>]*>', content)
        self.assertIsNotNone(m2)
        self.assertIn('selected', m2.group(0))


class FeedbackResponsesAndExportTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user('fb_exp_staff', password='pass', is_staff=True)
        self.responder = User.objects.create_user('fb_exp_user', password='pass', first_name='Resp', last_name='Onder')
        self.scenario = Scenario.objects.create(name='FB Export Scenario', created_by=self.staff, updated_by=self.staff)
        self.form = FeedbackForm.objects.create(title='Export form', audience='student', created_by=self.staff)
        self.q1 = FeedbackQuestion.objects.create(form=self.form, text='Useful?', question_type='choice', options=['Yes', 'No'], order=1)
        self.q2 = FeedbackQuestion.objects.create(form=self.form, text='Comments', question_type='text', order=2)
        self.response = FeedbackResponse.objects.create(form=self.form, user=self.responder, scenario=self.scenario)
        FeedbackAnswer.objects.create(response=self.response, question=self.q1, answer_text='Yes')
        FeedbackAnswer.objects.create(response=self.response, question=self.q2, answer_text='Nice tool')
        self.client.login(username='fb_exp_staff', password='pass')

    def test_responses_page_lists_answers(self):
        r = self.client.get(reverse('feedback_form_responses', args=[self.form.id]))
        self.assertContains(r, 'Resp Onder')
        self.assertContains(r, 'Nice tool')

    def test_delete_response(self):
        r = self.client.post(reverse('feedback_response_delete', args=[self.response.id]))
        self.assertRedirects(r, reverse('feedback_form_responses', args=[self.form.id]))
        self.assertFalse(FeedbackResponse.objects.filter(id=self.response.id).exists())
        self.assertFalse(FeedbackAnswer.objects.exists())

    def test_delete_response_requires_post(self):
        r = self.client.get(reverse('feedback_response_delete', args=[self.response.id]))
        self.assertEqual(r.status_code, 405)

    def test_csv_export_uses_comma_and_contains_answers(self):
        r = self.client.get(reverse('feedback_form_export_csv', args=[self.form.id]))
        self.assertEqual(r['Content-Type'], 'text/csv')
        content = r.content.decode('utf-8')
        header = content.splitlines()[0]
        self.assertIn('Username,', header)
        self.assertIn('Useful?', header)
        self.assertIn('Nice tool', content)

    def test_xlsx_export_contains_answers(self):
        import io
        import openpyxl
        r = self.client.get(reverse('feedback_form_export_xlsx', args=[self.form.id]))
        self.assertEqual(r['Content-Type'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        self.assertEqual(rows[0][0], 'Username')
        self.assertIn('Useful?', rows[0])
        self.assertIn('Yes', rows[1])
        self.assertIn('Nice tool', rows[1])

    def test_exports_blocked_for_non_staff(self):
        self.client.logout()
        self.client.login(username='fb_exp_user', password='pass')
        r = self.client.get(reverse('feedback_form_export_csv', args=[self.form.id]))
        self.assertEqual(r.status_code, 403)


class TeacherFeedbackTriggerTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.teacher = User.objects.create_user('fb_trig_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.scenario = Scenario.objects.create(name='FB Trigger Scenario', created_by=self.teacher, updated_by=self.teacher)
        self.form = FeedbackForm.objects.create(title='Post-creation form', audience='teacher', created_by=self.teacher)
        FeedbackQuestion.objects.create(form=self.form, text='Good proposals?', question_type='choice', options=['Yes', 'No'], order=1)
        self.client.login(username='fb_trig_teacher', password='pass')

    def _create_personal(self):
        from unittest.mock import patch
        with patch('authoringtool.views.apply_user_proposals_to_new_scenario.delay') as mock_delay:
            return self.client.post(
                reverse(
                    'create_personal_scenario',
                    args=[self.scenario.id],
                ),
                follow=True,
            )

    def test_modal_context_present_after_creation(self):
        r = self._create_personal()
        self.assertIsNotNone(r.context['feedback_form_json'])
        self.assertContains(r, 'feedbackModal')

    def test_no_modal_without_creation_flow(self):
        r = self.client.get(reverse('proposal_list', args=[self.scenario.id]))
        self.assertIsNone(r.context['feedback_form_json'])

    def test_no_modal_when_already_responded(self):
        FeedbackResponse.objects.create(form=self.form, user=self.teacher, scenario=self.scenario)
        r = self._create_personal()
        self.assertIsNone(r.context['feedback_form_json'])

    def test_no_modal_when_no_applicable_form(self):
        self.form.is_active = False
        self.form.save()
        r = self._create_personal()
        self.assertIsNone(r.context['feedback_form_json'])

    def test_popup_is_served_in_scenario_language(self):
        self.scenario.language = 'Greek'
        self.scenario.save()
        self.form.translations = {'el': {'title': 'Σχόλια', 'description': ''}}
        self.form.save()
        r = self._create_personal()
        data = json.loads(r.context['feedback_form_json'])
        self.assertEqual((data['language'], data['title']), ('el', 'Σχόλια'))

    def test_session_flag_consumed_after_one_render(self):
        self._create_personal()
        r = self.client.get(reverse('proposal_list', args=[self.scenario.id]))
        self.assertIsNone(r.context['feedback_form_json'])


class StudentFeedbackTriggerTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.teacher = User.objects.create_user('fb_sv_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.student = User.objects.create_user('fb_sv_student', password='pass')
        self.scenario = Scenario.objects.create(name='FB SV Scenario', created_by=self.teacher, updated_by=self.teacher)
        from authoringtool.models import ActivityType, Phase, Activity
        phase = Phase.objects.create(name='P1', scenario=self.scenario, created_by=self.teacher, updated_by=self.teacher)
        atype = ActivityType.objects.create(name='Explanation', created_by=self.teacher, updated_by=self.teacher)
        Activity.objects.create(name='A1', text='x', scenario=self.scenario, phase=phase,
                                activity_type=atype, created_by=self.teacher, updated_by=self.teacher)
        self.form = FeedbackForm.objects.create(title='Post-scenario form', audience='student', created_by=self.teacher)
        FeedbackQuestion.objects.create(form=self.form, text='Fun?', question_type='choice', options=['Yes', 'No'], order=1)
        # Students only reach scenarios assigned to their group.
        from usergroups.models import UserGroup, UserGroupMembership
        group = UserGroup.objects.create(name='FB class', prefix='fbc', number_of_users=1, created_by=self.teacher)
        UserGroupMembership.objects.create(group=group, user=self.student)
        group.assigned_scenarios.add(self.scenario)

    def test_student_gets_feedback_form_in_context(self):
        self.client.login(username='fb_sv_student', password='pass')
        r = self.client.get(reverse('studentView', args=[self.scenario.id]))
        self.assertIsNotNone(r.context['feedback_form_json'])
        self.assertContains(r, 'feedbackModal')

    def test_popup_is_served_in_scenario_language(self):
        self.scenario.language = 'Greek'
        self.scenario.save()
        self.form.translations = {'el': {'title': 'Σχόλια', 'description': ''}}
        self.form.save()
        self.client.login(username='fb_sv_student', password='pass')
        r = self.client.get(reverse('studentView', args=[self.scenario.id]))
        data = json.loads(r.context['feedback_form_json'])
        self.assertEqual((data['language'], data['title']), ('el', 'Σχόλια'))

    def test_teacher_gets_no_feedback_form(self):
        self.client.login(username='fb_sv_teacher', password='pass')
        r = self.client.get(reverse('studentView', args=[self.scenario.id]))
        self.assertIsNone(r.context['feedback_form_json'])
        self.assertNotContains(r, 'id="feedbackModal"')

    def test_already_responded_student_gets_no_form(self):
        FeedbackResponse.objects.create(form=self.form, user=self.student, scenario=self.scenario)
        self.client.login(username='fb_sv_student', password='pass')
        r = self.client.get(reverse('studentView', args=[self.scenario.id]))
        self.assertIsNone(r.context['feedback_form_json'])

    def test_no_applicable_form_gives_none(self):
        self.form.is_active = False
        self.form.save()
        self.client.login(username='fb_sv_student', password='pass')
        r = self.client.get(reverse('studentView', args=[self.scenario.id]))
        self.assertIsNone(r.context['feedback_form_json'])


class FormEditorScenarioControlsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user('editor_staff', password='pass', is_staff=True)
        Scenario.objects.create(name='Editor Scenario', created_by=self.staff, updated_by=self.staff)
        self.client.login(username='editor_staff', password='pass')

    def test_scenario_search_input_rendered(self):
        r = self.client.get(reverse('feedback_form_create'))
        self.assertContains(r, 'id="scenarioSearch"')

    def test_no_match_hint_rendered(self):
        r = self.client.get(reverse('feedback_form_create'))
        self.assertContains(r, 'id="scenarioNoMatch"')


class FeedbackLanguageTests(SimpleTestCase):
    def test_language_code_accepts_code_english_and_native_names(self):
        from feedback.languages import language_code
        self.assertEqual(language_code('Greek'), 'el')
        self.assertEqual(language_code('GREEK'), 'el')
        self.assertEqual(language_code(' ελληνικά '), 'el')
        self.assertEqual(language_code('pt'), 'pt')
        self.assertIsNone(language_code('Italian'))
        self.assertIsNone(language_code(''))
        self.assertIsNone(language_code(None))


class FeedbackTranslationServingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_tr', password='pass')

        def mk(name, lang):
            return Scenario.objects.create(name=name, language=lang, created_by=self.user, updated_by=self.user)

        self.greek, self.italian, self.english, self.french = (
            mk('TR el', 'Greek'), mk('TR it', 'Italian'), mk('TR en', 'English'), mk('TR fr', 'French'))
        self.form = FeedbackForm.objects.create(
            title='Feedback', description='Tell us', audience='student', created_by=self.user,
            translations={'el': {'title': 'Σχόλια', 'description': ''}})
        self.q1 = FeedbackQuestion.objects.create(
            form=self.form, text='Useful?', question_type='choice', options=['Yes', 'No', 'Maybe'], order=1,
            translations={'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', '']}})
        self.q2 = FeedbackQuestion.objects.create(form=self.form, text='Comments?', question_type='text', order=2)

    def test_display_language_uses_scenario_language_when_translated(self):
        from feedback.utils import display_language
        self.assertEqual(display_language(self.form, self.greek), 'el')

    def test_display_language_falls_back_to_source(self):
        from feedback.utils import display_language
        for scenario in (self.italian, self.english, self.french):  # French: no fr translation
            self.assertEqual(display_language(self.form, scenario), 'en', scenario.language)

    def test_serialize_translates_field_by_field(self):
        from feedback.utils import serialize_form
        data = serialize_form(self.form, 'el')
        self.assertEqual(data['language'], 'el')
        self.assertEqual(data['title'], 'Σχόλια')
        self.assertEqual(data['description'], 'Tell us')
        self.assertEqual(data['questions'][0]['text'], 'Χρήσιμο;')
        self.assertEqual(data['questions'][0]['options'], ['Ναι', 'No', 'Maybe'])
        self.assertEqual(data['questions'][1]['text'], 'Comments?')

    def test_serialize_without_language_is_original(self):
        from feedback.utils import serialize_form
        data = serialize_form(self.form)
        self.assertEqual((data['language'], data['title']), ('en', 'Feedback'))
        self.assertEqual(data['questions'][0]['options'], ['Yes', 'No', 'Maybe'])

    def test_translation_in_source_language_is_ignored(self):
        from feedback.utils import display_language, serialize_form
        self.form.source_language = 'el'
        self.form.save()
        self.assertEqual(display_language(self.form, self.greek), 'el')
        self.assertEqual(serialize_form(self.form, 'el')['title'], 'Feedback')
