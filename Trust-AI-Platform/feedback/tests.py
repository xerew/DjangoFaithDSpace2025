import json

from django.contrib.auth.models import Group, User
from django.db import IntegrityError, transaction
from django.test import Client, TestCase
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

    def _submit(self, form, answers):
        url = reverse('feedback_submit', args=[form.id, self.scenario.id])
        return self.client.post(url, json.dumps({'answers': answers}), content_type='application/json')

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

    def test_edit_replaces_questions(self):
        form = FeedbackForm.objects.create(title='Editable', audience='student', created_by=self.staff)
        FeedbackQuestion.objects.create(form=form, text='Old Q', question_type='text', order=1)
        self.client.login(username='fb_staff', password='pass')
        self.client.post(reverse('feedback_form_edit', args=[form.id]), {
            'title': 'Editable v2', 'audience': 'student', 'is_active': 'on', 'assign_to_all': 'on',
            'questions_json': json.dumps([{'text': 'New Q', 'type': 'text', 'options': [], 'required': True}]),
        })
        form.refresh_from_db()
        self.assertEqual(form.title, 'Editable v2')
        self.assertEqual(list(form.questions.values_list('text', flat=True)), ['New Q'])

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

    def test_student_gets_feedback_form_in_context(self):
        self.client.login(username='fb_sv_student', password='pass')
        r = self.client.get(reverse('studentView', args=[self.scenario.id]))
        self.assertIsNotNone(r.context['feedback_form_json'])
        self.assertContains(r, 'feedbackModal')

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
