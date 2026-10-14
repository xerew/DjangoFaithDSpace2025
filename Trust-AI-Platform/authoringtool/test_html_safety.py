from django.contrib.auth.models import Group, User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from authoringtool.html_safety import json_for_script, sanitize_html
from authoringtool.models import Activity, ActivityType, Phase, Scenario
from usergroups.models import UserGroup, UserGroupMembership

ATTACK = (
    '<p>Hi</p><script>alert(1)</script>'
    '<img src="x.png" onerror="alert(2)">'
    '<a href="javascript:alert(3)">click</a>'
)


class SanitizeHtmlTests(SimpleTestCase):
    def test_removes_scripts_event_handlers_and_javascript_links(self):
        clean = sanitize_html(ATTACK)
        self.assertNotIn('<script', clean)
        self.assertNotIn('alert(1)', clean)
        self.assertNotIn('onerror', clean)
        self.assertNotIn('javascript:', clean)
        self.assertIn('<p>Hi</p>', clean)

    def test_keeps_the_formatting_teachers_use(self):
        html = (
            '<p style="color: red; text-align: center;" class="lead">Text</p>'
            '<table border="1"><tbody><tr><td colspan="2" valign="top">Cell</td></tr></tbody></table>'
            '<img src="/media/tinymce/a.png" alt="graph" width="300" height="200">'
            '<img src="data:image/png;base64,iVBORw0KGgo=" alt="pasted">'
            '<ul><li><strong>Bold</strong> and <em>italic</em><sub>2</sub></li></ul>'
        )
        clean = sanitize_html(html)
        for fragment in (
            'style="color: red; text-align: center;"', 'class="lead"', 'colspan="2"',
            'src="/media/tinymce/a.png"', 'width="300"', 'data:image/png;base64',
            '<strong>Bold</strong>', '<sub>2</sub>',
        ):
            self.assertIn(fragment, clean)

    def test_keeps_embeds_from_known_hosts_only(self):
        youtube = sanitize_html(
            '<iframe src="https://www.youtube.com/embed/abc" width="560" '
            'allowfullscreen="allowfullscreen"></iframe>'
        )
        self.assertIn('src="https://www.youtube.com/embed/abc"', youtube)
        self.assertIn('allowfullscreen', youtube)
        phet = sanitize_html('<iframe src="https://phet.colorado.edu/sims/x.html"></iframe>')
        self.assertIn('phet.colorado.edu', phet)
        evil = sanitize_html('<iframe src="https://evil.example/steal"></iframe>')
        self.assertNotIn('evil.example', evil)
        plain_http = sanitize_html('<iframe src="http://www.youtube.com/embed/abc"></iframe>')
        self.assertNotIn('src=', plain_http)

    def test_handles_empty_values(self):
        self.assertEqual(sanitize_html(None), '')
        self.assertEqual(sanitize_html(''), '')

    def test_json_for_script_cannot_close_the_script_tag(self):
        payload = '{"text": "</script><script>alert(1)</script> & more"}'
        escaped = json_for_script(payload)
        self.assertNotIn('</script>', escaped)
        self.assertNotIn('<script>', escaped)
        self.assertIn('\\u003C/script\\u003E', escaped)
        self.assertIn('\\u0026', escaped)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class RenderedHtmlTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('xss_owner', password='pass')
        self.owner.groups.add(teachers)
        self.student = User.objects.create_user('xss_student', password='pass')
        self.scenario = Scenario.objects.create(
            name='XSS scenario', visibility_status='private',
            llm_context='<script>alert("ctx")</script>Context',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Phase', scenario=self.scenario,
            llm_context='<img src=x onerror=alert("phase")>',
            created_by=self.owner, updated_by=self.owner,
        )
        activity_type = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Attack', text=ATTACK, phase=self.phase, scenario=self.scenario,
            activity_type=activity_type, created_by=self.owner, updated_by=self.owner,
        )
        group = UserGroup.objects.create(
            name='XSS class', prefix='xss', number_of_users=1, created_by=self.owner,
        )
        UserGroupMembership.objects.create(group=group, user=self.student)
        group.assigned_scenarios.add(self.scenario)

    def assertSafe(self, body):
        self.assertNotIn('<script>alert', body)
        self.assertNotIn('onerror=', body)
        self.assertNotIn('javascript:alert', body)

    def test_student_receives_sanitised_activity_content(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse('get_activity', args=[self.activity.id]))
        self.assertEqual(response.status_code, 200)
        content = response.json()['content']
        self.assertSafe(content)
        self.assertIn('<p>Hi</p>', content)

    def test_teacher_activity_page_is_sanitised(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse(
            'activity', args=[self.scenario.id, self.phase.id, self.activity.id],
        ))
        self.assertEqual(response.status_code, 200)
        self.assertSafe(response.content.decode())

    def test_llm_context_on_phase_page_is_sanitised(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse('phase', args=[self.scenario.id, self.phase.id]))
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertNotIn('onerror=alert("phase")', body)
