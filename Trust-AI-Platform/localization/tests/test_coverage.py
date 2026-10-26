"""Whole-platform checks: pages render fully translated, scripts are safe, .mo files are current."""
import json
import re
from pathlib import Path

import polib
from django.conf import settings
from django.contrib.auth.models import Group, User
from django.contrib.auth.tokens import default_token_generator
from django.template.defaultfilters import escape
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from accounts.models import UserProfile
from authoringtool.models import (
    Activity, ActivityProposal, ActivityType, Phase, ProposalGenerationRun, Scenario,
    ScenarioCollaborator, ScenarioTranslationJob, UserProposalReview,
)
from feedback.models import FeedbackForm, FeedbackQuestion
from localization.catalog import load_po, po_path
from organization.models import Organization
from usergroups.models import UserGroup, UserGroupMembership

from . import test_marked_admin, test_marked_authoring, test_marked_groups_student
from . import test_marked_metrics, test_marked_shell

_SENTINEL_SOURCES = [
    test_marked_shell.SENTINELS,
    test_marked_authoring.SENTINELS,
    test_marked_metrics.SENTINELS,
    test_marked_groups_student.SENTINELS,
    test_marked_admin.SENTINELS,
]


def _sentinels(*keys):
    """The English sentinels of the named pages, without placeholder ones."""
    found = []
    for key in keys:
        for source in _SENTINEL_SOURCES:
            if key in source and not key.endswith('_msgids'):
                found += [s for s in source[key] if '%(' not in s and '{' not in s]
    return list(dict.fromkeys(found))


def _find(text, html):
    """Where ``text`` shows in ``html`` as a whole word, or None."""
    # Whole-word match, so "Username" does not match the id "yourUsername".
    for variant in {text, escape(text)}:
        pattern = re.escape(variant)
        if re.match(r'\w', variant[0]):
            # A term kept in brackets after the translation, "(Consumer Key)", is fine.
            # Nor a JS property name: row.Category, row['Next Moderate'].
            pattern = r"(?<![\w(.-])(?<!\[')" + pattern
        if re.match(r'\w', variant[-1]):
            pattern += r'(?![\w-])'
        match = re.search(pattern, html)
        if match:
            return match
    return None


def _present(text, html):
    return _find(text, html) is not None

_COMMENTS = re.compile(r'<!--.*?-->|/\*.*?\*/|(?<=\s)//[^\n]*|^//[^\n]*', re.S | re.M)


def _contexts(texts, html):
    out = []
    for text in texts:
        at = _find(text, html).start()
        out.append(repr(html[max(at - 70, 0):at + len(text) + 40]))
    return ' || '.join(out)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class GreekPagesTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        teachers = Group.objects.create(name='teachers')
        cls.teacher = User.objects.create_user('cov_teacher', 'cov_t@example.com', 'pass')
        cls.teacher.groups.add(teachers)
        cls.staff = User.objects.create_user(
            'cov_staff', 'cov_s@example.com', 'pass', is_staff=True, is_superuser=True,
        )
        cls.staff.groups.add(teachers)
        cls.student = User.objects.create_user('cov_student', 'cov_st@example.com', 'pass')
        cls.scenario = Scenario.objects.create(
            name='Zzq Pendulum', visibility_status='public',
            created_by=cls.teacher, updated_by=cls.teacher,
        )
        phase = Phase.objects.create(
            name='Zzq Phase', scenario=cls.scenario, created_by=cls.teacher, updated_by=cls.teacher,
        )
        kind = ActivityType.objects.create(
            name='Zzq Explanation', created_by=cls.teacher, updated_by=cls.teacher,
        )
        Activity.objects.create(
            name='Zzq Read', text='Zzq Read', phase=phase, scenario=cls.scenario,
            activity_type=kind, created_by=cls.teacher, updated_by=cls.teacher,
        )
        cls.activity = Activity.objects.get(scenario=cls.scenario)
        cls.phase = phase
        cls.kind = kind
        cls._extra_fixtures()
        cls.group = UserGroup.objects.create(
            name='Zzq Class', prefix='zzq', number_of_users=1, created_by=cls.teacher,
        )
        cls.group.assigned_scenarios.add(cls.scenario)
        UserGroupMembership.objects.create(group=cls.group, user=cls.student)
        cls.org = Organization.objects.create(
            name='Zzq School', short_name='zzqs', created_by=cls.teacher,
        )
        cls.org.admins.add(cls.teacher)
        cls.org.members.add(cls.teacher)
        form = FeedbackForm.objects.create(
            title='Zzq Form', audience='teacher', created_by=cls.teacher,
        )
        FeedbackQuestion.objects.create(form=form, text='Zzq?', question_type='text')

    @classmethod
    def _extra_fixtures(cls):
        """States that change what the scenario and proposal pages show."""
        teacher = cls.teacher
        # A scenario without phases.
        cls.empty = Scenario.objects.create(
            name='Zzq Empty', visibility_status='public', created_by=teacher, updated_by=teacher,
        )
        # Machine translations in each state.
        cls.translations = {}
        for status in ('done', 'waiting', 'failed'):
            target = Scenario.objects.create(
                name=f'Zzq Translated {status}', language='Ελληνικά', visibility_status='public',
                created_by=teacher, updated_by=teacher,
            )
            ScenarioTranslationJob.objects.create(
                source_scenario=cls.scenario, target_scenario=target,
                source_language='English', target_language='Ελληνικά', requested_by=teacher,
                status=status, error='boom' if status == 'failed' else '',
            )
            cls.translations[status] = target
        reviewer = User.objects.create_user('cov_reviewer', 'cov_r@example.com', 'pass')
        reviewer.groups.add(Group.objects.get(name='teachers'))
        cls.reviewer = reviewer
        ScenarioCollaborator.objects.create(
            scenario=cls.translations['done'], user=reviewer,
            role=ScenarioCollaborator.ROLE_TRANSLATION_REVIEWER, added_by=teacher,
        )
        verified = Scenario.objects.create(
            name='Zzq Verified', language='Ελληνικά', visibility_status='public',
            created_by=teacher, updated_by=teacher,
        )
        ScenarioTranslationJob.objects.create(
            source_scenario=cls.scenario, target_scenario=verified, source_language='English',
            target_language='Ελληνικά', requested_by=teacher, status='done', verified_by=reviewer,
        )
        cls.verified = verified
        # Proposals: one pending of each kind, and a scenario whose proposals are all reviewed.
        cls.proposal_scenarios = []
        for reviewed in (False, True):
            scenario = Scenario.objects.create(
                name=f'Zzq Proposals {reviewed}', visibility_status='public',
                created_by=teacher, updated_by=teacher,
            )
            ph = Phase.objects.create(
                name='Zzq P', scenario=scenario, created_by=teacher, updated_by=teacher,
            )
            act = Activity.objects.create(
                name='Zzq Act', text='Which is true?', phase=ph, scenario=scenario,
                activity_type=cls.kind, created_by=teacher, updated_by=teacher,
            )
            run = ProposalGenerationRun.objects.create(scenario=scenario, is_current=True)
            for action in ('revise', 'create', 'skip'):
                payload = json.dumps({
                    'action': action, 'activity_name': 'Zzq Act', 'activity_type': 'Question',
                    'content': 'Based on your data, which is true?', 'explanation': 'x',
                    'answers': [{'text': 'A', 'is_correct': True, 'weight': 3}],
                    'insert_location': 'after',
                })
                proposal = ActivityProposal.objects.create(
                    scenario=scenario, generation_run=run, phase=ph, activity=act,
                    proposal_type=action, suggested_action='x', translated_action='x',
                    json_action=payload, json_translated_action=payload,
                )
                if reviewed:
                    UserProposalReview.objects.create(
                        proposal=proposal, user=teacher, status='accepted',
                    )
            cls.proposal_scenarios.append(scenario)

    def render(self, url, user, lang):
        client = Client()
        if user is not None:
            # A signed-in user's saved language wins over the browser's.
            UserProfile.objects.update_or_create(user=user, defaults={'language': lang})
            client.force_login(user)
        response = client.get(url, HTTP_ACCEPT_LANGUAGE=lang, follow=True)
        self.assertEqual(response.status_code, 200, url)
        # Comments are not shown to anyone.
        return _COMMENTS.sub('', response.content.decode('utf-8'))

    def check(self, label, url, user, *keys, extra=()):
        # ``extra`` are more (url, user) renders of the same page in other states.
        renders = [(url, user), *extra]
        english = '\n'.join(self.render(u, who, 'en') for u, who in renders)
        greek = '\n'.join(self.render(u, who, 'el') for u, who in renders)
        unchanged = {
            e.msgid for e in load_po('el')
            if not e.obsolete and e.msgstr == e.msgid and not e.msgid_plural
        }
        usable = _sentinels(*keys)
        # Sentinels that only show in some states are not in the English render: drop them.
        shown = [s for s in usable if _present(s, english)]
        sentinels = [s for s in shown if s not in unchanged]
        counts = f'{label}: {len(sentinels)} checked, {len(shown)} shown in English, {len(usable)} usable'
        self.counts.append(counts)
        with self.subTest(page=label):
            self.assertTrue('lang="el"' in greek, f'{label}: not lang=el')
            # The check must not go vacuous: at most half the usable sentinels may be
            # absent from the English render, and enough of them must be checked.
            self.assertGreaterEqual(len(shown) * 2, len(usable), 'too few sentinels render. ' + counts)
            self.assertGreaterEqual(len(sentinels), min(5, len(usable)), 'too few checked. ' + counts)
            left = [s for s in sentinels if _present(s, greek)]
            self.assertEqual(
                left, [], f'{label}: English left in the Greek page: ' + _contexts(left, greek),
            )

    def test_pages_fully_translated_in_greek(self):
        uid = urlsafe_base64_encode(force_bytes(self.teacher.pk))
        token = default_token_generator.make_token(self.teacher)
        sid = self.scenario.id
        pages = [
            ('login', reverse('login'), None, 'login'),
            ('register', reverse('register'), None, 'register'),
            ('password reset form', reverse('password_reset'), None, 'password_reset_form'),
            ('password reset done', reverse('password_reset_done'), None, 'password_reset_done'),
            ('password reset confirm', reverse('password_reset_confirm', args=[uid, token]),
             None, 'password_reset_confirm'),
            ('password reset complete', reverse('password_reset_complete'), None,
             'password_reset_complete'),
            ('scenarios', reverse('scenarios'), self.teacher, 'scenarios'),
            ('scenario view', reverse('viewScenario', args=[sid]), self.teacher, 'view_scenario'),
            ('scenario edit', reverse('updateScenario', args=[sid]), self.teacher, 'update_scenario'),
            ('metrics and AI', reverse('ai_metrics', args=[sid]), self.teacher, 'ai_metrics'),
            ('proposal list', reverse('proposal_list', args=[sid]), self.teacher, 'proposal_list'),
            ('group list', reverse('list_groups'), self.teacher, 'list_groups'),
            ('organization list', reverse('list_organizations'), self.teacher, 'list_organizations'),
            ('feedback forms', reverse('feedback_form_list'), self.staff, 'feedback_forms'),
            ('student scenarios', reverse('studentScenarios'), self.student, 'scenario_selection'),
            ('admin dashboard', reverse('admin_dashboard'), self.staff, 'admin_dashboard'),
            ('documentation', reverse('documentation_and_tutorials'), self.teacher, 'documentation'),
            ('terms of service', reverse('tos'), self.teacher, 'tos'),
            ('terms of service (public)', reverse('tos'), None, 'tos_public'),
        ]
        view = lambda scenario: reverse('viewScenario', args=[scenario.id])  # noqa: E731
        proposals = lambda scenario: reverse('proposal_list', args=[scenario.id])  # noqa: E731
        extras = {
            'scenario view': [
                (view(self.empty), self.teacher),
                (view(self.translations['done']), self.staff),
                (view(self.verified), self.staff),
                (view(self.translations['done']), self.reviewer),
                (view(self.translations['waiting']), self.teacher),
                (view(self.translations['failed']), self.teacher),
            ],
            'proposal list': [(proposals(self.proposal_scenarios[0]), self.teacher),
                              (proposals(self.proposal_scenarios[1]), self.teacher)],
        }
        self.counts = []
        for label, url, user, *keys in pages:
            self.check(label, url, user, *keys, extra=extras.get(label, ()))
        print('COVERAGE TABLE\n' + '\n'.join(self.counts))


_SCRIPT = re.compile(r'<script\b[^>]*>(.*?)</script\s*>', re.S | re.I)
_TAG = re.compile(r'\{%\s*(translate|trans|blocktranslate|blocktrans)\b(.*?)%\}', re.S)
_AS_VAR = re.compile(r'\bas(?:var)?\s+(\w+)')
_QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'')
_SKIP_DIRS = {'staticfiles', 'static', 'docs', 'djangofaithvenv', 'node_modules'}
_SKIP_FILES = {'chatBot.html', 'pendulum_modified_imu.html'}


def _script_offenders(text, name):
    problems = []
    for block in _SCRIPT.finditer(text):
        body = block.group(1)
        for tag in _TAG.finditer(body):
            kind, rest = tag.group(1), tag.group(2)
            line = text[:block.start(1) + tag.start()].count('\n') + 1
            match = _AS_VAR.search(_QUOTED.sub('', rest))
            if not match:
                problems.append(f'{name}:{line} bare {kind} inside <script>')
                continue
            var = match.group(1)
            escaped = re.search(r'\{\{\s*' + re.escape(var) + r'\s*\|\s*escapejs\b', body)
            in_json = 'json_script' in body and var in body
            if not escaped and not in_json:
                problems.append(f'{name}:{line} "{var}" is not output with |escapejs')
    return problems


class ScriptLintTests(TestCase):
    def test_lint_catches_bare_tags(self):
        bad = '<script>var a = "{% translate "Hi" %}"; var b = "{% blocktranslate %}x{% endblocktranslate %}";</script>'
        self.assertEqual(len(_script_offenders(bad, 'bad.html')), 2)
        good = '<script>{% translate "Mark as OK" as t %}var a = "{{ t|escapejs }}";</script>'
        self.assertEqual(_script_offenders(good, 'good.html'), [])
        unescaped = '<script>{% translate "Hi" as t %}var a = "{{ t }}";</script>'
        self.assertEqual(len(_script_offenders(unescaped, 'u.html')), 1)

    def test_no_bare_translate_inside_script(self):
        root = Path(settings.BASE_DIR)
        offenders = []
        for path in root.rglob('*.html'):
            rel = path.relative_to(root)
            if set(rel.parts[:-1]) & _SKIP_DIRS or path.name in _SKIP_FILES:
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
            offenders += _script_offenders(text, rel.as_posix())
        self.assertEqual(offenders, [])


class CompiledCatalogTests(TestCase):
    def test_mo_files_are_compiled(self):
        # Content, not timestamps: every translated, non-fuzzy .po entry is in the .mo.
        def value(entry):
            if entry.msgid_plural:
                return {int(k): v for k, v in entry.msgstr_plural.items()}
            return entry.msgstr

        codes = [code for code, _name in settings.LANGUAGES if code != 'en']
        self.assertTrue(codes)
        for code in codes:
            with self.subTest(code=code):
                mo_path = po_path(code).with_suffix('.mo')
                self.assertTrue(mo_path.exists(), f'{mo_path} is missing')
                compiled = {
                    (e.msgctxt, e.msgid): value(e) for e in polib.mofile(str(mo_path))
                }
                stale = [
                    e.msgid for e in load_po(code)
                    if not e.obsolete and e.translated() and 'fuzzy' not in e.flags
                    and compiled.get((e.msgctxt, e.msgid)) != value(e)
                ]
                self.assertEqual(stale, [], 'the .mo is out of date: run compilemessages')
