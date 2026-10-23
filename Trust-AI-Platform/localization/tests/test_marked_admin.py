"""Admin pages, documentation and the Terms of Service wrappers are marked.

``SENTINELS`` maps a page name to exact English strings that page shows in a
normal state. Keys ending in ``_msgids`` hold strings that only show in some
states, sit inside loops, or have placeholders: they are msgids, not text
every reader sees, so a later render-in-Greek check skips them.
"""
from unittest import mock

from django.contrib.auth.models import User
from django.template import TemplateDoesNotExist
from django.template.loader import get_template, select_template
from django.test import TestCase
from django.urls import reverse

from localization.catalog import marked_msgids

SENTINELS = {
    'admin_dashboard': [
        'Platform Administration',
        'Administration',
        'Django Admin',
        'Feedback Forms',
        'Total Users',
        'Staff / Admins',
        'Users',
        'Roles',
        'Labs',
        'Email',
        'Maintenance',
        'Languages',
        'Search name, username or email…',
        'All Roles',
        'Clear filter',
        'Show inactive',
        'Last Login',
        'Add Role',
        'Can generate AI proposals for any scenario',
        'Search labs by name…',
        'Simulations',
        'Remote Labs',
        'Add Simulation',
        'Add Remote Lab',
        'Add VR Lab',
        'Iframe URL',
        'Consumer Key',
        'No simulations yet.',
        'No remote labs yet.',
        'No VR/AR labs yet.',
        'Edit User',
        'First Name',
        'Staff (can access admin)',
        'Save Changes',
        'Shared Secret',
        'QR code regenerates automatically on save',
        'Delete User',
        'Delete <strong id="deleteLabName"></strong>? Activities using this will lose their experiment link.',
        'Delete <strong id="deleteUserName"></strong>? This cannot be undone.',
    ],
    'admin_dashboard_script': [
        'User saved.',
        'User activated.',
        'User suspended.',
        'Delete this role?',
        'Role can now generate AI proposals.',
        'Role can no longer generate AI proposals.',
        'You will be logged in as this user. Continue?',
        'Leave blank to keep current secret.',
        'Edit Simulation',
        'Add VR/AR Lab',
        'Showing {start}–{end} of {total}',
    ],
    'admin_dashboard_msgids': [
        '%(n)d role selected',
        '%(n)d roles selected',
        'Showing {start}–{end} of %(n)d user',
        'Showing {start}–{end} of %(n)d users',
        '%(n)d item',
        '%(n)d items',
        '%(counter)s member',
        '%(counter)s members',
    ],
    'admin_email_tab': [
        'Compose message',
        'Maximum 200 characters.',
        'Choose recipients',
        'Review and send',
        'Recent campaigns',
        'No campaigns have been sent yet.',
        'Confirm campaign',
        'This action queues individual emails and cannot be undone.',
        'Queue emails',
        'Messages are sent individually to <strong>active users with the Teacher role and a valid email address</strong>. '
        'Recipient addresses are never shared, and duplicate email addresses receive only one copy.',
        'Could not count recipients.',
        'The selection has no deliverable teacher addresses.',
    ],
    'admin_email_tab_msgids': [
        'Sending paused until %(until)s: %(reason)s',
        'Up to %(limit)s emails go out per hour; sending starts within 5 minutes.',
        'by %(name)s',
        '%(n)s sent',
        '%(counter)s recipient',
        '%(counter)s recipients',
    ],
    'admin_languages_tab': [
        'No languages yet.',
        'Add language',
        'Language added.',
        'Remove this language?',
        'Click to rename',
        'The languages teachers can choose for scenarios and organizations. Renaming a language '
        'also renames it on every scenario and organization that uses it. Languages marked '
        '<i class="bi bi-lock-fill"></i> have built-in chat and help texts tied to their name and '
        'cannot be renamed. A language in use cannot be removed.',
    ],
    'admin_maintenance_tab': [
        'Maintenance notices',
        'The banner is automatically shown to everyone between the configured start and end times.',
        'Add notice',
        'No maintenance notices have been scheduled.',
    ],
    'admin_email_campaign': [
        'Campaign recipients',
        'Back to Email',
        'Resend to selected',
        'No recipients match this filter.',
        "Rebuilt from the campaign's totals, not recorded at sending time",
        'Tick the people to send the email to again.',
    ],
    'admin_email_campaign_msgids': [
        'Queue the email again for %(n)d person?',
        'Queue the email again for %(n)d people?',
        '%(n)d of these people already received it.',
        'Page %(number)s of %(total)s',
    ],
    'documentation': [
        'Documentation & Tutorials',
        'Help & Resources',
        'Guides, tutorials, external tools, and project information',
        'PDF Guide (English)',
        'Full platform user guide in English',
        'Platform Change Log',
        'Issue Reporting Form',
        'Associated Projects',
        "Visit Project's Site",
        'Discovery Space is an EU-funded Erasmus+ Partnerships for Innovation — Forward Looking project that '
        'aspires to develop a roadmap for the AI-Enhanced Classroom for Deeper Learning in STEM. It facilitates '
        'the transformation of the traditional classroom into an innovative environment that promotes scientific '
        'exploration and supports the development of key skills for all students.',
        'FAITH aims to provide the practitioners and stakeholders of AI systems not only with a comprehensive '
        'analysis of the foundations of AI trustworthiness, but also with an operational playbook for how to '
        'assess and build trustworthy AI systems and how to continuously measure their trustworthiness.',
    ],
    'documentation_msgids': [
        'Feature Request',
        'Available Scenarios',
        'Submit ideas and feature requests for the platform',
    ],
    'tos': [
        'Legal',
        'Terms of Service',
        'Please read these terms carefully before using the Trust AI Lab platform.',
        'Contents',
        'Description of the Platform',
        'Acceptance of Terms',
        'Communication & Consent',
        'EU Funding',
    ],
    'tos_public': [
        'Please read these terms carefully before creating an account.',
        'Back to Login',
        'Register',
        'This application was developed as part of the FAITH and DSPACE projects, which have received funding '
        'from the European Union under grant agreement No. 101135932 and No. 101086701. Views and opinions '
        'expressed are however those of the author(s) only and do not necessarily reflect those of the European '
        'Union or the European Education and Culture Executive Agency (EACEA). Neither the European Union nor '
        'EACEA can be held responsible for them.',
    ],
    'scenario_health': [
        'Scenario Health Check',
        'Total Scenarios',
        'No Issues',
        'Warnings',
        'Critical',
        'Fewer than 5 Phases',
        'Phases with No Activities',
        'No Evaluatable Activities',
        'Not Assigned to Any Student Group',
        'All scenarios have all 5 phases.',
        'All scenarios are assigned to at least one student group.',
        'Total Activities',
        'Implementations',
    ],
    'admin_base_site': [
        'Back to platform',
    ],
    'family_admin': [
        'Review rules',
        'Scenario family governance',
        'Associate scenarios manually',
        'Scan scenarios',
        'Ask Ollama to review pending',
        'Open candidate inbox',
        'Why not matched?',
        'Scan for translations',
        'Revision or scenario association?',
        'Choose scenarios and relationship',
        'Preview association',
        'Family that remains',
        'Selected scenario / source family',
        'Same family — translation',
        'Related topic only',
        'Passes every check.',
        'Side-by-side flow',
    ],
    'family_admin_msgids': [
        'Stopped by: %(label)s',
        'Confirm %(label)s',
        'Association preview: %(label)s',
        'revision v%(number)s',
        '%(counter)s variant',
        '%(counter)s variants',
        '%(counter)s student implementation',
        '%(counter)s student implementations',
    ],
}

PYTHON_SENTINELS = [
    'Access denied.',
    'Confirm the recipient count before sending.',
    'Subject is required.',
    'Role name is required.',
    'A role with this name already exists.',
    'Remove all members before deleting this role.',
    'Cannot impersonate a superuser.',
    'Language name is required.',
    'Language name must be at most %(max)d characters.',
    'A language called "%(name)s" already exists.',
    '"%(name)s" has built-in chat and help texts tied to its name, so it cannot be renamed.',
    'Renamed to "%(name)s" and updated %(scenarios)s and %(organizations)s.',
    '"%(name)s" is used by %(scenarios)s and %(organizations)s. Change their language first.',
    '%(n)d scenario',
    '%(n)d scenarios',
    '%(n)d organization',
    '%(n)d organizations',
    'Campaign queued for %(n)d recipient.',
    'Campaign queued for %(n)d recipients.',
    'Choose a valid recipient group.',
    'All teachers',
    '%(n)d selected teacher(s)',
    'Organizations: %(names)s',
]


class MarkedAdminTest(TestCase):
    def assertAllMarked(self, strings):
        marked = marked_msgids()
        missing = [s for s in strings if s not in marked]
        self.assertEqual(missing, [], f'not marked for translation: {missing}')

    def test_page_strings_are_marked(self):
        for page, strings in SENTINELS.items():
            with self.subTest(page=page):
                self.assertAllMarked(strings)

    def test_python_strings_are_marked(self):
        self.assertAllMarked(PYTHON_SENTINELS)

    def test_placeholder_strings_have_their_own_keys(self):
        for page, strings in SENTINELS.items():
            if any('%(' in s for s in strings):
                self.assertTrue(
                    page.endswith('_msgids'),
                    f'{page}: placeholder strings belong under a key ending in _msgids',
                )

    def test_every_touched_template_loads(self):
        for name in [
            'accounts/admin_dashboard.html',
            'accounts/_admin_email_tab.html',
            'accounts/_admin_languages_tab.html',
            'accounts/_admin_maintenance_tab.html',
            'accounts/admin_email_campaign.html',
            'accounts/documentation.html',
            'accounts/tos.html',
            'accounts/tos_public.html',
            'accounts/tos/en.html',
            'admin/base_site.html',
            'admin/authoringtool/scenario_health.html',
            'admin/authoringtool/scenariofamilycandidate/change_form.html',
            'admin/authoringtool/scenariofamilycandidate/change_list.html',
            'admin/authoringtool/scenariofamilyreviewproxy/associate_scenarios.html',
            'admin/authoringtool/scenariofamilyreviewproxy/change_list.html',
            'admin/authoringtool/translationmatch/change_list.html',
            'admin/authoringtool/translationmatch/explain.html',
        ]:
            with self.subTest(template=name):
                get_template(name)


class TermsLanguageTest(TestCase):
    def test_terms_fall_back_to_english(self):
        # tos/de.html does not exist yet: the English body is shown, with a 200.
        with self.assertRaises(TemplateDoesNotExist):
            get_template('accounts/tos/de.html')
        self.client.cookies['django_language'] = 'de'
        response = self.client.get(reverse('tos'), HTTP_ACCEPT_LANGUAGE='de')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Description of the Platform')
        self.assertContains(response, 'info@trust-ai-lab.eu')
        self.assertTemplateUsed(response, 'accounts/tos/en.html')

    def test_signed_in_terms_use_the_same_body(self):
        user = User.objects.create_user('tosuser', 'tos@example.com', 'pw')
        self.client.force_login(user)
        response = self.client.get(reverse('tos'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/tos.html')
        self.assertTemplateUsed(response, 'accounts/tos/en.html')
        self.assertContains(response, 'EU Funding Acknowledgment')

    def test_language_specific_body_wins_when_present(self):
        real = select_template

        def fake(names):
            if names[0] == 'accounts/tos/de.html':
                return real(['accounts/tos/en.html'])
            return real(names)

        with mock.patch('accounts.views.select_template', side_effect=fake) as patched:
            self.client.cookies['django_language'] = 'de'
            self.client.get(reverse('tos'))
        self.assertEqual(patched.call_args[0][0][0], 'accounts/tos/de.html')
