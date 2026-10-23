"""The shell, sign-in and account pages are marked for translation.

``SENTINELS`` maps a page name to exact English strings that page shows. A
later task renders each page in Greek and checks none of them remain.
"""
from django.test import SimpleTestCase

from localization.catalog import marked_msgids

PLATFORM_INTRO = 'Trust AI Lab Platform for STEM Education'
EU_FUNDED = 'Funded by the European Union'

SENTINELS = {
    'login': [
        'Login to Your Account',
        'Enter your username & password to login',
        'Username',
        'Password',
        'Login',
        PLATFORM_INTRO,
        EU_FUNDED,
        '<strong>Trust AI Lab Authoring Tool:</strong> Helps teachers design and implement inquiry-based learning scenarios.',
    ],
    'register': [
        'Create an Account',
        'Enter your personal details to create an account',
        'First Name',
        'Last Name',
        'Gender',
        'Prefer not to say',
        'Male',
        'Female',
        'Email',
        'Username',
        'Password',
        'Confirm Password',
        'Access Code',
        'Passwords do not match.',
        'Create Account',
        PLATFORM_INTRO,
    ],
    'profile': [
        'My Profile',
        'Account Settings',
        'Member since',
        'Last login',
        'Personal Info',
        'Change Password',
        'Current Password',
        'New Password',
        'Confirm New Password',
        'Username cannot be changed.',
        'Profile Picture',
        'Save Changes',
        'Update Password',
        'Change profile picture',
        'JPG, PNG, WEBP or GIF, up to 5 MB.',
        'Organizations',
        'Admin of',
        'Member of',
        'Start a conversation with this teacher.',
    ],
    'sidebar': [
        'Home',
        'Dashboard',
        'Authoring Tool',
        'Student Groups',
        'Student View',
        'Documentation & Tutorials',
        'Terms of Service',
        'Messages',
        'Platform Administration',
        'Sign Out',
        'Exit Impersonation',
    ],
    'footer': [
        EU_FUNDED,
    ],
    'main': [
        'Close',
    ],
    'maintenance_banner': [
        'Platform maintenance',
    ],
    'confirm_removal': [
        'Cancel',
    ],
    '404': [
        "The page you are looking for doesn't exist.",
        'Back to home',
    ],
    'password_reset_form': [
        'Reset Password',
        'Enter your email address to receive a password reset link',
        'Email Address',
        'Send Reset Link',
        'Back to Login',
    ],
    'password_reset_done': [
        'Reset Link Sent',
        'Check your email for a link to reset your password.',
        "If it doesn't arrive in a few minutes, please check your spam folder.",
        'Back to Login',
    ],
    'password_reset_confirm': [
        'Set New Password',
        'Your password must be at least 8 characters, not entirely numeric, and not a common password.',
        'New Password',
        'Confirm New Password',
        'Passwords do not match.',
    ],
    'password_reset_complete': [
        'Password Reset Complete',
        'Your password has been successfully reset.',
        'Login with New Password',
    ],
    'home': [
        'Get started',
        "You haven't created a scenario yet - start building your first one.",
        'Create your first scenario',
        'Your Scenarios',
        'Total Students',
        'About the Platform',
        'Inquiry-Based Learning with Trust AI Lab',
        'Latest Public Scenarios',
        'View all',
        'No public scenarios yet.',
        'Quick Actions',
        'Create Scenario',
        'Build a new IBL scenario for your students',
        'Manage your student groups and credentials',
        'Analytics Dashboard',
        'View student performance and scenario analytics',
    ],
    'messaging_threads': [
        'Organization Chats',
        'Team chat',
        "You're not part of any organization yet.",
        'No conversations yet.',
        "Message a colleague from their profile or an organization's member list.",
    ],
    'messaging_thread': [
        'View profile',
        'No messages yet. Say hello!',
        'Type a message…',
    ],
}

# Messages built in Python (flash messages and JSON errors).
PYTHON_SENTINELS = [
    'Invalid username or password.',
    'Username already taken.',
    'Email already in use.',
    'Invalid access code for Teacher registration.',
    'Access code is required for Teacher registration.',
    'Account created successfully. Please log in.',
    'AJAX required',
    'Unsupported file type. Use JPG, PNG, WEBP, or GIF.',
    'Image must be 5MB or smaller.',
    'Uploaded file is not a valid image.',
    'First name is required.',
    'Last name is required.',
    'Email is required.',
    'This email is already in use by another account.',
    'Bio must be 500 characters or fewer.',
    'Profile updated successfully.',
    'Choose an image to upload.',
    'Profile picture updated.',
    'Current password is incorrect.',
    'Password changed successfully.',
    'Unknown action.',
    'Teacher',
    'DSpace Partner',
    'Superuser',
    'Admin',
    'Message cannot be empty.',
    'Invalid recipient.',
]

# Strings used only inside <script> blocks.
SCRIPT_SENTINELS = [
    'New message from {name}',
    'Weak',
    'Fair',
    'Good',
    'Strong',
    'Creating account…',
    'Please fix the errors below.',
    'Network error. Please try again.',
    'The picture could not be saved.',
    'Sending…',
]


class MarkedShellTest(SimpleTestCase):
    def assertAllMarked(self, strings):
        marked = marked_msgids()
        missing = [s for s in strings if s not in marked]
        self.assertEqual(missing, [], f'not marked for translation: {missing}')

    def test_page_strings_are_marked(self):
        for page, strings in SENTINELS.items():
            with self.subTest(page=page):
                self.assertAllMarked(strings)

    def test_python_messages_are_marked(self):
        self.assertAllMarked(PYTHON_SENTINELS)

    def test_script_strings_are_marked(self):
        self.assertAllMarked(SCRIPT_SENTINELS)
