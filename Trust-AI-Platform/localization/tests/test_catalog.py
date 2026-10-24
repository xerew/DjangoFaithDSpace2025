import shutil
from unittest import skipUnless

from django.core.management import call_command
from django.test import SimpleTestCase

from localization.catalog import (
    EXTRA_DOUBLE_QUOTE, STRAY_PERCENT, UNBALANCED_BRACKETS, UNEXPECTED_LINE_BREAK,
    load_po, placeholder_problems, po_path, translation_problems,
)
from localization.languages import LANGUAGE_META

NON_ENGLISH = [m['code'] for m in LANGUAGE_META if m['code'] != 'en']


class CatalogTests(SimpleTestCase):
    def test_placeholder_problems_detects_dropped_and_added(self):
        self.assertTrue(any('%(name)s' in p for p in placeholder_problems('Hello %(name)s', 'Γεια')))
        self.assertIn('missing b', placeholder_problems('<b>Hi</b>', 'Γεια'))
        self.assertTrue(any('script' in p for p in placeholder_problems('Hi', 'Γεια <script>')))
        self.assertEqual(placeholder_problems('Hello %(name)s', 'Γεια %(name)s'), [])
        self.assertEqual(placeholder_problems('100%% sure', '100%% σίγουρα'), [])

    def test_translation_problems_keep_whole_tags(self):
        self.assertIn(UNBALANCED_BRACKETS, translation_problems('Hi', 'Γεια <img src=x onerror=alert(1) '))
        self.assertIn('unexpected <strong onmouseover=alert(1)>',
                      translation_problems('<strong>Hi</strong>', '<strong onmouseover=alert(1)>Γεια</strong>'))
        self.assertIn('missing <a href="%(url)s">',
                      translation_problems('<a href="%(url)s">Hi</a>', "<a href='%(url)s'>Γεια</a>"))
        self.assertEqual(translation_problems('<strong>Hi</strong>', '<strong>Γεια</strong>'), [])
        self.assertEqual(translation_problems('<a  href="x">Hi</a>', '<a href="x">Γεια</a>'), [])

    def test_translation_problems_refuse_extra_double_quotes(self):
        self.assertIn(EXTRA_DOUBLE_QUOTE, translation_problems('Search', 'S" autofocus onfocus="alert(1)'))
        self.assertEqual(translation_problems('Say "%(name)s"', 'Πες «%(name)s»'), [])
        self.assertEqual(translation_problems('Say "%(name)s"', 'Πες "%(name)s"'), [])

    def test_translation_problems_refuse_stray_percent_in_formatted_texts(self):
        self.assertIn(STRAY_PERCENT, translation_problems('%(name)s exists.', '%(name)s 100%.'))
        self.assertIn(STRAY_PERCENT, translation_problems('100%% wrong', '100% λάθος'))
        self.assertEqual(translation_problems('%(name)s exists.', '%(name)s 100%%.'), [])
        self.assertEqual(translation_problems('Done', 'Έγινε 100%'), [])

    def test_translation_problems_refuse_new_line_breaks(self):
        self.assertIn(UNEXPECTED_LINE_BREAK, translation_problems('Check again', 'Έλεγξε\nξανά'))
        self.assertEqual(translation_problems('One\nTwo', 'Ένα\nΔύο'), [])

    def test_translation_problems_include_placeholder_problems(self):
        self.assertIn('missing %(name)s', translation_problems('Hello %(name)s', 'Γεια'))

    def test_every_translation_keeps_placeholders(self):
        bad = []
        for code in NON_ENGLISH:
            for entry in load_po(code):
                if 'fuzzy' in entry.flags:
                    continue
                if entry.msgid_plural:
                    forms = entry.msgstr_plural
                    for idx, text in forms.items():
                        if not text:
                            continue
                        source = entry.msgid if int(idx) == 0 else entry.msgid_plural
                        bad += [(code, source, p) for p in translation_problems(source, text)]
                elif entry.msgstr:
                    bad += [(code, entry.msgid, p)
                            for p in translation_problems(entry.msgid, entry.msgstr)]
        self.assertEqual(bad, [])

    @skipUnless(shutil.which('xgettext'), 'GNU gettext is not installed')
    def test_update_translations_creates_po_for_all_languages(self):
        self.assertEqual(sorted(NON_ENGLISH), ['de', 'el', 'es', 'fr', 'pt', 'ro'])
        call_command('update_translations', verbosity=0)
        for code in NON_ENGLISH:
            self.assertTrue(po_path(code).exists(), code)
