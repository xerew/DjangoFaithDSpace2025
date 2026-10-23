import shutil
from unittest import skipUnless

from django.core.management import call_command
from django.test import SimpleTestCase

from localization.catalog import load_po, placeholder_problems, po_path
from localization.languages import LANGUAGE_META

NON_ENGLISH = [m['code'] for m in LANGUAGE_META if m['code'] != 'en']


class CatalogTests(SimpleTestCase):
    def test_placeholder_problems_detects_dropped_and_added(self):
        self.assertTrue(any('%(name)s' in p for p in placeholder_problems('Hello %(name)s', 'Γεια')))
        self.assertIn('missing b', placeholder_problems('<b>Hi</b>', 'Γεια'))
        self.assertTrue(any('script' in p for p in placeholder_problems('Hi', 'Γεια <script>')))
        self.assertEqual(placeholder_problems('Hello %(name)s', 'Γεια %(name)s'), [])
        self.assertEqual(placeholder_problems('100%% sure', '100%% σίγουρα'), [])

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
                        bad += [(code, source, p) for p in placeholder_problems(source, text)]
                elif entry.msgstr:
                    bad += [(code, entry.msgid, p)
                            for p in placeholder_problems(entry.msgid, entry.msgstr)]
        self.assertEqual(bad, [])

    @skipUnless(shutil.which('xgettext'), 'GNU gettext is not installed')
    def test_update_translations_creates_po_for_all_languages(self):
        call_command('update_translations', verbosity=0)
        for code in NON_ENGLISH:
            self.assertTrue(po_path(code).exists(), code)
