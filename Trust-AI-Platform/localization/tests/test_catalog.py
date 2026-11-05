import shutil
from unittest import skipUnless

import polib
from django.core.management import call_command
from django.test import SimpleTestCase

from localization.catalog import (
    EXTRA_DOUBLE_QUOTE, STRAY_PERCENT, UNBALANCED_BRACKETS, UNEXPECTED_LINE_BREAK, UNSAFE_BRACES,
    is_terms_only, load_po, placeholder_problems, po_path, translation_problems,
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

    def test_brace_fields_must_match_the_source(self):
        source = 'Your {language} translation of {name} is ready.'
        for bad in (
            'Η {language} μετάφραση του {name} {όνομα}',  # a name format_html has no value for
            'Η {language} μετάφραση του {name.__class__.__mro__} {name}',  # reads internals
            'Η {language} μετάφραση του {name[0]} {name}',  # index
            'Η {language} μετάφραση του {name!r}',  # conversion
            'Η {language} μετάφραση του {name:>10}',  # format spec
            'Η {language} μετάφραση του {name} {',  # lone opening brace
            'Η {language} μετάφραση του {name} }',  # lone closing brace
            'Η {language} μετάφραση του {name} {{',  # an added literal brace
            'Η {language} μετάφραση του {name}{{x}}',
        ):
            with self.subTest(bad=bad):
                self.assertIn(UNSAFE_BRACES, translation_problems(source, bad))

    def test_brace_fields_may_be_reordered(self):
        self.assertEqual(translation_problems(
            'Your {language} translation of {name} is ready.',
            'Το {name} είναι έτοιμο στα {language}.'), [])

    def test_text_without_brace_fields_gets_no_new_braces(self):
        for bad in ('Έλεγξε {ξανά}', 'Έλεγξε {', 'Έλεγξε }', 'Έλεγξε {{ξανά}}'):
            with self.subTest(bad=bad):
                self.assertIn(UNSAFE_BRACES, translation_problems('Check again', bad))
        self.assertEqual(translation_problems('Check again', 'Έλεγξε ξανά'), [])

    def test_literal_braces_in_the_source_may_be_kept(self):
        self.assertEqual(translation_problems('Use {{ and }} here', 'Χρησιμοποίησε {{ και }} εδώ'), [])
        self.assertEqual(translation_problems('Unbalanced { in source', 'Μονό { στην πηγή'), [])
        self.assertIn(UNSAFE_BRACES, translation_problems('Unbalanced { in source', 'Χωρίς'))

    def test_terms_only_entries_are_found_with_either_path_separator(self):
        def entry(*paths):
            return polib.POEntry(msgid='x', occurrences=[(path, '') for path in paths])

        for paths in (
            ['.\\accounts\\templates\\accounts\\tos.html'],
            ['./accounts/templates/accounts/tos.html'],
            ['accounts/templates/accounts/tos_public.html'],
            ['.\\accounts\\templates\\accounts\\tos\\pt.html'],
            ['./accounts/templates/accounts/tos/pt.html', './accounts/templates/accounts/tos.html'],
        ):
            with self.subTest(paths=paths):
                self.assertTrue(is_terms_only(entry(*paths)))
        for paths in (
            [],
            ['.\\templates\\head.html'],
            ['./accounts/templates/accounts/tos.html', './templates/head.html'],
            ['./accounts/templates/accounts/tos_other.html'],
            ['./other/templates/accounts/tos.html'],
        ):
            with self.subTest(paths=paths):
                self.assertFalse(is_terms_only(entry(*paths)))

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
