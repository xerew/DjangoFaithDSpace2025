"""Languages whose interface translation is finished stay finished.

Each code in ``DONE_LANGUAGES`` must have a complete, safe ``.po`` catalog, a
compiled ``.mo`` that matches it, and a translated Terms of Service body.
"""
import re
from pathlib import Path

import polib
from django.conf import settings
from django.test import SimpleTestCase
from django.utils import translation

from localization.catalog import load_po, po_path, translation_problems, untranslated

DONE_LANGUAGES = ['es', 'el', 'fr', 'de', 'ro', 'pt']

BINDING_NOTE = 'This is a translation. The English version of these Terms is the binding one.'

TOS_DIR = Path(settings.BASE_DIR) / 'accounts' / 'templates' / 'accounts' / 'tos'
_HEADING = re.compile(r'<h2[^>]*>(.*?)</h2>', re.S)
_ID = re.compile(r'\bid="([^"]+)"')


def _key(entry):
    return (entry.msgctxt, entry.msgid)


def _entry_problems(entry):
    if entry.msgid_plural:
        problems = []
        for idx, text in sorted(entry.msgstr_plural.items()):
            if not text:
                continue  # reported by the untranslated test
            source = entry.msgid if int(idx) == 0 else entry.msgid_plural
            problems += [f'[{idx}] {p}' for p in translation_problems(source, text)]
        return problems
    if not entry.msgstr:
        return []  # reported by the untranslated test
    return translation_problems(entry.msgid, entry.msgstr)


def _translation(entry):
    if entry.msgid_plural:
        return {int(k): v for k, v in entry.msgstr_plural.items()}
    return entry.msgstr


class LanguageCompleteTests(SimpleTestCase):
    def test_catalog_has_no_untranslated_or_fuzzy_entries(self):
        for code in DONE_LANGUAGES:
            with self.subTest(code=code):
                missing = [(e.msgctxt, e.msgid) for e in untranslated(code)]
                self.assertEqual(missing, [], f'{len(missing)} untranslated entries in {code}')

    def test_every_entry_is_safe_to_render(self):
        for code in DONE_LANGUAGES:
            with self.subTest(code=code):
                bad = []
                for entry in load_po(code):
                    if entry.obsolete:
                        continue
                    bad += [(entry.msgid, p) for p in _entry_problems(entry)]
                self.assertEqual(bad, [])

    def test_binding_note_is_translated(self):
        for code in DONE_LANGUAGES:
            with self.subTest(code=code):
                entry = load_po(code).find(BINDING_NOTE)
                self.assertIsNotNone(entry)
                self.assertTrue(entry.msgstr)
                self.assertNotEqual(entry.msgstr, BINDING_NOTE)

    def test_terms_of_service_body_is_translated(self):
        english = (TOS_DIR / 'en.html').read_text(encoding='utf-8')
        english_headings = [h.strip() for h in _HEADING.findall(english)]
        english_ids = _ID.findall(english)
        self.assertTrue(english_headings)
        for code in DONE_LANGUAGES:
            with self.subTest(code=code):
                path = TOS_DIR / f'{code}.html'
                self.assertTrue(path.exists(), f'{path} is missing')
                body = path.read_text(encoding='utf-8')
                leftovers = [h for h in english_headings if h in body]
                self.assertEqual(leftovers, [], 'English headings left in the body')
                self.assertEqual(_ID.findall(body), english_ids)

    def test_compiled_catalog_matches_po(self):
        for code in DONE_LANGUAGES:
            with self.subTest(code=code):
                mo_path = po_path(code).with_suffix('.mo')
                self.assertTrue(mo_path.exists(), f'{mo_path} is missing')
                compiled = {_key(e): _translation(e) for e in polib.mofile(str(mo_path))}
                stale = []
                for entry in load_po(code):
                    if entry.obsolete or not entry.translated():
                        continue
                    if compiled.get(_key(entry)) != _translation(entry):
                        stale.append(entry.msgid)
                self.assertEqual(stale, [], 'the .mo is out of date: run compilemessages')

    def test_compiled_catalog_is_used(self):
        # A variable, not a literal, so makemessages does not list this file
        # as a source of the msgid.
        msgid = 'Ask for help'
        for code in DONE_LANGUAGES:
            with self.subTest(code=code):
                expected = load_po(code).find(msgid).msgstr
                self.assertTrue(expected)
                with translation.override(code):
                    self.assertEqual(translation.gettext(msgid), expected)
