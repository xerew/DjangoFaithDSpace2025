"""Proposal answers carry exactly one Latin label (A., B., ...) in every language.

The translator used to turn "A." into the Greek "Α." and "C." into "Γ.";
the label stripper only knew Latin letters, so answers showed "A. Α. ...".
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from authoringtool.tasks import format_proposal_answer_text, translate_proposal_fields


class AnswerLabelTests(SimpleTestCase):
    def test_greek_label_is_replaced(self):
        self.assertEqual(format_proposal_answer_text('Γ. Αφαιρέστε τη κονσέρβα', 2),
                         'C. Αφαιρέστε τη κονσέρβα')

    def test_stacked_labels_are_collapsed(self):
        self.assertEqual(format_proposal_answer_text('A. Α. Τοποθετήστε', 0), 'A. Τοποθετήστε')
        self.assertEqual(format_proposal_answer_text('B. B) Text', 1), 'B. Text')

    def test_abbreviation_at_the_start_is_kept(self):
        self.assertEqual(format_proposal_answer_text('Π.χ. ένα εκκρεμές', 0),
                         'A. Π.χ. ένα εκκρεμές')

    def test_latin_labels_still_work(self):
        self.assertEqual(format_proposal_answer_text('D) Example answer', 0), 'A. Example answer')


class TranslationKeepsLatinLabelsTests(SimpleTestCase):
    def test_answers_are_translated_without_their_label(self):
        seen = []

        def fake_translate(text, language):
            seen.append(text)
            return f'Ελ {text}'

        data = {
            'activity_name': 'n', 'content': 'c', 'explanation': 'e',
            'answers': [{'text': 'A. First'}, {'text': 'C. Third'}],
        }
        with patch('authoringtool.tasks.translate_text', side_effect=fake_translate):
            out = translate_proposal_fields(data, 'el')

        self.assertIn('First', seen)
        self.assertNotIn('A. First', seen)
        self.assertEqual([a['text'] for a in out['answers']], ['A. Ελ First', 'B. Ελ Third'])
