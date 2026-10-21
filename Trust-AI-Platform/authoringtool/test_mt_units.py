from django.test import SimpleTestCase, TestCase

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.machine_translation.units import apply_unit, pending_units
from authoringtool.machine_translation.validation import validate_unit
from authoringtool.models import Activity, Answer
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher


class UnitTests(TestCase):
    def setUp(self):
        owner = make_teacher('mt_units_owner')
        self.source = build_source_scenario(owner)
        self.job = copy_for_translation(self.source, owner, 'Greek', 'English')
        self.target = self.job.target_scenario

    def test_pending_units_order_and_keys(self):
        units = pending_units(self.job)
        keys = [unit.key for unit in units]
        self.assertEqual(keys[0], 'scenario')
        self.assertTrue(keys[1].startswith('phase:') and keys[2].startswith('phase:'))
        self.assertTrue(all(key.startswith('activity:') for key in keys[3:]))
        self.assertEqual(len(units), 6)
        ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        unit = next(u for u in units if u.key == f'activity:{ask.id}')
        answer_keys = {f'answer:{a.id}' for a in ask.answers.all()}
        self.assertTrue(answer_keys <= set(unit.fields))
        self.assertEqual(unit.fields['name'], 'Doubling resistance')
        self.assertEqual(unit.label, 'Activity "Doubling resistance"')

    def test_unit_skips_fields_without_words(self):
        ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        ask.helper = ''
        ask.save()
        answer = ask.answers.first()
        answer.text = '42'
        answer.save()
        image_only = Activity.objects.get(scenario=self.target, name='Why it halves')
        image_only.text = '<p><img src="/media/a.png"></p>'
        image_only.save()
        units = {u.key: u for u in pending_units(self.job)}
        fields = units[f'activity:{ask.id}'].fields
        self.assertNotIn('helper', fields)
        self.assertNotIn(f'answer:{answer.id}', fields)
        self.assertNotIn('text', units[f'activity:{image_only.id}'].fields)

    def test_apply_unit_sets_plain_text_and_marks_done(self):
        ask = Activity.objects.get(scenario=self.target, name='Doubling resistance')
        unit = next(u for u in pending_units(self.job) if u.key == f'activity:{ask.id}')
        answer = ask.answers.order_by('id').first()
        translated = dict(unit.fields)
        translated['text'] = '<p>Τι συμβαίνει &amp; γιατί;</p>'
        translated[f'answer:{answer.id}'] = 'Υποδιπλασιάζεται'
        apply_unit(self.job, unit, translated)
        ask.refresh_from_db()
        self.assertEqual(ask.text, '<p>Τι συμβαίνει &amp; γιατί;</p>')
        self.assertEqual(ask.plain_text, 'Τι συμβαίνει & γιατί;')
        self.assertEqual(Answer.objects.get(pk=answer.pk).text, 'Υποδιπλασιάζεται')
        self.job.refresh_from_db()
        self.assertIn(unit.key, self.job.completed_units)

    def test_completed_units_are_not_pending(self):
        first = pending_units(self.job)[0]
        apply_unit(self.job, first, dict(first.fields))
        self.assertNotIn(first.key, [u.key for u in pending_units(self.job)])
        self.assertEqual(len(pending_units(self.job)), 5)

    def test_apply_scenario_header(self):
        header = pending_units(self.job)[0]
        apply_unit(self.job, header, {**header.fields, 'name': 'Νόμος του Ohm'})
        self.target.refresh_from_db()
        self.assertEqual(self.target.name, 'Νόμος του Ohm')


class ValidateUnitTests(SimpleTestCase):
    def test_validate_accepts_faithful_translation(self):
        self.assertEqual(
            validate_unit(
                {'text': '<p>Current is <b>2 A</b></p>'},
                {'text': '<p>Το ρεύμα είναι <b>2 A</b></p>'},
            ),
            [],
        )

    def test_validate_rejects_changed_tag(self):
        self.assertTrue(validate_unit(
            {'text': '<p>Current is <b>2 A</b></p>'},
            {'text': '<p>Το ρεύμα είναι <strong>2 A</strong></p>'},
        ))

    def test_validate_rejects_changed_src(self):
        self.assertTrue(validate_unit(
            {'text': '<p>See <img src="/media/a.png"></p>'},
            {'text': '<p>Δες <img src="/media/b.png"></p>'},
        ))

    def test_validate_rejects_lost_number(self):
        self.assertTrue(validate_unit(
            {'text': 'From 2 to 4 ohms'},
            {'text': 'Από δύο σε 4 ohm'},
        ))

    def test_validate_accepts_decimal_comma(self):
        self.assertEqual(
            validate_unit({'text': 'It is 0.5 A'}, {'text': 'Είναι 0,5 A'}),
            [],
        )

    def test_validate_rejects_changed_formula(self):
        self.assertTrue(validate_unit(
            {'helper': 'Think of $V=IR$'},
            {'helper': 'Σκέψου το $V=RI$'},
        ))

    def test_validate_rejects_missing_and_extra_keys(self):
        self.assertTrue(validate_unit({'name': 'A', 'text': 'B'}, {'name': 'Α'}))
        self.assertTrue(validate_unit({'name': 'A'}, {'name': 'Α', 'other': 'x'}))

    def test_validate_rejects_empty_value(self):
        self.assertTrue(validate_unit({'name': 'Doubling'}, {'name': '  '}))
