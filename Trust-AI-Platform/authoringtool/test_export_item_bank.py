"""export_item_bank writes per-activity aggregates for the simulated-students
study: real difficulty, option choices and results by ability group. It
never writes data about individual students."""
import io
import json
import os
import shutil
import tempfile

from django.contrib.auth.models import Group, User
from django.core.management import call_command
from django.test import TestCase, override_settings

from authoringtool.models import Activity, ActivityType, Answer, Phase, Scenario, UserAnswer


class ExportItemBankTests(TestCase):
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)
        os.makedirs(os.path.join(self.media, 'tinymce'))
        with open(os.path.join(self.media, 'tinymce', 'pendulum.png'), 'wb') as image:
            image.write(b'\x89PNG fake')
        settings_override = override_settings(MEDIA_ROOT=self.media)
        settings_override.enable()
        self.addCleanup(settings_override.disable)

        self.owner = User.objects.create_user('bank_owner')
        teacher = User.objects.create_user('bank_teacher')
        teacher.groups.add(Group.objects.create(name='teachers'))
        self.scenario = self.make_scenario('Pendulum', 'public')
        phase = self.scenario.phases.first()
        explanation = ActivityType.objects.create(name='Explanation', created_by=self.owner, updated_by=self.owner)
        Activity.objects.create(
            name='Read', text='<p>The period depends on the length.</p>',
            plain_text='The period depends on the length.', phase=phase,
            scenario=self.scenario, activity_type=explanation,
            created_by=self.owner, updated_by=self.owner,
        )
        self.q1, self.q1_options = self.question(phase, 'Which pendulum is faster? <img src="/media/tinymce/pendulum.png">')
        self.q2, self.q2_options = self.question(
            phase, 'Does mass matter? <img src="https://example.org/mass.png">',
        )
        # 30 students: student i answers q1 correctly when i < 18, q2 when i < 12;
        # each first answer counts, a later retry does not.
        for i in range(30):
            student = User.objects.create_user(f'bank_student_{i}')
            self.answer(student, self.q1, self.q1_options[0 if i < 18 else 1 + i % 2])
            self.answer(student, self.q1, self.q1_options[0])  # retry, ignored
            self.answer(student, self.q2, self.q2_options[0 if i < 12 else 1])
        self.answer(teacher, self.q1, self.q1_options[2])  # teachers are not students

        private = self.make_scenario('Private', 'private')
        hidden, options = self.question(private.phases.first(), 'Hidden?')
        for i in range(30):
            self.answer(User.objects.get(username=f'bank_student_{i}'), hidden, options[0])

    def make_scenario(self, name, visibility):
        scenario = Scenario.objects.create(
            name=name, visibility_status=visibility, language='English',
            created_by=self.owner, updated_by=self.owner,
        )
        Phase.objects.create(name='Explore', scenario=scenario, created_by=self.owner, updated_by=self.owner)
        return scenario

    def question(self, phase, text):
        question_type, _ = ActivityType.objects.get_or_create(
            name='Question', defaults={'created_by': self.owner, 'updated_by': self.owner},
        )
        activity = Activity.objects.create(
            name=text[:20], text=f'<p>{text}</p>', phase=phase, scenario=phase.scenario,
            activity_type=question_type, created_by=self.owner, updated_by=self.owner,
        )
        options = [
            Answer.objects.create(activity=activity, text=label, is_correct=(i == 0),
                                  created_by=self.owner, updated_by=self.owner)
            for i, label in enumerate(['A. Short one', 'b) Heavy one', 'Same'])
        ]
        return activity, options

    def answer(self, user, activity, option):
        UserAnswer.objects.create(user=user, activity=activity, answer=option, timing=20)

    def export(self, **options):
        output = os.path.join(self.out_dir, 'bank.jsonl')
        call_command('export_item_bank', output=output, min_answers=30, stdout=io.StringIO(), **options)
        with open(output, encoding='utf-8') as handle:
            text = handle.read()
        return text, {row['item_id']: row for row in map(json.loads, text.splitlines())}

    def test_exports_public_items_with_real_statistics(self):
        _, items = self.export()
        self.assertEqual(set(items), {self.q1.id, self.q2.id})
        q1 = items[self.q1.id]
        self.assertEqual(q1['real']['n'], 30)
        self.assertAlmostEqual(q1['real']['p_correct'], 18 / 30)
        shares = q1['real']['option_shares']
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares[str(self.q1_options[0].id)], 18 / 30)
        self.assertEqual(q1['language'], 'English')
        self.assertEqual([o['is_correct'] for o in q1['options']], [True, False, False])
        self.assertIn('The period depends on the length.', q1['preceding_material'])

    def test_results_by_ability_group(self):
        _, items = self.export()
        groups = items[self.q1.id]['real']['by_ability']
        self.assertEqual(set(groups), {'low', 'mid', 'high'})
        self.assertGreater(groups['high']['p_correct'], groups['low']['p_correct'])
        self.assertEqual(sum(g['n'] for g in groups.values()), 30)

    def test_images_are_copied_next_to_the_export(self):
        _, items = self.export()
        images = items[self.q1.id]['images']
        self.assertEqual(len(images), 1)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, images[0])))
        self.assertIn('[Image 1]', items[self.q1.id]['text'])
        self.assertEqual(items[self.q2.id]['images'], [])

    def test_letter_labels_typed_into_options_are_removed(self):
        _, items = self.export()
        self.assertEqual([o['text'] for o in items[self.q1.id]['options']], ['Short one', 'Heavy one', 'Same'])

    def test_images_that_could_not_be_copied_are_counted(self):
        _, items = self.export()
        self.assertEqual(items[self.q2.id]['missing_images'], 1)
        self.assertEqual(items[self.q1.id]['missing_images'], 0)

    def test_no_student_data_in_the_export(self):
        text, _ = self.export()
        self.assertNotIn('bank_student', text)
        self.assertNotIn('user', text.lower())
