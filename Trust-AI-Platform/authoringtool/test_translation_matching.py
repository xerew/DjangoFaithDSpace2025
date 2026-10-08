from io import StringIO

from django.contrib.auth.models import Group, User
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    NextQuestionLogic,
    Phase,
    QuestionBunch,
    Scenario,
    TranslationMatch,
)
from authoringtool.translation_matching import (
    build_flow_profile,
    compare_flows,
    scan_translation_matches,
)


ENGLISH_STEPS = [
    ('Explanation', 'A pendulum with a rope length L=0.5 m swings.', []),
    (
        'Question',
        'What is the period for a rope length of L=0.5 m?',
        [('1.4 s', True), ('2.8 s', False)],
    ),
    (
        'Question',
        'What happens to the period when the mass doubles?',
        [
            ('The period stays the same', True),
            ('The period doubles', False),
            ('The period halves', False),
        ],
    ),
    ('Explanation', 'The period depends only on L and g = 9.8 m/s2.', []),
]

GREEK_STEPS = [
    (
        'Explanation',
        'Ένα εκκρεμές με μήκος σχοινιού L=0.5 m ταλαντώνεται.',
        [],
    ),
    (
        'Question',
        'Ποια είναι η περίοδος για μήκος σχοινιού L=0.5 m;',
        [('1.4 s', True), ('2.8 s', False)],
    ),
    (
        'Question',
        'Τι συμβαίνει στην περίοδο όταν διπλασιάζεται η μάζα;',
        [
            ('Η περίοδος μένει ίδια', True),
            ('Η περίοδος διπλασιάζεται', False),
            ('Η περίοδος υποδιπλασιάζεται', False),
        ],
    ),
    (
        'Explanation',
        'Η περίοδος εξαρτάται μόνο από το L και το g = 9.8 m/s2.',
        [],
    ),
]

# Scenarios need at least 10 activities to be matched; extra explanations
# at the end keep the positions of the first four activities unchanged.
SHORT_ENGLISH_STEPS = list(ENGLISH_STEPS)
SHORT_GREEK_STEPS = list(GREEK_STEPS)
ENGLISH_STEPS = SHORT_ENGLISH_STEPS + [
    ('Explanation', f'Step {n}: the pendulum swings for {n}.5 seconds.', [])
    for n in range(5, 11)
]
GREEK_STEPS = SHORT_GREEK_STEPS + [
    ('Explanation', f'Βήμα {n}: το εκκρεμές ταλαντώνεται για {n}.5 δευτερόλεπτα.', [])
    for n in range(5, 11)
]


class TranslationMatchingBase(TestCase):
    def setUp(self):
        self.teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('translation_owner')
        self.owner.groups.add(self.teachers)
        self.types = {
            name: ActivityType.objects.create(
                name=name,
                created_by=self.owner,
                updated_by=self.owner,
            )
            for name in ('Explanation', 'Question', 'Guidance')
        }

    def build(self, name, language, steps):
        scenario = Scenario.objects.create(
            name=name,
            language=language,
            visibility_status='public',
            created_by=self.owner,
            updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Phase',
            scenario=scenario,
            created_by=self.owner,
            updated_by=self.owner,
        )
        activities = [
            self.add_activity(scenario, phase, kind, text, answers)
            for kind, text, answers in steps
        ]
        for current, following in zip(activities, activities[1:]):
            self.route(current, following)
        scenario.refresh_from_db()
        return scenario, activities

    def add_activity(self, scenario, phase, kind, text, answers=()):
        activity = Activity.objects.create(
            name=text[:40],
            text=text,
            plain_text=text,
            scenario=scenario,
            phase=phase,
            activity_type=self.types[kind],
            created_by=self.owner,
            updated_by=self.owner,
        )
        for answer_text, correct in answers:
            Answer.objects.create(
                activity=activity,
                text=answer_text,
                is_correct=correct,
            )
        return activity

    def route(self, activity, following):
        answers = list(activity.answers.order_by('id'))
        if not answers:
            NextQuestionLogic.objects.create(
                activity=activity,
                next_activity=following,
            )
        for answer in answers:
            NextQuestionLogic.objects.create(
                activity=activity,
                answer=answer,
                next_activity=following,
            )

    def compare(self, first, second):
        return compare_flows(
            build_flow_profile(first),
            build_flow_profile(second),
        )


class TranslationScoringTests(TranslationMatchingBase):
    def setUp(self):
        super().setUp()
        self.english, self.english_activities = self.build(
            'Pendulum', 'English', ENGLISH_STEPS,
        )
        self.greek, self.greek_activities = self.build(
            'Εκκρεμές', 'Ελληνικά', GREEK_STEPS,
        )

    def test_identical_translation_scores_100(self):
        result = self.compare(self.english, self.greek)

        self.assertEqual(result['confidence'], 100)
        self.assertTrue(result['exact'])
        self.assertEqual(result['differences'], [])
        self.assertEqual(
            result['activity_map'],
            [
                [english.id, greek.id]
                for english, greek in zip(
                    self.english_activities,
                    self.greek_activities,
                )
            ],
        )

    def test_unlinked_activity_is_ignored(self):
        self.add_activity(
            self.greek,
            self.greek_activities[0].phase,
            'Guidance',
            'Ζήτησε τη βοήθεια του δασκάλου.',
        )

        result = self.compare(self.english, self.greek)

        self.assertTrue(result['exact'])
        self.assertEqual(result['confidence'], 100)
        self.assertEqual(result['differences'], [])

    def test_linked_but_unreachable_leftover_scores_95_to_99(self):
        leftover = self.add_activity(
            self.greek,
            self.greek_activities[0].phase,
            'Guidance',
            'Ζήτησε τη βοήθεια του δασκάλου.',
        )
        self.route(leftover, self.greek_activities[3])

        result = self.compare(self.english, self.greek)

        self.assertTrue(result['exact'])
        self.assertGreaterEqual(result['confidence'], 95)
        self.assertLessEqual(result['confidence'], 99)
        self.assertEqual(len(result['differences']), 1)
        self.assertIn('unreachable', result['differences'][0])

    def test_unreachable_activity_without_phase_is_listed(self):
        without_phase = self.add_activity(
            self.greek,
            None,
            'Guidance',
            'Δραστηριότητα χωρίς φάση.',
        )
        with_phase = self.add_activity(
            self.greek,
            self.greek_activities[0].phase,
            'Guidance',
            'Ζήτησε τη βοήθεια του δασκάλου.',
        )
        self.route(without_phase, self.greek_activities[3])
        self.route(with_phase, self.greek_activities[3])

        result = self.compare(self.english, self.greek)

        self.assertTrue(result['exact'])
        self.assertEqual(
            sum('unreachable' in d for d in result['differences']),
            2,
        )

    def test_answer_weight_difference_is_named_and_not_exact(self):
        answer = self.greek_activities[1].answers.order_by('id')[0]
        Answer.objects.filter(pk=answer.pk).update(answer_weight=5)

        result = self.compare(self.english, self.greek)

        self.assertFalse(result['exact'])
        self.assertTrue(
            any('answer weights' in d for d in result['differences']),
            result['differences'],
        )

    def test_activity_setting_differences_are_named_and_not_exact(self):
        cases = [
            ('is_primary_ev', True, 'primary evaluation'),
            ('must_wait', True, 'must wait'),
            ('score_limit', 2.5, 'score limit'),
        ]
        for field, value, label in cases:
            with self.subTest(field=field):
                target = self.greek_activities[2]
                original = getattr(target, field)
                Activity.objects.filter(pk=target.pk).update(**{field: value})

                result = self.compare(self.english, self.greek)

                Activity.objects.filter(pk=target.pk).update(
                    **{field: original}
                )
                self.assertFalse(result['exact'])
                self.assertTrue(
                    any(label in d for d in result['differences']),
                    result['differences'],
                )

    def test_phase_placement_difference_is_named_and_not_exact(self):
        english_second = Phase.objects.create(
            name='Second', scenario=self.english,
            created_by=self.owner, updated_by=self.owner,
        )
        greek_second = Phase.objects.create(
            name='Δεύτερη', scenario=self.greek,
            created_by=self.owner, updated_by=self.owner,
        )
        Activity.objects.filter(pk=self.english_activities[3].pk).update(
            phase=english_second,
        )
        Activity.objects.filter(
            pk__in=[self.greek_activities[2].pk, self.greek_activities[3].pk],
        ).update(phase=greek_second)

        result = self.compare(self.english, self.greek)

        self.assertFalse(result['exact'])
        self.assertTrue(
            any('phase' in d for d in result['differences']),
            result['differences'],
        )

    def test_phase_holding_only_unlinked_activities_is_ignored(self):
        english_second = Phase.objects.create(
            name='Second', scenario=self.english,
            created_by=self.owner, updated_by=self.owner,
        )
        Activity.objects.filter(pk=self.english_activities[3].pk).update(
            phase=english_second,
        )
        leftover_phase = Phase.objects.create(
            name='Παλιά φάση', scenario=self.greek,
            created_by=self.owner, updated_by=self.owner,
        )
        self.add_activity(
            self.greek, leftover_phase, 'Guidance', 'Χωρίς συνδέσεις.',
        )
        greek_last = Phase.objects.create(
            name='Δεύτερη', scenario=self.greek,
            created_by=self.owner, updated_by=self.owner,
        )
        Activity.objects.filter(pk=self.greek_activities[3].pk).update(
            phase=greek_last,
        )

        result = self.compare(self.english, self.greek)

        self.assertTrue(result['exact'], result['differences'])
        self.assertEqual(result['confidence'], 100)

    def test_question_bunch_difference_is_named_and_not_exact(self):
        english, greek = self.english_activities, self.greek_activities
        QuestionBunch.objects.create(
            activity_primary=english[1],
            activity_ids=[english[1].id, english[2].id],
        )
        QuestionBunch.objects.create(
            activity_primary=greek[1],
            activity_ids=[greek[1].id],
        )

        result = self.compare(self.english, self.greek)

        self.assertFalse(result['exact'])
        self.assertTrue(
            any('question bunch' in d for d in result['differences']),
            result['differences'],
        )

    def test_matching_question_bunches_stay_exact(self):
        english, greek = self.english_activities, self.greek_activities
        QuestionBunch.objects.create(
            activity_primary=english[1],
            activity_ids=[english[1].id, english[2].id],
        )
        QuestionBunch.objects.create(
            activity_primary=greek[1],
            activity_ids=[greek[1].id, greek[2].id],
        )

        result = self.compare(self.english, self.greek)

        self.assertTrue(result['exact'])
        self.assertEqual(result['confidence'], 100)

    def test_route_difference_scores_below_95_and_names_the_route(self):
        question = self.greek_activities[2]
        second_answer = question.answers.order_by('id')[1]
        NextQuestionLogic.objects.filter(
            activity=question,
            answer=second_answer,
        ).update(next_activity=self.greek_activities[0])

        result = self.compare(self.english, self.greek)

        self.assertFalse(result['exact'])
        self.assertLess(result['confidence'], 95)
        self.assertTrue(
            any(
                difference.startswith('route at #3')
                and 'answer 2' in difference
                for difference in result['differences']
            ),
            result['differences'],
        )

    def test_changed_question_caps_at_60(self):
        first, second = self.greek_activities[1].answers.order_by('id')
        Answer.objects.filter(pk=first.pk).update(is_correct=False)
        Answer.objects.filter(pk=second.pk).update(is_correct=True)

        result = self.compare(self.english, self.greek)

        self.assertFalse(result['exact'])
        self.assertLessEqual(result['confidence'], 60)


class TranslationScanTests(TranslationMatchingBase):
    def test_scan_creates_pending_match_for_translation_pair(self):
        english, _ = self.build('Pendulum', 'English', ENGLISH_STEPS)
        greek, _ = self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)

        summary = scan_translation_matches()

        match = TranslationMatch.objects.get()
        self.assertEqual(match.scenario_a_id, min(english.id, greek.id))
        self.assertEqual(match.scenario_b_id, max(english.id, greek.id))
        self.assertEqual(match.confidence, 100)
        self.assertTrue(match.exact_match)
        self.assertEqual(match.status, 'pending')
        self.assertEqual(summary['matches'], 1)

    def test_scenarios_with_fewer_than_10_activities_are_not_matched(self):
        self.build('Short pendulum', 'English', SHORT_ENGLISH_STEPS)
        self.build('Σύντομο εκκρεμές', 'Ελληνικά', SHORT_GREEK_STEPS)

        summary = scan_translation_matches()

        self.assertFalse(TranslationMatch.objects.exists())
        self.assertEqual(summary['matches'], 0)

    def test_scenario_without_activities_is_not_matched(self):
        self.build('Pendulum', 'English', ENGLISH_STEPS)
        Scenario.objects.create(
            name='Empty', language='Ελληνικά',
            created_by=self.owner, updated_by=self.owner,
        )

        scan_translation_matches()

        self.assertFalse(TranslationMatch.objects.exists())

    def test_activities_count_across_all_phases(self):
        english, english_activities = self.build(
            'Pendulum', 'English', ENGLISH_STEPS,
        )
        greek, greek_activities = self.build(
            'Εκκρεμές', 'Ελληνικά', GREEK_STEPS,
        )
        for scenario, activities in (
            (english, english_activities),
            (greek, greek_activities),
        ):
            second = Phase.objects.create(
                name='Second', scenario=scenario,
                created_by=self.owner, updated_by=self.owner,
            )
            Activity.objects.filter(
                pk__in=[activity.pk for activity in activities[5:]],
            ).update(phase=second)

        scan_translation_matches()

        self.assertEqual(TranslationMatch.objects.count(), 1)

    def test_pending_match_is_removed_when_scenario_drops_below_10(self):
        english, english_activities = self.build(
            'Pendulum', 'English', ENGLISH_STEPS,
        )
        self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)
        scan_translation_matches()
        self.assertEqual(TranslationMatch.objects.count(), 1)
        Activity.objects.filter(
            pk__in=[activity.pk for activity in english_activities[4:]],
        ).delete()

        scan_translation_matches()

        self.assertFalse(TranslationMatch.objects.exists())

    def test_reviewed_match_is_kept_when_scenario_drops_below_10(self):
        english, english_activities = self.build(
            'Pendulum', 'English', ENGLISH_STEPS,
        )
        self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)
        scan_translation_matches()
        TranslationMatch.objects.update(status='confirmed')
        Activity.objects.filter(
            pk__in=[activity.pk for activity in english_activities[4:]],
        ).delete()

        scan_translation_matches()

        self.assertEqual(TranslationMatch.objects.get().status, 'confirmed')

    def test_same_language_copy_is_not_a_translation_candidate(self):
        self.build('Pendulum', 'English', ENGLISH_STEPS)
        self.build('Pendulum copy', 'English', ENGLISH_STEPS)

        scan_translation_matches()

        self.assertFalse(TranslationMatch.objects.exists())

    def test_rescan_keeps_review_decision(self):
        self.build('Pendulum', 'English', ENGLISH_STEPS)
        self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)
        scan_translation_matches()
        TranslationMatch.objects.update(status='confirmed')

        scan_translation_matches()

        match = TranslationMatch.objects.get()
        self.assertEqual(match.status, 'confirmed')

    def test_flow_change_after_review_marks_match_changed(self):
        self.build('Pendulum', 'English', ENGLISH_STEPS)
        greek, greek_activities = self.build(
            'Εκκρεμές', 'Ελληνικά', GREEK_STEPS,
        )
        scan_translation_matches()
        TranslationMatch.objects.update(status='confirmed')
        extra = self.add_activity(
            greek,
            greek_activities[0].phase,
            'Question',
            'Τι γίνεται στη Σελήνη με g = 1.6 m/s2;',
            [('Μεγαλώνει', True), ('Μικραίνει', False)],
        )
        self.route(greek_activities[-1], extra)

        scan_translation_matches()

        match = TranslationMatch.objects.get()
        self.assertEqual(match.status, 'changed')


class ScanTranslationsCommandTests(TranslationMatchingBase):
    def test_dry_run_reports_without_writing(self):
        english, _ = self.build('Pendulum', 'English', ENGLISH_STEPS)
        greek, _ = self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)
        output = StringIO()

        call_command('scan_translations', '--dry-run', stdout=output)

        self.assertFalse(TranslationMatch.objects.exists())
        self.assertIn('1 candidate', output.getvalue())
        self.assertIn(f'{english.id} <-> {greek.id}', output.getvalue())

    def test_scan_writes_matches(self):
        self.build('Pendulum', 'English', ENGLISH_STEPS)
        self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)

        call_command('scan_translations', stdout=StringIO())

        self.assertEqual(TranslationMatch.objects.count(), 1)


class TranslationVisibilityFilterTests(TestCase):
    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            'visibility_admin',
            password='pass',
        )
        self.pairs = {}
        for label, first, second in (
            ('public-public', 'public', 'public'),
            ('public-private', 'public', 'private'),
            ('org-org', 'org', 'org'),
        ):
            scenarios = [
                Scenario.objects.create(
                    name=f'{label} {side}',
                    visibility_status=visibility,
                    created_by=self.admin_user,
                    updated_by=self.admin_user,
                )
                for side, visibility in (('A', first), ('B', second))
            ]
            self.pairs[label] = TranslationMatch.objects.create(
                scenario_a=scenarios[0],
                scenario_b=scenarios[1],
                confidence=100,
                exact_match=True,
            )
        self.client.force_login(self.admin_user)
        self.url = reverse('admin:authoringtool_translationmatch_changelist')

    def shown(self, response):
        return {
            match.pk for match in response.context['cl'].result_list
        }

    def test_without_selection_all_pairs_are_shown(self):
        response = self.client.get(self.url)

        self.assertEqual(
            self.shown(response),
            {match.pk for match in self.pairs.values()},
        )

    def test_one_visibility_requires_both_scenarios_to_match(self):
        response = self.client.get(self.url, {'visibility': 'public'})

        self.assertEqual(
            self.shown(response),
            {self.pairs['public-public'].pk},
        )

    def test_several_visibilities_can_be_ticked(self):
        response = self.client.get(self.url, {'visibility': 'public,private'})

        self.assertEqual(
            self.shown(response),
            {self.pairs['public-public'].pk, self.pairs['public-private'].pk},
        )

    def test_filter_shows_a_checkbox_per_visibility(self):
        response = self.client.get(self.url, {'visibility': 'org'})

        self.assertContains(response, '☐ Public')
        self.assertContains(response, '☐ Private (In-Progress)')
        self.assertContains(response, '☑ Organization Users Only')
        self.assertContains(response, '☑', count=1)

    def test_unknown_visibility_values_are_ignored(self):
        response = self.client.get(self.url, {'visibility': 'secret'})

        self.assertEqual(
            self.shown(response),
            {match.pk for match in self.pairs.values()},
        )


class TranslationMatchAdminTests(TranslationMatchingBase):
    def setUp(self):
        super().setUp()
        self.admin_user = User.objects.create_superuser(
            'translation_admin',
            password='pass',
        )
        self.english, _ = self.build('Pendulum', 'English', ENGLISH_STEPS)
        self.greek, _ = self.build('Εκκρεμές', 'Ελληνικά', GREEK_STEPS)
        scan_translation_matches()
        self.match = TranslationMatch.objects.get()
        self.client.force_login(self.admin_user)

    def test_confirm_action_records_reviewer(self):
        response = self.client.post(
            reverse('admin:authoringtool_translationmatch_changelist'),
            {
                'action': 'confirm_selected',
                '_selected_action': [self.match.pk],
            },
        )

        self.assertEqual(response.status_code, 302)
        self.match.refresh_from_db()
        self.assertEqual(self.match.status, 'confirmed')
        self.assertEqual(self.match.reviewed_by, self.admin_user)
        self.assertIsNotNone(self.match.reviewed_at)

    def test_confirm_action_skips_non_exact_matches(self):
        TranslationMatch.objects.filter(pk=self.match.pk).update(
            exact_match=False,
            confidence=84,
        )

        response = self.client.post(
            reverse('admin:authoringtool_translationmatch_changelist'),
            {
                'action': 'confirm_selected',
                '_selected_action': [self.match.pk],
            },
            follow=True,
        )

        self.match.refresh_from_db()
        self.assertEqual(self.match.status, 'pending')
        self.assertIsNone(self.match.reviewed_by)
        self.assertContains(response, 'not exact 1:1')

    def test_change_form_refuses_confirming_non_exact_match(self):
        TranslationMatch.objects.filter(pk=self.match.pk).update(
            exact_match=False,
            confidence=84,
        )

        response = self.client.post(
            reverse(
                'admin:authoringtool_translationmatch_change',
                args=[self.match.pk],
            ),
            {'status': 'confirmed', 'review_notes': ''},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Only exact 1:1 matches')
        self.match.refresh_from_db()
        self.assertEqual(self.match.status, 'pending')

    def test_change_form_confirms_exact_match(self):
        response = self.client.post(
            reverse(
                'admin:authoringtool_translationmatch_change',
                args=[self.match.pk],
            ),
            {'status': 'confirmed', 'review_notes': ''},
        )

        self.assertEqual(response.status_code, 302)
        self.match.refresh_from_db()
        self.assertEqual(self.match.status, 'confirmed')
        self.assertEqual(self.match.reviewed_by, self.admin_user)

    def test_reject_action_records_reviewer(self):
        self.client.post(
            reverse('admin:authoringtool_translationmatch_changelist'),
            {
                'action': 'reject_selected',
                '_selected_action': [self.match.pk],
            },
        )

        self.match.refresh_from_db()
        self.assertEqual(self.match.status, 'rejected')
        self.assertEqual(self.match.reviewed_by, self.admin_user)

    def test_changelist_shows_confidence(self):
        response = self.client.get(
            reverse('admin:authoringtool_translationmatch_changelist')
        )

        self.assertContains(response, '100%')
        self.assertContains(response, 'Εκκρεμές')
        self.assertContains(response, 'Scan for translations')

    def test_detail_page_shows_flows_side_by_side(self):
        response = self.client.get(
            reverse(
                'admin:authoringtool_translationmatch_change',
                args=[self.match.pk],
            )
        )

        self.assertContains(response, 'What happens to the period')
        self.assertContains(response, 'Τι συμβαίνει στην περίοδο')

    def test_scan_button_starts_background_scan(self):
        response = self.client.post(
            reverse('admin:authoringtool_translationmatch_scan')
        )

        self.assertRedirects(
            response,
            reverse('admin:authoringtool_translationmatch_changelist'),
        )
