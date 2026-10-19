"""Proposal History page layout, the combined evidence it reports, and the
scenario page no longer linking to it."""
import re

from django.test import override_settings
from django.urls import reverse

from authoringtool.models import ProposalGenerationRun
from authoringtool.tasks import proposal_evidence_summary
from authoringtool.test_translation_pooling import PoolingBase


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ProposalHistoryPageTests(PoolingBase):
    def setUp(self):
        super().setUp()
        self.english = self.build('Pendulum', 'English')
        self.client.force_login(self.owner)
        self.url = reverse('proposal_history', args=[self.english[0].id])

    def start_run(self, **summary):
        return ProposalGenerationRun.start_new(
            self.english[0], self.owner,
            evidence_summary={'scope': 'local', 'implementation_count': 4, **summary},
        )

    def test_run_card_is_not_one_big_link(self):
        self.start_run()

        response = self.client.get(self.url)

        self.assertContains(response, '<div class="run-card')
        self.assertNotContains(response, '<a href="{}" class="run-card'.format(self.url))

    def test_no_link_is_nested_inside_another(self):
        self.start_run()

        html = self.client.get(self.url).content.decode()
        section = html[html.index('<section class="section">'):html.index('</section>')]

        depth = 0
        for tag in re.findall(r'<a\b|</a>', section):
            depth += 1 if tag == '<a' else -1
            self.assertLessEqual(depth, 1)

    def test_run_shows_who_proposals_and_a_view_button(self):
        run = self.start_run()

        response = self.client.get(self.url)

        self.assertContains(response, 'Triggered by')
        self.assertContains(
            response,
            reverse('proposal_history_run_detail', args=[self.english[0].id, run.id]),
        )
        self.assertContains(response, 'View proposals')

    def test_combined_translation_evidence_is_reported(self):
        self.start_run(pooled_implementation_count=712, translation_count=3)

        response = self.client.get(self.url)

        self.assertContains(response, '712')
        self.assertContains(response, 'this scenario and its 3 confirmed translations')

    def test_older_runs_without_combined_count_show_their_own(self):
        self.start_run()

        response = self.client.get(self.url)

        self.assertContains(response, '4')
        self.assertContains(response, 'this scenario')


class ProposalEvidenceSummaryTests(PoolingBase):
    def test_summary_records_the_combined_translation_count(self):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.student_answers(english, correct=True, timing=10)
        self.student_answers(greek, correct=True, timing=10)
        self.student_answers(greek, correct=False, timing=10)
        self.confirm(english, greek)

        summary = proposal_evidence_summary(english[0], {'implementation_count': 1})

        self.assertEqual(summary['pooled_implementation_count'], 3)
        self.assertEqual(summary['translation_count'], 1)
        self.assertEqual(summary['implementation_count'], 1)

    def test_without_translations_nothing_is_added(self):
        english = self.build('Pendulum', 'English')

        summary = proposal_evidence_summary(english[0], {'implementation_count': 0})

        self.assertNotIn('pooled_implementation_count', summary)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ScenarioPageProposalHistoryLinkTests(PoolingBase):
    def test_scenario_page_has_no_proposal_history_button(self):
        english = self.build('Pendulum', 'English')
        self.client.force_login(self.owner)

        response = self.client.get(reverse('viewScenario', args=[english[0].id]))

        self.assertNotContains(response, reverse('proposal_history', args=[english[0].id]))
