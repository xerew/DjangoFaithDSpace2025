import os
import tempfile

from django.test import override_settings

from authoringtool.models import TranslationMatch
from authoringtool.rag_sources import group_rag_pdfs, rag_pdf_signature
from authoringtool.test_translation_pooling import PoolingBase


class RagFilesBase(PoolingBase):
    def setUp(self):
        super().setUp()
        self.rag_root = tempfile.mkdtemp()
        override = override_settings(RAG_PDFS_ROOT=self.rag_root)
        override.enable()
        self.addCleanup(override.disable)

    def add_pdf(self, scenario, filename, payload=None):
        folder = os.path.join(self.rag_root, f'scenario_{scenario.id}')
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, filename), 'wb') as handle:
            handle.write(b'%PDF-' + (payload or filename.encode()))

    def pendulums(self, confirmed=True):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        if confirmed:
            self.confirm(english, greek)
        return english[0], greek[0]


class GroupRagPdfsTests(RagFilesBase):
    def test_own_and_translation_pdfs_are_listed_own_first(self):
        english, greek = self.pendulums()
        self.add_pdf(greek, 'b.pdf')
        self.add_pdf(english, 'a.pdf')

        names = [(pdf.scenario_id, pdf.filename) for pdf in group_rag_pdfs(english)]

        self.assertEqual(names, [(english.id, 'a.pdf'), (greek.id, 'b.pdf')])

    def test_translation_pdf_carries_its_scenario_name_and_language(self):
        english, greek = self.pendulums()
        self.add_pdf(greek, 'b.pdf')

        pdf = group_rag_pdfs(english)[0]

        self.assertEqual(pdf.scenario_name, 'Εκκρεμές')
        self.assertEqual(pdf.language, 'Ελληνικά')
        self.assertTrue(os.path.isfile(pdf.path))

    def test_same_file_in_two_translations_is_listed_once(self):
        english, greek = self.pendulums()
        self.add_pdf(english, 'forces.pdf', payload=b'same')
        self.add_pdf(greek, 'forces copy.pdf', payload=b'same')

        pdfs = group_rag_pdfs(english)

        self.assertEqual(len(pdfs), 1)
        self.assertEqual(pdfs[0].scenario_id, english.id)

    def test_unconfirmed_translation_pdfs_are_not_listed(self):
        english, greek = self.pendulums()
        TranslationMatch.objects.update(status='pending')
        self.add_pdf(english, 'a.pdf')
        self.add_pdf(greek, 'b.pdf')

        names = [pdf.filename for pdf in group_rag_pdfs(english)]

        self.assertEqual(names, ['a.pdf'])

    def test_upper_case_extension_is_listed_and_other_files_are_not(self):
        english, _ = self.pendulums()
        self.add_pdf(english, 'NOTES.PDF')
        self.add_pdf(english, 'notes.txt')

        names = [pdf.filename for pdf in group_rag_pdfs(english)]

        self.assertEqual(names, ['NOTES.PDF'])

    def test_new_pdf_changes_the_signature(self):
        english, greek = self.pendulums()
        self.add_pdf(english, 'a.pdf')
        before = rag_pdf_signature(group_rag_pdfs(english))

        self.add_pdf(greek, 'c.pdf')

        self.assertNotEqual(before, rag_pdf_signature(group_rag_pdfs(english)))

    def test_every_translation_has_the_same_signature(self):
        english, greek = self.pendulums()
        self.add_pdf(english, 'a.pdf')
        self.add_pdf(greek, 'b.pdf')

        self.assertEqual(
            rag_pdf_signature(group_rag_pdfs(english)),
            rag_pdf_signature(group_rag_pdfs(greek)),
        )

    def test_no_pdfs_gives_empty_signature(self):
        english, _ = self.pendulums()

        self.assertEqual(rag_pdf_signature(group_rag_pdfs(english)), '')
