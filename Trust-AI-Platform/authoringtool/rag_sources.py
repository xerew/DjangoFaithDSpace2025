"""The RAG PDFs of a scenario together with those of its confirmed translations.

Each scenario keeps its own folder (rag_pdfs/scenario_<id>/). Confirmed 1:1
translations are one scenario in several languages, so proposals read the
PDFs of every public member. A file uploaded to more than one member is used
once.
"""
import hashlib
import os
from typing import NamedTuple

from django.conf import settings

from .models import Scenario
from .translation_matching import translation_group_ids


class RagPdf(NamedTuple):
    scenario_id: int
    scenario_name: str
    language: str
    filename: str
    path: str
    sha256: str


def scenario_rag_folder(scenario_id):
    return os.path.join(settings.RAG_PDFS_ROOT, f'scenario_{scenario_id}')


def _read_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# The scenario page lists these on every view, so a file is read again only
# when its size or modification time changes.
_digests = {}


def _file_sha256(path):
    info = os.stat(path)
    key = (path, info.st_size, info.st_mtime_ns)
    if key not in _digests:
        _digests[key] = _read_sha256(path)
    return _digests[key]


def _scenario_pdfs(scenario):
    folder = scenario_rag_folder(scenario.id)
    if not os.path.isdir(folder):
        return []
    pdfs = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if not filename.lower().endswith('.pdf') or not os.path.isfile(path):
            continue
        try:
            sha256 = _file_sha256(path)
        except OSError as exc:
            # Deleted meanwhile or not readable: leave it out.
            print(f'[RAG] skipping unreadable PDF {path}: {exc}')
            continue
        pdfs.append(RagPdf(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            language=scenario.language or '',
            filename=filename,
            path=path,
            sha256=sha256,
        ))
    return pdfs


def group_rag_pdfs(scenario):
    """The scenario's own PDFs, then each translation's, each content once.

    Only public, non-archived translations share their PDFs; private and
    organization-only ones keep them to themselves.
    """
    members = Scenario.objects.filter(
        id__in=translation_group_ids(scenario),
        visibility_status='public',
        archived_at__isnull=True,
    ).exclude(id=scenario.id).order_by('id')
    seen = set()
    pdfs = []
    for member in [scenario, *members]:
        for pdf in _scenario_pdfs(member):
            if pdf.sha256 in seen:
                continue
            seen.add(pdf.sha256)
            pdfs.append(pdf)
    return pdfs


def rag_pdf_signature(pdfs):
    """Short id of the PDF contents; '' when there are none."""
    hashes = sorted({pdf.sha256 for pdf in pdfs})
    if not hashes:
        return ''
    return hashlib.sha256(','.join(hashes).encode('ascii')).hexdigest()[:12]
