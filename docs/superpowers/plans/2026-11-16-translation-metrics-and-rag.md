# Same Metrics & AI Data and Shared RAG PDFs Across Translations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirmed 1:1 translations show identical Metrics & AI numbers and flags (time included), and proposal runs read the PDFs of every translation, with an index that refreshes itself.

**Architecture:** Drop the same-language timing filter in the local metrics path and tag the cache name so old CSVs are not reused. A new `authoringtool/rag_sources.py` lists a translation group's PDFs (own first, de-duplicated by SHA-256) and signs the set; the RAG index folder name carries that signature, so a changed set builds a fresh index and old ones are removed. The scenario page lists translations' PDFs read-only.

**Tech Stack:** Django 5, Celery tasks, Chroma + LangChain (`HuggingFaceEmbeddings("all-MiniLM-L6-v2")`), Django test runner.

**Spec:** `docs/superpowers/specs/2026-11-16-translation-metrics-and-rag-design.md`

## Global Constraints

- Work in `C:/wt/translation-metrics-rag` on branch `feature/translation-metrics-rag` (from `main`).
- Run tests from `C:/wt/translation-metrics-rag/Trust-AI-Platform` with:
  `timeout 1200 "/c/Users/Nikos A. Grammatikos/Desktop/DjangoFaithDSpace2025-performance-test/djangofaithvenv/Scripts/python.exe" manage.py test <labels> --settings=faithDev.settings_test --noinput`
- Only confirmed `TranslationMatch` groups pool (`translation_group_ids`, `translation_activity_maps` in `authoringtool/translation_matching.py`). The family/"compatible" path stays untouched.
- No emojis in UI; use `bi-*` icons. Templates on `main` are plain English (no `{% translate %}` in `viewScenario.html`'s RAG card); keep that style.
- The embedding model stays `all-MiniLM-L6-v2`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A metrics CSV cached under the old same-language rule must not be served after deploy (Task 1 test `test_cache_name_marks_pooled_time`).
- The same PDF uploaded to two translations (local data: scenarios 23 and 90 share one file) is indexed once (Task 2 test `test_same_file_in_two_translations_is_listed_once`).
- A PDF uploaded after the first run is used by the next run without the force button (Task 2 test `test_new_pdf_changes_the_signature`), and the stale index folder is removed (Task 3 test `test_old_index_folders_are_removed`).
- A translation the viewer cannot see must not leak its PDF list on the scenario page (Task 4 test `test_hidden_translation_pdfs_are_not_listed`).
- Index folder clean-up must not touch another scenario whose id shares a prefix (scenario 1 vs 12) (Task 3 test `test_cleanup_leaves_other_scenarios_alone`).

---

### Task 1: Time pooled across all languages on Metrics & AI

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/tasks.py` (local path of `compute_category_metrics_per_phase_activity`, ~lines 2238-2400: the `same_language` dict, `member_is_same_language`, and the `if same_language.get(...)` guard)
- Modify: `Trust-AI-Platform/authoringtool/utils.py:28-65` (`get_scenario_evidence_cache_paths`)
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/ai_metrics_scenario.html:179`
- Test: `Trust-AI-Platform/authoringtool/test_translation_pooling.py`

**Interfaces:**
- Produces: `METRICS_RULES_TAG = 'pt'` in `authoringtool/utils.py`; cache prefix becomes `scenario_{id}_v{version}_{scope}_{language_slug}_{signature}_pt` + optional `_tr{pool}`.

- [ ] **Step 1: Write the failing tests** in `PooledMetricsTests` (replace `test_time_uses_same_language_students_only`):

```python
def test_time_uses_students_of_every_language(self):
    english = self.build('Pendulum', 'English')
    greek = self.build('Εκκρεμές', 'Ελληνικά')
    self.confirm(english, greek)
    self.student_answers(english, correct=True, timing=10)
    self.student_answers(greek, correct=True, timing=100)
    rows = self.question_rows(english)
    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0]['Total'], '2')
    self.assertEqual(float(rows[0]['Avg Time']), 55.0)

def test_every_translation_shows_the_same_numbers(self):
    english = self.build('Pendulum', 'English')
    greek = self.build('Εκκρεμές', 'Ελληνικά')
    self.confirm(english, greek)
    for timing in (10, 12, 14):
        self.student_answers(english, correct=True, timing=timing)
    for timing in (40, 60):
        self.student_answers(greek, correct=False, timing=timing)
    columns = ['Category', 'Total', 'Correct', 'Wrong', '% Correct', '% Wrong',
               'Avg Time', 'Timing Total', 'Next Low', 'Next Moderate', 'Next High']
    def strip(rows):
        return sorted(tuple(row[c] for c in columns) for row in rows)
    self.assertEqual(strip(self.question_rows(english)), strip(self.question_rows(greek)))

def test_every_translation_gets_the_same_flags(self):
    # same data as above; run calculate_activities_in_risk(scenario.id) for both,
    # read get_scenario_evidence_cache_paths(s, 'local')['flags'] and compare
    # sorted (Category, Flag, Reason) tuples: assertEqual(english_flags, greek_flags)

def test_cache_name_marks_pooled_time(self):
    english = self.build('Pendulum', 'English')
    path = get_scenario_evidence_cache_paths(english[0], 'local')['metrics']
    self.assertIn('_pt', os.path.basename(path))
```

In `CombinedThresholdTests.test_metrics_page_states_the_combined_count` add:
`self.assertContains(response, 'Metrics and AI proposals use them all.')` and `self.assertNotContains(response, 'students only')`.

- [ ] **Step 2: Run, expect failures** — `... test authoringtool.test_translation_pooling`. Expected: Avg Time 10.0 != 55.0, rows differ, `_pt` missing, header text missing.

- [ ] **Step 3: Implement.** Delete the `same_language` bookkeeping so every pooled implementation's `ua.timing` counts (keep the `if not ua.timing: continue`). Update the comment at the member loop to say time is pooled too. Add `METRICS_RULES_TAG` to the cache prefix. Template line 179 becomes: `Metrics and AI proposals use them all.`

- [ ] **Step 4: Run, expect PASS** — the whole `authoringtool.test_translation_pooling` module (17 existing + new).

- [ ] **Step 5: Commit** — `Count every translation's time on Metrics & AI`.

---

### Task 2: List a translation group's RAG PDFs

**Files:**
- Create: `Trust-AI-Platform/authoringtool/rag_sources.py`
- Test: `Trust-AI-Platform/authoringtool/test_rag_sources.py`

**Interfaces:**
- Produces:
  - `RagPdf` — `NamedTuple(scenario_id: int, scenario_name: str, language: str, filename: str, path: str, sha256: str)`
  - `scenario_rag_folder(scenario_id: int) -> str` — `os.path.join(settings.RAG_PDFS_ROOT, f'scenario_{scenario_id}')`
  - `group_rag_pdfs(scenario) -> list[RagPdf]` — own `*.pdf` files first (sorted by name), then each member of `translation_group_ids(scenario)` in id order; a file whose `sha256` was already seen is skipped. Missing folders give nothing. Case-insensitive `.pdf` match, as `viewScenario` does today.
  - `rag_pdf_signature(pdfs: list[RagPdf]) -> str` — first 12 hex chars of SHA-256 over the sorted hashes joined by `,`; `''` when the list is empty.

- [ ] **Step 1: Write the failing tests** (`TestCase` reusing `PoolingBase` from `test_translation_pooling`; `@override_settings(RAG_PDFS_ROOT=tempfile.mkdtemp())` per test class; helper writes `b'%PDF-' + payload` files):

```python
def test_own_and_translation_pdfs_are_listed_own_first(self):
    # english has a.pdf, greek has b.pdf, confirmed
    names = [(p.scenario_id, p.filename) for p in group_rag_pdfs(english)]
    self.assertEqual(names, [(english.id, 'a.pdf'), (greek.id, 'b.pdf')])

def test_same_file_in_two_translations_is_listed_once(self):
    # identical bytes in both folders under different names
    self.assertEqual(len(group_rag_pdfs(english)), 1)
    self.assertEqual(group_rag_pdfs(english)[0].scenario_id, english.id)

def test_unconfirmed_translation_pdfs_are_not_listed(self):
    # TranslationMatch status 'pending' -> only own files

def test_new_pdf_changes_the_signature(self):
    before = rag_pdf_signature(group_rag_pdfs(english))
    # add c.pdf to greek folder
    self.assertNotEqual(before, rag_pdf_signature(group_rag_pdfs(english)))

def test_no_pdfs_gives_empty_signature(self):
    self.assertEqual(rag_pdf_signature(group_rag_pdfs(english)), '')
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: authoringtool.rag_sources`).
- [ ] **Step 3: Implement** `rag_sources.py` as specified (hash with `hashlib.sha256` in 1 MB chunks).
- [ ] **Step 4: Run, expect PASS** — `authoringtool.test_rag_sources`.
- [ ] **Step 5: Commit** — `List the RAG PDFs of a scenario and its confirmed translations`.

---

### Task 3: Index the group's PDFs and rebuild when the set changes

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/tasks.py:3065-3138` (`get_index_dir`, `ensure_rag_index`, `chunk_and_index_pdfs`) and `:3881-3885` (force rebuild in `generate_llm_context_for_scenario`)
- Test: `Trust-AI-Platform/authoringtool/test_rag_sources.py`

**Interfaces:**
- Consumes: `group_rag_pdfs`, `rag_pdf_signature` (Task 2).
- Produces:
  - `get_index_dir(scenario_id: int, signature: str = '') -> str` — `scenario_{id}` or `scenario_{id}_{signature}` under `BASE_INDEX_DIR`.
  - `ensure_rag_index(scenario_id: int, signature: str = '')` — same Chroma setup, collection name `scenario_{id}`, persistent path `get_index_dir(scenario_id, signature)`.
  - `remove_rag_indexes(scenario_id: int, keep: str | None = None) -> None` — removes folders in `BASE_INDEX_DIR` named exactly `scenario_{id}` or starting `scenario_{id}_`, except `keep`; uses `shutil.rmtree(..., onerror=_on_rm_error)` and ignores folders it cannot remove (Chroma may hold a file open on Windows).
  - `chunk_and_index_pdfs(scenario_id: int)` — loads the scenario, takes `group_rag_pdfs`, signs it, calls `remove_rag_indexes(scenario_id, keep=<that dir>)`, opens `ensure_rag_index(scenario_id, signature)`; skips ingest if populated; otherwise loads each `RagPdf.path` with `PyPDFLoader`, sets `metadata['source_file']` and `metadata['source_scenario_id']`, keeps the existing chunk-text de-dup.
  - Force rebuild calls `remove_rag_indexes(scenario_id)` in place of the single `rmtree`.

- [ ] **Step 1: Write the failing tests** (patch `BASE_INDEX_DIR` to a temp dir via `mock.patch('authoringtool.tasks.BASE_INDEX_DIR', tmp)`; patch `authoringtool.tasks.ensure_rag_index` with a fake store whose `_collection.count()` returns 0 and whose `add_documents` records docs; patch `authoringtool.tasks.PyPDFLoader` with a fake returning one `Document(page_content=<filename>)` per file):

```python
def test_index_reads_translation_pdfs(self):
    chunk_and_index_pdfs(english.id)
    sources = {d.metadata['source_scenario_id'] for d in fake_store.added}
    self.assertEqual(sources, {english.id, greek.id})

def test_index_folder_follows_the_pdf_set(self):
    chunk_and_index_pdfs(english.id)
    first = fake_ensure.call_args.args
    # add a pdf to greek
    chunk_and_index_pdfs(english.id)
    self.assertNotEqual(first, fake_ensure.call_args.args)

def test_old_index_folders_are_removed(self):
    os.makedirs(os.path.join(tmp, f'scenario_{english.id}_oldsig'))
    os.makedirs(os.path.join(tmp, f'scenario_{english.id}'))
    remove_rag_indexes(english.id, keep=f'scenario_{english.id}_new')
    self.assertEqual(os.listdir(tmp), [])

def test_cleanup_leaves_other_scenarios_alone(self):
    os.makedirs(os.path.join(tmp, f'scenario_{english.id}0'))
    remove_rag_indexes(english.id)
    self.assertEqual(os.listdir(tmp), [f'scenario_{english.id}0'])
```

- [ ] **Step 2: Run, expect FAIL** (`remove_rag_indexes` missing; only own scenario indexed).
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run, expect PASS** — `authoringtool.test_rag_sources`, then `authoringtool` proposal tests that import tasks (`authoringtool.test_proposal_history authoringtool.test_translation_pooling`).
- [ ] **Step 5: Commit** — `Use the PDFs of every translation for proposals and rebuild the index when they change`.

---

### Task 4: Show the translations' PDFs on the scenario page

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/views.py:1145-1150` (`viewScenario` RAG listing) and the context dict (~line 1292)
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/viewScenario.html:450-512`
- Test: `Trust-AI-Platform/authoringtool/test_rag_sources.py`

**Interfaces:**
- Consumes: `group_rag_pdfs` (Task 2), `user_can_view_scenario` (views).
- Produces: context `translation_rag_files: list[RagPdf]` — entries of `group_rag_pdfs(myScenario)` whose `scenario_id != myScenario.id` and whose scenario the viewer can view. Own `rag_files` stays as today (all own files, including ones whose content matches a translation's).

- [ ] **Step 1: Write the failing tests** (log in as owner, `reverse('viewScenario', args=[english.id])`):

```python
def test_translation_pdfs_are_listed_read_only(self):
    response = self.client.get(url)
    self.assertContains(response, 'From translations')
    self.assertContains(response, 'b.pdf')
    self.assertContains(response, reverse('serve_rag_pdf', kwargs={'scenario_id': greek.id, 'filename': 'b.pdf'}))
    self.assertNotContains(response, reverse('delete_rag_pdf', kwargs={'scenario_id': greek.id, 'filename': 'b.pdf'}))

def test_hidden_translation_pdfs_are_not_listed(self):
    # greek built with visibility='private' and another owner
    self.assertNotContains(self.client.get(url), 'b.pdf')
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Under the own files, when `translation_rag_files` is non-empty: a small heading "From translations" and one row per file — `bi-file-earmark-pdf` link opening a modal (ids `pdfModalTr{{ forloop.counter }}`) with an iframe and Download button pointing at `serve_rag_pdf` with the translation's `scenario_id`, followed by muted text `{{ pdf.language }} · {{ pdf.scenario_name }}`; no delete button. Header count shows own + translation files. Empty-state text only when both lists are empty.
- [ ] **Step 4: Run, expect PASS** — `authoringtool.test_rag_sources authoringtool.test_translation_pooling`.
- [ ] **Step 5: Commit** — `List translations' RAG PDFs on the scenario page`.

---

### Task 5: Full check

- [ ] **Step 1:** Run the whole suite (`manage.py test --settings=faithDev.settings_test --noinput`, ~20 min). Expected: only the known failures from the 2026-10-22 baseline (2 NumericRange, `test_task_starter_can_read_result`, pinned version, 2 proposal history); compare by test name.
- [ ] **Step 2:** Fix any new failure caused by these changes and commit.
