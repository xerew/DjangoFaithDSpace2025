# Same Metrics & AI data and shared RAG PDFs across confirmed translations

Date: 2026-11-16 · Branch: `feature/translation-metrics-rag` (from `main`)

## Intent

Confirmed 1:1 translations of a scenario (e.g. English "Pendulum scenario" 90 and Greek
"Σενάριο για εκκρεμές" 23) are one scenario in different languages. Their Metrics & AI
pages must show the same data, and the AI proposals must draw on the same reference PDFs.
This is the first step toward one shared set of proposals per translation group (later
work, not part of this change).

## Goals

1. **Time pooled across languages.** Avg Time and Timing Total on Metrics & AI count the
   students of every confirmed translation, not only same-language students. Result:
   every member of a group shows the same numbers and the same risk flags (under its own
   activity names). The header's per-scenario split ("X from this scenario and Y from
   translations") still differs per page; the total is the same.
2. **Header text** drops "average time uses <language> students only"; it says metrics and
   AI proposals use all the implementations.
3. **Stale caches rebuilt once.** Metrics/flags CSVs computed under the old rule are not
   reused.
4. **RAG shared across the group.** A proposal run retrieves from the PDFs of the scenario
   and its confirmed translations that are public and not archived (user decision
   2026-11-16: private and organization-only translations do not share their PDFs). A PDF
   present in several members (same content) is indexed once.
5. **RAG index refreshes itself** when that PDF set changes (upload, delete, translation
   confirmed or undone), without the "force" button. Today the index is built once and
   never picks up later uploads or deletions.
6. **Scenario page lists the translations' PDFs** read-only, labelled with the
   translation they come from, so teachers see what the AI reads. Only the scenario's own
   PDFs can be deleted from its page.

## Non-goals

- Sharing one set of proposals across translations (next step).
- Pooling the per-scenario teacher analytics (Sankey, performance, time spent, paths,
  performers, student performance).
- The family/"compatible" evidence path (`SCENARIO_FAMILY_FEATURES`, off) is left as is.
- Changing the English-only embedding model (all-MiniLM-L6-v2).

## Approach

One sensible way, decided in conversation: change the existing local-scope metrics path
and the existing RAG helpers, keyed on the confirmed translation group
(`translation_group_ids`, `translation_activity_maps`). Alternatives (a separate group-level
metrics file, a shared PDF folder per group) were not needed: the per-scenario
computation already pools answers through the activity map, and keeping per-scenario
folders leaves upload/delete permissions as they are.

## Design

- `compute_category_metrics_per_phase_activity` (local path): drop the `same_language`
  filter so every pooled implementation's timing counts.
- Cache: add a fixed rule tag to the cache file prefix in
  `get_scenario_evidence_cache_paths` so existing CSVs are not reused.
- RAG: the index directory/collection key includes a signature of the group's PDF set
  (file content hashes across own + translation folders). A changed set means a new
  index; old index folders for the scenario are removed. Identical files (same hash) are
  loaded once. `generate_llm_context_for_scenario`'s force option still clears it.
- Scenario page (`viewScenario`): own PDFs as today, plus a read-only "From translations"
  list (file name, translation name/language, a link to view). The serve view allows
  viewing a translation's PDF only to users who can view that translation.
- Tests: update the same-language timing test; add tests that two confirmed translations
  give identical metrics/flags rows, that the RAG PDF set includes members and
  de-duplicates by content, and that a changed PDF set produces a new index key.

## Left to the builder

Exact helper names, how the index signature is stored, template styling (no emojis;
bi-* icons), wording of the new strings.
