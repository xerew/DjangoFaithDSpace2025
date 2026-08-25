# Scenario Families and Evidence: Phase 1–7 Test Guide

This guide audits the implementation against the agreed seven-phase plan and
explains how an administrator or teacher can manually test each available
feature.

## Current completion summary

| Phase | Status | Summary |
|---|---|---|
| 1. Family foundation | Implemented | Families, variant types, origin-chain backfill, copy inheritance, and family/language totals are available. |
| 2. Revision safety | Implemented | Published revisions are immutable; implemented scenarios require a protected draft, graph validation, and explicit publish. |
| 3. Implementation lineage | Implemented | Every new attempt has a `ScenarioImplementation`; scores, answers, progress, exact versions, quality, and lifecycle are linked. |
| 4. Activity comparability | Implemented | Activity concepts and immutable activity revisions are preserved by copies, with a dedicated admin matching screen. |
| 5. Family analytics | Implemented | Local/compatible/historical scopes, language filtering, exact counts, and evidence-aware CSV downloads are available. |
| 6. Proposal integration | Implemented | The 200 threshold, revision/evidence provenance, current-revision targeting, structural-failure separation, and bandit eligibility guards are enforced. |
| 7. Assisted family matching | Implemented | Explainable candidate discovery, optional multilingual similarity, and admin-only confirmation are available. |

All phases have automated coverage. The steps below provide an additional
manual acceptance path through the user interface.

## Test prerequisites

For a complete manual test, prepare:

1. An administrator account.
2. A teacher account.
3. At least one student account.
4. A small public canonical scenario with at least one question activity.
5. A translation and an adaptation copied from that scenario.
6. A test environment where Celery, Redis, PostgreSQL, and the Ollama tunnel
   are running.

Do not use production student records for destructive authoring tests.

## Phase 1 — Family foundation

### 1. Create an adaptation

As a teacher:

1. Open **Scenarios** from the authoring area.
2. Find the canonical scenario.
3. Click **Adapt**.
4. Open the newly created copy.
5. In **Scenario Family**, verify that the original and the copy are shown in
   the same family.
6. Verify that the copy is labelled **Teacher adaptation**.
7. Edit the copy and verify that its subjects initially match the source
   scenario.

Expected result: the copy has the same family and subjects, has the source in
its origin chain, and is classified as an adaptation.

### 2. Create a translation

As a teacher:

1. Return to **Scenarios**.
2. Click **Translate** on the canonical scenario.
3. Edit the copied scenario and select a different language.
4. Open the scenario view.

Expected result: the copy belongs to the same family and is labelled
**Official translation**. The actual text is copied for the teacher to
translate; the platform does not automatically translate every field.

### 3. Check family and language totals

1. Run the canonical scenario and its translation with test students.
2. Open either variant as a teacher.
3. Find the **Scenario Family** card.
4. Check:
   - family implementation total;
   - per-language totals;
   - visible variants and their variant types.
5. Find **Compatible Evidence Pool** and compare its count with the family
   total.

Expected result: family totals include visible family variants. Compatible
evidence can be lower than the historical family total because only exact,
approved evidence is eligible.

### 4. Administrator inspection

1. Open **Admin**.
2. Under **Authoring tool**, open **Scenario families**.
3. Open the family and verify its canonical scenario, subjects, variants,
   evidence-pool count, and implementation count.
4. Open **Scenarios** in Admin and verify the **Family** and **Variant type**
   fields.

## Phase 2 — Revision safety

### 1. Open a protected revision draft

1. Complete a scenario once as a student.
2. Sign in as its creator and open the scenario.
3. Click **Edit**.
4. Verify that **Save Changes** is disabled and the
   **Published evidence is protected** warning appears.
5. Click **Start Revision Draft**.
6. Edit scenario metadata, phases, activities, answers, or routes.

Expected result: editing is allowed only after the draft is opened. Student
access returns a temporary-unavailable message while the draft is open.

### 2. Validate and publish

1. Return to **Edit Scenario**.
2. Enter a short **Change summary**.
3. Click **Validate & Publish**.
4. If the scenario has an unreachable activity, missing answer route, broken
   branch, or cycle, verify that publishing is refused with a graph error.
5. Fix the graph and publish again.
6. Open the scenario as a student.

Expected result: a new current published revision is created and student
access resumes on that revision.

### 3. Inspect immutable history

1. Open **Admin → Authoring tool → Scenario versions**.
2. Verify version number, status, fingerprints, previous version, publisher,
   publish time, change summary, and read-only snapshot.
3. Open **Scenario revision drafts** to inspect drafts that are still open.

Expected result: published snapshots cannot be edited or deleted. Historical
unknown data has a version-zero **Legacy revision** boundary.

## Phase 3 — Implementation lineage

### 1. Inspect exact attempt lineage

1. Open a scenario as a student.
2. Answer at least one question.
3. As an administrator, open
   **Admin → Authoring tool → Scenario implementations**.
4. Find the attempt and verify:
   - status and start/completion timestamps;
   - scenario version;
   - version confidence;
   - data quality;
   - last activity.
5. Open **User & Scenario Scores** and verify its implementation link.
6. Open **User's Answers** and verify its implementation, scenario version,
   and activity revision links.
7. Confirm that teacher preview rows are absent from student evidence totals.

### 2. Test data-quality controls

In **Scenario implementations** or **User & Scenario Scores**:

1. Select one or more records.
2. Open the **Action** dropdown.
3. Test:
   - **Mark selected implementations as clean**;
   - **Mark selected implementations as suspect**;
   - **Exclude selected implementations from evidence**.

Expected result: suspect and excluded rows are not eligible for the compatible
AI evidence pool, and the quality value stays synchronized between the attempt
and score views.

## Phase 4 — Activity comparability

### 1. Inspect concepts and revisions

1. Create an adaptation or translation using **Adapt** or **Translate**.
2. Open **Admin → Authoring tool → Activity concepts**.
3. Open a concept and inspect its source/translated activity members.
4. Open **Activity revisions** and compare the immutable revision for each
   published scenario version.

Expected result: unchanged copied/translated activities retain the same
lineage key and concept. Each published version has an immutable activity
revision.

### 2. Manually correct a mapping

1. Open **Admin → Authoring tool → Activity matching**.
2. Filter by family or language.
3. Assign the same concept to truly equivalent activities.
4. Save.

Expected result: the admin rejects a concept from a different scenario family.

## Phase 5 — Family analytics

### 1. Test evidence scope filters

As a teacher:

1. Open a scenario.
2. Click **Metrics & AI**.
3. Under **Evidence scope**, switch between:
   - **Compatible family**;
   - **This scenario only**;
   - **Historical analytics**.
4. Compare the implementation total and evidence sources.

Expected result:

- **Compatible family** uses exact evidence from approved current versions.
- **This scenario only** uses exact evidence from the current scenario version.
- **Historical analytics** shows local legacy records whose exact version is
  unknown. It is read-only for LLM proposal generation.

For compatible evidence, correctness may pool compatible languages while
timing remains restricted to the target scenario language.

### 2. Test exact, compatible, and family counts

1. Return to the scenario view.
2. Check the **Scenario Family**, **Compatible Evidence Pool**, and
   **Evidence History** cards.
3. Verify that:
   - the local/current count can differ from the compatible count;
   - the compatible count can differ from the complete family history;
   - legacy records appear as historical rather than exact evidence.

### 3. Test permission-aware visibility

1. Put a private scenario in the same family as a public scenario.
2. Sign in as a different teacher who cannot view the private scenario.
3. Open the public scenario and **Metrics & AI**.

Expected result: the private scenario’s identifying source details and direct
link are not exposed to that teacher.

### 4. Filter by language and download CSV

1. On **Metrics & AI**, select a value from **Evidence language**.
2. Generate **Activity metrics** and **Risk flags**.
3. Click **Download metrics CSV** and **Download flags CSV**.
4. Verify that the filename and rows identify the selected scope/language and
   include revision/evidence provenance.
5. In Admin, export implementations, scores, or answers for full family,
   language, revision, concept, and attempt lineage.

Expected result: every scope/language selection has a separate cache and CSV,
so reports cannot be mixed accidentally.

## Phase 6 — Proposal integration

### 1. Approve compatible evidence

As an administrator:

1. Open
   **Admin → Authoring tool → Scenario version compatibilities**.
2. Filter by family, language, variant type, or status.
3. Select versions and run one of:
   - **Approve selected for family evidence**;
   - **Mark selected as needing review**;
   - **Exclude selected from family evidence**.
4. Reopen the scenario’s **Metrics & AI** page and check the compatible total.

Expected result: only exact, clean/unreviewed records from compatible current
versions count toward the proposal threshold.

### 2. Test the 200-evidence threshold

1. Open **Metrics & AI** with **Compatible family** selected.
2. If fewer than 200 eligible implementations exist, verify the warning.
3. Verify that the warning reports eligible compatible evidence, not the
   scenario’s full historical total.
4. Continue only for a manual low-data inspection.

### 3. Generate and inspect proposal provenance

Only the scenario creator or an administrator can generate:

1. On **Metrics & AI**, select **Compatible family**.
2. Click **Generate LLM Context** or **Force Rebuild**.
3. Wait for Celery to finish.
4. Click **View Proposals**.
5. Open **Proposal History** and then a run.
6. Verify the target scenario version, evidence scope, evidence version IDs,
   languages, implementation count, and source summary.

Expected result: a generation run is tied to the scenario’s current version.
Changing the scenario version or compatible source signature makes the old run
historical instead of silently applying it to the new version.

### 4. Test structural-failure separation

1. Open **Admin → Authoring tool → Proposal structural failures**.
2. Inspect generation, translation, acceptance, application, or graph-integrity
   failures.
3. In a proposal, reject with
   **Malformed or structurally invalid proposal**.
4. Open **Admin → Authoring tool → Q values**.

Expected result: structural feedback is recorded but does not add a pedagogical
bandit reward.

### 5. Test bandit evidence eligibility

1. Accept or pedagogically reject a proposal from the current compatible run.
2. In Admin, inspect its Q-value reward count.
3. Publish a new scenario revision or change the approved compatibility pool.
4. Rebuild the Q-value context.

Expected result: the now-stale review is not counted. Local-only,
historical, structurally invalid, archived, incompatible, or wrong-revision
runs never train the bandit.

## Phase 7 — Assisted family matching

This workflow is intentionally administrator-only.

### 1. Open the family governance dashboard

1. Open **Admin → Authoring Tool → Scenario family review dashboard**.
2. Verify that every family row shows:
   - its canonical scenario and subjects;
   - translations and adaptations;
   - the current immutable revision and revision history for each variant;
   - student implementation totals per variant and for the family;
   - the number of candidates waiting for review.
3. Use **Open candidate inbox** to see candidate pairs involving that family.

Expected result: implementations remain attributed to their original scenario
variant even when the family total combines them.

### 2. Associate scenarios manually

1. Open **Admin → Authoring Tool → Scenario family review dashboard**.
2. Click **Associate scenarios manually**.
3. Use the filter boxes to select:
   - the scenario whose family/canonical scenario should remain;
   - the scenario to associate.
4. Choose one relationship:
   - **Official translation**;
   - **Adaptation / revised copy**;
   - **Related topic only**;
   - **Not related**.
5. Add administrator review notes.
6. Click **Preview association**.
7. Inspect both families, languages, current immutable revisions, variants,
   implementation totals, warnings, and the merge impact.
8. Click **Confirm** only after verifying the preview.

Expected result: translation and adaptation decisions perform a logical family
merge. Related-topic and unrelated decisions are audited without changing
family membership. Implementations, answers, and immutable revisions remain
on their original scenario records.

Important: a revision is not an association between two scenario records.
Create a revision through the scenario draft/publish workflow. A separately
copied and edited scenario is an **Adaptation / revised copy**.

### 3. Scan for candidates

1. From the dashboard, or from
   **Admin → Authoring Tool → Scenario family candidates**, click
   **Scan scenarios**.
2. Use **Force profile refresh** if scenario content changed and profiles need
   rebuilding.
3. Filter the results by decision or suggested relationship.

Expected result: the scan creates suggestions; it does not merge families.

### 4. Request an optional Ollama second opinion

1. Make sure the Ollama tunnel and Celery worker are running.
2. On the dashboard or candidate inbox, click
   **Ask Ollama to review pending**. The batch size is controlled by
   `SCENARIO_FAMILY_REVIEW_LLM_BATCH_LIMIT`.
3. Alternatively, open one candidate and click
   **Ask Ollama for second opinion**.
4. Refresh after the Celery task completes.
5. Compare the deterministic suggestion with the LLM relationship,
   confidence, reasoning, cited evidence, and warnings.

Expected result: Ollama reviews the exact immutable revisions stored on the
candidate. Its structured output is saved for the administrator but never
changes a family automatically. A failed connection is shown as an LLM failure
and can be retried without affecting the deterministic candidate.

### 5. Review and classify a candidate

1. Open a pending candidate.
2. Compare:
   - scenario metadata and language;
   - structure and content fingerprints;
   - phase/activity counts;
   - ordered activity sequence;
   - shared activity-lineage signals;
   - lexical/content similarity;
   - multilingual embedding evidence, when configured.
3. Choose exactly one:
   - **Same family — translation**;
   - **Same family — adaptation**;
   - **Related topic only**;
   - **Not related**;
   - **Review later**.

Expected result: only the first two decisions join scenarios into one family.
Similarity is advisory; an administrator makes the final decision.

### 6. Verify the decision

1. Open **Admin → Authoring tool → Scenario family match decisions**.
2. Verify the actor, decision, previous families, resulting family, scores,
   evidence snapshot, and timestamp.
3. Open the affected scenario’s public/teacher view and inspect
   **Scenario Family**.

## Final acceptance checklist

All seven planned phases are implemented. Before production rollout:

1. Apply migrations through `0067`.
2. Restart web, Celery, Celery Beat, and Rasa action containers so all workers
   load the new schema and lineage-aware code.
3. Run the automated Django suite.
4. Perform one draft/publish smoke test and one student implementation smoke
   test on a non-production scenario.
5. Generate one language-filtered metrics CSV and one compatible-evidence
   proposal run, then inspect their provenance in Admin.
