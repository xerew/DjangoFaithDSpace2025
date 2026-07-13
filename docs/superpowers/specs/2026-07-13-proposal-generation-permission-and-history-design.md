# Proposal Generation Permission and History — Design Spec

## Goal

Two changes to the `authoringtool` app's AI-proposal review workflow:

1. Restrict proposal *generation* (regenerating a scenario's `ActivityProposal`s) to the scenario's creator (or an admin), matching the permission pattern already used on ~10 sibling views in this file.
2. Preserve past generation batches instead of deleting them, and add a read-only history UI so a scenario's creator (or admin) can look back at what was proposed, and what was accepted/rejected/edited, across previous regenerations.

## Current State (confirmed by reading the code)

- `trigger_llm_context_task` (`authoringtool/views.py:2342`) is the AJAX endpoint the "Generate"/"Force rebuild" button on `ai_metrics_scenario.html` calls. It only has `@login_required` — **any logged-in user can trigger regeneration for any scenario by ID today.**
- `generate_llm_context_for_scenario` (`authoringtool/tasks.py:2425`) unconditionally runs, near its end:
  ```python
  ActivityProposal.objects.filter(scenario=scenario).delete()
  UserProposalReview.objects.filter(proposal__scenario=scenario).delete()
  ```
  before recreating proposals from the scenario's current `ActivityFlag`s. This happens on **every** call, regardless of the `force_rebuild` flag (`force_rebuild` only controls RAG-index/LLM-context cache clearing, a separate concern). So every regeneration destroys the full prior review history for that scenario — every accept/reject decision, every edit diff, every rejection reason.
- `proposal_list_view` (`authoringtool/views.py:2371`) — the live per-scenario review page — is also only `@login_required`, with no creator/ownership restriction. It queries `ActivityProposal.objects.filter(scenario=myScenario)` with no status/date scoping, and auto-creates a `UserProposalReview` per proposal for whichever user is viewing it (supporting multiple independent reviewers per shared proposal, via `UserProposalReview`'s `unique_together = ('proposal', 'user')`).
- `apply_proposals_to_cloned_scenario` (`authoringtool/tasks.py:3077`, used by "Create Personalised Scenario") queries:
  ```python
  UserProposalReview.objects.filter(user=user, proposal__scenario=original_scenario, status='accepted')
  ```
  with **no scoping beyond scenario + user + status** — safe today only because old proposals are always deleted before new ones exist, so there's never more than one batch to accidentally mix together.

## Scope

- `Trust-AI-Platform/authoringtool/views.py` — permission fix on `trigger_llm_context_task`; new `proposal_history_view` and `proposal_history_run_detail_view`; scope `proposal_list_view` to the current run.
- `Trust-AI-Platform/authoringtool/tasks.py` — `generate_llm_context_for_scenario` stops deleting, creates a `ProposalGenerationRun` instead; `apply_proposals_to_cloned_scenario` gains current-run scoping.
- `Trust-AI-Platform/authoringtool/models.py` — new `ProposalGenerationRun` model; `ActivityProposal` gains a `generation_run` FK.
- `Trust-AI-Platform/authoringtool/urls.py` — two new routes.
- `Trust-AI-Platform/authoringtool/admin.py` — register `ProposalGenerationRun`.
- New templates: a history index (list of past runs for a scenario) and a run-detail page (read-only proposal cards for one past run).
- Migration: add `ProposalGenerationRun`, add `ActivityProposal.generation_run`, backfill existing `ActivityProposal` rows (if any exist in production) into one `is_current=True` run per scenario that has proposals, so nothing already in the database is orphaned or silently hidden from the now-current-scoped `proposal_list_view`.

## Global Constraints

- **Generation permission:** `@group_required('teachers')` plus `scenario.created_by != request.user and not is_admin_user(request.user)` → `403`, the exact pattern already used by `accept_proposal`/`reject_proposal`/`updateScenario`/etc. (`views.py` lines 429, 450, 505, 621, 634, 651, 808, 882, 1018, 2201, 2447, 2460). Response is `JsonResponse({"error": ...}, status=403)`, not `HttpResponseForbidden`, since `trigger_llm_context_task` is an AJAX/JSON endpoint (matching its own existing response shape, not the HTML-page pattern other views in that list use).
- **History/review viewing permission stays exactly as permissive as it is today** (`@login_required` only, no creator restriction) — this spec does not change who can view or make decisions on the live review page, only who can trigger a *new* generation. This is a deliberate asymmetry, not an oversight: it already exists today between generation (about to become creator-gated) and review (already open to any logged-in user), and tightening review/history access is a separate decision out of scope here.
- **Exactly one current run per scenario:** `ProposalGenerationRun` enforces this with a Postgres partial unique constraint (`UniqueConstraint(fields=['scenario'], condition=Q(is_current=True), name='...')`), not just application-level discipline — the production DB is Postgres, which supports partial/conditional unique indexes. The task must flip the old run to `is_current=False` and create the new run inside one `transaction.atomic()` block so the constraint is never violated mid-write.
- **No more deletion of `ActivityProposal`/`UserProposalReview` on regenerate.** The two `.delete()` calls in `generate_llm_context_for_scenario` are removed entirely. Old proposals, their reviews, and their `ActivityProposalEditEvent` audit trails simply stay in the database forever, linked to their (now-archived) `ProposalGenerationRun`. No pagination/pruning of history is in scope — matches this codebase's established "don't prematurely optimize for scale" pattern.
- **`proposal_list_view` must filter to the current run** (`ActivityProposal.objects.filter(scenario=myScenario, generation_run__is_current=True)`) so its behavior for active review is unchanged from today — same counts, same auto-created reviews, same everything, just scoped to the live batch instead of all-time.
- **`apply_proposals_to_cloned_scenario` must also filter to the current run** (`proposal__generation_run__is_current=True` added to its existing filter) — without this, "Create Personalised Scenario" would start silently materializing accepted decisions from old, superseded batches once history is no longer deleted. This is a required correctness fix bundled into this change, not optional.
- **Backfill migration for existing data:** any `ActivityProposal` rows that exist in the production database before this migration runs need a `ProposalGenerationRun` created for their scenario (one run per distinct scenario that has proposals, `is_current=True`, `created_at` = the earliest existing proposal's `created_at` for that scenario, `created_by` = the scenario's `created_by`) and their `generation_run` FK backfilled to it — otherwise `proposal_list_view`'s new current-run filter would show nothing for scenarios that already have in-progress reviews.

## Behavior

### Generation (`trigger_llm_context_task`)

- Non-creator, non-admin POST → `403` JSON, no task queued, no DB changes.
- Creator or admin POST → unchanged from today (queues `generate_llm_context_for_scenario.delay(...)`, returns the task id).

### Regeneration (`generate_llm_context_for_scenario`)

- Old behavior: delete all proposals/reviews for the scenario, then create fresh ones.
- New behavior: in one transaction, mark the scenario's current `ProposalGenerationRun` (if any) `is_current=False`, create a new `ProposalGenerationRun(scenario=scenario, created_by=<the user who triggered generation>, is_current=True)`, then create the new batch of `ActivityProposal`s tied to that new run (exactly as today, just with the added FK). Nothing about how proposals are generated/flagged changes — only what happens to the *previous* batch.

### Live review (`proposal_list_view`)

- Unchanged behavior, just scoped to `generation_run__is_current=True`. A scenario with no `ProposalGenerationRun` yet (never generated) behaves exactly as an empty-proposals scenario does today.

### History index (new: `proposal_history_view(scenario_id)`)

- `@login_required` (matching the live review page's own access level, per the constraint above).
- Lists the scenario's `ProposalGenerationRun`s where `is_current=False`, most recent first, each showing: created date, who triggered it, and decision counts (accepted / rejected / never-decided) computed from that run's proposals' `UserProposalReview`s for the *current viewing user* (mirroring how the live page's counts are per-user, not global).
- Empty state if the scenario has never been regenerated (i.e., at most one run ever existed, which is still current).

### Run detail (new: `proposal_history_run_detail_view(scenario_id, run_id)`)

- Same access level. 404s if the run doesn't belong to the given scenario (same IDOR-safe scoping pattern used everywhere else in this app).
- Read-only cards for every proposal in that run — title/suggested action, this user's decision (accepted/rejected/never decided), decided-at, whether it was edited (and if so, the edit count / final edited JSON), rejection reasons if rejected. Same visual card language as `proposal_list.html`, minus every action button (no accept/reject/edit controls — it's history, not an active queue).

## What Does NOT Change

- The shape or meaning of `ActivityProposal`/`UserProposalReview`/`ActivityProposalEditEvent`'s existing fields — only a new FK is added to `ActivityProposal`.
- `accept_proposal`/`reject_proposal`/`edit_proposal_json` — unchanged; they already operate on a specific review/proposal by ID, and since those IDs are only ever surfaced through the (now current-run-scoped) live review page, they naturally only ever act on the current run in practice.
- The Q-learning `QValue` reward signal wired to `UserProposalReview`'s `post_save` — untouched.
- Who can *view*/*decide on* proposals — only who can *generate* a new batch is being restricted.
