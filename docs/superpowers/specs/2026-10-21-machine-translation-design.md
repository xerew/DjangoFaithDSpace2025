# Machine Translation of Scenarios — Design

Date: 2026-10-21
Status: approved 2026-10-21

## Why

To use a scenario in another language today, a teacher presses **Translate**
on the scenario card (`authoringtool/templates/authoringtool/scenarios.html`,
which links to `duplicate_scenario` with `?variant=translation`). That makes an
untranslated copy, and the teacher retypes every phase, activity, answer and
feedback by hand. It takes a long time.

The platform already has access to an Ollama server on the NTUA machine. The
model to use for translation there is **gemma4:31b**. The admin wants **Translate**
to make a machine-translated copy in the background, and wants a person to be
able to check that translation and mark it verified.

Translation is different from duplication (**Adapt**):

- **Adapt** makes a new scenario owned by whoever pressed it, with a reference
  to the original. Its structure can change freely afterwards.
- **Translate** makes a scenario in another language whose **structure stays
  identical** to the original: same phases, activities, answers, routes,
  scoring and branching. Only the text differs.

## Goals

- **Translate** opens a modal. The user picks one target language from the
  `Language` table (`authoringtool/models.py: Language`), not including the
  scenario's own language, and presses **Start translation**.
- The job goes into a queue and runs in the background. The user doesn't have
  to stay on the page.
- The new scenario's card appears in the user's scenario list straight away,
  with a progress badge ("Translating… 12 / 30 activities"). When the job
  finishes, the badge becomes **Machine-translated**.
- When a job finishes or fails, the user sees a message at the top of the next
  page they open ("Your Greek translation of Ohm's Law is ready", with a link).
- The new scenario:
  - is created by, and belongs to, the user who pressed Translate (teacher or
    admin);
  - starts **private**, whatever the original's visibility, and the creator
    publishes it when ready;
  - has `variant_type='translation'`, the target language in `language`, and
    `origin_scenario` set to the original;
  - joins the original's family (`ensure_family()`), as the current
    `duplicate_scenario` does.
- If Ollama can't be reached from the Django/Celery containers, the modal shows
  a warning banner ("The translation service can't be reached right now.
  Machine translation is paused. Please try again later.") and **Start
  translation** is disabled.
- **Admins only** can assign a reviewer to a machine-translated scenario. The
  reviewer can be **any teacher or admin**. Users who aren't in the teachers
  group and aren't admins can't be picked, because the editor views are
  limited to them (`group_required('teachers')`). The assign box searches all
  teachers and admins. Those whose organization's `language`
  (`organization/models.py`) matches the target language are listed first as
  suggestions.
- The reviewer works in the **normal scenario editor in text-only mode**:
  controls that add, delete, reorder or reroute anything are hidden, and the
  original wording is shown in a small grey box under each text field. A
  **Mark translation verified** button sits at the top.
- When verified, the tag changes to **Translation verified**, with the
  verifier's name and the date.
- The reviewer stays on the scenario as a co-author after verifying, until the
  scenario's creator or an admin removes them.
- Anyone who presses Translate (any teacher who can see the scenario, or an
  admin) gets a machine translation. The old "untranslated copy" behaviour of
  the Translate button goes away. Adapt is unchanged.

## Non-goals

- No change when the original scenario changes later. A translation is a
  snapshot, and nothing is updated, flagged or re-translated.
- No translation of text inside images or videos. Images, video URLs,
  simulations, LabsLand and VR/AR experiments are copied as they are. They can
  be swapped by hand afterwards (see "Text-only mode").
- No automatic swap to a simulation in the target language. *(Builder's call
  to leave out of v1. Manual swaps are allowed.)*
- No "My translations" page and no email to the requester. The admin chose the
  card badge plus the next-page message.
- No change to Adapt (`duplicate_scenario` with `variant=adaptation`).
- No general co-author feature yet. That comes next (see "Co-author
  groundwork").

## Anti-goals (failure even if it "works")

- **Untranslated text left in a scenario marked Machine-translated.** The
  current `translate_text` (`authoringtool/tasks.py`) quietly returns the
  original text when a request fails. The new code must not. A piece of text
  that can't be translated makes the job fail, visibly.
- **A translation whose structure differs from the original.** Every phase,
  activity, answer, feedback, route, branching and bunch is copied 1:1. Nobody,
  including the creator, the reviewer or admins, can change a machine
  translation's structure through the editor.
- **Broken content.** HTML tags and attributes, image `src`s, links, formulas
  (LaTeX/MathML) and numbers come out of translation unchanged.
- **A machine translation visible to students before its creator publishes
  it.**
- **An Ollama outage marking jobs failed.** If the service drops in the middle
  of a job, the job waits and carries on later, keeping the work already done.
- **Several translations running at once on the NTUA GPU.** At most one job
  runs at a time across the whole platform, oldest first.
- **Emojis in the UI.** Use plain text labels, or the Bootstrap icons
  (`bi-*`) the platform already uses.

## Constraints and existing code

- Ollama URL: `settings.OLLAMA_URL` (default
  `http://host.docker.internal:11434`). `tasks.py` also defines its own
  `OLLAMA_URL` from the environment.
- **Model: `gemma4:31b`** on the NTUA server. The admin confirmed it with
  `ollama list` / `ollama show`, and it is the model the server's
  `OLLAMA_MODEL` names. It is a general instruction model, not a translator
  fine-tune with a fixed template. That means it can return a whole activity's
  fields as JSON. Use Ollama's structured output (`format` with a JSON schema)
  and validate the result anyway. Add a new setting
  `LLM_MACHINE_TRANSLATION_MODEL`, read from the environment, default
  `gemma4:31b`. Leave `LLM_TRANSLATION_MODEL` (`aya-expanse:32b`, used by
  proposals) alone. The local `gemma3-translator-4b.modelfile` is not relevant.
- Don't reuse `translate_text`. It assumes English is the source, translates
  each string with no context, has no timeout, and hides failures.
  `ollama_text` also has no timeout. The new client sets one.
- Celery: one worker (`celery -A faithDev worker`), Redis broker, beat already
  running (`CELERY_BEAT_SCHEDULE` in `settings.py`, which has the email-outbox
  drain). The email outbox (`accounts/tasks.py`, spec
  `2026-10-20-email-outbox-design.md`) already solves "one run at a time,
  claim rows safely, recover stale claims". Follow the same pattern.
- `Scenario.language` is free text, not a foreign key. Its values were
  normalised to `Language` names by migration `0045`. `Language` holds names
  only.
- `Scenario` has a unique constraint on (`created_by`, `name`). The translated
  name must not collide: add a suffix such as "(2)" if it does.
- The current `duplicate_scenario` (`views.py:2649`) is the reference for what
  a structural copy involves. It copies phases, activities (including
  `lineage_key` and `concept`), answers, `NextQuestionLogic`,
  `EvQuestionBranching`, `QuestionBunch`, `start_activity`, and calls
  `ensure_current_version`. **It does not copy `AnswerFeedback`.** The
  translation copy must copy and translate those as well. (This is also a bug
  in Adapt, but that is a separate fix.)
- Permissions live in `views.py`: `is_admin_user`, `user_can_view_scenario`,
  `user_can_edit_scenario`, `user_manages_scenario`.
- Confirmed translation pairs live in `TranslationMatch`
  (`models.py:1300`). Proposals pool evidence across `status='confirmed'` pairs
  (`translation_matching.py`). `scan_translation_matches` marks a confirmed
  pair `changed` when either side's flow signature changes.

## What gets translated

For each scenario: `name`, `description`, `learning_goals`.
For each phase: `name`, `description`.
For each activity: `name`, `text` (HTML), `helper`.
For each answer: `text`.

`AnswerFeedback.text` and the three `EvQuestionBranching.*_feedback` texts are
**copied unchanged and not translated**. *(Builder's call: they can only be
edited in Django admin, and neither the student view nor the Rasa actions read
them, so translating them would add failure risk and an unreviewable field for
no visible gain.)* They are still copied, so the structure stays identical.

`plain_text` is not translated separately. It is rebuilt from the translated
`text` the same way the editor builds it today.

Not copied: the `llm_context`, `related_act_llm_context`,
`llm_image_description` and `short_llm_summary` fields. They are regenerated
later by the normal LLM-context task if needed. `correct_count` and
`incorrect_count` start at 0. *(Builder's call: these describe the original's
students, not the translation's.)*

Each **activity is translated as one unit**, with its answers, so terms stay
consistent within an activity. The
scenario header and each phase are units of their own. Including a short
scenario-level context (scenario name and learning goals) in each prompt is
recommended.

The source language is the scenario's `language`. If it is blank, the modal
also asks for the source language. *(Builder's call.)*

## Translation job and queue

A new model, for example `ScenarioTranslationJob`:

- `source_scenario`, `target_scenario` (created when the job is queued, so the
  card can appear straight away), `source_language`, `target_language`,
  `requested_by`
- `status`: `waiting`, `translating`, `done`, `failed`
- `units_total`, `units_done` (for the badge), `error`, `attempts`,
  `claimed_at`, `started_at`, `finished_at`
- `notified_at` (when the "ready" or "failed" message was shown)
- `model` (the Ollama model tag actually used)
- `verified_by`, `verified_at`

The target scenario's structure (all rows, with the source text) is copied when
the job is queued, inside one transaction. The job then replaces text unit by
unit. Each unit's result is saved as soon as it's validated, so a resumed job
continues where it stopped. *(Builder's call: copying first means the
structure is guaranteed identical and progress is easy to count. A scenario
whose job isn't `done` can't be opened, shown to students, or published.)*

**Running:**

- Queueing the job also starts the drain task. A beat entry (every minute)
  catches anything missed.
- The drain task runs only when no other job is `translating` (with a stale
  claim timeout). It claims the oldest `waiting` job and works through its
  units.
- If Ollama is unreachable or times out, the job goes back to `waiting` with
  its progress kept, and is retried on a later drain. This never counts as a
  failure.
- If the model's output fails validation, the unit is retried a few times. If
  it still fails, the job becomes `failed` with a readable error naming the
  activity. The card shows **Translation failed** with **Retry** (carries on
  from the failed unit) and **Delete**.

**Validation of each unit's output (required):**

- Every expected field is present and non-empty when the source was non-empty.
  The answer and feedback counts match.
- HTML tag structure, attributes, image `src`s and link `href`s are identical
  to the source.
- Formulas and numbers in the source appear unchanged in the output.
- How the HTML is kept intact is the builder's choice: send whole HTML and
  check it afterwards, or extract text segments, translate them, and put them
  back. Either way the checks above apply.

**Finishing:**

- Status becomes `done`, the target scenario's `current_version` is refreshed
  (`ensure_current_version`), and a **confirmed** `TranslationMatch` is created
  between source and target. It has `exact_match=True`, an `activity_map`
  pairing every original activity with its copy, flow signatures filled in, and
  `scenario_a`/`scenario_b` ordered by id as the model's constraint requires.
  This lets proposals pool evidence across the two straight away, because the
  structure is identical by construction. If the original's structure changes
  later, the scan marks the pair `changed` and pooling stops for it. The
  translation itself is not touched.
- **Scan change needed** *(builder's call)*: today `scan_translation_matches`
  rewrites `activity_map` with reachable activities only, and marks a confirmed
  pair `changed` whenever the scorer doesn't produce it as a candidate. That
  can happen with no structural change at all, for example from word overlap
  between close languages. For pairs created by a machine-translation job, the
  scan must:
  - keep `activity_map` as created;
  - mark the pair `changed` only when a flow signature actually differs from
    the stored one.
  Other pairs keep today's behaviour.

## Ollama reachability check

- In `docker-compose.yml` only the `celery` service points at the SSH tunnel
  (`OLLAMA_URL: http://ollama-tunnel:11434`). `web` uses the
  `host.docker.internal` default, which can't see the NTUA machine. So **the
  check runs in Celery, where translation runs**. *(Builder's call: no compose
  change or server deploy step needed.)*
- A beat task, about every minute, calls `GET {OLLAMA_URL}/api/tags` with a
  short timeout (about 3 seconds) and checks that the translation model is
  listed. It writes `{reachable, model_present, checked_at}` to the Redis
  cache. The drain task updates the same entry whenever a request succeeds or
  fails.
- `web` only reads that cache entry. If it is missing or older than about 3
  minutes, it counts as unreachable.
- The Translate modal reads the status when it opens. If the service is down,
  the modal shows the warning banner and disables Start. The POST that queues a
  job checks again and refuses if the service is down.
- *(Builder's call)* Cards of jobs that are waiting because the service is
  down read "Waiting — translation service unavailable".

## Tags on scenario cards and pages

- While running: "Translating… N / M" (or "Waiting in queue").
- Done, not verified: **Machine-translated**.
- Verified: **Translation verified**, with "Verified by <name> on <date>" on the
  card or scenario page.
- Failed: **Translation failed**, with Retry and Delete.
- Shown on teacher-facing scenario lists and the scenario's own page. Students
  don't see the tags. *(Builder's call.)*
- No emojis. Use a `bi-*` icon (for example `bi-translate`, `bi-patch-check`)
  if an icon is wanted.

## "Ready" message on the next page

There is no notification model in the platform. On any page load by the
requester, finished or failed jobs with `notified_at` empty are turned into a
Django `messages` entry ("Your Greek translation of Ohm's Law is ready —
open it." / "…failed — see details.") and stamped `notified_at`. A middleware
or context processor does this. It must be cheap: a single indexed query, and
skipped for anonymous users and students.

## Reviewer and co-author groundwork

- A new model, for example `ScenarioCollaborator`: `scenario`, `user`, `role`
  (`translation_reviewer` now; `co_author` later), `added_by`, `added_at`.
  Unique on (`scenario`, `user`). This is the start of the co-author feature
  that comes next, so it is general rather than translation-specific.
- `user_can_edit_scenario` returns True for collaborators. Sharing, deleting,
  archiving and visibility (`user_manages_scenario`) stay with the owner and
  admins.
- Only admins can add a reviewer, and only teachers or admins can be picked
  (enforced on the server, not only in the picker). The scenario's creator or
  an admin can remove one.
- A collaborator must be able to see the private translation in their
  scenario list and open it. `user_can_view_scenario` and the list query must
  include collaborators.
- *(Builder's call)* The reviewer is told by email, sent directly with
  `send_mail` and not through the campaign outbox (the outbox spec says
  notification emails skip the queue). The scenario also appears in the
  reviewer's own scenario list.
- **Mark translation verified** is available to an assigned reviewer and to
  admins. It sets `verified_by` and `verified_at`. Later text edits keep it
  verified. An admin can undo verification. *(Builder's call.)*

## Text-only mode in the editor

- Applies to every scenario made by machine translation (it has a
  `ScenarioTranslationJob` as target) and to everyone, including the creator
  and admins.
- In the templates, controls that add, delete, reorder or reroute phases,
  activities or answers, and that change correctness, weights, score limits,
  activity types, branching targets, bunches or the start activity, are hidden.
- **Media can be swapped.** In text-only mode, people who can edit the
  translation may still change an activity's simulation, LabsLand experiment
  and VR/AR experiment, and the images and video URLs of the scenario, phases,
  activities, answers and feedback. A typical case is picking the Greek PhET
  version of a simulation.
- **On the server**, every view that changes structure refuses the request for
  these scenarios (403 or an error message). Hiding buttons alone is not
  enough. Media changes are allowed through.
- **Media swaps must not break the pairing.** The flow signature used by the
  translation scan includes simulation, lab and VR ids
  (`translation_matching._fields`). For machine-translation pairs, the "did
  the structure change" comparison in the scan must ignore those media fields
  (and image anchors), so a manual swap doesn't mark the pair `changed`. How
  is the builder's choice, for example a media-free signature stored on the
  job. A swap does change the translation's own `ScenarioVersion` structure
  fingerprint, as any edit of those fields does today. That is expected.
- Under each translatable field, the original wording is shown read-only in a
  small grey box. The source row is found through `lineage_key` or the job's
  activity map.
- Older scenarios with `variant_type='translation'` made by hand are not
  affected.

## Testing

- Use TDD (`superpowers:test-driven-development`), with Ollama mocked.
- Cover at least:
  - the structural copy is identical, including `AnswerFeedback`;
  - a failed or unreachable request never leaves source text marked done;
  - validation rejects changed tags, `src`s or numbers;
  - only one job runs at a time;
  - an outage puts the job back to waiting with progress kept;
  - the new scenario is private;
  - structure-changing views refuse machine translations;
  - only admins can assign reviewers;
  - the creator and admins can remove reviewers;
  - the confirmed `TranslationMatch` is created with a full activity map;
  - the next-page message appears once.
- Test running instructions: see the project's settings_test / Docker notes.

## Left to the builder

The exact prompt (after checking the NTUA model's expected format), segment
versus whole-HTML strategy, retry counts and timeouts, migration names, URL
names, modal implementation (the platform's existing modal and Bootstrap
patterns), how the badge refreshes (simple polling while any job is waiting or
translating is enough), and file layout (for example a new
`authoringtool/machine_translation.py`).
