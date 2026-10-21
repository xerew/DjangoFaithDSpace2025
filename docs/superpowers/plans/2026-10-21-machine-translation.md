# Machine Translation of Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The scenario card's **Translate** button queues a background machine translation (Ollama `gemma4:31b` on the NTUA machine). The result is a private, structurally identical scenario tagged Machine-translated. An admin-assigned reviewer can correct its text and mark it Translation verified.

**Architecture:** Queueing does three things in one transaction: it copies the whole structure into a new scenario, records an id map, and creates a `ScenarioTranslationJob`. A Celery drain task (one job at a time, cache lock as in the email outbox) then replaces the text unit by unit (scenario header, each phase, each activity with its answers). Each unit is validated before it is saved. Finishing creates a confirmed `TranslationMatch`. Translations are text-only in the editor. Structural views refuse them, and the mixed views save only the text and media fields. A new general `ScenarioCollaborator` model gives reviewers edit access and is the groundwork for co-authors.

**Tech Stack:** Django (venv: 5.2 locally; Docker image: 4.0), Celery + Redis, Ollama HTTP API (`/api/chat`, `/api/tags`), Bootstrap 5 modals, `requests`.

**Spec:** `docs/superpowers/specs/2026-10-21-machine-translation-design.md` (approved 2026-10-21). Read it before starting any task.

## Global Constraints

- Model setting `LLM_MACHINE_TRANSLATION_MODEL`, read from the environment, default `gemma4:31b`. Leave `LLM_TRANSLATION_MODEL` alone.
- Never use `translate_text` or `ollama_text` from `authoringtool/tasks.py`. Every Ollama request sets a timeout.
- Untranslated source text must never be left in a job marked `done`. A failure is visible (`failed` or `waiting`), never silent.
- At most one job is `translating` across the platform. The oldest `waiting` job goes first.
- If Ollama can't be reached, the job goes back to `waiting` with its progress kept. That is never `failed`.
- New translations start `visibility_status='private'`, `variant_type='translation'`, `origin_scenario=<source>`, family via `source.ensure_family()`, and are owned (`created_by`) by whoever pressed Translate.
- A machine translation's structure can never change through the editor, for anyone, admins included. Media fields (simulation, LabsLand, VR/AR, images, video URLs) may change.
- `AnswerFeedback.text` and `EvQuestionBranching.*_feedback` are copied unchanged and never sent to the model.
- Only admins (`is_admin_user`) assign reviewers. Only users in the `teachers` group or admins can be assigned (checked on the server). The creator or an admin removes collaborators.
- No emojis anywhere in UI text. Use plain labels or `bi-*` icons.
- Exact UI copy:
  - Banner: `The translation service can't be reached right now. Machine translation is paused. Please try again later.`
  - Tags: `Machine-translated`, `Translation verified`, `Translation failed`, `Waiting in queue`, `Waiting — translation service unavailable`, `Translating… {done} / {total}`
  - Ready message: `Your {language} translation of {name} is ready.`
  - Failed message: `Your {language} translation of {name} failed. Open your scenarios to retry.`
  - Buttons: `Start translation`, `Mark translation verified`, `Assign reviewer`, `Retry`
- Tests: from `Trust-AI-Platform/`, run `../djangofaithvenv/Scripts/python.exe manage.py test --settings=faithDev.settings_test --noinput <label>`. `settings_test` uses a dummy cache and eager Celery. Tests that need the cache use `@override_settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}})`. Tests that queue a job patch `authoringtool.tasks.drain_translation_queue_task.delay`. The known baseline failures are listed in memory `reference_running_tests` (6 on 2026-10-19). Compare by test name.

## Review Focus

1. **Fields with no translatable words** (empty `helper`, activity text that is only an `<img>`, or an answer that is just `"42"`) must not reach the model, and must not fail validation. They are copied as they are. Test: `test_unit_skips_fields_without_words` (Task 3), `test_drain_completes_scenario_with_image_only_activity` (Task 7).
2. **A request that times out while Ollama is up** (one huge activity) must not loop in `waiting` forever. After a timeout, re-check `/api/tags`: if it answers, count the timeout as a failed attempt for that unit. Test: `test_timeout_with_service_up_counts_as_unit_failure` (Task 7).
3. **The target scenario deleted mid-job** (the creator deletes the "Waiting in queue" card): the drain must finish its run cleanly and move on to the next job. Test: `test_drain_skips_job_whose_target_was_deleted` (Task 7).
4. **The same scenario translated twice into the same language by the same user** must not hit the (`created_by`, `name`) unique constraint. The second gets `" (2)"`. Test: `test_second_copy_gets_numbered_name` (Task 2).
5. **A source scenario with blank `language`**: the modal must ask for the source language, and the POST must refuse when source equals target. Test: `test_start_requires_source_language_when_scenario_has_none` (Task 8).

---

## File Structure

New package `authoringtool/machine_translation/`:

| File | Responsibility |
|---|---|
| `__init__.py` | empty |
| `copying.py` | `copy_for_translation()`: structural copy, id map, job creation |
| `units.py` | build units from target rows, apply translated units, `strip_html` |
| `validation.py` | `validate_unit()`: tags, attributes, numbers, formulas |
| `client.py` | Ollama `/api/chat` call with JSON schema, errors |
| `health.py` | service check (Celery side) and cached status read (web side) |
| `queue.py` | `drain_translation_queue()`, finalize, retry, stuck-job release |
| `editing.py` | text-only saves for `updateActivityData` and `updateAnswers`, guard helpers, originals lookup |
| `collaborators.py` | assign, remove, verify rules and reviewer email |

Modified: `authoringtool/models.py`, a new migration `0074_machine_translation.py`, `authoringtool/tasks.py` (three Celery tasks), `faithDev/settings.py` (settings, beat entries, middleware), `authoringtool/translation_matching.py` (structure signature, scan rule), `authoringtool/views.py` (new views, guards, permission helpers, list view), `authoringtool/urls.py`, a new `authoringtool/middleware.py`, and the templates `scenarios.html`, `viewScenario.html`, `viewPhase.html`, `viewActivity.html`, `updateActivity.html`, `updateAnswers.html`, `updatePhase.html` and `updateScenario.html`.

Tests: one file per task, `authoringtool/test_mt_<area>.py`.

---

### Task 1: Settings, models and migration

**Files:**
- Modify: `faithDev/settings.py` (after `LLM_TRANSLATION_MODEL`, line ~229)
- Modify: `authoringtool/models.py` (new models after `TranslationMatch`; new fields on `TranslationMatch`)
- Create: `authoringtool/migrations/0074_machine_translation.py` (via `makemigrations`)
- Test: `authoringtool/test_mt_models.py`

**Interfaces:**
- Produces:
  - `settings.LLM_MACHINE_TRANSLATION_MODEL` (str, default `'gemma4:31b'`)
  - `settings.MACHINE_TRANSLATION_TIMEOUT` (int seconds, env, default `600`)
  - `ScenarioTranslationJob` with fields:
    - `source_scenario` FK Scenario `SET_NULL` null, related `translation_jobs_as_source`
    - `target_scenario` OneToOne Scenario `CASCADE`, related `translation_job`
    - `source_language`, `target_language` Char(100)
    - `requested_by` FK User `SET_NULL` null, related `requested_translation_jobs`
    - `status` Char(20), choices `waiting|translating|done|failed`, default `waiting`, indexed
    - `id_map` JSON, default `dict`: `{"phases": {new_id_str: old_id}, "activities": {...}, "answers": {...}}`
    - `completed_units` JSON, default `list`
    - `units_total` PositiveInt, default 0
    - `error` Text blank
    - `attempts` PositiveInt, default 0
    - `claimed_at`, `started_at`, `finished_at`, `notified_at` DateTime null
    - `model` Char(255) blank
    - `verified_by` FK User `SET_NULL` null, related `verified_translations`
    - `verified_at` DateTime null
    - `created_at` auto_now_add
    - Index on (`requested_by`, `notified_at`). Constants `STATUS_WAITING`, `STATUS_TRANSLATING`, `STATUS_DONE`, `STATUS_FAILED`, `PENDING_STATUSES = (waiting, translating)`. Property `units_done -> int` (`len(completed_units)`).
  - `ScenarioCollaborator`:
    - `scenario` FK `CASCADE`, related `collaborators`
    - `user` FK `CASCADE`, related `scenario_collaborations`
    - `role` Char(32), choices `translation_reviewer` ("Translation reviewer"), `co_author` ("Co-author")
    - `added_by` FK User `SET_NULL` null
    - `added_at` auto_now_add
    - Unique (`scenario`, `user`), name `unique_scenario_collaborator`
  - `TranslationMatch.origin` Char(32), choices `scan|machine_translation`, default `scan`
  - `TranslationMatch.structure_signature_a` / `structure_signature_b` Char(64) blank

- [ ] **Step 1: Write the failing tests** in `authoringtool/test_mt_models.py`:
  - `test_job_defaults`: a new job has `status == 'waiting'`, `units_done == 0`, `id_map == {}`.
  - `test_collaborator_unique_per_scenario`: a second `ScenarioCollaborator` for the same (scenario, user) raises `IntegrityError`.
  - `test_translation_match_origin_defaults_to_scan`.
  - `test_machine_translation_model_setting_default`: `settings.LLM_MACHINE_TRANSLATION_MODEL == 'gemma4:31b'` (when the env var is unset).
- [ ] **Step 2: Run them and confirm they fail** with ImportError or AttributeError: `... test authoringtool.test_mt_models`
- [ ] **Step 3: Add the settings and models**, then `../djangofaithvenv/Scripts/python.exe manage.py makemigrations authoringtool -n machine_translation --settings=faithDev.settings_test`
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git add faithDev/settings.py authoringtool/models.py authoringtool/migrations/0074_machine_translation.py authoringtool/test_mt_models.py`, then `git commit -m "Add translation job and collaborator models"`

---

### Task 2: Structural copy

**Files:**
- Create: `authoringtool/machine_translation/__init__.py`, `authoringtool/machine_translation/copying.py`
- Test: `authoringtool/test_mt_copying.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces: `copy_for_translation(source: Scenario, user: User, target_language: str, source_language: str) -> ScenarioTranslationJob`. It runs inside `transaction.atomic()` and returns the saved job (`status='waiting'`, `units_total` set, `id_map` filled).

Rules (reference: `views.duplicate_scenario`, `views.py:2649-2796`):
- **New scenario.** Name is `f"{source.name} ({target_language})"`. If the user already has a scenario with that name, append `" (2)"`, `" (3)"` and so on. Language is `target_language`. Copy `learning_goals`, `description`, `age_of_students`, `subject_domains`, `suggested_learning_time`, `image` and `subjects`. Set `visibility_status='private'`, `variant_type='translation'`, `origin_scenario=source`, `family=source.ensure_family()`, and `created_by` and `updated_by` to `user`.
- **Rows to copy.** Copy phases (with `image` and `order`). Copy activities with every structural and media field, `lineage_key` and `concept`. Set `correct_count=incorrect_count=0` and leave the four `llm_*` fields empty. Copy answers (with `image` and `vid_url`) and **`AnswerFeedback` rows** (text, image and `vid_url` unchanged). Copy `NextQuestionLogic`, `EvQuestionBranching` (feedback texts unchanged), `QuestionBunch` (ids remapped) and `start_activity`.
- **Text.** Rows keep the **source text** for now. Units replace it later.
- **Job.** `id_map` maps each new phase, activity and answer id (as a string) to the source id. `units_total = 1 + phases + activities`. Do not call `ensure_current_version` here (Task 7 does).

- [ ] **Step 1: Write the failing tests**, building a source fixture with 2 phases, 3 activities (one Question with 2 answers, one with an `AnswerFeedback`, one evaluatable with `EvQuestionBranching` and `QuestionBunch`) and an answer route:
  - `test_copy_is_private_translation_owned_by_requester`: visibility `private`, variant `translation`, `origin_scenario == source`, `created_by == user`, language is the target.
  - `test_copy_matches_structure`: `build_flow_profile(copy)['signature'] == build_flow_profile(source)['signature']`. Import it from `authoringtool.translation_matching`.
  - `test_copy_includes_answer_feedback`: the copy's answer has 1 feedback with the same text.
  - `test_copy_maps_every_row`: `id_map['activities']` has 3 entries, each pointing at a source activity with the same `lineage_key`. The same holds for phases and answers.
  - `test_units_total_counts_header_phases_activities`: equals `1 + 2 + 3 == 6`.
  - `test_second_copy_gets_numbered_name`: two copies to Greek by the same user are named `"<name> (Greek)"` and `"<name> (Greek) (2)"`.
  - `test_copy_does_not_carry_counts_or_llm_context`: `correct_count == 0` and `llm_context` empty.
- [ ] **Step 2: Run them and confirm they fail**: `... test authoringtool.test_mt_copying`
- [ ] **Step 3: Implement `copy_for_translation`**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Copy a scenario's structure for machine translation"`

---

### Task 3: Units and validation

**Files:**
- Create: `authoringtool/machine_translation/units.py`, `authoringtool/machine_translation/validation.py`
- Test: `authoringtool/test_mt_units.py`

**Interfaces:**
- Consumes: Task 2 job and target rows.
- Produces:
  - `Unit` = dataclass `(key: str, label: str, fields: dict[str, str])`. Keys are `'scenario'`, `'phase:<id>'` and `'activity:<id>'`, with ids of **target** rows. Field keys:
    - scenario: `name`, `description`, `learning_goals`
    - phase: `name`, `description`
    - activity: `name`, `text`, `helper`, plus `answer:<answer_id>` per answer
    - `label` is a readable name, e.g. `Activity "Measure the current"`.
  - `pending_units(job) -> list[Unit]`: units whose key isn't in `job.completed_units`, in the order scenario, phases by `order`, activities by phase order then `order`. Text is read from the **target** rows. Fields with no letters (`not re.search(r'[^\W\d_]', strip_html(value))`) are left out of `fields`.
  - `apply_unit(job, unit: Unit, translated: dict[str, str]) -> None`: writes the translated fields to the target rows. For activities it also sets `plain_text = strip_html(text)`. It then appends `unit.key` to `job.completed_units`, all in one atomic block.
  - `strip_html(html: str) -> str`: `html.unescape(django.utils.html.strip_tags(html))`. This is the same as `views.strip_html_tags`; don't import views.
  - `validate_unit(source: dict[str, str], translated: dict[str, str]) -> list[str]`: human-readable errors, empty when valid. It rejects:
    - a missing key, or an empty value where the source was non-empty;
    - extra keys;
    - a different sequence of `(tag, sorted attrs)` from `html.parser.HTMLParser` (start tags and end tags);
    - a different multiset of digit runs (`re.findall(r'\d+', ...)`), so `0.5` and `0,5` count as equal;
    - different formula segments matched by `\$\$.+?\$\$|\$.+?\$|\\\(.+?\\\)|\\\[.+?\\\]` (DOTALL), compared as ordered lists.

- [ ] **Step 1: Write the failing tests**:
  - `test_pending_units_order_and_keys`: the first unit is `scenario`, then phases, then activities. An activity unit has its `answer:<id>` fields.
  - `test_unit_skips_fields_without_words`: `helper=''` and an answer `"42"` are left out. An activity whose text is `<p><img src="/media/a.png"></p>` has no `text` field.
  - `test_apply_unit_sets_plain_text_and_marks_done`
  - `test_completed_units_are_not_pending`
  - `test_validate_accepts_faithful_translation`: `<p>Current is <b>2 A</b></p>` → `<p>Το ρεύμα είναι <b>2 A</b></p>` returns `[]`.
  - `test_validate_rejects_changed_tag`: `<b>` becomes `<strong>`.
  - `test_validate_rejects_changed_src`
  - `test_validate_rejects_lost_number`
  - `test_validate_accepts_decimal_comma`
  - `test_validate_rejects_changed_formula`: `$V=IR$` becomes `$V=RI$`.
  - `test_validate_rejects_missing_and_extra_keys`
- [ ] **Step 2: Run them and confirm they fail**: `... test authoringtool.test_mt_units`
- [ ] **Step 3: Implement `units.py` and `validation.py`**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Build, validate and apply translation units"`

---

### Task 4: Ollama client

**Files:**
- Create: `authoringtool/machine_translation/client.py`
- Test: `authoringtool/test_mt_client.py`

**Interfaces:**
- Produces:
  - `class TranslationServiceUnavailable(Exception)`: raised on a connection error, a 5xx, a 404 "model not found" envelope, or an `error` that mentions the model.
  - `class TranslationTimeout(Exception)`: raised on `requests.Timeout`.
  - `class TranslationOutputInvalid(Exception)`: raised on non-JSON content, or JSON that isn't an object of strings.
  - `translate_fields(fields: dict[str, str], source_language: str, target_language: str, context: str) -> dict[str, str]`

`translate_fields` POSTs to `f"{settings.OLLAMA_URL}/api/chat"` with timeout `settings.MACHINE_TRANSLATION_TIMEOUT` and this body:
- `model`: `settings.LLM_MACHINE_TRANSLATION_MODEL`
- `stream`: False
- `options`: `{"temperature": 0.2}`
- `format`: a JSON schema of an object whose required string properties are exactly `fields`' keys, with `additionalProperties: false`
- `messages`: a system message and a user message (below)

The system message states these rules:
- translate from `{source_language}` into `{target_language}`;
- translate only human-readable text;
- keep every HTML tag and attribute, every number, unit and formula (`$...$`, `\(...\)`) exactly as written;
- keep the same keys;
- return only the JSON object.

The user message holds `context` (scenario name and learning goals) and the JSON of `fields`. The function returns the parsed `message.content`.

- [ ] **Step 1: Write the failing tests** (patch `authoringtool.machine_translation.client.requests.post`):
  - `test_sends_model_schema_and_timeout`: the payload's `model == settings.LLM_MACHINE_TRANSLATION_MODEL`, `format['required']` is sorted equal to the fields' keys, and `timeout=settings.MACHINE_TRANSLATION_TIMEOUT`.
  - `test_returns_parsed_fields`
  - `test_connection_error_raises_unavailable`
  - `test_timeout_raises_translation_timeout`
  - `test_non_json_raises_output_invalid`
  - `test_model_missing_raises_unavailable`: the response is 404 with `{"error": "model 'gemma4:31b' not found"}`.
- [ ] **Step 2: Run them and confirm they fail**: `... test authoringtool.test_mt_client`
- [ ] **Step 3: Implement `client.py`**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Add Ollama client for machine translation"`

---

### Task 5: Service health check

**Files:**
- Create: `authoringtool/machine_translation/health.py`
- Modify: `authoringtool/tasks.py` (new task), `faithDev/settings.py` (beat entry)
- Test: `authoringtool/test_mt_health.py`

**Interfaces:**
- Produces:
  - `STATUS_KEY = 'authoringtool:translation-service-status'`
  - `check_translation_service() -> dict`: Celery side. Calls `GET {OLLAMA_URL}/api/tags` with timeout 3. `model_present` is true if `settings.LLM_MACHINE_TRANSLATION_MODEL` is among `models[].name`. Calls `record_service_status` and returns the dict.
  - `record_service_status(reachable: bool, model_present: bool) -> None`: writes `{reachable, model_present, checked_at: iso}` to the cache with timeout 600.
  - `translation_service_available(now=None) -> bool`: web side. True only if the cache entry exists, `checked_at` is less than 180 seconds old, and `reachable and model_present`.
  - Celery task `check_translation_service_task()` in `authoringtool/tasks.py`. Beat entry `'check-translation-service'`: schedule `60.0`, `options={'expires': 50}`.

- [ ] **Step 1: Write the failing tests** (locmem cache override, `requests.get` patched):
  - `test_available_when_model_listed`
  - `test_unavailable_when_model_missing`
  - `test_unavailable_on_connection_error`
  - `test_unavailable_when_status_missing`
  - `test_unavailable_when_status_stale`: `now = checked_at + 181 s`.
  - `test_beat_entry_registered`: `settings.CELERY_BEAT_SCHEDULE['check-translation-service']['task'] == 'authoringtool.tasks.check_translation_service_task'`.
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Check translation service health from Celery"`

---

### Task 6: Structure signature and the scan rule for machine-translation pairs

**Files:**
- Modify: `authoringtool/translation_matching.py` (`build_flow_profile` ~line 257; `scan_translation_matches` ~lines 875-972)
- Test: `authoringtool/test_mt_scan.py`

**Interfaces:**
- Produces: `build_flow_profile(scenario)` also returns `'structure_signature'`. It is the same sha256 as `signature`, but built from tokens whose `simulation`, `lab` and `vr` fields are blanked.

Scan rule for `match.origin == 'machine_translation'`:
- Never overwrite `activity_map`, `status` or the scores from candidate results.
- On every scan, whether or not the pair is a candidate, build both profiles and compare their `structure_signature`s with the stored `structure_signature_a/b`. If either differs and the status is `confirmed`, set `changed`. Otherwise leave the match untouched.
- Never delete these matches.
- Pairs with `origin='scan'` keep today's behaviour exactly.

- [ ] **Step 1: Write the failing tests**, creating an MT pair from `copy_for_translation` plus a hand-made confirmed `TranslationMatch(origin='machine_translation', structure_signature_a/b=...)`:
  - `test_structure_signature_ignores_simulation`: swapping an activity's simulation changes `signature` but not `structure_signature`.
  - `test_scan_keeps_mt_pair_confirmed_when_not_a_candidate`: patch `find_translation_candidates` to return `[]`. The status stays `confirmed` and `activity_map` is unchanged.
  - `test_scan_keeps_mt_pair_after_media_swap`
  - `test_scan_marks_mt_pair_changed_after_structure_change`: add an answer to the source.
  - `test_scan_behaviour_unchanged_for_scan_pairs`: an `origin='scan'` confirmed pair that is no longer a candidate becomes `changed`, as today.
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the new tests and `authoringtool.test_translation_matching`, and confirm they pass**. The known baseline failures may remain.
- [ ] **Step 5: Commit**: `git commit -m "Keep machine-translation pairs unless their structure changes"`

---

### Task 7: Queue drain and finishing

**Files:**
- Create: `authoringtool/machine_translation/queue.py`
- Modify: `authoringtool/tasks.py` (`drain_translation_queue_task`), `faithDev/settings.py` (beat entry `'drain-translation-queue'`: schedule `60.0`, `options={'expires': 50}`)
- Test: `authoringtool/test_mt_queue.py`

**Interfaces:**
- Consumes:
  - Task 3: `pending_units`, `apply_unit`, `validate_unit`
  - Task 4: `translate_fields` and its exceptions
  - Task 5: `check_translation_service`, `record_service_status`
  - Task 6: `build_flow_profile(...)['structure_signature']`
- Produces:
  - `LOCK_KEY = 'authoringtool:drain-translation-queue'`, `LOCK_TIMEOUT = 4 * 60 * 60`, `STUCK_AFTER = timedelta(hours=3)`, `MAX_UNIT_ATTEMPTS = 3`
  - `drain_translation_queue(now=None) -> dict` (counts: `done`, `failed`, `requeued`)
  - `retry_job(job) -> None`: `failed` → `waiting`, clears `error`, keeps `completed_units`
  - `finish_job(job) -> None`
  - Celery task `drain_translation_queue_task()`

How the drain runs:
1. **Lock.** Take the lock with `cache.add(LOCK_KEY, '1', LOCK_TIMEOUT)`. If it's held, return. Always release it in `finally`.
2. **Release stuck jobs.** Move `translating` jobs whose `claimed_at` is older than `STUCK_AFTER` back to `waiting`.
3. **One job at a time.** If any job is still `translating`, return.
4. **Claim.** Loop: claim the oldest `waiting` job by `created_at` with a filtered `update(status='translating', claimed_at=now, attempts=F+1)`, and set `started_at` if it's empty. Wrap the job's handling in `ScenarioTranslationJob.DoesNotExist` / `Scenario.DoesNotExist` handling. If the job or target was deleted, skip it.
5. **For each pending unit**, try up to `MAX_UNIT_ATTEMPTS` times:
   - On `TranslationServiceUnavailable`: `record_service_status(False, ...)`, set the job back to `waiting`, and **stop the whole run**.
   - On `TranslationTimeout`: call `check_translation_service()`. If it's unreachable, treat it like Unavailable. If it's reachable, count it as one failed attempt.
   - On `TranslationOutputInvalid` or a non-empty `validate_unit` result: count it as one failed attempt.
   - After 3 failed attempts: the job becomes `failed`, with `error = f'{unit.label}: {"; ".join(errors)}'` and `finished_at` set. Move on to the next job.
   - On success: `apply_unit`.
   - The `context` passed to the client is the target scenario's (already translated) name and learning goals once the header is done. Before that, it's the source text.
6. **When no units are pending**, call `finish_job`:
   - `target.ensure_current_version(created_by=job.requested_by, change_summary='Machine translation')`
   - Create a `TranslationMatch` with:
     - `origin='machine_translation'`, `status='confirmed'`, `exact_match=True`, `confidence=100`, `structure_score=route_score=anchor_score=1.0`
     - `scenario_a`/`scenario_b` ordered by id
     - `activity_map` = `[[a_activity_id, b_activity_id], ...]` from `id_map['activities']`, oriented to the a/b order
     - `flow_signature_*` and `structure_signature_*` from `build_flow_profile` of each side
   - Skip creating it if the source is gone.
   - Set `status='done'`, `finished_at`, and `model=settings.LLM_MACHINE_TRANSLATION_MODEL`.

- [ ] **Step 1: Write the failing tests** (patch `queue.translate_fields`; locmem cache):
  - `test_drain_translates_all_units_and_finishes`: status is `done`, every target activity's text is the fake translation, and `current_version` is set.
  - `test_finish_creates_confirmed_mt_match_with_full_map`: the `activity_map` length equals the activity count, and the origin is `machine_translation`.
  - `test_unavailable_requeues_and_keeps_progress`: the fake raises Unavailable on the 3rd unit. Status is `waiting` and `units_done == 2`. A second drain with a working fake finishes it.
  - `test_invalid_output_three_times_fails_with_label`: status `failed`, and `error` starts with `Activity "`.
  - `test_failed_job_never_marked_done_with_source_text`
  - `test_retry_job_resumes_from_failed_unit`: the fake receives only the units that weren't done.
  - `test_only_one_job_translating`: with a job already `translating` (recent `claimed_at`), the drain translates nothing.
  - `test_stuck_job_released`: `claimed_at` is 4 h old, so the job runs again.
  - `test_jobs_run_oldest_first`
  - `test_timeout_with_service_up_counts_as_unit_failure`: the fake always raises Timeout and `check_translation_service` is patched to reachable, so the job ends `failed`.
  - `test_drain_skips_job_whose_target_was_deleted`: delete job 1's target before draining. Job 2 still finishes.
  - `test_drain_completes_scenario_with_image_only_activity`
  - `test_beat_entry_registered`
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Drain the machine-translation queue one job at a time"`

---

### Task 8: Starting a translation (modal and views)

**Files:**
- Modify: `authoringtool/views.py`, `authoringtool/urls.py`, `authoringtool/templates/authoringtool/scenarios.html` (Translate button at lines 277-279, plus a modal per card or one shared modal)
- Test: `authoringtool/test_mt_start.py`

**Interfaces:**
- Consumes:
  - Task 2: `copy_for_translation`
  - Task 5: `translation_service_available`
  - Task 7: `drain_translation_queue_task`
- Produces (urls):
  - `machine_translation_status` → GET `translations/service-status/` returns JSON `{"available": bool}`
  - `start_machine_translation` → POST `scenarios/<int:scenario_id>/translate/` with `target_language` and an optional `source_language`

View rules for `start_machine_translation`:
- `@require_POST`, `@group_required('teachers')`.
- Refuse unless `user_can_view_scenario`.
- The source scenario must not have a pending job of its own.
- Source language is `scenario.language`, or `source_language` from the POST if that is blank. It must be a `Language` name.
- `target_language` must be a `Language` name different from the source.
- `translation_service_available()` must be true, otherwise show the banner copy as an error message.
- On success: `copy_for_translation(...)`, then `transaction.on_commit(drain_translation_queue_task.delay)`, then a success message `Translation queued. You can leave this page.`, then redirect to `scenarios`.

**Modal:**
- Opened by the Translate button, which becomes a `<button>` with the same `bi-translate` icon and label.
- When it opens, JS fetches `machine_translation_status`. If the service is unavailable, it shows the banner copy (`alert alert-warning`) and disables the select and **Start translation**.
- The select lists `Language.objects` minus the scenario's language.
- If the scenario's `language` is blank, a second select asks for the source language.
- Leave `duplicate_scenario` unchanged. Only the button changes.

- [ ] **Step 1: Write the failing tests** (patch `drain_translation_queue_task.delay` and `translation_service_available`):
  - `test_teacher_can_start_translation`: one job exists, the target is private, and the response redirects to `scenarios`.
  - `test_refused_when_service_unavailable`: no job, and the message contains the banner copy.
  - `test_refused_when_target_equals_source`
  - `test_refused_for_unknown_language`
  - `test_refused_when_user_cannot_view_scenario`
  - `test_start_requires_source_language_when_scenario_has_none`
  - `test_status_endpoint_reports_availability`
  - `test_scenarios_page_has_translate_modal_without_emojis`: the response contains `Start translation`, and no character in it is in the emoji ranges `U+1F300–U+1FAFF` or `U+2600–U+27BF`.
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement the views, URLs and modal**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Start machine translation from the Translate modal"`

---

### Task 9: Text-only enforcement on the server

**Files:**
- Create: `authoringtool/machine_translation/editing.py`
- Modify: `authoringtool/views.py` (add the guard calls next to the existing `authoring_revision_guard` calls)
- Test: `authoringtool/test_mt_editing.py`

**Interfaces:**
- Consumes: Task 1 job.
- Produces:
  - `machine_translation_job(scenario) -> ScenarioTranslationJob | None` (no exception if none)
  - `translation_pending_guard(request, scenario) -> HttpResponse | None`: if a job exists and isn't `done`, add the message `This translation is still being prepared.` and redirect to `scenarios`.
  - `translation_structure_guard(request, scenario) -> HttpResponse | None`: if a job exists, add the message `Machine translations keep the original structure. Only text and media can be changed.` and redirect to `viewScenario`. Also returns the pending redirect when relevant.
  - `save_translation_activity_text(request, activity) -> None`: saves `activity_name`, `activity_text` (plus `plain_text`) and `helping_quote`. If the activity's type is `Experiment`, it also saves the experiment selection (`experiment_type` with `simulation`, `remote_lab` or `VRAR_lab`, as in `updateActivityData` lines 1637-1649). Everything else in the POST is ignored.
  - `save_translation_answer_text(request, activity) -> None`: for each posted `answers[i][id]` that belongs to the activity, saves `text`, `vid_url`, image upload and `clear_image_i`. It ignores `is_correct`, `answer_weight` and `next_activity_*`, entries without an id, and never deletes anything.

Wiring in `views.py`:
- `translation_pending_guard` goes at the top of `viewScenario`, `updateScenario`, `updateScenarioData`, `updatePhase`, `updatePhaseData`, `viewPhase`, `updateActivity`, `viewActivity` and `updateAnswers`.
- `translation_structure_guard` goes in `createPhase`, `createPhaseData`, `deletePhase`, `createActivity`, `createActivityData`, `deleteActivity`, `move_phase`, `move_activity`, `set_start_activity`, `createAnswers`, `deleteAnswer`, `createCriterion`, `createCriterionData`, `updateCriterion`, `updateCriterionData`, `accept_proposal` and the revision begin/publish views.
- In `updateActivityData`, and in `updateAnswers` on POST: if `machine_translation_job(scenario)` is set, call the text-only save, then `record_edit` and `refresh_version_if_initialized`, and redirect as the normal path does (`phase` and `activity` respectively).

- [ ] **Step 1: Write the failing tests** on a finished MT scenario (job `done`):
  - `test_delete_activity_refused`: the activity still exists. The same check applies to `deletePhase`, `createActivityData`, `move_activity`, `set_start_activity`, `deleteAnswer` and `createCriterionData`. Use a `subTest` loop over the URL names.
  - `test_admin_also_refused_structural_change`
  - `test_activity_update_saves_text_only`: the POST changes the name, text, `is_evaluatable`, `activity_type` and `next_activity_id`. Only the name, text and `plain_text` change, and the flow signature is unchanged.
  - `test_activity_update_allows_simulation_swap`
  - `test_answers_update_saves_text_only`: the POST flips `is_correct`, omits one answer and adds a new one without an id. Both answers keep `is_correct`, none is deleted or created, and the text is updated.
  - `test_pending_translation_cannot_be_opened`: job `waiting`, so `viewScenario` redirects to `scenarios`.
  - `test_normal_scenarios_unaffected`: `deleteActivity` on a plain scenario still works.
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests, plus `authoringtool.tests` and `authoringtool.test_authoring_improvements`, to confirm nothing else broke**
- [ ] **Step 5: Commit**: `git commit -m "Keep machine translations text-only on the server"`

---

### Task 10: Card tags, progress, the next-page message and list bundling

**Files:**
- Create: `authoringtool/middleware.py`
- Modify: `faithDev/settings.py` (`MIDDLEWARE`: add `'authoringtool.middleware.MachineTranslationNoticeMiddleware'` after `accounts.middleware.LoginRequiredMiddleware`), `authoringtool/views.py` (`scenarios` view lines 605-759, new status view, retry view), `authoringtool/urls.py`, `scenarios.html`, `viewScenario.html`
- Test: `authoringtool/test_mt_listing.py`

**Interfaces:**
- Consumes:
  - Task 5: `translation_service_available`
  - Task 7: `retry_job`, `drain_translation_queue_task`
- Produces (urls):
  - `machine_translation_job_status` → GET `translations/jobs/status/?ids=1,2` returns `{"jobs": {"<id>": {"status", "units_done", "units_total"}}}`, only for jobs whose target the user can view
  - `retry_machine_translation` → POST `translations/jobs/<int:job_id>/retry/`, for the requester or an admin; it calls `retry_job` and then the drain on commit

Requirements:
- **`scenarios` view.** Attach `scenario.translation_job` to page items (`select_related`, one query). The same goes for each version in `translation_versions`.
- **Bundling.** Machine-translation targets that are still `private` are left out of `hidden_ids`, so they stay their own card for whoever can see them. Once they're published, they bundle as usual.
- **Card tags** (plain text pills with an optional `bi-*` icon):
  - `waiting` → `Waiting in queue`, or `Waiting — translation service unavailable` when `translation_service_available()` is false
  - `translating` → `Translating… {done} / {total}`
  - `done` + not verified → `Machine-translated`
  - verified → `Translation verified` plus `Verified by {username} on {date}`
  - `failed` → `Translation failed`, with **Retry** and the existing Delete
- **Card buttons.** While a job is pending, hide View, Edit, Adapt and Translate.
- **Polling.** The same tags appear in the Translations modal rows and on the `viewScenario` header. A small script polls `machine_translation_job_status` every 15 seconds while any pending job is on the page, updates the badges, and reloads the page when one turns `done` or `failed`.
- **Middleware.** For authenticated users on non-AJAX GET requests, it runs one query: jobs with `requested_by=user`, `status in (done, failed)` and `notified_at=None`. For each, it adds `messages.success` (ready copy, with the target's name linked to `viewScenario`) or `messages.error` (failed copy), then bulk-updates `notified_at`.

- [ ] **Step 1: Write the failing tests**:
  - `test_card_shows_progress_badge`: contains `Translating… 2 / 6`.
  - `test_card_shows_machine_translated_tag`
  - `test_card_shows_verified_tag_with_name`
  - `test_failed_card_has_retry`
  - `test_pending_card_hides_view_and_edit`
  - `test_private_translation_not_bundled_into_original_card`
  - `test_published_translation_bundles`
  - `test_ready_message_shown_once`: the first GET shows `is ready.` and the second doesn't.
  - `test_failed_message_shown`
  - `test_job_status_endpoint_hides_other_users_jobs`
  - `test_retry_requeues_failed_job`
  - `test_waiting_tag_says_service_unavailable_when_down`
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests, plus `authoringtool.test_translation_lists`, and confirm they pass** (apart from the baseline)
- [ ] **Step 5: Commit**: `git commit -m "Show translation progress, tags and ready messages"`

---

### Task 11: Text-only editor templates with the original wording

**Files:**
- Modify: `authoringtool/machine_translation/editing.py` (originals lookup), `authoringtool/views.py` (context in `viewScenario`, `viewPhase`, `viewActivity`, `updateActivity`, `updateAnswers`, `updatePhase`, `updateScenario`), and the templates `viewScenario.html`, `viewPhase.html`, `viewActivity.html`, `updateActivity.html`, `updateAnswers.html`, `updatePhase.html`, `updateScenario.html`
- Test: `authoringtool/test_mt_templates.py`

**Interfaces:**
- Consumes: Task 9 `machine_translation_job`.
- Produces: `original_of(job, obj) -> model instance | None`. It handles Scenario (`job.source_scenario`), Phase, Activity and Answer via `job.id_map`, and returns None if the source row is gone.
- Context keys: `is_machine_translation` (bool) and `original` (the object for the page). On `updateAnswers`, each answer gets `answer.original_text`.

Template rules when `is_machine_translation`:
- **Hidden:** add/delete phase, add/delete activity, move up/down, "make first", add/delete answer, criterion links, and the inputs for activity type, `is_evaluatable`, `is_primary_ev`, next activity, bunch selection, `is_correct` and `answer_weight`.
- **Kept:** simulation, lab and VR pickers, image and video inputs.
- **Original wording:** under each translatable input, a read-only `<div class="text-muted small border rounded p-2 mt-1">Original: …</div>` shows the original's value. Render HTML fields with `|safe`, the same way the page already renders activity text.

- [ ] **Step 1: Write the failing tests**:
  - `test_view_scenario_hides_structure_controls`: the `createPhase` URL is not in the response for an MT scenario, but is present for a normal one.
  - `test_update_activity_shows_original_text`
  - `test_update_activity_hides_type_and_routing_inputs`: `name="activity_type"` and `name="next_activity_id"` are absent.
  - `test_update_answers_hides_correctness_and_shows_original`
  - `test_original_missing_when_source_deleted`: the page still renders, with no "Original:" box.
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests and confirm they pass**
- [ ] **Step 5: Commit**: `git commit -m "Text-only editor for machine translations with original wording"`

---

### Task 12: Reviewer, collaborators and verification

**Files:**
- Create: `authoringtool/machine_translation/collaborators.py`
- Modify: `authoringtool/views.py` (`user_can_view_scenario` line 96, `user_can_edit_scenario` line 185, `scenarios_visible_to_user` line 305, the `visible_scenarios` query in `scenarios`, and new views), `authoringtool/urls.py`, `scenarios.html` (Assign reviewer modal, admins only), `viewScenario.html` (reviewer list with Remove, and the **Mark translation verified** button)
- Test: `authoringtool/test_mt_reviewers.py`

**Interfaces:**
- Consumes: Task 1 models, Task 9 `machine_translation_job`.
- Produces:
  - `can_be_reviewer(user) -> bool`: true if `is_admin_user(user)` or the user is in the `teachers` group.
  - `assign_reviewer(scenario, reviewer, added_by) -> ScenarioCollaborator`: raises `PermissionDenied` unless `added_by` is an admin and `can_be_reviewer(reviewer)`. It uses `get_or_create` with role `translation_reviewer` and sends the email (below).
  - `remove_collaborator(scenario, user, removed_by) -> None`: allowed for `scenario.created_by` or an admin.
  - `can_verify(user, scenario) -> bool`: the job is `done` and the user is an admin or a collaborator.
  - `verify_translation(scenario, user)` and `unverify_translation(scenario, admin)`: set or clear `job.verified_by` and `verified_at`. Unverify is for admins only.
  - URLs, all POST except the first:
    - `translation_reviewer_candidates`: GET `translations/<int:scenario_id>/reviewer-candidates/?q=`, admins only. Returns JSON `{"users": [{"id", "username", "name", "institution", "suggested"}]}` for teachers and admins matching `q` (username, name or email), up to 20. `suggested` is true when one of the user's `member_of_organizations` has `language` equal to the target language (case-insensitive). Suggested users come first.
    - `assign_translation_reviewer`: `translations/<int:scenario_id>/reviewers/`
    - `remove_scenario_collaborator`: `scenarios/<int:scenario_id>/collaborators/<int:user_id>/remove/`
    - `verify_machine_translation`: `translations/<int:scenario_id>/verify/`
    - `unverify_machine_translation`: `translations/<int:scenario_id>/unverify/`

Permission changes:
- `user_can_view_scenario` and `user_can_edit_scenario` return True for a `ScenarioCollaborator` of a non-archived scenario. `user_manages_scenario` is unchanged.
- `scenarios_visible_to_user` and the non-admin `visible_scenarios` query add `Q(collaborators__user=user)`.

Reviewer email, sent with `send_mail` (not the outbox):
- Subject: `You have been asked to review a translation`
- Body: `{admin} asked you to review the {language} translation "{name}" on Trust AI Lab. Open it here: {SITE_URL}{viewScenario url}`
- An email failure is logged and doesn't undo the assignment.

- [ ] **Step 1: Write the failing tests**:
  - `test_admin_assigns_teacher_reviewer`: a collaborator exists and `mail.outbox` has 1 message with the subject above.
  - `test_non_admin_cannot_assign`
  - `test_student_cannot_be_assigned`: user has no group, so the response is 403 and no collaborator is created.
  - `test_reviewer_can_view_and_edit_private_translation`
  - `test_reviewer_sees_translation_in_list`
  - `test_reviewer_cannot_change_visibility_or_delete`: `user_manages_scenario` is false and `deleteScenario` is 403.
  - `test_creator_and_admin_can_remove_reviewer`, `test_reviewer_cannot_remove_self_or_others`
  - `test_reviewer_marks_verified`: `verified_by` is set and the card shows `Translation verified`.
  - `test_verification_kept_after_text_edit`
  - `test_admin_can_unverify`, `test_non_admin_cannot_unverify`
  - `test_candidates_lists_teachers_suggested_first`: the matching-org-language teacher comes first, and a student isn't listed.
  - `test_reviewer_stays_after_verification`
- [ ] **Step 2: Run them and confirm they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the full suite**: `... manage.py test --settings=faithDev.settings_test --noinput`. Compare failures by name with the baseline in memory `reference_running_tests`.
- [ ] **Step 5: Commit**: `git commit -m "Assign translation reviewers and mark translations verified"`

---

## Deployment notes (for the final summary, not a task)

- On the server: run `migrate`, then recreate `celery` and `celerybeat` (new beat entries). Set `LLM_MACHINE_TRANSLATION_MODEL=gemma4:31b` in `.env` if it differs from the default.
- No compose change is needed. The health check runs in Celery, which already uses `http://ollama-tunnel:11434`.
