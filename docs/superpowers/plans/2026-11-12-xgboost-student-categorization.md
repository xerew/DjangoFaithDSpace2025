# XGBoost Student Categorization (admin only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two XGBoost models (final categorization, early prediction) trained on demand from all finished runs of public scenarios, with an admin-only dashboard of out-of-sample statistics next to PISA.

**Architecture:** The PISA formula's pure parts move into `authoringtool/performance_formula.py`, used by the two existing callers and by a new Django app `categorization`. That app turns runs into a table of scenario-independent inputs (`dataset.py`, `features.py`), trains and evaluates with plain pandas/XGBoost (`training.py`, no Django), stores everything on a `TrainingRun` row (Celery task), and shows it on a custom admin page (`views.py`, `report.py`).

**Tech Stack:** Django 4.0.10, Celery 5.5, pandas 2.3, numpy 2.2, XGBoost 2.1 (CPU wheel), scikit-learn 1.6, ECharts (already vendored), openpyxl.

**Spec:** `docs/superpowers/specs/2026-11-12-xgboost-student-categorization-design.md`

## Global Constraints

- PISA output must not change anywhere: `compute_final_performance` and `compute_student_performance_metrics` give identical results before and after Task 2.
- Thresholds: **High ≥ 83.3**, **Moderate ≥ 49.7**, else **Low**. Phase weights 20/20/45/15 for up to 4 phases, 20/20/30/15/15 for 5 or more.
- Public scenarios only: `Scenario.visibility_status == 'public'` at training time.
- Every dashboard number comes from out-of-fold predictions; a student never appears in both training and test data (`GroupKFold` on `user_id`).
- The early model sees nothing from after its checkpoint.
- Staff or superuser only (`user.is_staff or user.is_superuser`); everyone else who is logged in gets 403.
- No emojis in the UI; plain labels or `bi-*` icons. Template strings wrapped in `{% translate %}` / `gettext`.
- `Trust-AI-Platform/requirements.txt` is UTF-16 LE with CRLF; keep it that way.
- New models live in the new app `categorization` (own migrations from `0001`), never in `authoringtool`, because the translation branch already owns `authoringtool` migrations 0076/0077.
- Fixed seed `42`.
- Test command (from `C:/wt/xgboost/Trust-AI-Platform`):
  `timeout 1800 "C:/Users/Nikos A. Grammatikos/Desktop/DjangoFaithDSpace2025-performance-test/djangofaithvenv/Scripts/python.exe" manage.py test <labels> --settings=faithDev.settings_test --noinput`
  Full-suite baseline: 6 known failures (listed in Task 9); no new ones.

## Review Focus

1. **A training job that dies mid-way** (worker restart) leaves a run `running` forever and the button stays disabled. Expected: a run active for more than 6 hours is marked failed when the next one starts. (Test in Task 1.)
2. **Too little data** (no public scenarios, a handful of finished runs, a single category). Expected: the run ends `failed` with a plain message, not a traceback. (Test in Task 6.)
3. **A cross-validation fold missing a category** (no High students in one fold). The sklearn `XGBClassifier` raises on non-consecutive labels. Expected: three probability columns every time. (Test in Task 5.)
4. **Legacy runs whose activities were deleted** (`UserAnswer.activity` is NULL, or answers point at activities no longer in the scenario). Expected: those answers are ignored; the run falls back to the last-phase rule and does not crash. (Test in Task 3.)
5. **A scenario or language the model never saw** (leave-one-scenario-out, blank language). Expected: prediction still works; categories are fixed from the full table before splitting. (Test in Task 5.)

---

### Task 1: `categorization` app, dependencies and the `TrainingRun` model

**Files:**
- Create: `Trust-AI-Platform/categorization/__init__.py`, `apps.py`, `models.py`, `migrations/0001_initial.py` (via `makemigrations`), `tests/__init__.py`, `tests/test_models.py`
- Modify: `Trust-AI-Platform/faithDev/settings.py` (INSTALLED_APPS; `CATEGORIZATION_MODEL_DIR`)
- Modify: `Trust-AI-Platform/requirements.txt` (UTF-16)
- Modify: `docker-compose.yml` (`CATEGORIZATION_MODEL_DIR: /data/categorization_models` for `web` and `celery`; both already mount `./data:/data`)

**Interfaces:**
- Produces:
  - `TrainingRun` with fields:
    - `started_by` (FK User, SET_NULL, null)
    - `created_at` (auto_now_add), `started_at`, `finished_at` (nullable)
    - `status`: `queued` | `running` | `finished` | `failed`
    - `progress` (CharField 255, blank), `error` (TextField, blank)
    - `versions`, `parameters`, `counts`, `results` (JSONField, default dict)
    - `final_model_path`, `early_model_path` (CharField 500, blank)
    - `Meta.ordering = ['-created_at', '-id']`
  - `TrainingRun.ACTIVE = ('queued', 'running')`, `TrainingRun.STALE_AFTER = timedelta(hours=6)`
  - `TrainingRun.start(user) -> TrainingRun`: inside `transaction.atomic()`, marks active runs older than `STALE_AFTER` as `failed` with error `"Stopped without finishing (the worker may have restarted)."`. Raises `TrainingAlreadyRunning` if another active run remains; otherwise creates a `queued` run.
  - `TrainingRun.set_progress(text)` saves only `progress`.
  - `exception TrainingAlreadyRunning(Exception)`
  - `settings.CATEGORIZATION_MODEL_DIR = os.getenv('CATEGORIZATION_MODEL_DIR', str(BASE_DIR / 'categorization_models'))`. Model files go here, not under MEDIA_ROOT, which nginx serves publicly.

- [ ] **Step 1: Add the dependencies.** Append `scikit-learn==1.6.1` and `xgboost-cpu==2.1.4` to `requirements.txt`, decoding and re-encoding as UTF-16 LE with CRLF. A Python one-off is fine: `open(p, encoding='utf-16')`, then write it back with `encoding='utf-16', newline='\r\n'`. Check `file requirements.txt` still says "UTF-16, little-endian … CRLF". `xgboost-cpu` is the official CPU-only wheel (import name `xgboost`); it avoids the ~200 MB CUDA/NCCL dependency the plain `xgboost` wheel pulls on Linux. Install both into the main checkout's venv with `pip install scikit-learn==1.6.1 xgboost-cpu==2.1.4`. If `xgboost-cpu` has no wheel for the platform, use `xgboost==2.1.4` in both places and say so in the report. This adds untracked files under `djangofaithvenv/`; never commit them.
- [ ] **Step 2: Write the failing tests** in `categorization/tests/test_models.py`:
  - `test_start_creates_queued_run`: `TrainingRun.start(admin)` gives status `queued`, `started_by == admin`.
  - `test_second_start_while_running_raises`: start, set status `running`, then `start` again → `TrainingAlreadyRunning`.
  - `test_stale_running_run_is_failed_and_new_run_starts`: create a run `running` with `created_at` 7 hours ago (use `.update()`). `start` succeeds; the old run is `failed` with the exact error text above.
  - `test_finished_runs_do_not_block`.
- [ ] **Step 3: Run them** (label `categorization`) and confirm they fail: the app does not exist yet.
- [ ] **Step 4: Implement** the app, model and settings. Then run `python manage.py makemigrations categorization --settings=faithDev.settings_test`.
- [ ] **Step 5: Run the tests.** Expected: 4 pass.
- [ ] **Step 6: Commit** ("Add the categorization app and its training-run model").

### Task 2: One shared PISA formula

**Files:**
- Create: `Trust-AI-Platform/authoringtool/performance_formula.py`, `Trust-AI-Platform/authoringtool/test_performance_formula.py`
- Modify: `Trust-AI-Platform/authoringtool/tasks.py`: `compute_final_performance` (~379–535) and `compute_student_performance_metrics` (~1882–2116)

**Interfaces:**
- Produces (all pure, no queries):
  - `HIGH_THRESHOLD = 83.3`, `MODERATE_THRESHOLD = 49.7`, `CATEGORIES = ('Low', 'Moderate', 'High')`
  - `phase_weights(phase_count: int) -> list[float]`: `[0.2, 0.2, 0.45, 0.15]`, or `[0.2, 0.2, 0.3, 0.15, 0.15]` when `phase_count > 4`
  - `categorize(percentage: float) -> str`
  - `scored_activity_ids(primary_ids: list[int], bunch_map: dict[int, list[int]], known_ids: set[int] | None = None) -> list[int]`: for each primary in order, its bunch members (or `[primary]`), first occurrence only. With `known_ids`, ids not in it are dropped (the CSV's existing behaviour).
  - `phase_score(scored_ids: list[int], chosen_weight: dict[int, int], max_weight: dict[int, int]) -> tuple[float, float]` returns `(total, maximum)`. Missing answers add 0; missing max adds 0.
  - `weighted_percentage(phase_scores: list[tuple[float, float]], weights: list[float]) -> float | None`: phases with `maximum == 0` are skipped. A position beyond `len(weights)` has weight 0 (the CSV's behaviour). The result is `sum(total/max × w) / sum(w) × 100`, or `None` when no phase counted.

- [ ] **Step 1: Write the characterisation tests first**, in `test_performance_formula.py`, against the **unchanged** code. Build the fixture like `test_help_requests.HelpRequestBase`:
  - 4 phases; evaluatable primary activities; one `QuestionBunch` with 2 members; answers with weights 0/1/2.
  - 3 students whose expected finals are High, Moderate and Low. Every student answers the start activity.
  
  The tests:
  - `test_chart_counts_unchanged`: `compute_final_performance(s.id, None, None, None)['waterfall_data']` == `[{'name': 'Total', 'value': 3}, {'name': 'High', 'value': 1}, {'name': 'Moderate', 'value': 1}, {'name': 'Low', 'value': 1}]`.
  - `test_csv_categories_unchanged`: parse `csv_content` and check each username's `Final Categorization` and each phase's `Categorization` against hand-computed values. Write them down in the test.
  - `test_csv_sixth_phase_gets_no_weight`: add phases 5 and 6 with scored activities; the final category still matches the 5-weight hand calculation.
  - Pure unit tests:
    - `categorize`: 83.3 → High, 83.29 → Moderate, 49.7 → Moderate, 49.69 → Low.
    - `phase_weights`: 3, 4, 5 and 7 phases.
    - `scored_activity_ids`: de-duplication and the `known_ids` filter.
    - `weighted_percentage`: rescaling when a phase has no scored activity, and `None` when nothing counts.
- [ ] **Step 2: Run the tests.** Characterisation tests PASS on the old code; unit tests FAIL (module missing).
- [ ] **Step 3: Implement `performance_formula.py`** and switch both callers to it.
  - In `compute_final_performance`, keep its own `phase_weights[index]` indexing, so a 6th phase still raises `IndexError` as today. That bug is a spec non-goal. Use `phase_score` and `categorize` only.
  - In `compute_student_performance_metrics`, use `scored_activity_ids(..., known_ids=set(bunch_activity_objs))`, `phase_score`, `weighted_percentage` (a `None` result becomes `0`, as today) and `categorize`.
  - Leave the other `83.3`/`49.7` sites in `tasks.py` alone.
- [ ] **Step 4: Run** `authoringtool.test_performance_formula authoringtool.test_help_requests authoringtool.test_analytics_access authoringtool.tests`. Expected: all new tests pass; the others show no new failures (two `ProposalHistoryViewTests` and `test_task_starter_can_read_result` are baseline failures).
- [ ] **Step 5: Commit** ("Share the PISA formula between the chart and the CSV").

### Task 3: Runs, finished rule and labels

**Files:**
- Create: `Trust-AI-Platform/categorization/dataset.py`, `categorization/tests/test_dataset.py`, `categorization/tests/fixtures.py` (scenario builder shared by Tasks 3–7)

**Interfaces:**
- Consumes: `performance_formula.*` (Task 2); `authoringtool.utils.get_last_answers(scenario_id)` and `get_eligible_user_answers(scenario_id)` (default scope `local`); `authoringtool.help_time.help_totals(scenario)`; `authoringtool.translation_matching.translation_groups()`.
- Produces:
  - `@dataclass AnswerEvent`: `activity_id: int`, `phase_position: int` (1-based, phases ordered by `('order', 'id')`), `created_on: datetime`, `active_seconds: float | None` (timing minus help on that (user, activity), floored at 0; `None` when timing is NULL or 0), `chosen_weight: int | None`, `is_last: bool`.
  - `@dataclass Run`:
    - identity: `implementation_id`, `user_id`, `username`, `scenario_id`, `scenario_name`
    - `language` (stripped, `''` allowed)
    - `group_key: int` (smallest scenario id in its confirmed translation group, else its own id)
    - `is_current_version: bool`: `version_confidence == 'exact'` and `scenario_version_id == scenario.current_version_id`
    - `phase_count: int`
    - `phase_scores: list[tuple[float, float]]` (one per phase position)
    - `answers: list[AnswerEvent]` (all eligible answers, time order)
    - `help_by_position: dict[int, dict]`: `{'seconds', 'requests'}` per phase position, from `HelpRequest` rows of this user and scenario with `ended_at` set
    - `final_percentage: float | None`
    - `label: str | None`
    - `finished_rule: str`: `'route_end'`, `'last_phase'` or `'unfinished'`
  - `@dataclass ScenarioContext`: `scenario`, `phases` (ordered list), `position_by_phase_id`, `scored_ids_by_position: dict[int, list[int]]`, `max_weight: dict[int, int]`, `terminal_ids: set[int]`, `activity_ids: set[int]`.
  - `public_scenarios() -> list[Scenario]`: `visibility_status='public'` that have at least one scored activity, ordered by id.
  - `scenario_context(scenario) -> ScenarioContext`
  - `terminal_activity_ids(scenario) -> set[int]`: activities with no outgoing route. Routes are `NextQuestionLogic.next_activity` (with or without an answer, i.e. the default next) and the three `EvQuestionBranching` targets. Non-primary members of a `QuestionBunch` are excluded, because the primary carries the route (same graph as `graph_validation.validate_scenario_graph`).
  - `finished_rule(answered_ids: set[int], ctx) -> str`:
    - `'route_end'` if the run answered any terminal activity.
    - Else `'last_phase'` when no answered id is in `ctx.activity_ids`, or `ctx.terminal_ids` is empty, and the run answered something in the last position. In this branch, `answered_ids` are judged on the activities' current phase.
    - Else `'unfinished'`.
  - `load_runs(progress: Callable[[str], None] = lambda text: None) -> list[Run]`: one `Run` per implementation of every public scenario, labelled only when `finished_rule != 'unfinished'` and `final_percentage is not None`.

- [ ] **Step 1: Write the failing tests**:
  - `test_only_public_scenarios`: a private and an org scenario with finished runs contribute nothing.
  - `test_label_matches_csv`: for the Task 2 fixture made public, each run's `label` equals the CSV's `Final Categorization` for that username. This is the spec's required cross-check.
  - `test_branch_ending_early_counts_as_finished`: a scenario where Low students are routed to a remediation activity in phase 2 that ends the route. A student who ends there gets `finished_rule == 'route_end'` and a label.
  - `test_dropout_is_unfinished`: answered only phase 1 → `label is None`, `finished_rule == 'unfinished'`.
  - `test_deleted_activity_answers_fall_back` (Review Focus 4): the run's answers have `activity=None`, or point at an activity deleted from the scenario, plus one answer in the last phase → `'last_phase'`, no exception.
  - `test_teacher_and_excluded_runs_left_out`: a teacher's run and a run with `data_quality_status='excluded'` are missing from `load_runs()`.
  - `test_translation_group_key`: two public scenarios with a confirmed `TranslationMatch` share `group_key`.
  - `test_current_version_flag`: a `legacy_unknown` implementation has `is_current_version False`.
  - `test_help_seconds_removed_from_time`.
- [ ] **Step 2: Run them** and see them fail (module missing).
- [ ] **Step 3: Implement `dataset.py`.** Key answers by `implementation_id`, not user. Use `get_last_answers` for `is_last` and labels, and `get_eligible_user_answers` for all answers (retries, time order). Prefetch per scenario, with no queries per run (pattern: `compute_student_performance_metrics`).
- [ ] **Step 4: Run** `categorization`. Expected: pass.
- [ ] **Step 5: Commit** ("Build labelled runs for categorization training").

### Task 4: Model inputs and the checkpoint table

**Files:**
- Create: `Trust-AI-Platform/categorization/features.py`, `categorization/tests/test_features.py`

**Interfaces:**
- Consumes: `Run`, `AnswerEvent` (Task 3); `performance_formula.categorize`, `weighted_percentage`, `phase_weights`.
- Produces:
  - `MAX_POSITIONS = 6`, `FAST_ANSWER_RATIO = 0.1`
  - `FAST_ANSWER_RULE = 'An answer is very fast when its time is under 10% of the median time on that activity.'` (shown on the dashboard)
  - `META_COLUMNS = ['implementation_id', 'user_id', 'username', 'scenario_id', 'scenario_name', 'language', 'group_key', 'is_current_version', 'label', 'final_percentage']`
  - `Norms`: `time_median: dict[int, float]` (activity → median `active_seconds` of last answers), `visit_share: dict[tuple[int, int], float]` ((scenario, activity) → share of that scenario's runs that answered it). Built by `build_norms(runs) -> Norms` from all runs. These are label-free statistics; this is my call, written in the plan.
  - `run_inputs(run, norms, upto: int | None) -> dict`: `upto=None` means the whole run; otherwise only answers and help with `phase_position <= upto`. Keys:
    - Per position `i` in 1..6: `p{i}_pct`, `p{i}_answered_share`, `p{i}_category` (0/1/2 for Low/Moderate/High), `p{i}_weight`, `p{i}_time_ratio` (median of `active_seconds / time_median`). Positions beyond `upto` or `phase_count`, or with nothing to measure, are NaN.
    - `phase_count`, `fast_share`, `retries` (eligible answers minus distinct activities answered), `help_requests`, `help_seconds`, `path_commonness` (mean `visit_share` of answered activities), `is_current_version` (0/1)
    - `scenario`, `language` (strings; `''` becomes `'Unspecified'`)
    - Checkpoint rows only: `checkpoint` (k) and `pct_so_far` (`weighted_percentage` over positions 1..k with `phase_weights(phase_count)`)
  - `CATEGORICAL = ['scenario', 'language']`
  - `final_table(runs, norms) -> pd.DataFrame`: labelled runs only; META_COLUMNS + inputs with `upto=None`. No `checkpoint` and no `pct_so_far`: the final model must learn the formula from the phase scores.
  - `checkpoint_table(runs, norms) -> pd.DataFrame`: for each labelled run and each `k` in `1..phase_count-1` where the run answered in position `k`: META_COLUMNS + inputs with `upto=k` + `baseline_label = categorize(pct_so_far)` (row skipped when `pct_so_far is None`).
  - `input_columns(table) -> list[str]`: every column except META_COLUMNS and `baseline_label`.
  - `as_model_frame(table, columns, categories: dict[str, list[str]]) -> pd.DataFrame`: casts CATEGORICAL columns to `pd.CategoricalDtype(categories)`, so categories never depend on the split.

- [ ] **Step 1: Write the failing tests**:
  - `test_checkpoint_sees_nothing_later` (anti-goal): build the checkpoint row at `k=1`, add answers, retries and help in phase 2 and 3, rebuild. Every input column of the `k=1` row is unchanged.
  - `test_final_table_has_no_final_percentage_input`: `'final_percentage'` and `'pct_so_far'` not in `input_columns(final_table(...))`.
  - `test_baseline_label_uses_reached_phases_only`: positions 1–2 scored at 100% and 40%, weights 0.2/0.2 → `pct_so_far == 70.0`, `baseline_label == 'Moderate'`.
  - `test_fast_share`: 2 of 4 last answers under 10% of the median → `0.5`.
  - `test_retries_counts_changed_answers`.
  - `test_blank_language_is_unspecified`.
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement `features.py`.**
- [ ] **Step 4: Run** `categorization`. Expected: pass.
- [ ] **Step 5: Commit** ("Turn runs into scenario-independent model inputs").

### Task 5: Training and evaluation (no Django)

**Files:**
- Create: `Trust-AI-Platform/categorization/training.py`, `categorization/tests/test_training.py` (plain `SimpleTestCase` with synthetic DataFrames, no database)

**Interfaces:**
- Consumes: `features.input_columns`, `as_model_frame`, `CATEGORICAL`, `META_COLUMNS`; `performance_formula.CATEGORIES`.
- Produces:
  - `SEED = 42`, `FOLDS = 5`, `LOSO_MIN_RUNS = 30`, `MIN_LABELLED_RUNS = 50`, `EARLY_STOPPING_ROUNDS = 30`
  - `PARAMS = {'objective': 'multi:softprob', 'num_class': 3, 'eta': 0.08, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'tree_method': 'hist', 'eval_metric': 'mlogloss', 'seed': 42}`, with `num_boost_round = 400`
  - `class NotEnoughData(Exception)`: message shown on the dashboard as-is.
  - `fit_booster(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> xgb.Booster`:
    - Uses the native API, `xgb.DMatrix(X, label=y, weight=balanced weights, enable_categorical=True)`, so `num_class=3` is fixed even if a class is missing.
    - Balanced weights: `sklearn.utils.class_weight.compute_sample_weight('balanced', y)`.
    - Early stopping on a `GroupShuffleSplit(test_size=0.15, random_state=SEED)` slice when there are at least 10 groups; otherwise fixed 200 rounds.
  - `predict_proba(booster, X) -> np.ndarray` with shape `(n, 3)`, columns in `CATEGORIES` order.
  - `metrics(y_true: list[str], y_pred: list[str]) -> dict`:
    - `accuracy`, `macro_f1`, `runs`
    - `per_class: {Low|Moderate|High: {precision, recall, f1, support}}`
    - `confusion: 3×3 list` (rows truth, columns prediction, `CATEGORIES` order)
    - Uses `zero_division=0`.
  - `grouped_oof(table, columns, categories, progress) -> pd.DataFrame`: `GroupKFold(n_splits=min(FOLDS, n_students))` on `user_id`. Returns `table.index`-aligned columns `pred`, `confidence`, `prob_Low`, `prob_Moderate`, `prob_High`.
  - `leave_one_group_out(table, columns, categories, progress) -> list[dict]`: one entry per `group_key` with at least `LOSO_MIN_RUNS` rows: `{group_key, scenario_names, runs, accuracy, macro_f1}`. When the table has `checkpoint`, it also carries `by_checkpoint: {k: {runs, accuracy, baseline_accuracy}}`.
  - `importances(booster, X, columns, top=15) -> list[{'input', 'mean_abs_shap'}]`: `booster.predict(DMatrix, pred_contribs=True)` has shape `(n, 3, f+1)`. Take the mean absolute value over rows and classes, drop the bias column, sort descending.
  - `train_all(final: pd.DataFrame, checkpoints: pd.DataFrame, progress) -> tuple[dict, dict[str, xgb.Booster]]`:
    - Raises `NotEnoughData` when `len(final) < MIN_LABELLED_RUNS`, or when `final.label` has fewer than 2 categories. Message: `f"Not enough finished runs to train: {n} found, at least {MIN_LABELLED_RUNS} needed, with at least two categories."`
    - Returns the `results` dict below, plus `{'final': booster, 'early': booster}` fitted on all rows.

  `results` schema (the dashboard and export read exactly this):
  ```
  {"final": {"overall": metrics, "by_version": {"current": metrics, "older": metrics},
             "importances": [...], "loso": [...],
             "predictions": [{implementation_id, user_id, username, scenario_id, scenario_name,
                              language, group_key, is_current_version, pisa, model, confidence,
                              final_percentage}]},
   "early": {"checkpoints": [{"checkpoint": k, "xgb": metrics, "baseline": metrics}],
             "importances": {"<k>": [...]}, "loso": [...],
             "predictions": [{implementation_id, scenario_id, scenario_name, language,
                              is_current_version, checkpoint, truth, model, baseline}]},
   "seconds": {"total": float, "loso": float}}
  ```

- [ ] **Step 1: Write the failing tests**. A helper builds a synthetic table of 200 runs, 80 students and 4 `group_key`s, with labels from a known rule on `p1_pct`.
  - `test_no_student_in_train_and_test`: patch `fit_booster` to record training `user_id`s per fold. No predicted row's `user_id` was in its own fold's training set.
  - `test_fold_missing_a_category_still_gives_three_columns` (Review Focus 3): train on rows with only Low/Moderate, predict → shape `(n, 3)`.
  - `test_unseen_scenario_and_blank_language_predict` (Review Focus 5): hold out a `group_key` whose `scenario` and `language` values never occur in training. `leave_one_group_out` returns an entry for it.
  - `test_same_data_same_results`: `train_all` twice gives identical `results` (excluding `seconds`).
  - `test_too_little_data_raises`: 20 runs → `NotEnoughData` with the exact message.
  - `test_metrics_confusion_order`: truth `['High', 'Low']`, prediction `['High', 'Moderate']` → `confusion[2][2] == 1`, `confusion[0][1] == 1`.
  - `test_importances_sorted_and_capped`: at most 15 entries, descending, and the rule column `p1_pct` comes first.
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement `training.py`.** `progress` texts: `"Final model: cross-validation {i} / {n}"`, `"Final model: unseen scenarios {i} / {n}"`, then the same with `"Early model"`.
- [ ] **Step 4: Run** `categorization.tests.test_training`. Expected: pass, in under a minute.
- [ ] **Step 5: Commit** ("Train and evaluate both categorization models").

### Task 6: Celery job, start/status endpoints and a management command

**Files:**
- Create: `Trust-AI-Platform/categorization/tasks.py`, `categorization/management/__init__.py`, `categorization/management/commands/__init__.py`, `categorization/management/commands/train_categorization.py`, `categorization/tests/test_job.py`

**Interfaces:**
- Consumes: `TrainingRun` (Task 1), `dataset.load_runs` (Task 3), `features.*` (Task 4), `training.train_all`, `NotEnoughData` (Task 5).
- Produces:
  - `run_training(run_id: int) -> None`:
    - Sets `running` and `started_at`, then builds runs, norms and tables, calling `run.set_progress` throughout ("Building inputs…").
    - Calls `train_all`, then saves the boosters as `{CATEGORIZATION_MODEL_DIR}/run_{id}_final.json` and `run_{id}_early.json` (creating the folder).
    - Fills `versions` (`xgboost.__version__`, `sklearn.__version__`), `parameters` (`PARAMS`, `num_boost_round`, input columns, `FAST_ANSWER_RULE`) and `counts`. `counts` holds: `students`, `runs_total`, `labelled`, `unfinished`, `route_end`, `last_phase`, `current_version`, `older_version`, `scenarios` as a list of `{id, name, language, runs, labelled}`.
    - On success: `results`, `finished` and `finished_at`.
    - On `NotEnoughData`: `failed` with its message.
    - On any other exception: `failed` with `"{type}: {message}"`, plus the traceback in the log.
  - `@shared_task train_models_task(run_id)`, which calls `run_training`.
  - `manage.py train_categorization`: creates a run with `started_by=None` via `TrainingRun.start(None)`, calls `run_training` inline, and prints status, counts and `results['seconds']`. Used to measure run time (Task 9).

- [ ] **Step 1: Write the failing tests**, using the Task 3 fixture builder:
  - `test_job_with_too_little_data_fails_cleanly` (Review Focus 2): one public scenario, 3 finished runs → status `failed`, and `error` starts with `"Not enough finished runs to train"`.
  - `test_job_success_fills_run`: patch `MIN_LABELLED_RUNS` to 5 and build 12 finished runs over 2 scenarios → `finished`; `results['final']['overall']['runs'] == counts['labelled']`; both model files exist in a temp `CATEGORIZATION_MODEL_DIR` (`override_settings` + `tempfile.TemporaryDirectory`).
  - `test_unexpected_error_marks_failed`: patch `load_runs` to raise `RuntimeError('boom')` → `failed`, `error == 'RuntimeError: boom'`.
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement** the task, `run_training` and the command.
- [ ] **Step 4: Run** `categorization`. Expected: pass.
- [ ] **Step 5: Commit** ("Run categorization training as a background job").

### Task 7: Admin dashboard

**Files:**
- Create: `Trust-AI-Platform/categorization/report.py`, `categorization/views.py`, `categorization/templates/categorization/dashboard.html`, `categorization/tests/test_dashboard.py`
- Modify: `Trust-AI-Platform/faithDev/urls.py` (paths before `admin/`, like `ai_server_connection`), `Trust-AI-Platform/templates/admin/base_site.html` (link "Student categorization" next to "AI server connection")

**Interfaces:**
- Consumes: `TrainingRun`, `train_models_task`, `training.metrics`, `features.FAST_ANSWER_RULE`.
- Produces:
  - `report.filter_rows(rows: list[dict], scenario: str, language: str, version: str) -> list[dict]`: `version` is `''`, `'current'` or `'older'`. Empty strings mean no filter.
  - `report.final_view(results, filters) -> dict`:
    - `overall`, `confusion`, `importances`
    - `disagreements`: rows where `pisa != model`, sorted by `confidence` descending. Each row adds `threshold_distance = min(|pct − 83.3|, |pct − 49.7|)`.
    - `per_scenario`: agreement per `group_key` from the filtered predictions, with the matching LOSO accuracy or `None`.
    - `version_counts`
  - `report.early_view(results, filters, checkpoint: int | None) -> dict`:
    - `rows`: per checkpoint, recomputed from filtered predictions with `training.metrics` for both `model` and `baseline`.
    - `picked` checkpoint (default: the first), with its `confusion` and `importances`.
    - `loso`: for the picked checkpoint.
    - `chart`: lists for ECharts.
  - URL names:
    - `categorization_dashboard`: GET `admin/student-categorization/`, query `run`, `tab` (`final`|`early`), `scenario`, `language`, `version`, `checkpoint`.
    - `categorization_train`: POST `admin/student-categorization/train/`. `TrainingRun.start(request.user)`, then `train_models_task.delay(run.id)`, with message "Training started." or, on `TrainingAlreadyRunning`, "A training run is already in progress." Redirects to the dashboard.
    - `categorization_status`: GET `admin/student-categorization/status/` → JSON `{id, status, progress, error}` of the newest run, or `{}`.
  - `admin_only(view)`: anonymous → redirect to `admin:login` with `next`; logged in but not staff/superuser → `HttpResponseForbidden`. Renders with `admin.site.each_context(request)` and extends `admin/base_site.html`.

  **Page content:** the spec's Dashboard section is the source.
  - Header:
    - the last finished run's date, who started it, duration and counts (including `route_end`/`last_phase`/unfinished)
    - a collapsible list of the public scenarios used, with runs and labelled runs per scenario (`counts['scenarios']`)
    - the Train button: disabled with a progress line while a run is active; JS polls `categorization_status` every 5 s and reloads when the run ends
    - the last failed run's error, if newer than the last finished one
    - a run dropdown, newest first
    - filters for scenario, language and version
    - `FAST_ANSWER_RULE`
  - Tab 1: headline numbers, confusion matrix (ECharts heatmap, PISA rows), disagreements table, top-15 importances (horizontal bar), per-scenario agreement next to LOSO.
  - Tab 2: accuracy and macro F1 per checkpoint (line chart, XGBoost vs baseline, one y-axis 0–1), per-checkpoint table, checkpoint dropdown driving the confusion matrix, importances and LOSO table.
  - Charts load `{% static 'vendor/echarts/echarts.min.js' %}`, and data is passed with `json_script`.
  - With no finished run, the page shows a single line ("No training has finished yet.") and the Train button.

- [ ] **Step 1: Write the failing tests**:
  - `test_staff_sees_page`, `test_superuser_sees_page`
  - `test_teacher_gets_403`
  - `test_anonymous_redirected_to_admin_login`
  - `test_empty_state`: no runs → 200, contains "No training has finished yet."
  - `test_train_starts_job`: patch `train_models_task.delay`; POST → one queued run, delay called with its id.
  - `test_train_while_running_shows_message`: a running run exists → no new run, message shown.
  - `test_filters_recompute_metrics`: a finished run with hand-written `results` (2 scenarios, 4 predictions, one disagreement in scenario A). `?scenario=<B>` shows agreement 100% and no disagreement rows; no filter shows 75%.
  - `test_status_json`
  - `test_header_link_present` on the admin index.
  - `test_no_emoji_in_template`: the rendered HTML has no character in the emoji ranges `U+1F300–U+1FAFF` or `U+2600–U+27BF`.
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement** `report.py`, `views.py`, the template, urls and the header link.
- [ ] **Step 4: Run** `categorization`. Expected: pass.
- [ ] **Step 5: Check it in a browser.**
  - Run the dev server against the test settings plus fixture data, or Docker.
  - Load a finished run made by `manage.py train_categorization` on synthetic or local data.
  - Look at both tabs at desktop and phone width. Check that charts render in light mode, labels don't overlap and nothing scrolls horizontally.
- [ ] **Step 6: Commit** ("Add the admin dashboard for categorization models").

### Task 8: Excel export

**Files:**
- Modify: `Trust-AI-Platform/categorization/views.py`, `categorization/urls` entries in `faithDev/urls.py`, `dashboard.html`
- Create: `categorization/export.py`, `categorization/tests/test_export.py`

**Interfaces:**
- Consumes: `report.final_view`, `report.early_view`.
- Produces:
  - `export.workbook(run, tab, filters) -> bytes` (openpyxl):
    - `final` sheets: `Summary`, `Confusion matrix`, `Disagreements`, `Importance`, `Per scenario`
    - `early` sheets: `Checkpoints`, `Confusion matrix` (picked checkpoint), `Importance`, `Unseen scenarios`
    - Filters applied, and written on the `Summary`/`Checkpoints` sheet.
  - URL `categorization_export`: GET `admin/student-categorization/<int:run_id>/export/<tab>/`, `admin_only`. Returns `Content-Disposition: attachment; filename="categorization-run-{id}-{tab}.xlsx"`. 404 for an unknown run or tab, or a run not `finished`.

- [ ] **Step 1: Write the failing tests**:
  - `test_final_export_sheets`: load with `openpyxl.load_workbook(BytesIO(response.content))` and check the sheet names; the Disagreements row count matches the page.
  - `test_early_export_sheets`
  - `test_export_respects_filters`
  - `test_teacher_cannot_export` (403)
  - `test_unfinished_run_404`
- [ ] **Step 2: Run** and see them fail.
- [ ] **Step 3: Implement** the export and add one button per tab.
- [ ] **Step 4: Run** `categorization`. Expected: pass.
- [ ] **Step 5: Commit** ("Export categorization results to Excel").

### Task 9: Real-data run, timing and full suite

**Files:**
- Modify: `deploy/` notes only if a deploy step changes (none expected besides rebuilding `web`, `celery` and `celerybeat` and running `migrate`)

- [ ] **Step 1: Train on the local data copy.** Explain to the user before starting Docker or building an image, since they asked for heavier checks to be explained first. Then run `docker compose run --rm web python manage.py migrate categorization` and `docker compose run --rm web python manage.py train_categorization`. Record:
  - total seconds and LOSO seconds
  - labelled, unfinished, `route_end`/`last_phase` counts
  - final-model agreement and macro F1
  - early accuracy per checkpoint against the baseline

  If the run takes more than 10 minutes, report it. Possible changes are fewer boosting rounds, or skipping LOSO for the final model; the user decides.
- [ ] **Step 2: Run the full suite** (no label, output to a file, `timeout 3600`). Expected: the 6 known failures only.
  - `test_org_editor_edits_details_but_not_sharing` and `test_two_teachers_can_use_the_same_name` (sqlite NumericRange errors)
  - `test_task_starter_can_read_result`
  - `test_implementations_and_answers_are_pinned_to_current_version`
  - `ProposalHistoryViewTests.test_history_index_lists_only_past_runs` and `test_history_index_shows_decision_counts`
- [ ] **Step 3: Commit** any fixes, then report to the user:
  - the numbers from Step 1
  - the deploy steps: pull; build `web celery celerybeat`; `docker compose run --rm web python manage.py migrate`; `up -d web celery celerybeat`
  - the translation-branch merge note: `base_site.html` gains one more header link, and the new templates need their strings in the `.po` files.
