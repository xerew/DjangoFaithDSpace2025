# XGBoost Student Categorization (admin only) — Design

Date: 2026-11-12
Status: approved by the admin on 2026-11-12

## Why

Every student run of a scenario gets a final category, **High**, **Moderate**
or **Low**, from a fixed formula (called "PISA" below, because the thresholds
are the PISA-aligned ones also used by the risk flags):

- Only the **last answer** per activity counts.
- **Phase %** = sum of `answer_weight` of the chosen answers ÷ sum of the
  highest `answer_weight` of each scored activity × 100. Scored activities are
  those with `is_evaluatable` and `is_primary_ev`, plus the other activities
  in their `QuestionBunch`. Unanswered scored activities count as 0.
- **Final %** = weighted average of phase % by phase position: 20/20/45/15 for
  up to 4 phases, 20/20/30/15/15 for 5 or more. Phases with no scored activity
  are skipped and the remaining weights rescaled.
- **High** ≥ 83.3, **Moderate** ≥ 49.7, otherwise **Low**. The same
  thresholds give a category for each phase.

The formula lives twice today: `compute_final_performance`
(`authoringtool/tasks.py`, the waterfall chart) and
`compute_student_performance_metrics` (same file, the "Final Categorization"
column of the student CSV).

The admin wants XGBoost used in two ways, as an alternative view next to PISA:

1. **Final categorization.** A model that reproduces the final category from a
   student's whole run. The point is to show that a learned model arrives at
   the same categorization, and to show exactly where and how it disagrees.
2. **Early prediction.** A model that predicts the final category while the
   student is still partway through: after phase 1, after phase 2, and so on.

Both are for checking model quality (accuracy and related statistics) on a
small dashboard that only admins can see.

## Goals

- Two XGBoost models trained on **all students of all scenarios together**.
  The admin asked for one pooled model: the formula is the same everywhere,
  and most scenarios are too small for a model of their own.
- Training starts **only when an admin presses a button**. There is no
  schedule. It runs as a Celery job in the background; the dashboard shows
  that it is running and refreshes when it finishes.
- An **admin-only dashboard** with two tabs, described below.
- **PISA is not removed or changed.** Teachers' pages, charts, CSV exports and
  risk flags keep giving exactly the same results as before.

## Non-goals

- Teachers and students never see the models or their output.
- Predictions are not written back onto students, runs or scores. They exist
  only in the saved results of a training run, shown on the dashboard.
- No automatic or scheduled retraining.
- No model per scenario, and no model per language.
- Not fixing the two problems found in the existing formula code while
  designing this (they are separate work):
  - the waterfall chart raises `IndexError` on a 6th phase with scored
    activities, while the CSV gives it weight 0;
  - the chart and the CSV include slightly different students (the chart
    requires the scenario's first activity to be answered; the CSV takes
    anyone with an answer in the date range).
- No year-by-year cohort comparison (train on earlier years, test on 2026).
  It was discussed for research on 2026-10-20 and may come later.

## Anti-goals (a failure even if it technically works)

- Any change to a PISA category anywhere on the platform.
- Metrics measured on data the model was trained on. Every number on the
  dashboard comes from students the model did not train on.
- The same student appearing in both training and test data in any
  evaluation.
- The early model using anything from after its checkpoint (later answers,
  later times, the final score).

## Data

**Unit:** one student run, `ScenarioImplementation` (one student, one
scenario version). The local copy from 2026-10-20 had 4,481 student–scenario
runs, 3,292 students and 102,911 answers across 2024-05 to 2026-10, in
English, Portuguese, Greek, Romanian and Spanish.

**Which scenarios:** **public scenarios only** (the admin's decision):
`Scenario.visibility_status == 'public'` at the moment training starts.
Private and any other non-public scenarios are left out entirely, including
their runs. A translation counts only if the translation itself is public.
The dashboard lists the scenarios used.

**Which answers:** the same eligible answers the analytics already use
(`get_eligible_user_answers` / `get_evidence_answers` in
`authoringtool/evidence.py` and `authoringtool/utils.py`), with whatever
scope those return under the running settings, so runs marked `suspect` or
`excluded` are left out. Teachers are excluded, as in the existing code.

- **Old and legacy runs stay in** (my call). With family pooling off, which is
  production today (`SCENARIO_FAMILY_FEATURES=0`), this includes
  `legacy_unknown` runs and runs on older scenario versions, scored against
  today's phases and activities. That is exactly how PISA scores them on the
  platform today, and the goal is to reproduce the platform's categories.
- Each run carries a flag: **exact current version** or **older / legacy**.
  It is an input to both models. The dashboard shows how many runs fall in
  each group, adds it as a filter next to scenario and language, and reports
  the main measures for each group separately, so it is visible if old runs
  behave differently.
- Tests and production may therefore train on different populations when the
  pooling setting differs; that is acceptable.
- There is one live implementation per student per scenario
  (`start_or_resume`), so "run" and "student–scenario pair" are the same
  thing.

**Labels:** the PISA final category of each run, computed by the formula
above.

- **The formula must be the same code the platform uses** (builder's
  approach, my call): extract it into one shared function used by
  `compute_final_performance`, `compute_student_performance_metrics` and the
  new training code. Tests must show the two existing callers give
  identical output before and after the extraction. If that turns out to be
  risky, the training code may call a copy, but then a test must check the
  copy against both existing functions on the same data.
- **Finished runs only for labels.** Completion is not recorded:
  `ScenarioImplementation.complete()` exists but nothing calls it, and 0 of
  4,860 implementations were ever marked `completed`. Without a rule, every
  dropout would be labelled Low, because unanswered activities count as 0.
  A run counts as finished if **the student answered an activity at which
  their route ends**: an activity with no next activity, following the same
  routing the student view uses (`NextQuestionLogic`, `EvQuestionBranching`,
  and the default next activity). This matters because a branch, such as a
  remediation path for weaker students, may end before the phase with the
  highest `order`; a "last phase" rule would quietly drop those students and
  push the labels towards High and Moderate. Where the route can't be worked
  out (for example a legacy run whose activities no longer exist), fall back
  to "answered at least one activity in the scenario's last phase". Runs that
  stopped earlier get no label and are not used for training or testing. The
  dashboard shows how many runs were labelled by each rule and how many were
  left out as unfinished.

**Inputs (features).** They must mean the same thing in every scenario, so
no activity IDs, phase names or raw points. The builder decides the exact
list; it must cover at least:

- Per phase, by **phase position** (1st, 2nd, 3rd …): phase %, the share of
  scored activities answered, and the PISA phase category.
- Number of phases in the scenario, and the weight of each position (so the
  model can see the 4-phase vs 5-phase weight difference).
- **Time**: active time per phase with help pauses removed (as
  `active_time` / `help_totals` already do), compared with other students on
  the same activity (for example the student's time ÷ the median time on that
  activity), not raw seconds.
- **Very fast answers** that look like guesses: the share of answers much
  faster than is usual for that activity. The rule (for example under 10% of
  the activity's median time) is the builder's call and must be written on
  the dashboard.
- **Retries**: how many times the student changed an answer.
- **Help**: help requests and help time. (Very few exist today, so expect
  this to matter little.)
- **Path**: whether the student took the scenario's most common route, or how
  far from it.
- **Scenario** and **language** as categorical inputs.

**The final categorization model sees the scores** (the admin's decision).
It is therefore expected to agree with PISA almost every time; the interest
is in the disagreements.

**Early checkpoints.** For a scenario with *n* phases there are checkpoints
after phase 1 … phase *n*−1. At checkpoint *k* the inputs use only phases
1…*k*: answers, times, retries, help and path up to the end of phase *k*. A
run contributes to checkpoint *k* only if the student answered something in
phase *k*. The early model is **one model for all checkpoints**, with the
checkpoint number as an input, unless the builder finds separate models per
checkpoint clearly better; either way the dashboard reports each checkpoint
separately.

## Evaluation

- **5-fold cross-validation grouped by student** (`GroupKFold` on `user_id`).
  Every run gets an out-of-fold prediction from a model that never saw that
  student. All dashboard metrics, the confusion matrices and the
  disagreements table use these out-of-fold predictions.
- **Leave-one-scenario-out**, for **both models**: for each scenario group
  with at least 30 labelled runs, train on all other groups and test on that
  one. This shows how the model does on a scenario it has never seen. It
  matters most for the early model; for the final model, which sees the
  scores, expect it to be close to perfect.
  - A **scenario group** is a scenario plus its confirmed 1:1 translations
    (`TranslationMatch`), held out together. Holding out the English version
    while training on its Greek copy would not be an unseen scenario.
  - The builder measures the extra run time this adds and reports it.
- **Baseline for early prediction**: at checkpoint *k*, apply the PISA
  formula to phases 1…*k* only (rescaling the weights of the phases reached)
  and use the thresholds. XGBoost is only useful where it beats this.
- **Measures**: accuracy, macro F1 (each category counts equally), and
  precision, recall and F1 for High, Moderate and Low. For the final model,
  also **agreement with PISA** (the same number as accuracy, labelled in
  plain words).
- **Class imbalance**: use balanced sample weights, and always show macro F1
  next to accuracy.
- **Input importance**: from XGBoost's own SHAP values (`pred_contribs=True`),
  averaged as mean absolute value per input. No `shap` package needed.
- Fixed random seed, so the same data gives the same results.
- After evaluation, each model is also fitted once on all labelled data and
  saved with the run (see Storage), so it can be inspected or reused later.

## Dashboard

**Who:** Django staff or superusers, the same rule as machine translation
(`authoringtool/machine_translation/collaborators.py:28`). Everyone else gets
403. It is reached from the Django admin area (a custom admin view, like the
existing `get_urls` views in `authoringtool/admin.py`).

**Header (both tabs)**

- Last finished training: date and time, who started it, how long it took,
  number of students, labelled runs, runs left out as unfinished, scenarios.
- **Train models** button. While a job is running, the button is disabled and
  a progress line is shown ("Building inputs…", "Cross-validation 3 / 5…").
  Only one training job may run at a time. A failed job shows its error.
- Dropdown of earlier training runs (newest first), so results can be
  compared over time. Shows the latest finished run by default.
- Filters for **scenario** and **language**. They narrow the tables and
  charts to those runs' out-of-fold predictions; they do not retrain.

**Tab 1: Final categorization**

- Headline numbers: agreement with PISA, macro F1, F1 for each category.
- Confusion matrix: PISA category (rows) against model category (columns).
- **Disagreements table**: username, scenario, language, PISA category, model
  category, the model's confidence, the run's PISA final %, and its distance
  to the nearest threshold (83.3 or 49.7). Most confident disagreements first.
- Input importance (top 15).
- Agreement per scenario, with the leave-one-scenario-out result beside it.

**Tab 2: Early prediction**

- Main chart: accuracy (and macro F1) at each checkpoint, XGBoost next to the
  baseline.
- Table per checkpoint: number of runs, accuracy, macro F1, F1 for each
  category, for XGBoost and the baseline.
- Confusion matrix for a checkpoint picked from a dropdown.
- Input importance for the picked checkpoint.
- Leave-one-scenario-out results: accuracy per held-out scenario group at the
  picked checkpoint, XGBoost next to the baseline.

**Export**: one button per tab that downloads the tab's tables as Excel
(`openpyxl` is already installed).

**Look**: the platform's existing styling; charts with ECharts, which the
platform already uses. No emojis; plain labels or `bi-*` icons (standing
rule for the platform UI). Strings marked for translation like the rest of
the interface.

## Storage

A new model for training runs (name is the builder's call), with at least:
who started it, when it started and finished, status (queued, running,
finished, failed), error text, a progress text, XGBoost and scikit-learn
versions, the parameters and input list used, data counts, and the full
results as JSON (all dashboard numbers, both confusion matrices, importances,
per-scenario results, the per-checkpoint table, and the disagreements as run
IDs plus the displayed fields). The fitted models are saved as XGBoost JSON
files under the media or data volume, linked from the run. All runs are kept.

## Constraints

- Server: Django in Docker with a separate Celery worker
  (`docker-compose.yml`, `celery -A faithDev worker`) and Redis. The job runs
  on that worker. It must not block the queue for long, so the builder should
  measure its run time on the full local copy and say what it is.
- Add `xgboost` and `scikit-learn` to `Trust-AI-Platform/requirements.txt`.
  That file is **UTF-16**; keep its encoding. Pin versions that install in the
  Python 3.10 Docker image. `pandas` and `numpy` are already there.
- Tests run with the existing setup (Docker, or local SQLite
  `settings_test`). The baseline has 6 known failures; no new ones.

## Left to the builder

- The exact input list, within the rules above.
- XGBoost parameters (sensible defaults with early stopping are fine; no big
  search).
- One early model with checkpoint as an input, or one per checkpoint.
- Names of modules, models and URLs; page layout within the description
  above.

## Calls I made (not discussed with the admin)

- Unit of analysis is a run, not a student.
- Old and legacy runs are included, flagged, and reported separately.
- "Finished" = reached the end of the student's route, falling back to
  "answered something in the last phase" when the route can't be worked out.
- Grouped 5-fold cross-validation; leave-one-scenario-out for both models,
  only for scenario groups with at least 30 labelled runs, with translations
  held out together with their original.
- Admin = staff or superuser.
- Usernames are shown in the disagreements table.
- All training runs are kept, with a dropdown to view earlier ones.
- Excel export per tab.
- Input importance from XGBoost's built-in SHAP values.
