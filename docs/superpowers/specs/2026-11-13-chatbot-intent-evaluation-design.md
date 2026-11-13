# Chatbot intent evaluation on the Student categorization page

Date: 2026-11-13. Branch: `feature/xgboost-categorization` (the page lives there).

## Why

The chatbot (Rasa) intent evaluation was run by hand on 12 November:
`rasa test nlu --cross-validation`, then `RasaFaith/RasaFaith/tools/export_intent_report.py`
turned the output into an Excel file and charts under `results/export/`. Those
files sit in folders on one machine. The admin wants to see the results on the
platform and to run the evaluation again from there, for example after changing
the chatbot's training sentences.

## Goals

- A third tab, **Chatbot intents**, on the admin Student categorization page
  (`/admin/student-categorization/?tab=chatbot`), next to Final and Early.
- **Run evaluation** button with a choice of 3 or 5 folds. One run at a time.
  A run takes about 30 minutes (3 folds, measured locally), so the page shows
  progress and the admin can leave and come back.
- For the selected run, show:
  - headline numbers: accuracy (all folds pooled), macro F1, weighted F1,
    example sentences, intents, wrong predictions, wrong predictions above the
    fallback threshold (the fallback would not catch them), and the per-fold
    mean (std) of test accuracy and F1 when available;
  - F1 per intent (bars, worst first), with precision, recall, examples and
    wrong count in a table;
  - confusion matrix (true intent rows, predicted columns), same heatmap style
    as the PISA matrix;
  - confidence chart: right and wrong predictions by confidence, with the
    fallback threshold marked;
  - table of wrong predictions (sentence, true, predicted, confidence),
    highest confidence first.
- **Download Excel** with the same sheets the script makes now: Summary,
  Per intent, Confusion matrix, Wrong predictions, All predictions.
- A dropdown of earlier runs (date and folds).
- Admins only (staff or superuser), like the rest of the page.

## Non-goals

- Training or deploying a new live chatbot model. The evaluation trains
  throw-away models inside cross-validation only; the running chatbot is not
  touched.
- Story / dialogue (TEDPolicy) evaluation. NLU intents only.
- Importing the 12 November results. History starts with the first run on the page.
- Editing the training sentences from the page.

## Anti-goals

- Slowing the live chatbot for students while an evaluation runs.
- Opening the Rasa HTTP API (`--enable-api`) or giving any container the
  Docker socket.

## How it works

### The data that is evaluated

The server's own chatbot folder, `RasaFaith/RasaFaith` (the same folder the
live `rasa` service mounts): `data/nlu.yml` and `config.yml`. Each run stores a
copy of the `nlu.yml` text and its SHA-256, so a result always says which
sentences it was measured on.

### Job record (Django)

A new model in the `categorization` app (migration `0002`), e.g.
`ChatbotEvaluation`, following `TrainingRun`:

- `started_by` (FK, SET_NULL), `created_at`, `started_at`, `finished_at`
- `status`: queued, running, finished, failed; `progress` text; `error` text
- `folds` (3 or 5)
- `nlu_text`, `nlu_sha256`, `rasa_version`, `fallback_threshold`
- `report` (Rasa `intent_report.json`), `errors` (`intent_errors.json`),
  `successes` (`intent_successes.json`): JSON fields
- `log_tail`: the last ~300 lines of Rasa's console output (holds the per-fold
  mean (std) lines and helps when a run fails)
- the same "only one active run" and "stale after N hours" rules as
  `TrainingRun` (N = 3 hours here), so a lost run never blocks the button.

### Worker (new container)

A new compose service, `rasa-eval`:

- image `rasa/rasa:3.6.0` (same as the live chatbot, so the same Rasa
  version); it already has `psycopg2`, so no extra packages;
- entrypoint `python /worker/worker.py`; the script lives in
  `deploy/rasa-eval/worker.py`, mounted read-only;
- mounts `./RasaFaith/RasaFaith` read-only;
- database credentials from the same variables `rasa_actions` uses;
- `cpus: "2"` so the evaluation cannot take the CPU from the live chatbot;
- `restart: unless-stopped`.

The worker loop:

1. Every 10 seconds, claim the oldest queued row atomically
   (`UPDATE ... SET status='running' WHERE id = (SELECT id ... WHERE
   status='queued' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
   RETURNING ...`).
2. Copy `data/nlu.yml` and `config.yml` into a fresh temporary folder (Rasa
   writes a cache next to its data, and the mount is read-only).
3. Run `rasa test nlu --nlu data/nlu.yml --config config.yml
   --cross-validation --folds N --successes --out out` there, capturing the
   console output; update `progress` as folds go by.
4. On success: write the three JSON files, `log_tail`, `nlu_text`,
   `nlu_sha256`, `rasa_version`, `fallback_threshold`, `finished_at`,
   status finished. On any error: status failed, a short error, `log_tail`.
5. Delete the temporary folder.

The worker stays thin: it moves files into the database. All reading of the
results (tables, matrix, charts, Excel) happens in Django.

### Django side

- `report` module: from the stored JSON build the summary, per-intent rows,
  confusion matrix, prediction rows, wrong rows and confidence buckets. Port
  the logic of `export_intent_report.py` (that script is not in git; its
  logic is described here and in the plan).
- Views: the `chatbot` tab inside the existing dashboard view; a POST
  endpoint to queue a run (folds 3 or 5 only); the status endpoint reports
  the active evaluation too, so the page refreshes when it finishes; an
  Excel export endpoint.
- Excel: openpyxl, same sheet names as the script, every cell stored as
  text/number only (sentences are typed by people; reuse the "never a
  formula" rule of the categorization export).
- Template: the new tab with ECharts, same look, dark mode and phone layout
  rules as the other tabs. No emojis.

### Fallback threshold

Read from `config.yml` when it sets a `FallbackClassifier` threshold;
otherwise Rasa's default 0.3 (the current `config.yml` uses the default
pipeline).

## Local testing

Add `rasa-eval` to the `faith-xgb` test stack on the PC and run a 3-fold
evaluation from the page.

## Server deploy (after merge, not part of this work)

Pull, then `docker compose up -d --build web celery rasa-eval` (web applies
the migration).

## Calls made by Claude (check these)

- The tab lives on the Student categorization page and in the same branch.
- The model lives in the `categorization` app instead of a new app.
- Hand-off through a database table polled by the worker, not Celery or Redis:
  the Rasa image has no Celery, and the table also keeps the history.
- CPU cap of 2 for the worker (the server's core count is not known here).
- Stale after 3 hours.
- Only NLU, not stories.
- A queued run that no worker has picked up shows "Waiting for the chatbot
  evaluation worker" (for example when `rasa-eval` is not running yet); it
  fails as stale after 3 hours like any lost run.
- Chatbot evaluations and categorization training are independent: one of
  each may run at the same time.
- Every run is kept (each is a few hundred kilobytes); no automatic clean-up.

## Left to the builder

- How progress is read from Rasa's output (fold lines), as long as the
  admin sees something move.
- Exact chart details, within the existing page's style.
- Testing approach for the worker script (it runs outside Django).
