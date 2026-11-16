# Proposal generation progress on Metrics & AI

Date: 2026-11-16 · Branch: `feature/proposal-progress` (from `main` ebae276f)

## Why

Generating proposals takes minutes and the page only said "Still generating…".
Teachers want to see what the job is doing (which activity, which action) and how
far along it is. A reload lost even that, and a second teacher could start a
second run on the same scenario.

## What (user decisions, 2026-11-16)

- A progress bar with the job's stages in order, each ticked off with a short
  summary, and the current item: activity name, chosen action, step
  (choosing, writing, translating).
- The server remembers the running job per scenario: a reload or another teacher
  sees the same progress ("started by X"), and Generate is locked until it ends.

## Design

- `authoringtool/proposal_progress.py`: one record per scenario in the shared
  Django cache (Redis in production; Celery and web both reach it).
  `reserve()` (view, before queueing), `JobProgress` (the Celery job:
  `stage`, `step`, `detail`, `summary`, `finish`), `job_status()` (readers: adds
  percent, the stage list and whether it still runs).
- Stages and weights (time share): check 2, PDFs 4, activity summaries 25,
  scenario/phase summaries 6, learning paths 15, risk detection 4, proposals 40,
  saving 4. A stage with nothing to do is ticked "already up to date".
- A record not updated for 30 minutes counts as stuck: it no longer locks
  Generate (a worker restart must not lock a scenario for hours).
- Generate while a job runs: 409 with the starter's name; the page then follows
  that job. New endpoint `proposal_job_status/<scenario>` (view rights needed).
- Page: progress panel for everyone who can open Metrics & AI; it polls every
  3 seconds while a job runs; the Generate buttons stay disabled meanwhile.
  Text from the job is inserted as text, never HTML.

## Builder's calls

- Only a running job is shown on page load; a finished one is reported to the
  tab that followed it (the new proposals are the lasting result).
- The old task-id status endpoint stays for compatibility.
