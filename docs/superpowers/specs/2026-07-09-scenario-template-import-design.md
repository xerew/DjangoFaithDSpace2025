# Scenario Template Import — Design

**Date:** 2026-07-09
**Branch:** `improvements/performance-and-responsive`

---

## Overview

Teachers can create a new scenario by uploading a pre-formatted Excel (`.xlsx`) file instead of building it manually in the authoring tool. The system validates the file fully, reports all errors before saving anything, then creates the complete scenario — phases, activities, answers, answer feedback, next-activity routing, evaluation groups, and score-based branches — in one atomic operation.

A downloadable blank template (with instructions, dropdown validation, and example rows) is provided so teachers do not need to figure out column names themselves.

---

## Entry Point (UI)

The existing "Create New Scenario" button in the authoring tool opens a modal with two tabs:

- **Upload Template** — file picker (`.xlsx` only), "Download blank template" link, Import button
- **Manual Setup** — existing creation flow, unchanged

On import:
- A spinner replaces the Import button while the request is in flight
- On error: the modal body becomes a scrollable error list grouped by sheet + row
- On success: modal closes, page redirects to the new scenario in the authoring tool

---

## Template File Structure (7 sheets)

### Sheet 1: README
Plain-text instructions. Contains:
- Column descriptions for each sheet
- Referencing rules: "Activity names must match exactly — copy-paste from the Activities sheet to avoid typos"
- Example rows for Next Activity and Evaluation sheets
- Notes on markdown support in text fields

### Sheet 2: Scenario
Single data row. All scenario metadata.

| Column | Required | Notes |
|---|---|---|
| Name | Yes | Must be unique across all scenarios in the system |
| Description | No | Markdown supported |
| Learning Goals | No | Markdown supported |
| Language | No | Free text (e.g. English, Greek) |
| Subject Domains | No | Comma-separated (e.g. Physics, STEM) |
| Age Min | No | Integer |
| Age Max | No | Integer |
| Suggested Time (min) | No | Integer, minutes |
| Video URL | No | |
| Visibility | No | `private` / `org` / `public` (default: `private`) |

### Sheet 3: Phases
One row per phase. Row order determines phase order.

| Column | Required | Notes |
|---|---|---|
| Phase Name | Yes | Used as reference in Activities sheet |
| Description | No | Markdown supported |
| Video URL | No | |

### Sheet 4: Activities
One row per activity. Row order determines activity order within each phase.

| Column | Required | Notes |
|---|---|---|
| Activity Name | Yes | Must be unique within the file; used as reference everywhere |
| Phase Name | Yes | Must match a Phase Name from the Phases sheet |
| Text | Yes | Markdown supported; converted to HTML on import |
| Activity Type | No | Must match an existing ActivityType name in the DB |
| Helper | No | Free text |
| Is Evaluatable | No | `Yes` / `No` (default: `No`) |
| Is Primary Evaluation | No | `Yes` / `No` (default: `No`); requires Is Evaluatable = Yes |
| Must Wait | No | `Yes` / `No` (default: `No`) |
| Score Limit | No | Float (default: `0`) |
| Simulation Name | No | Must match an existing Simulation name in the DB |
| Remote Lab Name | No | Must match an existing ExperimentLL name in the DB |
| VR Lab Name | No | Must match an existing VRARExperiment name in the DB |
| Image URL | No | |
| Video URL | No | |

### Sheet 5: Answers
One row per answer. Multiple rows per activity allowed.

| Column | Required | Notes |
|---|---|---|
| Activity Name | Yes | Must match an Activity Name from the Activities sheet |
| Answer Text | Yes | Markdown supported |
| Is Correct | No | `Yes` / `No` (default: `No`) |
| Answer Weight | No | Integer (default: `0`) |
| Image URL | No | |
| Video URL | No | |
| Feedback Text | No | Markdown supported; creates AnswerFeedback record if present |
| Feedback Image URL | No | |
| Feedback Video URL | No | |

### Sheet 6: Next Activity
One row per routing rule. Handles `NextQuestionLogic`.

| Column | Required | Notes |
|---|---|---|
| Source Activity Name | Yes | Must match an Activity Name |
| Answer Text | No | Leave blank for the default (unconditional) next activity; otherwise must match an Answer Text for this activity |
| Next Activity Name | No | Leave blank to end the scenario; otherwise must match an Activity Name |

Each `(Source Activity Name, Answer Text)` pair must be unique within this sheet.

### Sheet 7: Evaluation
One row per evaluatable activity. Handles `QuestionBunch` + `EvQuestionBranching`.

| Column | Required | Notes |
|---|---|---|
| Primary Activity Name | Yes | Must match an Activity Name with Is Evaluatable = Yes |
| Grouped Activities | Yes | Comma-separated Activity Names that form the scoring bunch (include the primary itself) |
| High Branch Activity | No | Activity Name for the high-score branch |
| High Branch Feedback | No | Markdown supported |
| Mid Branch Activity | No | Activity Name for the mid-score branch |
| Mid Branch Feedback | No | Markdown supported |
| Low Branch Activity | No | Activity Name for the low-score branch |
| Low Branch Feedback | No | Markdown supported |

---

## Blank Template Download

**Route:** `GET /authoringtool/template/download/` — any authenticated user

The view generates (or serves a cached) `.xlsx` with:
- All 7 sheets with correct names
- Row 1 headers: bold, light blue background
- Required columns marked with `*` in the header
- Boolean columns have data-validation dropdowns (`Yes` / `No`) applied to rows 2–200
- Column widths auto-sized to fit header text
- README sheet populated with the full reference guide and example rows

---

## Import Pipeline

**Route:** `POST /authoringtool/import/` — any authenticated user

Implemented as a service class `ScenarioImporter` (in `authoringtool/importer.py`). The view calls it and returns JSON.

### Stage 1 — Parse
Read all 6 content sheets with `openpyxl`. Skip rows where all cells are blank. Build in-memory structures:
- `scenario_data` — dict of field → value
- `phases` — list of dicts, in row order
- `activities` — list of dicts, in row order; build `name → dict` lookup
- `answers` — list of dicts; build `(activity_name, answer_text) → dict` lookup
- `routing` — list of dicts
- `evaluation` — list of dicts

### Stage 2 — Validate (collect ALL errors before touching the DB)

Errors carry `sheet`, `row`, `column`, and `message`. All errors are collected; validation does not stop at the first failure.

**Structural:**
- All 7 sheets present (by name, case-insensitive)
- All required columns present per sheet

**Field-level:**
- Required fields not empty
- Boolean fields are `yes` / `no` (case-insensitive)
- Numeric fields (`Age Min`, `Age Max`, `Suggested Time`, `Score Limit`, `Answer Weight`) are valid numbers
- `Visibility` is one of `private`, `org`, `public`
- `(Source Activity, Answer Text)` pairs in Next Activity sheet are unique

**Cross-reference:**
- Phase Names in Activities sheet all exist in Phases sheet
- Activity Names in Answers, Next Activity, and Evaluation sheets all exist in Activities sheet
- Answer Texts in Next Activity sheet (non-blank) match an answer for that source activity
- Branch Activity Names in Evaluation sheet exist in Activities sheet
- Grouped Activities names in Evaluation sheet all exist in Activities sheet
- Duplicate Activity Names within the file flagged
- Scenario Name does not already exist in the DB

**Logic:**
- Activities with `Is Primary Evaluation = Yes` also have `Is Evaluatable = Yes`
- Every activity with `Is Evaluatable = Yes` has exactly one row in the Evaluation sheet

**DB existence checks:**
- Simulation Name (if given) matches an existing `Simulation.name`
- Remote Lab Name (if given) matches an existing `ExperimentLL.name`
- VR Lab Name (if given) matches an existing `VRARExperiment.name`
- Activity Type (if given) matches an existing `ActivityType.name`

### Stage 3 — Create (inside `transaction.atomic`)

If any errors were collected, raise them immediately — no DB writes.

Creation order:
1. `Scenario` — set `created_by` / `updated_by` to the uploading user
2. `Phase` objects — in row order; build `phase_name → Phase` mapping
3. `Activity` objects — in row order; convert `text` markdown → HTML (`markdown2`), strip tags for `plain_text`; build `activity_name → Activity` mapping; resolve Simulation / Lab / VR FKs by name
4. `Answer` + `AnswerFeedback` objects — in row order; convert markdown; build `(activity_name, answer_text) → Answer` mapping
5. `NextQuestionLogic` rows — resolve activity and answer names via mappings
6. `QuestionBunch` — resolve `activity_ids` list from grouped activity names
7. `EvQuestionBranching` — resolve all three branch FK activity names

### Response contracts

**Success:**
```json
{"success": true, "scenario_id": 42, "redirect": "/authoringtool/scenario/42/"}
```

**Error:**
```json
{
  "success": false,
  "errors": [
    {"sheet": "Activities", "row": 5, "column": "Activity Name", "message": "Required field is empty"},
    {"sheet": "Next Activity", "row": 12, "column": "Next Activity Name", "message": "\"Intro Activty\" does not match any activity name"},
    {"sheet": "Evaluation", "row": 3, "column": "Primary Activity Name", "message": "\"Quiz 1\" is not marked Is Evaluatable = Yes"}
  ]
}
```

---

## New Files

| File | Purpose |
|---|---|
| `authoringtool/importer.py` | `ScenarioImporter` service class — all validation and creation logic |
| `authoringtool/template_generator.py` | `generate_blank_template()` — returns an `openpyxl.Workbook` |

## Modified Files

| File | Change |
|---|---|
| `authoringtool/views.py` | Two new views: `import_scenario` (POST) and `download_template` (GET) |
| `authoringtool/urls.py` | Two new URL patterns |
| `authoringtool/templates/authoringtool/` | Modal tab added to the create-scenario entry point |

---

## Dependencies

Both are new additions to `Trust-AI-Platform/requirements.txt`:

- `openpyxl` — reads uploaded `.xlsx` files and generates the blank template
- `markdown2` — converts markdown cell content to HTML for storage in `Activity.text` and `Answer.text` / `AnswerFeedback.text`

---

## Access Control

Both routes require `@login_required`. No staff restriction — any teacher can import and download the template. The created scenario is attributed to `request.user`.

---

## Out of Scope

- Images embedded in the Excel file (images are URL references only)
- Updating an existing scenario via template (import always creates new)
- Exporting an existing scenario to template format
- LLM-assisted parsing of free-form documents (Word, PDF)
