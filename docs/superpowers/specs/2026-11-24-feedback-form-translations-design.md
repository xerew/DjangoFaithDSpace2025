# Feedback form translations

Date: 2026-11-24
Status: draft, awaiting approval

## Why

Scenarios exist in several languages. Feedback forms are written once, in one
language, and every scenario shows that text. The team now has to run feedback
forms in two scenarios of different languages, and needs the answers from both
analysed together.

## Goals

- One feedback form can carry hand-written translations of its title,
  description, question texts and answer options.
- Students and teachers see the form in the scenario's language; anything not
  translated falls back to the original text, field by field.
- Answers from every language pool per question: one column per question in
  the responses page and export, and choice answers recorded in the original
  wording.
- Adding or correcting a translation never erases answers already collected.

## Non-goals

- Machine-translated drafts ("Translate with AI"). Translations are typed by
  hand; the button can come later.
- Translating free-text answers. They stay as people wrote them.
- Languages outside the platform's seven (en, es, el, fr, de, ro, pt).
- Showing the form in the viewer's interface language. The scenario's
  language decides.

## Anti-goals

- Any save that silently deletes or changes collected answers.
- A Greek and an English answer to the same question ending up in different
  export columns or with different wording for the same choice.

## Constraints

- Built on `main`, on its own branch, so it can deploy before the interface
  translation (`feature/interface-translation`, not yet on `main`). The two
  forms need to run soon. When the interface translation merges, the
  conflicts in `feedback/views.py` and the two popup templates are resolved
  then (user's decision).
- Existing forms and responses keep working with no admin action.

## Current state

- `FeedbackForm` / `FeedbackQuestion` store plain text; `options` is a list of
  strings (`feedback/models.py`).
- Every save deletes all questions and recreates them
  (`feedback/views.py`, `_save_form_from_post`), which cascades to
  `FeedbackAnswer`. The editor warns about this.
- The popup (student: `studentview/templates/studentview/scenarioView.html`;
  teacher: `authoringtool/templates/authoringtool/proposal_list.html`) sends the
  chosen option's text, and `submit_feedback` checks it against
  `question.options`.
- `Scenario.language` is free text ("Greek", "Ελληνικά", "el");
  `localization.languages.code_for_table_name` maps it to one of the seven
  codes, or `None`.

## Approaches

- **A. JSON `translations` field on the form and each question (chosen).**
  One migration, no new tables; the editor reads and writes it directly;
  options line up with the original options by position.
- B. Separate translation tables, one row per question per language. Cleaner
  to query by language, but two more models and more code to keep in sync for
  seven languages and a handful of questions.
- C. django-modeltranslation (`text_el`, `text_fr` columns). New dependency,
  fits the options list badly, and a migration for every new language.
  Dropped.

The user picked A.

## Design

### Data

- `FeedbackForm.source_language`: language code of the original text,
  default `'en'`, limited to the seven codes. *(Claude's call: an admin may
  write the original in Greek; without this there is no English tab.)*
- `FeedbackForm.translations`: `JSONField(default=dict, blank=True)`, shape
  `{"el": {"title": "...", "description": "..."}}`.
- `FeedbackQuestion.translations`: same, shape
  `{"el": {"text": "...", "options": ["...", "..."]}}`. `options[i]`
  translates the original `options[i]`; a missing or blank entry falls back to
  the original.
- `FeedbackResponse.language`: code of the version the person saw, blank
  allowed. Responses given before this change stay blank and are shown and
  exported as their form's current `source_language`, so correcting a form's
  original language also corrects its older responses. *(Changed after the
  final review: a one-time backfill would have stamped every old response
  `en`, even for forms written in another language.)*

### Choosing the language

`display_language(form, scenario)`: `code_for_table_name(scenario.language)`;
if that is `None` or equals `form.source_language`, the original is shown.
`serialize_form(form, language)` returns each field translated where a
non-blank translation exists, otherwise the original. Both popups call it
with the scenario.

### Popup and submission

- Radio buttons send the option's position (`"0"`, `"1"`, ...) instead of its
  text, in both popups.
- `submit_feedback` accepts a position and stores the original wording
  `question.options[i]` in `FeedbackAnswer.answer_text`, so stored answers are
  identical in every language. For pages loaded before the deploy it also
  accepts the exact original wording. *(Claude's call.)*
- The response records `language = display_language(form, scenario)`.
- Error messages that quote a question ("Question ... is required") quote it
  in the language the person saw.
- The popup's own buttons (Skip, Submit, Close) and server error messages are
  not part of this work: on `main` they are plain English. Once the interface
  translation merges, they should use the form's display language (not the
  viewer's interface language), so a Greek form never has English buttons.
  That is a follow-up for the merge. *(Claude's call.)*

### Saving in place

- The editor sends each existing question's `id` in `questions_json`.
- On save, questions with an `id` belonging to this form are updated; those
  without an `id` are created; questions of the form missing from the payload
  are deleted (with their answers). Unknown or foreign ids are an error.
- For a question that already has answers *(Claude's call)*:
  - question text, required flag, order and all translations stay editable;
  - its type and its original options (count and wording) are locked; the
    editor shows them read-only with a short note, and the server rejects a
    change. To change them, the admin deletes the question and adds a new one.
- The editor's warning banner changes from "saving erases all answers" to
  "removing a question erases its answers".

### Editor

- A "Original language" select next to Title.
- Language tabs above the form body: "Original" plus the other six languages.
  The Original tab is today's editor. Each translation tab lists title,
  description, then every question with the original text shown above an
  input for the translation, and for choice questions one input per option
  with the original as placeholder.
- Each translation tab shows how many fields are still empty, so a missing
  translation is visible before the form goes live.
- Translations travel in `questions_json` (per question) and a new
  `form_translations_json` hidden field (title, description). Server-side
  validation: language keys must be among the seven; a translated options
  list may not be longer than the original.

### Responses page and export

- New "Language" column after "Submitted" (code, e.g. `el`; blank for
  none).
- Question headers and choice answers use the original wording.

## Testing

- Serialization: full translation, partial translation (field-by-field
  fallback), unknown scenario language, scenario language equal to the
  source language.
- Submission: by position in a translated form stores the original wording
  and the language; old-style text submission still accepted; bad positions
  rejected.
- Saving: editing translations or question text keeps existing answers;
  removing a question deletes only its answers; changing type or options of an
  answered question is rejected; foreign question ids rejected.
- Export: Language column present, answers from two languages in one column.
- `test_edit_replaces_questions` is rewritten for in-place saving.

## Left to the builder

- Whether the popup script, duplicated in the two templates, moves to one
  static file.
- Exact layout and wording of the tabs and the empty-field count.
- Showing which languages a form has on the forms list (nice to have).
