# Feedback Form Translations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feedback forms carry hand-typed translations and are shown in the scenario's language, with answers from every language pooled per question.

**Architecture:** JSON `translations` fields on `FeedbackForm` and `FeedbackQuestion`, a `source_language` on the form and a `language` on each response. `serialize_form` picks the scenario's language field by field; popups send choice positions and the server stores the original wording. Saving updates questions in place instead of delete-and-recreate.

**Tech Stack:** Django (feedback app), plain JS in Django templates, SQLite test settings.

**Spec:** `docs/superpowers/specs/2026-11-24-feedback-form-translations-design.md`

## Global Constraints

- Branch `feature/feedback-form-translations` off `main`, in a worktree. Not on the interface-translation branch.
- `main` has no `localization` app and no gettext in the feedback app: user-facing strings stay plain English, matching the surrounding code.
- Languages: exactly en, es, el, fr, de, ro, pt. Do not add others.
- No emojis in the UI; plain labels or `bi-*` icons.
- A choice answer is always stored as the original option wording in `FeedbackAnswer.answer_text`.
- No save may delete an answer except by removing its question.
- Commit messages: plain sentences in the repo's style, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Running tests

From `C:/wt/feedback-translations/Trust-AI-Platform`:

```
timeout 1800 "C:/Users/Nikos A. Grammatikos/Desktop/DjangoFaithDSpace2025-performance-test/djangofaithvenv/Scripts/python.exe" manage.py test feedback --settings=faithDev.settings_test --noinput
```

(The worktree's own venv is stale; use the main checkout's interpreter.)

## Review Focus

1. **Numeric option wording.** Likert options `1`…`5`: position `0` must store `"1"`, and a legacy string `"5"` must store `"5"`. Positions are JSON integers, wording is a JSON string; never guess from a string. Test: Task 2 `test_numeric_option_text_is_not_mistaken_for_position`.
2. **Source language changed after translating.** A form switched from `en` to `el` that still holds an `el` translation shows the original to a Greek scenario and records `el`, without crashing. Test: Task 1 `test_translation_in_source_language_is_ignored`.
3. **Whitespace-only translation entries.** These are saved as empty and fall back to the original; they never show a blank question. Test: Task 3 `test_whitespace_translations_are_saved_empty`.
4. **Pages opened before the deploy.** They post option wording as strings, which is still accepted. Test: Task 2 `test_legacy_text_submission_still_accepted`.
5. **Scenario language spelling.** `"GREEK"`, `" ελληνικά "` and `"el"` all map to `el`; `"Italian"` and blank map to the original. Test: Task 1 `test_language_code_accepts_code_english_and_native_names`.

---

### Task 1: Language mapping, model fields, translated serialization

**Files:**
- Create: `Trust-AI-Platform/feedback/languages.py`
- Modify: `Trust-AI-Platform/feedback/models.py`
- Create: `Trust-AI-Platform/feedback/migrations/0003_feedback_translations.py`
- Modify: `Trust-AI-Platform/feedback/utils.py`
- Test: `Trust-AI-Platform/feedback/tests.py` (new classes `FeedbackLanguageTests`, `FeedbackTranslationServingTests`)

**Interfaces:**
- Produces:
  - `feedback.languages.FEEDBACK_LANGUAGES: list[tuple[str, str, str]]`: `(code, english_name, native_name)` in order en, el, fr, de, es, ro, pt.
  - `feedback.languages.LANGUAGE_CODES: frozenset[str]`
  - `feedback.languages.language_code(name: str | None) -> str | None`: case- and whitespace-insensitive match on code, English name or native name.
  - `feedback.languages.language_name(code: str) -> str`: English name, `''` for unknown or blank.
  - Model fields: `FeedbackForm.source_language` (`CharField(max_length=8, default='en', choices=[(c, english) ...])`); `FeedbackForm.translations` and `FeedbackQuestion.translations` (`JSONField(default=dict, blank=True)`); `FeedbackResponse.language` (`CharField(max_length=8, blank=True, default='')`).
  - Translation shapes: form `{"el": {"title": str, "description": str}}`, question `{"el": {"text": str, "options": [str, ...]}}`.
  - `feedback.utils.display_language(form, scenario) -> str`
  - `feedback.utils.serialize_form(form, language: str | None = None) -> dict`: same keys as today plus `'language'`.

- [ ] **Step 1: Create the worktree and commit the docs**

```bash
cd "C:/Users/Nikos A. Grammatikos/Desktop/DjangoFaithDSpace2025-performance-test"
git -c core.longpaths=true worktree add -b feature/feedback-form-translations C:/wt/feedback-translations main
git -C C:/wt/feedback-translations config core.longpaths true
cp docs/superpowers/specs/2026-11-24-feedback-form-translations-design.md C:/wt/feedback-translations/docs/superpowers/specs/
mkdir -p C:/wt/feedback-translations/docs/superpowers/plans
cp docs/superpowers/plans/2026-11-24-feedback-form-translations.md C:/wt/feedback-translations/docs/superpowers/plans/
cd C:/wt/feedback-translations && git add docs/superpowers && git commit -m "Add the design and plan for translated feedback forms"
```

Then delete the two uncommitted copies from the co-authors checkout, so they aren't committed there by accident.

- [ ] **Step 2: Write the failing tests**

```python
class FeedbackLanguageTests(SimpleTestCase):
    def test_language_code_accepts_code_english_and_native_names(self):
        from feedback.languages import language_code
        self.assertEqual(language_code('Greek'), 'el')
        self.assertEqual(language_code('GREEK'), 'el')
        self.assertEqual(language_code(' ελληνικά '), 'el')
        self.assertEqual(language_code('pt'), 'pt')
        self.assertIsNone(language_code('Italian'))
        self.assertIsNone(language_code(''))
        self.assertIsNone(language_code(None))


class FeedbackTranslationServingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_tr', password='pass')
        mk = lambda name, lang: Scenario.objects.create(name=name, language=lang, created_by=self.user, updated_by=self.user)
        self.greek, self.italian, self.english, self.french = (
            mk('TR el', 'Greek'), mk('TR it', 'Italian'), mk('TR en', 'English'), mk('TR fr', 'French'))
        self.form = FeedbackForm.objects.create(
            title='Feedback', description='Tell us', audience='student', created_by=self.user,
            translations={'el': {'title': 'Σχόλια', 'description': ''}})
        self.q1 = FeedbackQuestion.objects.create(
            form=self.form, text='Useful?', question_type='choice', options=['Yes', 'No', 'Maybe'], order=1,
            translations={'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', '']}})
        self.q2 = FeedbackQuestion.objects.create(form=self.form, text='Comments?', question_type='text', order=2)

    def test_display_language_uses_scenario_language_when_translated(self):
        self.assertEqual(display_language(self.form, self.greek), 'el')

    def test_display_language_falls_back_to_source(self):
        for scenario in (self.italian, self.english, self.french):  # French: no fr translation
            self.assertEqual(display_language(self.form, scenario), 'en')

    def test_serialize_translates_field_by_field(self):
        data = serialize_form(self.form, 'el')
        self.assertEqual(data['language'], 'el')
        self.assertEqual(data['title'], 'Σχόλια')
        self.assertEqual(data['description'], 'Tell us')
        self.assertEqual(data['questions'][0]['text'], 'Χρήσιμο;')
        self.assertEqual(data['questions'][0]['options'], ['Ναι', 'No', 'Maybe'])
        self.assertEqual(data['questions'][1]['text'], 'Comments?')

    def test_serialize_without_language_is_original(self):
        data = serialize_form(self.form)
        self.assertEqual((data['language'], data['title']), ('en', 'Feedback'))
        self.assertEqual(data['questions'][0]['options'], ['Yes', 'No', 'Maybe'])

    def test_translation_in_source_language_is_ignored(self):
        self.form.source_language = 'el'
        self.form.save()
        self.assertEqual(display_language(self.form, self.greek), 'el')
        self.assertEqual(serialize_form(self.form, 'el')['title'], 'Feedback')
```

Add `SimpleTestCase` to the existing `django.test` import, and `display_language` and `serialize_form` to the imports from `feedback.utils`.

- [ ] **Step 3: Run the tests to make sure they fail**

Run: the feedback test command, limited to `feedback.tests.FeedbackLanguageTests feedback.tests.FeedbackTranslationServingTests`
Expected: ImportError / FAIL (no `feedback.languages`, no `display_language`).

- [ ] **Step 4: Implement**

- `feedback/languages.py` holds the constants and the two helpers. Native names: English, Ελληνικά, Français, Deutsch, Español, Română, Português. Its docstring says it's to be replaced by `localization.languages.code_for_table_name` when the interface translation merges.
- Add the model fields.
- `display_language`: let `code = language_code(scenario.language)`. Return `code` if it's in `LANGUAGE_CODES`, differs from `form.source_language`, and the form or any of its questions has a translation entry for it with at least one non-blank value. Otherwise return `form.source_language`.
- `serialize_form`: if `language` is falsy or equals `form.source_language`, serve the originals. Otherwise, for each field, use the stripped translation if it's non-blank, or else the original. Options are matched by position, and a missing or blank option falls back to the original. Set `'language'` to the language actually requested if it was applied, or else to `form.source_language`.

- [ ] **Step 5: Create the migration**

Run `manage.py makemigrations feedback --name feedback_translations --settings=faithDev.settings_test`. Then add a `RunPython` step that sets `language=form.source_language` on every existing `FeedbackResponse`, with a no-op reverse.

- [ ] **Step 6: Run the feedback tests to make sure they pass**

Run: the feedback test command
Expected: all pass, including the existing `test_serialize_form_shape`.

- [ ] **Step 7: Commit**

```bash
git add Trust-AI-Platform/feedback
git commit -m "Store feedback form translations and serve forms in the scenario's language"
```

---

### Task 2: Popups send positions; submission stores the original wording and the language

**Files:**
- Modify: `Trust-AI-Platform/feedback/views.py` (`submit_feedback`, `_clean_answer`)
- Modify: `Trust-AI-Platform/studentview/views.py:107-110`
- Modify: `Trust-AI-Platform/authoringtool/views.py:4077-4080`
- Modify: `Trust-AI-Platform/studentview/templates/studentview/scenarioView.html` (popup script)
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/proposal_list.html` (popup script)
- Test: `Trust-AI-Platform/feedback/tests.py` (`FeedbackSubmitEndpointTests`, `StudentFeedbackTriggerTests`, `TeacherFeedbackTriggerTests`)

**Interfaces:**
- Consumes: `display_language`, `serialize_form(form, language)` from Task 1.
- Produces: the submit payload contract `{"answers": {"<question id>": <int position> | "<text>"}}`. Choice positions are JSON integers, free text is a string, and a legacy choice is its original wording as a string.

- [ ] **Step 1: Write the failing tests** (in `FeedbackSubmitEndpointTests`)

In `setUp`, add a `self.greek` scenario (`language='Greek'`) and give `q_choice` the translation `{'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', 'Όχι']}}`. Add a helper `_submit_to(scenario, form, answers)`.

```python
def test_choice_by_position_stores_original_wording_and_language(self):
    self.client.login(username='fb_student', password='pass')
    r = self._submit_to(self.greek, self.student_form, {str(self.q_choice.id): 1})
    self.assertTrue(r.json()['success'])
    response = FeedbackResponse.objects.get(user=self.student, scenario=self.greek)
    self.assertEqual(response.language, 'el')
    self.assertEqual(response.answers.get(question=self.q_choice).answer_text, 'No')

def test_numeric_option_text_is_not_mistaken_for_position(self):
    q = FeedbackQuestion.objects.create(form=self.student_form, text='Rate', question_type='choice',
                                        options=['1', '2', '3', '4', '5'], is_required=False, order=3)
    self.client.login(username='fb_student', password='pass')
    self._submit_to(self.scenario, self.student_form, {str(self.q_choice.id): 0, str(q.id): 0})
    self.assertEqual(FeedbackAnswer.objects.get(question=q).answer_text, '1')
    FeedbackResponse.objects.all().delete()
    self._submit_to(self.scenario, self.student_form, {str(self.q_choice.id): 0, str(q.id): '5'})
    self.assertEqual(FeedbackAnswer.objects.get(question=q).answer_text, '5')

def test_legacy_text_submission_still_accepted(self):
    self.client.login(username='fb_student', password='pass')
    r = self._submit_to(self.greek, self.student_form, {str(self.q_choice.id): 'Yes'})
    self.assertTrue(r.json()['success'])
    self.assertEqual(FeedbackAnswer.objects.get(question=self.q_choice).answer_text, 'Yes')

def test_bad_positions_rejected(self):
    self.client.login(username='fb_student', password='pass')
    for bad in (2, -1, True, 1.0, 'Ναι'):
        r = self._submit_to(self.scenario, self.student_form, {str(self.q_choice.id): bad})
        self.assertEqual(r.status_code, 400, bad)
    self.assertFalse(FeedbackResponse.objects.exists())

def test_required_error_quotes_question_in_shown_language(self):
    self.client.login(username='fb_student', password='pass')
    r = self._submit_to(self.greek, self.student_form, {})
    self.assertIn('Χρήσιμο;', r.json()['error'])
```

In `StudentFeedbackTriggerTests` and `TeacherFeedbackTriggerTests`, using each class's existing setup with a Greek scenario and a translated form:

```python
def test_popup_is_served_in_scenario_language(self):
    ...
    data = json.loads(response.context['feedback_form_json'])
    self.assertEqual((data['language'], data['title']), ('el', 'Σχόλια'))
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: the feedback test command
Expected: the new tests FAIL. Positions are rejected as invalid, and the trigger tests get the original title.

- [ ] **Step 3: Implement**

- `_clean_answer(answers, question) -> str | None` returns the stored text: `''` if there's no answer, or `None` if it's invalid. For a choice question, an `int` that isn't a `bool` and is in range gives `question.options[i]`, and a `str` whose stripped value is in `question.options` gives that value. For a text question, it takes a `str`, stripped.
- In `submit_feedback`, compute `language = display_language(form, scenario)` and `shown = serialize_form(form, language)`. Error messages quote the question text from `shown`, and the response is created with `language=language`.
- Both call sites: `serialize_form(fb_form, display_language(fb_form, scenario))`.
- Both popups: set `input.value = String(i)`, and send `parseInt(el.value, 10)` for choice answers.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: the feedback test command
Expected: all pass. `test_student_can_submit_student_form` is unchanged and still passes through the legacy path.

- [ ] **Step 5: Commit**

```bash
git add Trust-AI-Platform/feedback Trust-AI-Platform/studentview Trust-AI-Platform/authoringtool
git commit -m "Record feedback choices in their original wording, whatever language the form was shown in"
```

---

### Task 3: Save forms in place, with translations and locks on answered questions

**Files:**
- Modify: `Trust-AI-Platform/feedback/views.py` (`_parse_questions_json`, `_save_form_from_post`, `feedback_form_edit` GET context)
- Test: `Trust-AI-Platform/feedback/tests.py` (`FeedbackManagementViewTests`; replaces `test_edit_replaces_questions`)

**Interfaces:**
- Consumes: `LANGUAGE_CODES` from Task 1.
- Produces, as the POST contract with the editor (Task 4):
  - `source_language`: one of `LANGUAGE_CODES`.
  - `form_translations_json`: `{"el": {"title": str, "description": str}}`.
  - `questions_json`: a list of `{"id": int | null, "text", "type", "options", "required", "translations": {"el": {"text": str, "options": [str]}}}`.
  - The GET edit context's `questions_json` items also carry `"answer_count": int`. The context gains `form_translations_json` (str), `source_language` (str) and `languages` (`FEEDBACK_LANGUAGES`).
- Error copy (exact):
  - `'Question "{text}" already has answers, so its type and options can\'t be changed. Remove it and add a new question instead.'`
  - `'Unknown translation language: {code}.'`
  - `'The {english_name} translation of "{text}" has more options than the original.'`
  - `'Invalid original language.'`
  - existing `'Invalid question data.'` for foreign or unknown ids

- [ ] **Step 1: Write the failing tests**

Add a helper `_post_edit(form, questions, form_translations=None, source_language='en')` that posts title, audience, `is_active`, `assign_to_all` and the JSON fields.

```python
def test_edit_keeps_answers_when_translating(self):
    form, q, answer = self._answered_form()          # choice Yes/No with one answer 'No'
    self._post_edit(form, [{'id': q.id, 'text': 'Useful?', 'type': 'choice', 'options': ['Yes', 'No'],
                            'required': True, 'translations': {'el': {'text': 'Χρήσιμο;', 'options': ['Ναι', 'Όχι']}}}],
                    form_translations={'el': {'title': 'Σχόλια', 'description': ''}})
    q.refresh_from_db(); form.refresh_from_db()
    self.assertTrue(FeedbackAnswer.objects.filter(pk=answer.pk).exists())
    self.assertEqual(q.translations['el']['options'], ['Ναι', 'Όχι'])
    self.assertEqual(form.translations['el']['title'], 'Σχόλια')

def test_edit_creates_new_and_deletes_missing_questions(self):
    # q_keep and q_drop each answered once; post [q_keep with id, a new question without id]
    # -> q_drop and its answer are gone, q_keep's answer stays, texts == ['Keep', 'New']

def test_answered_question_options_and_type_are_locked(self):
    # posting options ['Yes','No','Maybe'], then separately type 'text', for the answered q
    # -> status 200, 'already has answers' in content, q.options still ['Yes','No'], answer still there

def test_unanswered_question_options_can_change(self):
    # same post on a question with no answers -> options saved

def test_foreign_question_id_rejected(self):
    # id of another form's question -> 'Invalid question data.', that question unchanged

def test_translation_validation(self):
    # translations {'it': {...}} -> 'Unknown translation language: it.'
    # el options ['Ναι','Όχι','Ίσως'] for a 2-option question -> 'The Greek translation of "Useful?" has more options than the original.'

def test_whitespace_translations_are_saved_empty(self):
    # el text '   ', options [' ', 'Όχι'] -> stored {'text': '', 'options': ['', 'Όχι']}

def test_source_language_saved_and_validated(self):
    # create with source_language 'el' -> form.source_language == 'el'; 'xx' -> 'Invalid original language.'

def test_edit_page_lists_ids_translations_and_answer_counts(self):
    # GET edit -> json.loads(response.context['questions_json'])[0] has id == q.id, translations, answer_count == 1
```

Delete `test_edit_replaces_questions`.

- [ ] **Step 2: Run the tests to make sure they fail**

Run: the feedback test command
Expected: the new tests FAIL. Answers are erased, and `translations`, `source_language` and `answer_count` are missing.

- [ ] **Step 3: Implement**

- `_clean_translations(raw, fields: tuple[str, ...], option_count: int | None, label: str) -> tuple[dict | None, str | None]` is shared by the form and the questions. It strips every string. Option lists longer than `option_count` are an error. Entries whose values are all empty are dropped.
- `_parse_questions_json` also returns `id` and the cleaned `translations` for each question.
- In `_save_form_from_post`, validate before writing anything. Check `source_language`. Every non-null `id` must belong to `form`. For each id with answers, `type` and `options` must equal the stored values. Then, inside the existing transaction, update the questions that have ids, create the ones without, delete the form's questions that weren't posted, and set `order` from the list position.
- Edit GET builds `questions_json` with an answer count per question (one `annotate(Count('answers'))` query).

- [ ] **Step 4: Run the tests to make sure they pass**

Run: the feedback test command
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Trust-AI-Platform/feedback
git commit -m "Save feedback forms in place so adding translations never erases answers"
```

---

### Task 4: Editor tabs for translations

**Files:**
- Modify: `Trust-AI-Platform/feedback/templates/feedback/form_edit.html`
- Test: `Trust-AI-Platform/feedback/tests.py` (`FeedbackManagementViewTests`)

**Interfaces:**
- Consumes: the Task 3 POST contract and GET context.

- [ ] **Step 1: Write the failing tests**

```python
def test_edit_page_has_language_controls_and_new_banner(self):
    form, q, _ = self._answered_form()
    content = self.client.get(reverse('feedback_form_edit', args=[form.id])).content.decode()
    self.assertIn('name="source_language"', content)
    self.assertIn('name="form_translations_json"', content)
    self.assertIn('data-lang="el"', content)
    self.assertIn('Removing a question erases its answers', content)
    self.assertNotIn('permanently erase the answers already collected', content)
```

- [ ] **Step 2: Run the test to make sure it fails**

Expected: FAIL on `source_language`.

- [ ] **Step 3: Implement**

- **Original language.** An "Original language" `<select name="source_language">` next to Title, built from `languages`.
- **Tabs.** A tab strip above Title: "Original", then one tab per language other than the source, each a button with `data-lang="<code>"` labelled with the native name. Changing the original language rebuilds the strip.
- **Translation state.** Each question row keeps its translations and `answer_count` on the row element (for example `row._translations`). Form translations live in one JS object.
- **Translation tabs.** Opening a tab hides the Original body and renders a panel from the *current* rows in order: the title and description, then each question with its original text shown above a text input, and for choice questions one input per original option with that option as placeholder. Inputs write back into the row's or the form's translation object on `input`. Rendering from the current rows means questions added on the Original tab show up.
- **Empty count.** Each tab label shows how many of its fields are empty, for example "Ελληνικά (3 empty)", and refreshes as you type.
- **Answered rows.** The type select is `disabled` and the options textarea `readonly`. A muted note reads "Has answers: type and options are locked." Remove asks `confirm('Removing this question erases its N answers. Continue?')`.
- **Banner.** The warning banner now reads: "This form already has responses. Removing a question erases its answers; editing text and translations keeps them."
- **Submit.** The submit handler adds `id` and `translations` to each question and fills `form_translations_json`.
- **After a validation error.** Re-rendering keeps the posted `questions_json` and `form_translations_json`; pass `form_translations_json` and `source_language` back in the error branches of create and edit.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: the feedback test command
Expected: all pass.

- [ ] **Step 5: Check it in the browser**

Use the `run` skill to start the app and open the editor. Create a form with two choice questions. Fill the Greek tab, leaving one option empty, and save. Reopen the form and check that the Greek text is still there and the empty count shows 1. Add a question on the Original tab and check that it appears on the Greek tab. Answer the form from a Greek scenario as a student, then check that type and options are locked in the editor.

- [ ] **Step 6: Commit**

```bash
git add Trust-AI-Platform/feedback
git commit -m "Add language tabs to the feedback form editor"
```

---

### Task 5: Language on the responses page and in exports

**Files:**
- Modify: `Trust-AI-Platform/feedback/views.py` (`_export_rows`, `feedback_form_responses`)
- Modify: `Trust-AI-Platform/feedback/templates/feedback/form_responses.html`
- Test: `Trust-AI-Platform/feedback/tests.py` (`FeedbackResponsesAndExportTests`)

**Interfaces:**
- Consumes: `FeedbackResponse.language` and `language_name` from Task 1.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_has_language_column_and_pools_answers(self):
    # one 'en' and one 'el' response, both answering q with stored 'No'
    rows = _export_rows(self.form)
    self.assertEqual(rows[0][:4], ['Username', 'Scenario', 'Submitted', 'Language'])
    self.assertEqual(sorted(r[3] for r in rows[1:]), ['el', 'en'])
    self.assertEqual([r[4] for r in rows[1:]], ['No', 'No'])

def test_responses_page_shows_language(self):
    # el response -> 'Greek' in the page
```

Update the existing export tests if they index answer columns by position.

- [ ] **Step 2: Run the tests to make sure they fail**

Expected: FAIL. There's no Language column yet.

- [ ] **Step 3: Implement**

- `_export_rows` inserts `response.language` after Submitted.
- In the responses page, the meta line becomes `scenario · date · Greek`, with the language part left out when the language is blank.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: the feedback test command
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Trust-AI-Platform/feedback
git commit -m "Show which language each feedback response was given in"
```

---

### Task 6: Whole-branch verification

- [ ] **Step 1:** `manage.py makemigrations --check --dry-run --settings=faithDev.settings_test` should report no changes.
- [ ] **Step 2:** Run the full suite with the same interpreter and settings. Compare it to `main` by test name: the only failures should be the known ones (2 NumericRange, `test_task_starter_can_read_result`, pinned version, 2 proposal history).
- [ ] **Step 3:** Use superpowers:finishing-a-development-branch. The server deploy needs a migrate (feedback 0003) and a restart of `web`. Celery isn't involved.
