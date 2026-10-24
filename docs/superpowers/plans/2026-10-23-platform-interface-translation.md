# Multilingual Platform Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate every fixed piece of interface text in the Trust AI Lab
platform into 13 languages. Add a flag language switcher, per-account
language memory, and reviewer corrections that need no deploy.

**Architecture:**
- A new `localization` Django app owns:
  - the language list and flags
  - the switcher and the user-language middleware
  - translation tooling, reviewer overrides and display labels for
    reference values
- Interface text is marked with Django's gettext tooling. It's extracted into
  `Trust-AI-Platform/locale/<code>/LC_MESSAGES/django.po`, translated by the
  builder, compiled to `.mo` and committed.
- Risk flags keep their English sentences, which are AI-prompt input and
  `QValue` keys. They gain a reason code plus numbers so the display can be
  rendered in any language.

**Tech Stack:** Django (4.0 in production Docker, 5.2 locally), GNU gettext
locally (`xgettext`, `msgfmt` in `C:/Program Files/gettext-iconv/bin`),
`polib`, Bootstrap 5, Celery.

**Spec:** `docs/superpowers/specs/2026-10-22-platform-interface-translation-design.md`

## Global Constraints

> **Scope change (2026-10-24):** only en, es, el, fr, de, ro and pt are offered.
> Task 12-scope reduces the language list. Task 12 then translates es, el, fr,
> de, ro and pt only. Basque stays in the repo, unused.

- **Languages: 13, in this order:**

  | Code | Native name | Table name | Flag |
  |---|---|---|---|
  | en | English | English | gb |
  | eu | Euskara | Basque | es-pv |
  | ca | Català | Catalan | es-ct |
  | et | Eesti | Estonian | ee |
  | fr | Français | French | fr |
  | de | Deutsch | German | de |
  | el | Ελληνικά | Greek | gr |
  | it | Italiano | Italian | it |
  | pt | Português | Portuguese | pt |
  | ro | Română | Romanian | ro |
  | es | Español | Spanish | es |
  | tr | Türkçe | Turkish | tr |
  | uk | Українська | Ukrainian | ua |

  Persian is deferred and doesn't appear anywhere. `LANGUAGE_CODE = 'en'`.
- **Flags are SVG images** in
  `Trust-AI-Platform/static/vendor/flag-icons/4x3/<flag>.svg`, from the
  MIT-licensed flag-icons project (copy its LICENSE alongside). No emoji
  anywhere in the UI.
- **No language prefix in URLs.**
- **What must never be wrapped for translation:** user content (scenario,
  phase, activity, answer and feedback text, group, organization and campaign
  text, notice reasons, usernames) and anything stored in the database,
  except the reference values Task 8 lists.
- **Downloaded files stay English:** CSV and Excel exports, and the import
  template whose headers `importer.py` reads back. Plato (`chatBot.html`,
  `static/chatbot_static/`) and Ollama output are not touched.
- **Marking rules for every template task:**
  - `{% load i18n %}` in each template that uses tags.
  - `{% translate "…" %}` for short text. `{% blocktranslate trimmed %}`
    when the text contains variables or sentence-level markup.
  - `context="…"` when one English word means different things (for example
    "Start" for a button vs. a date).
  - **Text inside `<script>`:** `"{% translate '…' as s %}{{ s|escapejs }}"`
    or a `json_script` value. A bare `{% translate %}` inside JS is never
    allowed (a French apostrophe would break the script).
  - `messages.*`, form labels, help texts, `choices` labels, and JSON
    `error` strings shown to users use `gettext` / `gettext_lazy`, imported
    as `_`.
  - Visible text stays the same English. Only wrapping changes.
- **Tone:** student pages are informal (as in `studentview/help_labels.py`).
  Teacher and admin pages are neutral and polite. Placeholders and HTML tags
  are kept exactly. Plural forms follow each language's gettext plural rule.
- **Tests run from `Trust-AI-Platform/`:**
  `../djangofaithvenv/Scripts/python.exe manage.py test <labels> --settings=faithDev.settings_test --noinput`
  - The known baseline has 6 failures in `authoringtool`; no new ones are
    allowed.
  - Docker: `MSYS_NO_PATHCONV=1 docker compose run --rm --no-deps -v "$(pwd -W)/Trust-AI-Platform:/app" web python manage.py test <labels> --noinput`,
    run from the repo root.
- **After any marking change:** run
  `../djangofaithvenv/Scripts/python.exe manage.py update_translations` (Task 2).
- **Commits end with:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **A translated string in a script with an apostrophe or quote (French,
  Catalan, Italian):** the page's JS must keep working. Task 13 adds
  `test_no_bare_translate_inside_script`, a lint over all templates.
- **A logged-in user whose saved language is no longer offered,** for
  example `fa` saved by hand: they get English, not a crash. Task 1:
  `test_unknown_saved_language_falls_back_to_english`.
- **A scenario whose free-text language is "greek ", "Ελληνικά" or unknown:**
  matched case-insensitively and trimmed. Unknown falls back to the
  student's own language. Task 6:
  `test_scenario_language_matching_is_lenient`.
- **A reviewer correction that adds HTML (`<script>`, `<b>`) not in the
  source:** refused, because `blocktranslate` output isn't escaped. Task 11:
  `test_override_cannot_add_html`.
- **`next` on the switcher pointing to another site:** ignored, so the user
  goes to `/` instead. Task 1: `test_switch_ignores_foreign_next`.

---

### Task 1: Languages, switcher, remembered choice

**Files:**
- Create app `Trust-AI-Platform/localization/`: `__init__.py`, `apps.py`,
  `languages.py`, `middleware.py`, `views.py`, `urls.py`,
  `templatetags/localization_tags.py`, `tests/__init__.py`,
  `tests/test_language_choice.py`
- Create: `Trust-AI-Platform/templates/includes/language_switcher.html`
- Create: `Trust-AI-Platform/static/vendor/flag-icons/4x3/{gb,es-pv,es-ct,ee,fr,de,gr,it,pt,ro,es,tr,ua}.svg`
  and `LICENSE`, downloaded from
  `https://cdnjs.cloudflare.com/ajax/libs/flag-icons/7.2.3/flags/4x3/<flag>.svg`
- Modify: `faithDev/settings.py`
  - `INSTALLED_APPS` += `'localization'`
  - make sure `USE_L10N` isn't set to False, so dates and numbers follow the
    language
  - `LANGUAGE_CODE = 'en'`
  - `LANGUAGES` from the table above, using native names
  - `LOCALE_PATHS = [BASE_DIR / 'locale']`
  - add `django.template.context_processors.i18n`
  - `MIDDLEWARE`: `django.middleware.locale.LocaleMiddleware` after
    `SessionMiddleware`, and `localization.middleware.UserLanguageMiddleware`
    right after `AuthenticationMiddleware`
- Modify: `faithDev/urls.py` (include `localization.urls` at `/language/`)
- Modify: `accounts/models.py` (`UserProfile.language = CharField(max_length=10, blank=True)`)
  and generate the migration
- Modify: `templates/head.html`
  - `<html lang="{{ LANGUAGE_CODE }}" dir="{% if LANGUAGE_BIDI %}rtl{% else %}ltr{% endif %}">`
  - the switcher included before the profile `nav-item` (line ~101)
- Modify these standalone pages, which have their own `<html>`: add the same
  `lang`/`dir` and the switcher top-right.
  - `accounts/templates/accounts/login.html`
  - `accounts/templates/accounts/register.html`
  - `accounts/templates/accounts/tos_public.html`
  - `templates/registration/password_reset_{form,done,confirm,complete}.html`
  - `templates/404.html`

**Interfaces:**
- Produces, in `localization/languages.py`:
  - `LANGUAGE_META: list[dict]` with keys `code`, `native`, `table_name`, `flag`
  - `supported_codes() -> set[str]`
  - `code_for_table_name(name: str | None) -> str | None`: case-insensitive
    and trimmed. It also accepts the native name or the code, and returns
    None if unknown.
  - `meta_for(code: str) -> dict`: falls back to English
- Produces, in `localization/views.py`: `switch_language(request)`, POST
  only, named `switch_language`.
  - fields `language` and `next`
  - validates the code and sets Django's language cookie
    (`settings.LANGUAGE_COOKIE_NAME`)
  - saves `request.user.profile.language`, creating the profile if missing
  - redirects to `next` if `url_has_allowed_host_and_scheme`, else `/`
- Produces, in `localization/middleware.py`: `UserLanguageMiddleware`.
  - For a signed-in user, the saved language, if supported, is activated and
    set as `request.LANGUAGE_CODE`. The cookie is set on the response when it
    differs.
  - If none is saved, the current language is saved to the profile.
  - Unsupported saved values are treated as none.
- Produces the template tag `{% language_switcher %}` (inclusion tag) that
  renders `includes/language_switcher.html`. The context has `current` meta,
  `languages` meta, and `next` = `request.get_full_path`.

- [ ] **Step 1: Write failing tests** in `localization/tests/test_language_choice.py` (Django `TestCase`):
  - `test_browser_language_used_on_first_visit`: GET `/accounts/login/` with
    `HTTP_ACCEPT_LANGUAGE='el'` gives `response.context['LANGUAGE_CODE'] == 'el'`
    and `'lang="el"'` in the HTML.
  - `test_unsupported_browser_language_falls_back_to_english`: `'ja'` gives `'en'`.
  - `test_switch_sets_cookie_and_profile`: a signed-in user POSTs
    `{'language': 'de', 'next': '/accounts/profile/'}`. Expect a redirect to
    `/accounts/profile/`, the cookie set to `'de'`, and the profile language
    `'de'`.
  - `test_switch_rejects_unknown_code`: `'fa'` gives 400 and no cookie.
  - `test_switch_ignores_foreign_next`: `next='https://evil.test/'` redirects to `/`.
  - `test_saved_language_applies_after_login`: the profile language is
    `'it'` and the client has a cookie `'en'`. After login, GET the
    scenarios page: `LANGUAGE_CODE == 'it'` and the response cookie is `'it'`.
  - `test_first_login_saves_browser_language`: the profile language is
    empty and the cookie `'fr'`. After a request, the profile language is
    `'fr'`.
  - `test_unknown_saved_language_falls_back_to_english`: the profile
    language is `'fa'`, giving `LANGUAGE_CODE == 'en'` and a 200 response.
  - `test_switcher_lists_all_languages_with_flags`: the scenarios page
    contains `flag-icons/4x3/es-pv.svg`, `Ελληνικά` and the `switch_language`
    URL, and no other language codes appear in the switcher.
  - `test_code_for_table_name`: `'Greek'`, `' greek '`, `'Ελληνικά'` and
    `'el'` give `'el'`. `'Persian'` and `'Klingon'` give None.
- [ ] **Step 2: Run them** and check they fail (ImportError or 404).
- [ ] **Step 3: Implement** the app, settings, profile field and migration,
  switcher and page edits, and download the flags. The switcher is a
  Bootstrap dropdown:
  - the button shows the current flag image (20×15) and the uppercase code
  - each item shows flag, code and native name as a small POST form with
    the CSRF token
  - at phone width it shows only flag + code
- [ ] **Step 4: Run** `... test localization accounts` and check it passes.
- [ ] **Step 5: Commit:** `Add the language switcher and remember each user's language`

---

### Task 2: Translation tooling and checks

**Files:**
- Modify: `Trust-AI-Platform/requirements.txt` (`polib==1.2.0`); install it
  into `djangofaithvenv`
- Create: `localization/management/commands/update_translations.py`
- Create: `localization/catalog.py`
- Create: `Trust-AI-Platform/locale/<code>/LC_MESSAGES/` for the 12 non-English codes
- Test: `localization/tests/test_catalog.py`

**Interfaces:**
- Produces the command `update_translations`.
  - Runs `makemessages` for the 12 non-English codes with
    `no_obsolete=True` and `add_location='file'`.
  - Ignores `staticfiles/*`, `static/*`, `docs/*`, `media/*`,
    `ai_metrics_cache/*` and `*/migrations/*`.
  - Then runs `compilemessages` for the same codes.
- Produces, in `localization/catalog.py`:
  - `po_path(code: str) -> Path`
  - `load_po(code: str) -> polib.POFile`
  - `placeholders(text: str) -> collections.Counter`, which counts
    `%(name)s`, `%s`, `%d`, `{name}` and `{0}` tokens plus HTML tag names
    (opening, closing and self-closing)
  - `placeholder_problems(msgid: str, msgstr: str) -> list[str]`, which is
    empty when the placeholder counters are equal
  - `marked_msgids(code='el') -> set[str]`: every msgid and
    `msgid_plural` in that `.po`
  - `untranslated(code: str) -> list[polib.POEntry]`: empty msgstr, or fuzzy

- [ ] **Step 1: Write failing tests:**
  - `test_placeholder_problems_detects_dropped_and_added`:
    - `('Hello %(name)s', 'Γεια')` reports `%(name)s`
    - `('<b>Hi</b>', 'Γεια')` reports `b`
    - `('Hi', 'Γεια <script>')` reports `script`
    - `('Hello %(name)s', 'Γεια %(name)s')` reports nothing
  - `test_every_translation_keeps_placeholders`: for every non-English
    `.po`, every non-empty, non-fuzzy entry (singular and each plural form)
    has no placeholder problems.
  - `test_update_translations_creates_po_for_all_languages`: after
    `call_command('update_translations')`, `po_path(code)` exists for the 12
    codes.
- [ ] **Step 2: Run** and check they fail.
- [ ] **Step 3: Implement** and run `update_translations` once. The `.po`
  files will hold Django's few existing strings.
- [ ] **Step 4: Run** `... test localization` and check it passes.
- [ ] **Step 5: Commit** the tooling plus the generated `locale/`:
  `Add translation tooling and placeholder checks`

---

### Task 3: Mark shell and account pages

**Method (same for Tasks 3–6):**
- Follow the marking rules in Global Constraints.
- The test lives in `localization/tests/test_marked_<area>.py` and asserts
  that a set of sentinel strings is in `marked_msgids()` (from
  `localization/catalog.py`, Task 2) after running `update_translations`.
- Sentinels are exact English strings copied from this area's templates and
  code:
  - at least one heading, one button and one message per page
  - every Python message string
  - every script string
- Step 1 writes the sentinel test. Step 2 sees it fail, because nothing is
  marked yet. Step 3 marks the text and runs `update_translations`. Step 4
  sees the test pass and runs the app's existing tests (no new failures).
  Step 5 commits, including the regenerated `locale/` files.

**Scope:**
- `templates/` (`head.html`, `sidebar.html`, `footer.html`, `main.html`,
  `404.html`, `confirm_removal.html`, `includes/maintenance_banner.html`,
  `registration/*.html` including `password_reset_email.html` and
  `password_reset_subject.txt`)
- `accounts/templates/accounts/{login,register,profile}.html`
- `home/templates/home/home.html`, `messaging/templates/messaging/*.html`
- Python strings in `accounts/views.py`, `accounts/forms.py`, `home/`,
  `messaging/`
- Sentinels include "Invalid username or password.", the sidebar items
  ("Home", "Dashboard", "Authoring Tool", "Student Groups", "Student View",
  "Documentation & Tutorials", "Terms of Service") and the register form
  labels.
- Commit: `Mark shell, sign-in and account pages for translation`

### Task 4: Mark authoring pages

**Method (same for Tasks 3–6):**
- Follow the marking rules in Global Constraints.
- The test lives in `localization/tests/test_marked_<area>.py` and asserts
  that a set of sentinel strings is in `marked_msgids()` (from
  `localization/catalog.py`, Task 2) after running `update_translations`.
- Sentinels are exact English strings copied from this area's templates and
  code:
  - at least one heading, one button and one message per page
  - every Python message string
  - every script string
- Step 1 writes the sentinel test. Step 2 sees it fail, because nothing is
  marked yet. Step 3 marks the text and runs `update_translations`. Step 4
  sees the test pass and runs the app's existing tests (no new failures).
  Step 5 commits, including the regenerated `locale/` files.

**Scope:**
- `authoringtool/templates/authoringtool/`: `scenarios.html`,
  `archived_scenarios.html`, `create*/update*/view*` for Scenario, Phase,
  Activity, Answer(s) and Criterion, `includes/next_activity_picker.html`,
  `_translation_tag.html`, `_translation_review_panel.html`,
  `_assign_reviewer_modal.html`, `_original_text.html`, and the LTI pages
- Python strings in `authoringtool/views.py`, `authoringtool/forms.py`,
  `authoringtool/machine_translation/*.py` (for example
  `TRANSLATION_SERVICE_DOWN` and `REVIEW_SUBJECT`), and
  `authoringtool/middleware.py`
- Model `choices` labels shown in these pages via `get_*_display`
  (`variant_type`, `visibility_status`) get `gettext_lazy`. Confirm
  `makemigrations --check` reports no changes.
- Sentinels include "Adapt", "Create copy", "Creating your copy…",
  "Translate", "Start translation" and "The copy couldn't be created.
  Nothing was changed. (%(error)s)". When marking it, convert that f-string
  to `%(error)s` form.
- Commit: `Mark authoring pages for translation`

### Task 5: Mark Metrics & AI, proposals and analytics

**Method (same for Tasks 3–6):**
- Follow the marking rules in Global Constraints.
- The test lives in `localization/tests/test_marked_<area>.py` and asserts
  that a set of sentinel strings is in `marked_msgids()` (from
  `localization/catalog.py`, Task 2) after running `update_translations`.
- Sentinels are exact English strings copied from this area's templates and
  code:
  - at least one heading, one button and one message per page
  - every Python message string
  - every script string
- Step 1 writes the sentinel test. Step 2 sees it fail, because nothing is
  marked yet. Step 3 marks the text and runs `update_translations`. Step 4
  sees the test pass and runs the app's existing tests (no new failures).
  Step 5 commits, including the regenerated `locale/` files.

**Scope:**
- `ai_metrics_scenario.html`, `_ai_server_banner.html`,
  `proposal_list.html`, `proposal_history*.html`, and `index.html` (the
  analytics dashboard, the largest file)
- `machine_translation/health.py` banner titles and details: convert them to
  `gettext` with named placeholders. Render the time values before
  substitution.
- The rest of the proposal and metrics strings in `authoringtool/views.py`.
- Sentinels include "The AI server is offline, so proposals and LLM context
  can't be generated right now.", "Check again", "Generate LLM Context",
  "Force Rebuild" and "Proposal History".
- Commit: `Mark metrics, proposals and analytics pages for translation`

### Task 6: Mark groups, organizations, feedback and student pages

**Method (same for Tasks 3–6):**
- Follow the marking rules in Global Constraints.
- The test lives in `localization/tests/test_marked_<area>.py` and asserts
  that a set of sentinel strings is in `marked_msgids()` (from
  `localization/catalog.py`, Task 2) after running `update_translations`.
- Sentinels are exact English strings copied from this area's templates and
  code:
  - at least one heading, one button and one message per page
  - every Python message string
  - every script string
- Step 1 writes the sentinel test. Step 2 sees it fail, because nothing is
  marked yet. Step 3 marks the text and runs `update_translations`. Step 4
  sees the test pass and runs the app's existing tests (no new failures).
  Step 5 commits, including the regenerated `locale/` files.

**Scope:**
- `usergroups/templates/usergroups/*.html`,
  `organization/templates/organization/*.html`,
  `feedback/templates/feedback/*.html`
- `studentview/templates/studentview/{scenarioSelection,scenarioForm,scenarioView}.html`
  (not `chatBot.html` or `pendulum_modified_imu.html`)
- Python strings in those apps
- **Scenario-language rule (spec):**
  - In `studentview/views.py` (`scenario_viewer` and the help JSON views),
    render inside `translation.override(code)`, where
    `code = code_for_table_name(scenario.language)`, when that's supported.
    Otherwise use the user's language.
  - Delete `studentview/help_labels.py`. Its English strings become msgids,
    and its 13 existing translations seed those msgids in Task 12. Keep its
    Persian entry in this task's commit message for later.
  - Templates use `{% translate %}` and drop `help_text`.
- Extra tests in `studentview/tests` (update the existing
  `authoringtool/test_help_requests.py` assertions that used `help_text`):
  - `test_scenario_page_follows_scenario_language`: the scenario language
    is `'Greek'` and the user's cookie `'de'`. The response is rendered
    with `LANGUAGE_CODE == 'el'` (check the `<html lang>`).
  - `test_scenario_language_matching_is_lenient`: `' greek '` gives `'el'`.
    `'Klingon'` gives the user's `'de'`.
  - `test_scenario_list_follows_user_language`: the selection page uses
    `'de'`.
- Commit: `Mark groups, organizations, feedback and student pages for translation`

---

### Task 7: Admin pages, documentation and Terms of Service

**Files:**
- `accounts/templates/accounts/admin_dashboard.html`,
  `_admin_email_tab.html`, `_admin_languages_tab.html`,
  `_admin_maintenance_tab.html`, `admin_email_campaign.html`, and
  `accounts/admin_views.py` strings
- The custom Django-admin templates under `authoringtool/templates/admin/`
  and `templates/admin/` (`scenario_health.html`, `base_site.html`)
- `accounts/templates/accounts/documentation.html`: mark it in
  paragraph-sized `{% blocktranslate trimmed %}` blocks, so reviewers can
  correct it
- **Terms of Service:**
  - Move the terms body out of `tos.html` and `tos_public.html` into
    `accounts/templates/accounts/tos/en.html`. Both pages include
    `tos/<LANGUAGE_CODE>.html`, falling back to `tos/en.html` (use
    `select_template`).
  - The wrapper pages' own chrome is marked normally.
  - The `tos/<code>.html` bodies are written in Task 12. Each non-English one
    begins with a translated "This is a translation. The English version of
    these Terms is the binding one." paragraph.
- Test `localization/tests/test_marked_admin.py`: sentinels, plus
  `test_terms_fall_back_to_english` (`tos/de.html` missing gives the English
  body with a 200 response).
- Commit: `Mark admin pages and documentation; split the Terms per language`

---

### Task 8: Display labels for reference values

**Files:**
- Create: `localization/reference_labels.py`, and a filter `ui_label` in
  `localization/templatetags/localization_tags.py`
- Modify: the templates that print these values. Find them with:
  - `grep -rn "subject.name\|\.category\|activity_type.name\|\.language\b\|get_language_display" --include=*.html`
- Modify: `authoringtool/models.py` `Simulation.LANGUAGE_CHOICES` labels to
  `gettext_lazy`, and confirm no migration
- Test: `localization/tests/test_reference_labels.py`

**Interfaces:**
- Produces `ui_label(value: str) -> str`, also available as the template
  filter `{{ value|ui_label }}`.
  - It returns the translated label for a known English value, else the
    value unchanged.
  - Known values (all `gettext_lazy`, with `pgettext_lazy` contexts
    `subject`, `subject category`, `activity type`, `performance group`):
    - **subjects:** Physics, Mathematics, Chemistry, Biology, Computer
      Science, Environmental Science, Earth & Space Science, Engineering &
      Technology, Statistics & Data Science
    - **categories:** STEM, Humanities, Social Sciences, Arts, Other
    - **activity types:** Explanation, Experiment, Guidance, Question
    - **groups:** High, Moderate, Low
  - **Language table names** map to the native name from `LANGUAGE_META`,
    for example "Greek" to "Ελληνικά". "Persian" maps to "فارسی".
- Tests:
  - `test_known_values_are_marked`: after `update_translations`, every known
    subject, category, activity type and group name is in `marked_msgids()`.
  - `test_language_names_are_native`: `ui_label('Greek') == 'Ελληνικά'`
    under any language.
  - `test_unknown_value_unchanged`: `ui_label('Robotics') == 'Robotics'`.
  - `test_scenarios_page_uses_ui_label`: a scenario with subject Physics
    renders through the filter. Patch `localization.reference_labels.ui_label`
    to return `'PHYSICS-LABEL'` and assert it appears.
- Commit: `Show built-in subjects, activity types and language names in the interface language`

---

### Task 9: Risk flags rendered in any language

**Files:**
- Create: `authoringtool/risk_flag_text.py`. It has no model imports, so
  migrations can use it.
- Modify: `authoringtool/models.py` `ActivityFlag`: add
  `reason_code = CharField(max_length=40, blank=True)` and
  `reason_params = JSONField(default=dict, blank=True)`. Generate the schema
  migration, plus a data migration whose `RunPython` backfills from
  `parse_reason`.
- Modify: `authoringtool/tasks.py`
  - `calculate_activities_in_risk` (around lines 2463–2810): each flag dict
    gains `"Code"` and `"Params"`, and `"Reason"` is built with
    `reason_text(code, params, language='en')`. `ActivityFlag.objects.create`
    stores both.
  - `get_data_insight` (around line 2812): read the numbers from
    `flag.reason_params`, falling back to `parse_reason`. The English output
    stays the same, because insights are AI-prompt input.
- Modify: `authoringtool/views.py`
  - `ai_metrics`: the flag rows get `FlagDisplay` and `ReasonDisplay` from
    `display_flag`
  - `risk_flags_status`: add the same keys to each row of a successful
    result
- Modify: `ai_metrics_scenario.html` (lines ~470–478 and ~702–708) to show
  the `*Display` fields
- Test: `authoringtool/test_risk_flag_text.py`

**Interfaces:**
- `REASONS: dict[str, gettext_lazy template]`: one code per distinct Reason
  sentence in `calculate_activities_in_risk`. That's the 9 codes
  `engagement_slow`, `engagement_fast`, `v_shape`, `low_beats_moderate`,
  `high_worse_than_moderate`, `timing_all_wrong`, `timing_all_right`,
  `systemic_failure` and `dynamic_correctness`. Each template uses named
  placeholders.
- `FLAG_TYPES: dict[str, gettext_lazy template]` for the 6 flag-type
  strings. The engagement ones carry `%(activity_type)s`.
- `reason_text(code: str, params: dict[str, str], language: str | None = None) -> str`
  - Params are the already-formatted strings, so the English output is
    byte-identical to today.
  - Group names inside params are passed through `ui_label` when rendering
    in a language other than English.
- `parse_reason(flag_type: str, reason: str) -> tuple[str, dict[str, str]] | None`:
  matches each English template, with a regex built from it, and returns
  None if nothing matches.
- `display_flag(flag_type: str, reason: str, code: str = '', params: dict | None = None) -> tuple[str, str]`:
  in the active language, it uses code and params, else `parse_reason`, else
  returns `(flag_type, reason)` unchanged.

- [ ] **Step 1: Write failing tests:**
  - `test_english_is_byte_identical`: for each of the 9 codes, a fixture
    copied from today's f-string output gives
    `reason_text(code, params, 'en') == fixture`.
  - `test_parse_round_trips`: `parse_reason(type, fixture)` gives
    `(code, params)` for all 9.
  - `test_unparseable_reason_displays_unchanged`.
  - `test_insight_numbers_unchanged`: `get_data_insight` on a flag built
    from each fixture returns the same strings as before the change.
    Capture today's output in the fixture first.
  - `test_backfill_sets_code_and_params`: run the data migration function on
    legacy rows.
  - `test_metrics_page_shows_display_text`: the rendered page uses
    `display_flag` output.
- [ ] **Step 2: Run** and check they fail.
- [ ] **Step 3: Implement** the module, model fields, migrations, task and
  view changes.
- [ ] **Step 4: Run** `... test authoringtool.test_risk_flag_text authoringtool.tests`
  and check no new failures.
- [ ] **Step 5: Commit:** `Store risk flags as code plus numbers so they display in any language`

---

### Task 10: Emails in the recipient's language

**Files:**
- Modify: `accounts/views.py`. `CustomPasswordResetView.form_class` becomes a
  `LocalizedPasswordResetForm` (in `accounts/forms.py`) whose
  `send_mail(...)` wraps the parent call in
  `translation.override(recipient_language(context['user']))`.
- Modify: `accounts/outbox.py` `build_message`: render subject is untouched
  (admin content), while the template and the `plain_text_message` frame
  ("Hello", "Trust AI Lab") are rendered inside
  `override(recipient_language(delivery.user))`.
- Modify: `authoringtool/machine_translation/collaborators.py` `_send_review_email`
- Modify: `templates/email/admin_bulk_email.html`: mark the frame text
- Create: `localization/languages.py` gets
  `recipient_language(user) -> str`: the profile language if supported, else
  `'en'`.
- Test: `localization/tests/test_emails.py`. It uses a test email backend,
  `localization.tests.utils.LanguageRecordingBackend`, a locmem subclass
  whose `send_messages` stores `translation.get_language()` with each
  message.
  - `test_password_reset_email_in_users_language`: a user with `'el'` and a
    request made in `'en'` give a recorded language of `'el'`.
  - `test_campaign_frame_in_recipients_language`: two recipients, `'de'`
    and none, after `drain_outbox()`, record `['de', 'en']`.
  - `test_reviewer_email_in_reviewers_language`: `_send_review_email` to a
    reviewer with `'fr'` records `'fr'`.
- Commit: `Send emails in each recipient's language`

---

### Task 11: Reviewer corrections

**Files:**
- Create in `localization/`:
  - `models.py`:
    - `TranslationReviewer(user FK, language CharField(10), unique together)`
    - `TranslationOverride(language, msgctxt default '', msgid, plural_index
      default 0, msgstr, updated_by FK, updated_at)`, unique on (language,
      msgctxt, msgid, plural_index)
    - `TranslationOverrideLog(override FK SET_NULL, language, msgid, old,
      new, user FK, at)`
  - `overrides.py`
  - `views.py` additions
  - templates `localization/review.html`
  - `admin.py`
- Modify: `accounts/templates/accounts/_admin_languages_tab.html` (assign and
  remove reviewers per language) and `accounts/admin_views.py`
- Modify: `faithDev/settings.py`: add
  `localization.middleware.TranslationOverrideMiddleware` after
  `UserLanguageMiddleware`
- Test: `localization/tests/test_review.py`

**Interfaces:**
- `apply_overrides(language: str) -> None`: writes each override into
  `django.utils.translation.trans_real.translation(language)._catalog`. The
  key is `msgid`, or `f"{msgctxt}\x04{msgid}"`, and `(key, plural_index)`
  for plurals.
- `overrides_version() -> int` from the cache key
  `localization:overrides-version`. `bump_overrides_version()` runs on every
  save.
- The middleware calls `apply_overrides(request.LANGUAGE_CODE)` once per
  process per version.
- Views (reviewers see only their languages; staff see all):
  - `review_translations(request, code)` (GET, paged 100, `q` search,
    `filter=needs_review`, where needs review means
    `catalog.untranslated(code)`)
  - `save_translation(request, code)` (POST). It returns 400 with the
    problems when `placeholder_problems` isn't empty.
- The ToS isn't listed: it's a template, not msgids.

- [ ] **Step 1: Write failing tests:**
  - `test_only_assigned_reviewer_can_edit_their_language` (403 for others
    and for other languages, 200 for staff)
  - `test_override_applies_immediately` (save the Greek msgstr for "Check
    again", then a GET of a page under `'el'` shows it)
  - `test_override_cannot_drop_placeholder`
  - `test_override_cannot_add_html`
  - `test_override_is_logged`
  - `test_admin_assigns_reviewer_on_languages_tab`
- [ ] **Step 2: Run** and check they fail.
- [ ] **Step 3: Implement.** The page lists the source, the current
  translation (override, else `.po`) and an edit box. The sidebar link
  "Translations" shows only for reviewers and staff.
- [ ] **Step 4: Run** `... test localization` and check it passes.
- [ ] **Step 5: Commit:** `Let assigned reviewers correct translations without a deploy`

---

### Task 12: Write the translations

Run `update_translations` first. Then for each of the 12 codes:
- fill every msgid and plural form in `locale/<code>/LC_MESSAGES/django.po`
- remove `#, fuzzy`
- write `accounts/templates/accounts/tos/<code>.html`: the full Terms in that
  language, starting with the binding-English paragraph

Seed the scenario-page help strings from `help_labels.py`'s history
(`git show HEAD~N:Trust-AI-Platform/studentview/help_labels.py`). Then run
`compilemessages` (part of `update_translations`).

Split into three commits so each is reviewable:
- **12a:** `eu`, `ca`, `et`, `fr`
- **12b:** `de`, `el`, `it`, `pt`
- **12c:** `ro`, `es`, `tr`, `uk`

**Test, added in 12a:** `localization/tests/test_complete.py`,
`test_language_complete` for the codes done so far. Extend the list in 12b
and 12c. It asserts `catalog.untranslated(code) == []`, and that
`tos/<code>.html` exists and contains the binding paragraph's msgstr.
`test_every_translation_keeps_placeholders` (Task 2) must stay green.

Commits:
- `Translate the interface into Basque, Catalan, Estonian and French`
- `… German, Greek, Italian and Portuguese`
- `… Romanian, Spanish, Turkish and Ukrainian`

---

### Task 13: Whole-platform checks

**Files:** `localization/tests/test_coverage.py`

- `test_pages_fully_translated_in_greek`: render each page below under
  `HTTP_ACCEPT_LANGUAGE='el'`, with a teacher, a student and a staff user as
  needed. Assert `lang="el"` and that none of that page's English sentinels
  (reuse the lists from Tasks 3–7) appear in the HTML. The pages:
  - login, register, the four password reset pages
  - scenarios list, scenario view, scenario edit, Metrics & AI, proposal list
  - group list, organization list, feedback forms
  - student scenario selection, admin dashboard, documentation, both ToS
    pages
- `test_no_bare_translate_inside_script`: a lint over every template. Inside
  `<script>…</script>`, any `{% translate %}` or `{% blocktranslate %}` must
  be the `as` form whose variable is used with `|escapejs`, or be inside
  `json_script`.
- `test_mo_files_are_compiled`: each `django.mo` is newer than its `.po`.
- **Run the full suite locally.** Only the 6 known failures may appear.
- **Run the `localization`, `accounts` and `studentview` suites in Docker
  (Django 4.0):** with Docker Desktop started, everything must pass.
- **Write the deploy note in the hand-off:**
  1. pull and build `web`, `celery` and `celerybeat`
  2. migrate (accounts, authoringtool and localization migrations)
  3. recreate the containers
  4. no gettext is needed on the server, because the `.mo` files are
     committed
- Commit: `Check every page is fully translated`
