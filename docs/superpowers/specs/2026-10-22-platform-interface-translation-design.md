# Multilingual Platform Interface — Design

Date: 2026-10-22
Status: approved 2026-10-23

## Why

Trust AI Lab is used by partners across Europe and beyond. Every piece of
interface text is written directly in English in 90 Django templates, view
code and page scripts. There is no translation setup: `USE_I18N = True`, but
there's no `LocaleMiddleware`, no `.po` files, and only 2 templates use
`{% trans %}`. Teachers and students who don't read English can't use the
platform comfortably.

The admin wants the whole platform's fixed text available in every language
the platform already knows, with a language switcher showing flags and codes.

## Goals

- **Everyone gets a language switcher with a flag and a code:** visitors on
  login, register and password reset, students, teachers and platform admins.
- **All fixed interface text is translated:**
  - menus, buttons, headings, help text, form labels and placeholders
  - validation, error and success messages (including Django `messages`)
  - text inside page scripts (alerts, toasts, button states)
  - the login, register and password reset pages
  - the 404 page
  - the Documentation & Tutorials page and the Terms of Service (public and
    signed-in versions)
- **Languages:** every language in the authoring tool's `Language` table
  except Persian. That's 13: English, Basque, Catalan, Estonian, French,
  German, Greek, Italian, Portuguese, Romanian, Spanish, Turkish and
  Ukrainian.
  - The production table was checked on 2026-10-23 and has exactly these
    plus Persian.
  - **Persian is deferred** (admin, 2026-10-23). It's right-to-left and needs
    the whole layout mirrored, so it comes as a separate step later.
- **Dates and numbers are formatted for the chosen language,** using
  Django's built-in formats.
- **Emails follow the recipient's saved language:**
  - password reset email and subject
  - the fixed frame around admin campaign emails (greeting, "Platform
    Announcement", footer)
  - any other automatic email
- **Partners can correct wording:** admins assign a teacher as reviewer for a
  language, and that reviewer can edit that language's wording on a
  translations page. Edits take effect without a deploy.

## Non-goals

- **Content people create stays as written:** scenarios, phases, activities,
  answers, feedback, group names, organization names, campaign message
  bodies, maintenance notice reasons, model names in the database.
- **Plato (the Rasa chatbot) is out of scope.** It keeps following the
  scenario's language.
- **No language in URLs.** Addresses stay as they are (`/accounts/login/`,
  not `/el/accounts/login/`).
- **PDFs aren't translated,** for example `documentation_en.pdf`.
- **Downloaded files stay English** (builder's call). This covers CSV and
  Excel exports (organization reports, group, feedback and analytics
  exports, the flags CSV) and the scenario import template. The importer
  reads that template's column headers back, and research scripts may
  expect fixed English column names.
- **AI-generated text stays as generated:** proposals and Ollama summaries.
- **Plato's chat page inside the scenario** (`chatBot.html`) stays as it is,
  with the chatbot.
- **Django admin keeps Django's own translations** for its built-in screens.
  Our model and field names there stay English.

## Anti-goals

- **A half-translated page in a language the switcher offers.** If a language
  is listed, its pages are complete. Only reviewer edits change wording
  afterwards.
- **Translating user content by accident.** For example, wrapping a
  scenario's name in a translation tag so that a scenario named "Energy" gets
  shown as "Ενέργεια".
- **A wrong language sticking.** Someone who picked Greek must not see English
  after logging in on another device, and vice versa.

## Decided by the admin

- Audience: everyone, including admin pages.
- Translations: drafted by the builder (Claude) for all languages. Native
  speakers at the partners review later.
- Plato: not in this project.
- Remembering the choice: browser language before login, remembered in that
  browser when changed. After login, saved on the account and applied on any
  device.
- Flags: best fit, each shown with its code:
  - English: UK flag
  - Basque: ikurriña
  - Catalan: senyera
- Flags may be emoji (admin, 2026-10-23); see "Languages and codes" for why
  the builder still uses images.
- Persian: deferred to a later step.
- Emails: password reset, the campaign frame, and all other automatic emails.
- Reviewers: assigned per language by admins.
- **On a student's scenario page the interface follows the scenario's
  language**, as "Ask for help" does today (`studentview/help_labels.py`).
  Every other page follows the switcher.
- Documentation & Tutorials and the Terms of Service are translated into every
  language. The admin confirmed the Terms again on 2026-10-23.
- Rollout: everything at once.
- **Risk flags are translated.** The sentences the system writes for flags
  are reworked so they display in any language (see "System-written text").
- **Known reference values are translated for display only:**
  - seeded subjects and subject categories, activity types, and the
    simulation language choices
  - language names, which show in their own language ("Ελληνικά")

  Stored values stay English. Values admins add later show as typed.
- **Reviewers can correct Documentation & Tutorials,** which is split into
  reviewable strings. **They can't edit the Terms of Service:** its
  translations are fixed in code, changes go through the admin, and each one
  carries the "English version is binding" line.
- **New languages:** when the admin later adds a language to the table, they
  ask Claude to translate it. The switcher therefore lists exactly the
  languages that have complete translation files, not the database table.

## How it works

### Languages and codes

A single list in settings (`LANGUAGES`) holds the interface languages. Each
entry has:
- an ISO 639-1 code
- the language's own name, for example "Ελληνικά"
- the English name used in the `Language` table
- a flag code for the switcher

| Table name | Code | Flag |
|---|---|---|
| English | en | gb |
| Basque | eu | es-pv (ikurriña) |
| Catalan | ca | es-ct (senyera) |
| Estonian | et | ee |
| French | fr | fr |
| German | de | de |
| Greek | el | gr |
| Italian | it | it |
| Portuguese | pt | pt |
| Romanian | ro | ro |
| Spanish | es | es |
| Turkish | tr | tr |
| Ukrainian | uk | ua |

- **Flags are images, not emoji** (builder's call). The admin allows emoji
  flags here, but Windows browsers don't draw flag emoji: they show the
  letters "GR", "DE"… instead. Basque and Catalan also have no emoji flag at
  all. Images look the same everywhere. The
  `flag-icons` set includes `es-pv` and `es-ct`. Serve it from the project's
  static vendor folder, like the other vendor assets.
- **All 13 languages are left-to-right.**
- **Source language:** English (`en`), which is also the final fallback.

### Choosing and remembering the language

1. **The first visit uses the browser's preferred language.** Django's
   `LocaleMiddleware` reads `Accept-Language`, limited to `LANGUAGES`, and
   falls back to English.
2. **The switcher changes the language.**
   - It posts to Django's `set_language`, which sets the language cookie.
   - For a signed-in user, the choice is also saved on the account: a new
     `language` field on `accounts.UserProfile`.
3. **On login, the account's saved language becomes the active one** and sets
   the cookie, so it follows the user across devices. If the account has
   none, the current browser language is saved to the account.
4. **Users with no profile row:** create the profile, or skip saving without
   crashing.
5. **Emails** go out in the recipient's saved language, or English if they
   have none (builder's call).

### Student scenario page

The scenario page (`studentview` scenario view and the parts of it rendered
around the chat iframe) activates the scenario's language for that request,
when that language is one of `LANGUAGES`. Otherwise it keeps the user's
language.

`help_labels.py` then becomes redundant: its strings move into the normal
translation files. Its 14 existing translations seed those entries.

### Marking text for translation

- **Templates:** fixed text is wrapped in `{% translate %}` /
  `{% blocktranslate %}`, using context where the same English word has
  different meanings.
- **Python:** user-facing strings use `gettext` / `gettext_lazy`. That covers
  `messages.*`, form labels and errors, JSON error strings shown to users,
  and email subjects.
- **Page scripts:** strings shown to users come through Django's
  `JavaScriptCatalog`. Use `gettext()` in JS, or pass translated values in
  `data-` attributes or `json_script`.
- **User content stays unwrapped,** in templates and code.
- **Emails** render inside `translation.override(<recipient's language>)`.
  The campaign outbox renders each message in its recipient's saved language.

### System-written text (risk flags)

Today, risk flags are stored as English sentences in `ActivityFlag.flag_type`
and `flag_reason`, for example "Engagement risk: High group's Avg Time
(31.0s) is > 1 std above mean (23.40 ± 6.09)". Later code reads numbers back
out of them by matching the English (`authoringtool/tasks.py` around lines
2824–2904). The insights and resolutions built from flags (`tasks.py`
around 4052) are also put together in English.

- **Flags are stored as a code plus their numbers,** for example
  `engagement_avg_time_high` with `{group, value, mean, std}`. They're
  rendered as a sentence in the viewer's language when displayed.
- **The string matching is rewritten** to read the stored numbers, not the
  sentence.
- **Existing flag rows are migrated** by parsing today's English sentences
  into code + numbers. Rows that can't be parsed keep their original text,
  shown as is, so no flag disappears.
- **Insights and resolutions** built from flags get the same treatment
  (builder's call: they're system-written like flags).
- **Caching:**
  - Rendered flag text that's cached (Redis entries, cached pages) is
    either cached per language or stored as codes and rendered on read.
  - Flags in the CSV download stay English, like other downloads.

### Reference values in the database

Seeded subjects, subject categories, activity types, simulation language
choices and language names get display-only translations:
- a gettext entry per known value, applied where the value is shown
- language names come from the language list (each in its own language)

Stored values and code that looks them up by English name (for example
activity types in `tasks.py` and `export_item_bank.py`) are unchanged.

### Translation files

- Use standard gettext `.po`/`.mo` files under a project `locale/`
  directory, built with `makemessages` and `compilemessages`, for every
  language in the table.
- Compile the `.mo` files in the Docker build or commit them, so production
  needs no extra step at runtime.
- **The builder drafts every string for all 13 non-English languages.**
  - Placeholders (`%(name)s`, `{count}`) and HTML must be kept intact.
  - Plural forms must be correct per language.
  - Students and teachers are addressed in the platform's existing tone:
    informal where the student-facing pages already use it (see
    `help_labels.py`), neutral and polite elsewhere.
- **Long pages:**
  - Documentation & Tutorials is translated in full as normal strings
    (paragraph-sized where sensible), so reviewers can correct it.
  - The Terms of Service (public and signed-in) uses one fixed template per
    language. These are not reachable from the reviewer page. Each starts
    with a line saying the English version is binding.

### Reviewer corrections

- **A `TranslationReviewer` assignment (user + language code), managed by
  platform admins** on the admin dashboard.
- **Reviewers get a translations page** listing every source string for their
  language, with the current translation, a search, and a filter for strings
  changed in English since last reviewed. They save corrections there.
- **Corrections are stored in the database** as overrides keyed by
  (language, message id, context). They apply at runtime on top of the `.po`
  catalogs, cached and invalidated on save. Edits survive rebuilds and need
  no deploy.
- **Saving checks the placeholders.** If a correction drops or invents a
  placeholder (`%(name)s`, `{0}`, HTML tags), the save is refused.
- **Edits are recorded:** who changed what and when, so a bad edit can be
  undone.

### Getting ready for Persian later

- `<html lang="…">` is set from the active language on every page, including
  standalone ones (login, register, password reset, the chat page and emails).
- `dir` is set from Django's bidi flag for the language: always `ltr` for
  the 13. Then adding Persian later needs RTL styles and translations, not a
  template sweep.
- No RTL stylesheet in this project.

### The switcher

- **Signed-in pages:** in the top bar, next to the user menu. It shows the
  current language's flag and code. The dropdown lists every language with
  flag, code and native name.
- **Login, register and password reset:** the same control in the top-right
  corner.
- **Phone width:** collapses to flag + code and stays reachable.

## Constraints

- **Django versions:** production runs Django 4.0 in Docker, and local
  development uses 5.2. Use APIs available in both.
- **Tests:** they run locally with `faithDev.settings_test` (SQLite, dummy
  cache) and in Docker. The baseline is 6 known authoring-tool failures.
- **Every page keeps working in English** exactly as today.
- **Rules from earlier work still apply:**
  - no emoji in the UI (the flags are images anyway)
  - "Ask for help" and other student strings must keep working for scenario
    languages

## Left to the builder

- How to split the work into tasks, and the order of templates.
- Message ids and contexts, and whether long pages use per-language templates
  or many strings.
- How to implement the runtime override layer, provided the rules above hold.
- Whether `.mo` files are committed or compiled in the Docker build.

## Testing

Required tests:
- **Language choice:**
  - browser language picked on first visit
  - the switcher sets the cookie and saves to the profile
  - logging in applies the saved language
  - an unsupported language falls back to English
- **Student scenario page:** uses the scenario's language whatever the
  switcher says, and the user's language when the scenario language isn't
  supported.
- **Risk flags:**
  - parsing migrated flags back into code + numbers, with each existing
    sentence pattern covered
  - flags render in Greek
  - the reworked numeric checks give the same results as today's string
    matching
- **Coverage:** a test that renders every main page (login, register,
  password reset, scenarios list, scenario edit, Metrics & AI, admin
  dashboard, student pages) in Greek and checks that no known English-only UI
  string remains. Use a small list of sentinel strings per page.
- **Placeholders:** a test that compiles every `.po` file and checks no
  translation drops or adds placeholders compared with its English source
  (`msgfmt --check`, or the equivalent in Python).
- **Emails:** the password reset and campaign frame render in the
  recipient's language.
- **Reviewers:**
  - only an assigned reviewer can edit, and only their own language
  - placeholder validation on save
  - an override takes effect immediately
