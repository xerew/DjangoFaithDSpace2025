# Admin User List and Form Editor Tweaks — Design Spec

## Goal

Three small refinements from production use:

1. The User Management table no longer lists accounts with no role (students) — they clutter the admin's user list.
2. In the feedback form editor, toggling "Assign to all scenarios" syncs every scenario checkbox in both directions (checking checks all — already the case; unchecking now unchecks all — currently a no-op).
3. The scenario assignment list gets a search box so admins can find and check a specific scenario without scrolling.

## Scope

- `Trust-AI-Platform/accounts/admin_views.py` — `admin_dashboard`'s user queryset.
- `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html` — remove the now-dead "No Role" role-filter option and its JS value handling.
- `Trust-AI-Platform/feedback/templates/feedback/form_edit.html` — checkbox-sync JS + scenario search input/JS.
- Tests in `accounts` (queryset) and `feedback` (render-level, where meaningful).

## Behavior

### 1. User Management excludes no-role accounts

- `admin_dashboard`'s `users` queryset becomes: users with at least one group, OR `is_staff`, OR `is_superuser` — i.e. `User.objects.filter(Q(groups__isnull=False) | Q(is_staff=True) | Q(is_superuser=True)).distinct()`, keeping the existing `prefetch_related('groups').order_by('username')`. `.distinct()` is required — the groups join duplicates multi-group users.
- The role-filter dropdown's "No Role" option (`value="__none__"`, `id="rfNone"`) is removed from the template, since no listed row can match it anymore. The filter JS keys off whatever `.role-filter-check` checkboxes exist, so removing the input removes the behavior; also delete any `__none__`-specific branch in the row-matching JS if one exists (verify by reading the filter function).
- The stats cards are UNCHANGED: "Total"/"Active" keep counting all accounts, and the "Students" card (the `no_role` aggregate) stays — they are platform-wide counts, deliberately not tied to the table's contents.
- Existing per-user actions (edit/delete/impersonate) are unaffected — they operate on listed rows only, and no-role accounts simply aren't listed.

### 2. Assign-to-all checkbox syncs both ways

In `form_edit.html`'s existing `assignAll` change handler, add the missing else-branch: when unchecked, all `.scenario-cb` checkboxes become unchecked. (Semantics: with assign-to-all off, checked boxes are inclusions — so "uncheck all" means "no scenarios included," matching the requested behavior.) The `updateHint()` text logic is unchanged.

### 3. Scenario search box

- A text input above the `.scenario-box` list (`placeholder="Search scenarios…"`), full-width, standard `form-control`, with `autocomplete="off"`.
- Typing filters the checkbox rows client-side: case-insensitive substring match against the scenario's label text; non-matching rows get `display:none`.
- **Filtering must never change checked state** — hidden rows keep their checked value and still submit with the form (hidden inputs still post in HTML forms; only `disabled` would drop them, which we never set).
- Clearing the input shows all rows again. When nothing matches, a small muted "No scenarios match." hint row is shown.
- The assign-to-all sync (item 2) applies to ALL scenario checkboxes, including currently-hidden ones — the master toggle is an all-or-nothing operation regardless of an active search filter.

## What Does NOT Change

- Feedback form visibility/permissions — every staff admin already sees every form; confirmed working, explicitly out of scope.
- No change-history feature — considered and scratched.
- Stats cards, other dashboard tabs, all server-side assignment semantics (`_save_form_from_post` untouched).
