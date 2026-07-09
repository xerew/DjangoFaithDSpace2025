# Teacher Home Page — Design

**Date:** 2026-07-09
**Branch:** `improvements/performance-and-responsive`

---

## Overview

A dedicated home page for teachers, served by a new `home` Django app. It becomes the post-login landing page for all teacher-group users. Students are unaffected and still land on `studentScenarios`. The page is informational, welcoming, and action-oriented — not an analytics dashboard.

---

## Entry Point

- URL: `/home/`
- View name: `teacher_home`
- Access: `@group_required('teachers')` (same decorator used across all authoring views)
- Login redirect for teachers changes from `index` to `teacher_home`
- Root URL `''` redirects to `/home/` instead of `/authoringtool/`

Students continue to redirect to `studentScenarios` on login.

---

## Page Sections

### 1. Hero / Welcome

- Personalised greeting: "Welcome back, [user.get_full_name or username]"
- Today's date displayed subtly beneath the name
- One-sentence platform subtitle: "Trust AI Lab — Inquiry-Based Learning powered by FAITH & DSpace, funded by the European Union."

### 2. Quick Stats Bar

Three inline stat counters, pulled live from the DB in the view:

| Stat | Query |
|---|---|
| Your Scenarios | `Scenario.objects.filter(created_by=request.user).count()` |
| Your Groups | `UserGroup.objects.filter(teacher=request.user).count()` |
| Total Students | Sum of members across all teacher's groups |

Empty-state safe — shows `0` with no errors.

### 3. About IBL

Two-column layout (Bootstrap `col-12 col-md-6`):

- **Left:** Short text block — what Inquiry-Based Learning is, how the platform supports it (scenario-based digital experiments, adaptive evaluation, progress tracking)
- **Right:** Classroom image at `{% static 'img/home_ibl.jpg' %}` — a placeholder image is added to static files; the teacher/admin can replace it with a real photo

### 4. Latest Public Scenarios

Up to 5 most recently created public scenarios across all teachers:

```python
Scenario.objects.filter(visibility_status='public').order_by('-created_at')[:5]
```

Displayed as Bootstrap cards showing: scenario name, subject domains, age range, and a "View" button linking to `viewScenario`.

If no public scenarios exist, shows a subtle empty-state message.

### 5. Quick Actions

Three action cards with icon, title, and description:

| Action | URL name | Icon |
|---|---|---|
| Create Scenario | `createScenario` | `bi-plus-circle` |
| Student Groups | `list_groups` | `bi-people-fill` |
| Analytics Dashboard | `index` | `bi-bar-chart-line` |

### 6. Get Started Strip

Shown **only** when `my_scenario_count == 0`. A highlighted banner (blue, matching the hero palette):

> "You haven't created a scenario yet — start building your first one."

Single CTA button: "Create your first scenario" → `createScenario`.

---

## Routing Changes

| Location | Before | After |
|---|---|---|
| `accounts/views.py` login redirect | `redirect('index')` | `redirect('teacher_home')` |
| `faithDev/urls.py` root `''` | `redirect('/authoringtool/', permanent=True)` | `redirect('/home/', permanent=True)` |
| `faithDev/urls.py` app include | — | `path('home/', include('home.urls'))` |
| Sidebar `head.html` | No Home item | New **Home** nav item at top (teachers only) |

---

## New Files

| File | Purpose |
|---|---|
| `home/__init__.py` | App package |
| `home/apps.py` | `HomeConfig` with `name = 'home'` |
| `home/views.py` | `teacher_home` view |
| `home/urls.py` | Single pattern: `path('', views.teacher_home, name='teacher_home')` |
| `home/templates/home/home.html` | Page template extending `main.html` |

## Modified Files

| File | Change |
|---|---|
| `faithDev/settings.py` | Add `'home'` to `INSTALLED_APPS` |
| `faithDev/urls.py` | Add home include; update root redirect |
| `accounts/views.py` | Teacher login redirect → `teacher_home` |
| `templates/head.html` | Add Home sidebar item (teachers only, above Dashboard) |

## Static Assets

| File | Notes |
|---|---|
| `static/img/home_ibl.jpg` | Placeholder classroom image — replaceable |

---

## Access Control

- `teacher_home` view: `@group_required('teachers')` — non-teachers who reach `/home/` get a 403
- Students are never redirected here; their login redirect remains `studentScenarios`
- No admin-only restriction — all teachers can access

---

## Out of Scope

- Student-facing home page (students use `studentScenarios`)
- Notification feed or real-time activity stream
- Editable page content from admin (all copy is in the template)
- IBL image upload via UI (static file swap only)
