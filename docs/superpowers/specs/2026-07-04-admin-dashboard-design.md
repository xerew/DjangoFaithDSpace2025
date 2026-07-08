# Admin Dashboard Design

**Date:** 2026-07-04  
**App:** `accounts`  
**Branch:** `improvements/performance-and-responsive`

---

## Overview

A staff-only user management dashboard accessible from the header dropdown under the logged-in user's name. Provides a single-page interface with two tabs — Users and Roles — for managing platform users and Django Groups without leaving the app.

---

## Architecture

### URL structure

All routes under `/accounts/admin/`:

| URL | View | Method |
|-----|------|--------|
| `/accounts/admin/` | `admin_dashboard` | GET |
| `/accounts/admin/edit_user/<id>/` | `admin_edit_user` | POST |
| `/accounts/admin/delete_user/<id>/` | `admin_delete_user` | POST |
| `/accounts/admin/toggle_user/<id>/` | `admin_toggle_user` | POST |
| `/accounts/admin/create_role/` | `admin_create_role` | POST |
| `/accounts/admin/delete_role/<id>/` | `admin_delete_role` | POST |
| `/accounts/admin/rename_role/<id>/` | `admin_rename_role` | POST |
| `/accounts/admin/impersonate/<id>/` | `admin_impersonate` | POST |
| `/accounts/admin/impersonate_exit/` | `admin_impersonate_exit` | POST |

### Access control

All views check `request.user.is_staff or request.user.is_superuser`. Non-staff users receive a 403. The nav link in `head.html` is only rendered for staff/superusers.

### New files

- `accounts/admin_views.py` — all admin view functions (keeps `views.py` from growing further)
- `accounts/templates/accounts/admin_dashboard.html` — single template for the full page

### Modified files

- `accounts/urls.py` — add all `/accounts/admin/` routes
- `templates/head.html` — add "User Management" nav item + impersonation banner

---

## Users Tab

### Stats bar

Four stat cards at the top of the tab:
- **Total Users** — `User.objects.count()`
- **Active** — `User.objects.filter(is_active=True).count()`
- **Staff / Admins** — `User.objects.filter(is_staff=True).count()`
- **Inactive** — `User.objects.filter(is_active=False).count()`

### Search & filter

- Text input: instant client-side filter on username, full name, email
- Dropdown: filter by group (All | Teacher | DSpace Partner | No Role | any other group)
- Toggle: "Show inactive users" (inactive rows hidden by default)

### User table

Columns: Avatar initials · Name / @username · Email · Role badges · Last Login · Status badge · Actions

**Inline quick actions (no modal):**
- **Suspend / Unsuspend** — AJAX POST to `admin_toggle_user`, flips `is_active`, row updates in place
- **Impersonate** — AJAX POST to `admin_impersonate`, page redirects to home as that user
- **Edit** — opens slide-in edit panel from the right
- **Delete** — opens a confirmation modal; AJAX POST on confirm, row removed from table

### Edit panel (slide-in from right)

Opens alongside the table (does not cover the full screen). Contains:
- First name, last name, email — editable text fields
- Toggle: **Staff** (`is_staff`)
- Toggle: **Superuser** (`is_superuser`) — only editable by superusers
- Checkboxes: group memberships (one per existing Django Group)
- **Save** button — AJAX POST to `admin_edit_user`, returns JSON, shows success toast; panel stays open

---

## Roles Tab

### Role cards

One card per Django Group showing:
- Group name
- Member count badge
- Collapsible member list (avatar + name; clicking a name switches to Users tab and opens that user's edit panel)

### Inline rename

Clicking the group name turns it into a text input. Pressing Enter or clicking away triggers AJAX POST to `admin_rename_role`. Card updates in place.

### Delete

Delete button on each card. Disabled (with tooltip "Remove all members first") if the group has any members. On confirm: AJAX POST to `admin_delete_role`, card removed.

### Create new role

Input + "Add Role" button at the top of the Roles tab. AJAX POST to `admin_create_role`, new card appended immediately.

---

## Impersonation

### Starting impersonation

- Admin clicks **Impersonate** on any user row
- AJAX POST to `admin_impersonate/<id>/`
- View stores `request.session['impersonator_id'] = request.user.id`, then switches session to the target user
- Redirects to the home/index page as that user

### Impersonation banner

Injected in `head.html` when `request.session.get('impersonator_id')` is set:

```
⚠ You are viewing as @username — [ Exit Impersonation ]
```

Fixed bar above the header, visible on every page. "Exit Impersonation" POSTs to `admin_impersonate_exit`, which restores the original user from the session and redirects to the admin dashboard.

### Security

- `admin_impersonate` requires the requesting user to be staff/superuser **before** the session switch
- Superusers cannot be impersonated (returns 403)
- `admin_impersonate_exit` only works if `impersonator_id` is present in the session

---

## Navigation

In `head.html`, inside the profile dropdown (after Organizations, before the divider above Sign Out), add:

```html
{% if request.user.is_staff or request.user.is_superuser %}
<li>
  <a class="dropdown-item d-flex align-items-center" href="{% url 'admin_dashboard' %}">
    <span class="profile-icon-wrap profile-icon-purple"><i class="bi bi-shield-lock-fill"></i></span>
    <span>User Management</span>
  </a>
</li>
{% endif %}
```

---

## Interaction model

All mutations (edit, delete, toggle, role actions) use `fetch()` AJAX POST with `X-CSRFToken` header and return JSON `{"success": true}` or `{"success": false, "error": "..."}`. No full-page reloads except for impersonation start/exit (which redirect intentionally).

Client-side search and tab switching require no server round-trips.
