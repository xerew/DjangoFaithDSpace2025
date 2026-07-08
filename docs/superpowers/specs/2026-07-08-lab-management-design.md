# Lab Management Admin Dashboard — Design

**Date:** 2026-07-08  
**Branch:** `improvements/performance-and-responsive`

---

## Overview

Extend the existing staff-only admin dashboard (`/accounts/admin/`) with a third tab — **Labs** — for managing the three experiment/simulation content types used across scenarios: Simulations (PhET iframe), Remote Labs (LabsLand LTI), and VR/AR Labs (QR-code-launched). Staff can list, create, edit, and delete all three types without using Django admin.

---

## Models (all in `authoringtool/models.py`)

### `Simulation` (line 145)
- `name` (CharField 200)
- `iframe_url` (URLField)
- `width` (PositiveIntegerField, default 800)
- `height` (PositiveIntegerField, default 600)
- `allow_fullscreen` (BooleanField, default True)

### `ExperimentLL` (line 161) — Remote Lab (LabsLand LTI)
- `name` (CharField 200)
- `description` (TextField)
- `launch_url` (URLField)
- `consumer_key` (CharField 100)
- `shared_secret` (CharField 100) — sensitive; show in edit panel only, never in table
- `picture` (ImageField) — **not managed by this dashboard**

### `VRARExperiment` (line 178) — VR/AR Lab
- `name` (CharField 200)
- `description` (TextField)
- `launch_url` (URLField)
- `qr_code` (ImageField, auto-generated on `save()` from `launch_url`) — display as thumbnail in table
- `picture` (ImageField) — **not managed by this dashboard**

---

## Architecture

### New file

`accounts/admin_lab_views.py` — nine POST views for lab CRUD, all decorated with `@require_POST` and `@staff_required` (imported from `admin_views.py`). Imports `Simulation`, `ExperimentLL`, `VRARExperiment` from `authoringtool.models`.

### Modified files

- `accounts/admin_views.py` — `admin_dashboard` view gains three extra context vars
- `accounts/urls.py` — nine new URL patterns under `/accounts/admin/`
- `accounts/templates/accounts/admin_dashboard.html` — third "Labs" tab added

---

## URL Structure

All routes under `/accounts/admin/`, all POST:

| URL | View | Action |
|-----|------|--------|
| `admin/simulations/create/` | `admin_create_simulation` | Create simulation |
| `admin/simulations/<id>/edit/` | `admin_edit_simulation` | Edit simulation |
| `admin/simulations/<id>/delete/` | `admin_delete_simulation` | Delete simulation |
| `admin/remote_labs/create/` | `admin_create_remote_lab` | Create remote lab |
| `admin/remote_labs/<id>/edit/` | `admin_edit_remote_lab` | Edit remote lab |
| `admin/remote_labs/<id>/delete/` | `admin_delete_remote_lab` | Delete remote lab |
| `admin/vr_labs/create/` | `admin_create_vr_lab` | Create VR/AR lab |
| `admin/vr_labs/<id>/edit/` | `admin_edit_vr_lab` | Edit VR/AR lab |
| `admin/vr_labs/<id>/delete/` | `admin_delete_vr_lab` | Delete VR/AR lab |

---

## Dashboard GET Context Changes

`admin_dashboard` adds to its existing context:

```python
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment

simulations = Simulation.objects.all().order_by('name')
remote_labs  = ExperimentLL.objects.all().order_by('name')
vr_labs      = VRARExperiment.objects.all().order_by('name')
```

---

## Labs Tab — UI

### Simulation section

Table columns: **Name** | **Iframe URL** (truncated, `text-truncate` max-width) | **Size** (`WxH px`) | **Fullscreen** (badge Yes/No) | **Actions** (Edit · Delete)

Header: "Simulations" + "Add Simulation" button (blue).

### Remote Lab section

Table columns: **Name** | **Description** (truncated, 60 chars) | **Launch URL** (truncated) | **Consumer Key** | **Actions** (Edit · Delete)

`shared_secret` does **not** appear in this table. Header: "Remote Labs (LabsLand)" + "Add Remote Lab" button (cyan).

### VR/AR Lab section

Table columns: **Name** | **Description** (truncated, 60 chars) | **Launch URL** (truncated) | **QR Code** (24×24 px thumbnail `<img>`, or "—" if not yet generated) | **Actions** (Edit · Delete)

Header: "VR/AR Labs" + "Add VR Lab" button (red, matching existing VR color from authoring tool).

---

## Lab Panel (slide-in)

A second slide-in panel (`#lab-panel`, `.lab-panel-overlay`) independent of the existing user edit panel. Same visual style: slides in from the right, overlay behind, close button.

Panel content is swapped by JavaScript based on which Add/Edit button was clicked. The panel `data-type` attribute stores the current resource type (`simulation` | `remote_lab` | `vr_lab`) and `data-id` stores the record ID for edits (`""` for new).

### Simulation fields in panel
- Name (text, required)
- Iframe URL (url, required)
- Width px (number, required, default 800)
- Height px (number, required, default 600)
- Allow Fullscreen (checkbox, default checked)

### Remote Lab fields in panel
- Name (text, required)
- Description (textarea)
- Launch URL (url, required)
- Consumer Key (text, required)
- Shared Secret (text, required) — `type="password"` with reveal toggle

### VR/AR Lab fields in panel
- Name (text, required)
- Description (textarea)
- Launch URL (url, required)
- Note below the URL field: "QR code regenerates automatically on save"

---

## AJAX Response Contracts

### Create simulation — success
```json
{"success": true, "id": 1, "name": "Pendulum Lab", "iframe_url": "https://...", "width": 800, "height": 600, "allow_fullscreen": true}
```

### Edit simulation — success
Same shape as create.

### Create/edit remote lab — success
```json
{"success": true, "id": 2, "name": "LabsLand Pendulum", "description": "...", "launch_url": "https://...", "consumer_key": "abc"}
```
`shared_secret` is never returned in the response.

### Create/edit VR lab — success
```json
{"success": true, "id": 3, "name": "Mars VR", "description": "...", "launch_url": "https://...", "qr_code_url": "/media/qr_codes/...png"}
```
`qr_code_url` is `null` if QR generation failed (non-blocking).

### Any delete — success
```json
{"success": true}
```

### Any error
```json
{"success": false, "error": "Human-readable message"}
```

---

## Interaction Flows

### Add
1. Click "Add [Type]" button
2. `#lab-panel` opens with empty fields for that type, title "Add [Type Name]"
3. Fill in fields, click Save
4. AJAX POST to create URL
5. On success: prepend new `<tr>` to the appropriate table, show success toast, keep panel open
6. On error: show error message in panel

### Edit
1. Each `<tr>` stores all non-secret field values as `data-*` attributes
2. Click Edit button → JS reads data attributes, fills panel fields, sets `data-id` on panel, opens panel with title "Edit [Name]"
3. Modify fields, click Save
4. AJAX POST to edit URL
5. On success: update `<tr>` cells in-place, update data attributes, show success toast

### Delete
1. Click Delete → confirmation modal: "Delete [Name]? This cannot be undone. Any activities using this lab/simulation will lose their experiment link."
2. Confirm → AJAX POST to delete URL
3. On success: remove `<tr>` from table

### VR Lab QR update
After create or edit of a VR lab, the JS updates the QR thumbnail `<img src>` from `qr_code_url` in the response (or hides the thumbnail if null).

---

## Access Control

All nine views use `@require_POST` + `@staff_required` (same as existing admin views). No superuser-only operations — all staff can manage lab content.

Delete is blocked with a `{"success": false, "error": "..."}` response if:
- The record does not exist (`get_object_or_404`)
- Any other DB-level error

No referential integrity block on delete. All three FKs on `Activity` (`simulation`, `experiment_ll`, `vr_ar_experiment`) use `on_delete=SET_NULL` — confirmed in `authoringtool/models.py:230-232`. Deleting a lab resource sets those activity FKs to NULL without cascading.

---

## Tests

New test class `AdminLabViewsTest` in `accounts/tests.py`:

1. `test_anonymous_redirected` — GET to `/accounts/admin/` without login redirects
2. `test_regular_user_forbidden` — staff=False, is_active=True → 403 or redirect on POST
3. `test_create_simulation` — staff POST creates record, returns success JSON
4. `test_edit_simulation` — staff POST updates record
5. `test_delete_simulation` — staff POST deletes record
6. `test_create_remote_lab` — staff POST creates record
7. `test_edit_remote_lab` — shared_secret not in response
8. `test_delete_remote_lab`
9. `test_create_vr_lab` — response includes `qr_code_url` (may be null in test env without media)
10. `test_edit_vr_lab`
11. `test_delete_vr_lab`
