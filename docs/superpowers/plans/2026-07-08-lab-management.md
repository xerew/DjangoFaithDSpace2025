# Lab Management Admin Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Labs" tab to the existing staff admin dashboard (`/accounts/admin/`) with full CRUD for Simulations, Remote Labs (LabsLand LTI), and VR/AR Labs.

**Architecture:** New `accounts/admin_lab_views.py` holds nine POST views. The existing `admin_dashboard` GET gains three context vars. The existing `admin_dashboard.html` template gains a third tab with three table sections plus a dedicated slide-in panel and delete modal — all using the existing `postJSON` / `showToast` / `escHtml` JS helpers.

**Tech Stack:** Django 4.x, Bootstrap 5, Bootstrap Icons, vanilla JS fetch API. Models in `authoringtool/models.py`.

## Global Constraints

- All new AJAX mutation views use `@require_POST` + `@staff_required` (import from `admin_views.py`).
- All AJAX endpoints return `{"success": true, ...fields}` or `{"success": false, "error": "..."}`.
- No page reloads — DOM updates in-place after every AJAX call.
- `shared_secret` must never appear in any AJAX response (neither create nor edit).
- For remote lab edit: if the `shared_secret` POST field is blank, keep the existing value in the DB.
- For VR lab edit: if `launch_url` changes and a QR code already exists, delete the old file and set `qr_code = None` before `save()` so the model's `save()` regenerates it.
- All new tests run under `--settings=faithDev.settings_test` (SQLite in-memory).
- The venv Python is `c:\Users\Nikos A. Grammatikos\Desktop\DjangoFaithDSpace2025-performance-test\djangofaithvenv\Scripts\python.exe`.
- Work on branch `improvements/performance-and-responsive`.
- The working directory for manage.py is `Trust-AI-Platform/`.

---

### Task 1: Backend — views, URLs, context, tests

**Files:**
- Create: `Trust-AI-Platform/accounts/admin_lab_views.py`
- Modify: `Trust-AI-Platform/accounts/admin_views.py` (add context vars to `admin_dashboard`)
- Modify: `Trust-AI-Platform/accounts/urls.py` (9 new routes + import)
- Modify: `Trust-AI-Platform/accounts/tests.py` (new `AdminLabViewsTest` class)

**Interfaces:**
- Produces: nine URL names used verbatim in Task 2's JS:
  - `admin_create_simulation`, `admin_edit_simulation`, `admin_delete_simulation`
  - `admin_create_remote_lab`, `admin_edit_remote_lab`, `admin_delete_remote_lab`
  - `admin_create_vr_lab`, `admin_edit_vr_lab`, `admin_delete_vr_lab`
- Produces: three context vars passed to the template:
  - `simulations` — evaluated list of all `Simulation` objects ordered by name
  - `remote_labs` — evaluated list of all `ExperimentLL` objects ordered by name
  - `vr_labs` — evaluated list of all `VRARExperiment` objects ordered by name
  - `lab_count` — integer: `len(simulations) + len(remote_labs) + len(vr_labs)`

- [ ] **Step 1: Create `accounts/admin_lab_views.py`**

```python
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment
from .admin_views import staff_required


def _qr_url(vr):
    try:
        return vr.qr_code.url if vr.qr_code else None
    except Exception:
        return None


# ── Simulations ────────────────────────────────────────────────────────

@require_POST
@staff_required
def admin_create_simulation(request):
    name = request.POST.get('name', '').strip()
    iframe_url = request.POST.get('iframe_url', '').strip()
    if not name or not iframe_url:
        return JsonResponse({'success': False, 'error': 'Name and iframe URL are required.'})
    try:
        width = int(request.POST.get('width') or 800)
        height = int(request.POST.get('height') or 600)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Width and height must be numbers.'})
    allow_fullscreen = request.POST.get('allow_fullscreen') == 'true'
    sim = Simulation.objects.create(
        name=name, iframe_url=iframe_url,
        width=width, height=height, allow_fullscreen=allow_fullscreen,
    )
    return JsonResponse({
        'success': True, 'id': sim.id, 'name': sim.name,
        'iframe_url': sim.iframe_url, 'width': sim.width,
        'height': sim.height, 'allow_fullscreen': sim.allow_fullscreen,
    })


@require_POST
@staff_required
def admin_edit_simulation(request, sim_id):
    sim = get_object_or_404(Simulation, id=sim_id)
    name = request.POST.get('name', '').strip()
    iframe_url = request.POST.get('iframe_url', '').strip()
    if not name or not iframe_url:
        return JsonResponse({'success': False, 'error': 'Name and iframe URL are required.'})
    try:
        width = int(request.POST.get('width') or 800)
        height = int(request.POST.get('height') or 600)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Width and height must be numbers.'})
    sim.name = name
    sim.iframe_url = iframe_url
    sim.width = width
    sim.height = height
    sim.allow_fullscreen = request.POST.get('allow_fullscreen') == 'true'
    sim.save()
    return JsonResponse({
        'success': True, 'id': sim.id, 'name': sim.name,
        'iframe_url': sim.iframe_url, 'width': sim.width,
        'height': sim.height, 'allow_fullscreen': sim.allow_fullscreen,
    })


@require_POST
@staff_required
def admin_delete_simulation(request, sim_id):
    sim = get_object_or_404(Simulation, id=sim_id)
    sim.delete()
    return JsonResponse({'success': True})


# ── Remote Labs (LabsLand) ─────────────────────────────────────────────

@require_POST
@staff_required
def admin_create_remote_lab(request):
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    consumer_key = request.POST.get('consumer_key', '').strip()
    shared_secret = request.POST.get('shared_secret', '').strip()
    if not name or not launch_url or not consumer_key or not shared_secret:
        return JsonResponse({'success': False, 'error': 'Name, launch URL, consumer key, and shared secret are required.'})
    lab = ExperimentLL.objects.create(
        name=name,
        description=request.POST.get('description', '').strip(),
        launch_url=launch_url,
        consumer_key=consumer_key,
        shared_secret=shared_secret,
    )
    return JsonResponse({
        'success': True, 'id': lab.id, 'name': lab.name,
        'description': lab.description, 'launch_url': lab.launch_url,
        'consumer_key': lab.consumer_key,
        # shared_secret intentionally omitted
    })


@require_POST
@staff_required
def admin_edit_remote_lab(request, lab_id):
    lab = get_object_or_404(ExperimentLL, id=lab_id)
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    consumer_key = request.POST.get('consumer_key', '').strip()
    if not name or not launch_url or not consumer_key:
        return JsonResponse({'success': False, 'error': 'Name, launch URL, and consumer key are required.'})
    shared_secret = request.POST.get('shared_secret', '').strip()
    lab.name = name
    lab.description = request.POST.get('description', '').strip()
    lab.launch_url = launch_url
    lab.consumer_key = consumer_key
    if shared_secret:          # blank = keep existing
        lab.shared_secret = shared_secret
    lab.save()
    return JsonResponse({
        'success': True, 'id': lab.id, 'name': lab.name,
        'description': lab.description, 'launch_url': lab.launch_url,
        'consumer_key': lab.consumer_key,
        # shared_secret intentionally omitted
    })


@require_POST
@staff_required
def admin_delete_remote_lab(request, lab_id):
    lab = get_object_or_404(ExperimentLL, id=lab_id)
    lab.delete()
    return JsonResponse({'success': True})


# ── VR/AR Labs ────────────────────────────────────────────────────────

@require_POST
@staff_required
def admin_create_vr_lab(request):
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    if not name or not launch_url:
        return JsonResponse({'success': False, 'error': 'Name and launch URL are required.'})
    vr = VRARExperiment(
        name=name,
        description=request.POST.get('description', '').strip(),
        launch_url=launch_url,
    )
    vr.save()   # triggers QR generation
    return JsonResponse({
        'success': True, 'id': vr.id, 'name': vr.name,
        'description': vr.description, 'launch_url': vr.launch_url,
        'qr_code_url': _qr_url(vr),
    })


@require_POST
@staff_required
def admin_edit_vr_lab(request, vr_id):
    vr = get_object_or_404(VRARExperiment, id=vr_id)
    name = request.POST.get('name', '').strip()
    launch_url = request.POST.get('launch_url', '').strip()
    if not name or not launch_url:
        return JsonResponse({'success': False, 'error': 'Name and launch URL are required.'})
    # If URL changed, clear old QR so save() regenerates it
    if launch_url != vr.launch_url and vr.qr_code:
        vr.qr_code.delete(save=False)
        vr.qr_code = None
    vr.name = name
    vr.description = request.POST.get('description', '').strip()
    vr.launch_url = launch_url
    vr.save()
    return JsonResponse({
        'success': True, 'id': vr.id, 'name': vr.name,
        'description': vr.description, 'launch_url': vr.launch_url,
        'qr_code_url': _qr_url(vr),
    })


@require_POST
@staff_required
def admin_delete_vr_lab(request, vr_id):
    vr = get_object_or_404(VRARExperiment, id=vr_id)
    vr.delete()
    return JsonResponse({'success': True})
```

- [ ] **Step 2: Update `admin_views.py` — add lab context to `admin_dashboard`**

At the top of `admin_views.py`, add the import:
```python
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment
```

In the `admin_dashboard` view, replace the return statement:
```python
# Before (existing return):
    return render(request, 'accounts/admin_dashboard.html', {
        'all_users': users,
        'groups': groups,
        'stats': stats,
        'is_superuser': is_superuser,
    })

# After:
    simulations = list(Simulation.objects.all().order_by('name'))
    remote_labs = list(ExperimentLL.objects.all().order_by('name'))
    vr_labs = list(VRARExperiment.objects.all().order_by('name'))
    return render(request, 'accounts/admin_dashboard.html', {
        'all_users': users,
        'groups': groups,
        'stats': stats,
        'is_superuser': is_superuser,
        'simulations': simulations,
        'remote_labs': remote_labs,
        'vr_labs': vr_labs,
        'lab_count': len(simulations) + len(remote_labs) + len(vr_labs),
    })
```

- [ ] **Step 3: Update `accounts/urls.py` — add import + 9 new routes**

Add after the existing `from . import admin_views` line:
```python
from . import admin_lab_views
```

Add after the existing impersonate URL patterns:
```python
    path('admin/simulations/create/', admin_lab_views.admin_create_simulation, name='admin_create_simulation'),
    path('admin/simulations/<int:sim_id>/edit/', admin_lab_views.admin_edit_simulation, name='admin_edit_simulation'),
    path('admin/simulations/<int:sim_id>/delete/', admin_lab_views.admin_delete_simulation, name='admin_delete_simulation'),
    path('admin/remote_labs/create/', admin_lab_views.admin_create_remote_lab, name='admin_create_remote_lab'),
    path('admin/remote_labs/<int:lab_id>/edit/', admin_lab_views.admin_edit_remote_lab, name='admin_edit_remote_lab'),
    path('admin/remote_labs/<int:lab_id>/delete/', admin_lab_views.admin_delete_remote_lab, name='admin_delete_remote_lab'),
    path('admin/vr_labs/create/', admin_lab_views.admin_create_vr_lab, name='admin_create_vr_lab'),
    path('admin/vr_labs/<int:vr_id>/edit/', admin_lab_views.admin_edit_vr_lab, name='admin_edit_vr_lab'),
    path('admin/vr_labs/<int:vr_id>/delete/', admin_lab_views.admin_delete_vr_lab, name='admin_delete_vr_lab'),
```

- [ ] **Step 4: Write failing tests**

Add to `accounts/tests.py` after the existing `AdminDashboardAccessTest` class. Add `import json` at the top if not already present.

```python
import json
from authoringtool.models import Simulation, ExperimentLL, VRARExperiment

class AdminLabViewsTest(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('stafflab', password='x', is_staff=True)
        self.regular = User.objects.create_user('regularlab', password='x')
        self.client.force_login(self.staff)

    # Access control
    def test_regular_user_forbidden(self):
        self.client.force_login(self.regular)
        r = self.client.post('/accounts/admin/simulations/create/', {
            'name': 'X', 'iframe_url': 'https://x.com', 'width': '800', 'height': '600', 'allow_fullscreen': 'true'
        })
        self.assertEqual(r.status_code, 403)

    # Simulation CRUD
    def test_create_simulation(self):
        r = self.client.post('/accounts/admin/simulations/create/', {
            'name': 'PhET Pendulum', 'iframe_url': 'https://phet.colorado.edu/',
            'width': '800', 'height': '600', 'allow_fullscreen': 'true',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertTrue(Simulation.objects.filter(name='PhET Pendulum').exists())
        self.assertEqual(data['width'], 800)

    def test_create_simulation_missing_fields(self):
        r = self.client.post('/accounts/admin/simulations/create/', {'name': '', 'iframe_url': ''})
        data = json.loads(r.content)
        self.assertFalse(data['success'])

    def test_edit_simulation(self):
        sim = Simulation.objects.create(name='Old', iframe_url='https://old.com', width=800, height=600)
        r = self.client.post(f'/accounts/admin/simulations/{sim.id}/edit/', {
            'name': 'New', 'iframe_url': 'https://new.com', 'width': '1024', 'height': '768', 'allow_fullscreen': 'false',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        sim.refresh_from_db()
        self.assertEqual(sim.name, 'New')
        self.assertEqual(sim.width, 1024)
        self.assertFalse(sim.allow_fullscreen)

    def test_delete_simulation(self):
        sim = Simulation.objects.create(name='ToDelete', iframe_url='https://x.com', width=800, height=600)
        r = self.client.post(f'/accounts/admin/simulations/{sim.id}/delete/')
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertFalse(Simulation.objects.filter(id=sim.id).exists())

    # Remote Lab CRUD
    def test_create_remote_lab(self):
        r = self.client.post('/accounts/admin/remote_labs/create/', {
            'name': 'LabsLand Pendulum', 'launch_url': 'https://labsland.com/lti',
            'consumer_key': 'key123', 'shared_secret': 'secret456', 'description': 'A lab',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertNotIn('shared_secret', data)
        self.assertTrue(ExperimentLL.objects.filter(name='LabsLand Pendulum').exists())

    def test_edit_remote_lab_blank_secret_preserved(self):
        lab = ExperimentLL.objects.create(
            name='Lab', launch_url='https://x.com', consumer_key='ck', shared_secret='original_secret'
        )
        r = self.client.post(f'/accounts/admin/remote_labs/{lab.id}/edit/', {
            'name': 'Lab Updated', 'launch_url': 'https://x.com',
            'consumer_key': 'ck', 'shared_secret': '',  # blank — keep existing
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertNotIn('shared_secret', data)
        lab.refresh_from_db()
        self.assertEqual(lab.shared_secret, 'original_secret')  # not changed

    def test_delete_remote_lab(self):
        lab = ExperimentLL.objects.create(name='X', launch_url='https://x.com', consumer_key='ck', shared_secret='ss')
        r = self.client.post(f'/accounts/admin/remote_labs/{lab.id}/delete/')
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertFalse(ExperimentLL.objects.filter(id=lab.id).exists())

    # VR Lab CRUD
    def test_create_vr_lab(self):
        r = self.client.post('/accounts/admin/vr_labs/create/', {
            'name': 'Mars VR', 'launch_url': 'https://vr.example.com/mars', 'description': 'VR lab',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertIn('qr_code_url', data)   # key present (may be None in test env)
        self.assertTrue(VRARExperiment.objects.filter(name='Mars VR').exists())

    def test_edit_vr_lab(self):
        vr = VRARExperiment.objects.create(name='Old VR', launch_url='https://old.com', description='')
        r = self.client.post(f'/accounts/admin/vr_labs/{vr.id}/edit/', {
            'name': 'New VR', 'launch_url': 'https://new.com', 'description': 'Updated',
        })
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertEqual(data['name'], 'New VR')
        self.assertIn('qr_code_url', data)

    def test_delete_vr_lab(self):
        vr = VRARExperiment.objects.create(name='X', launch_url='https://x.com', description='')
        r = self.client.post(f'/accounts/admin/vr_labs/{vr.id}/delete/')
        data = json.loads(r.content)
        self.assertTrue(data['success'])
        self.assertFalse(VRARExperiment.objects.filter(id=vr.id).exists())
```

- [ ] **Step 5: Run tests — all should fail (views not yet wired to URLs)**

```
cd Trust-AI-Platform
djangofaithvenv\Scripts\python.exe manage.py test accounts.tests.AdminLabViewsTest --settings=faithDev.settings_test -v 2
```

Expected: most tests FAIL with 404 or ImportError. That's correct — views not wired yet.

- [ ] **Step 6: Wire up — run tests again, all should pass**

After Steps 1–3 are complete, re-run:
```
djangofaithvenv\Scripts\python.exe manage.py test accounts.tests.AdminLabViewsTest --settings=faithDev.settings_test -v 2
```

Expected: all 11 tests PASS. Also run existing suite to confirm no regressions:
```
djangofaithvenv\Scripts\python.exe manage.py test accounts.tests.AdminDashboardAccessTest --settings=faithDev.settings_test -v 2
```

Expected: 9/9 PASS.

- [ ] **Step 7: Commit**

```bash
git add Trust-AI-Platform/accounts/admin_lab_views.py \
        Trust-AI-Platform/accounts/admin_views.py \
        Trust-AI-Platform/accounts/urls.py \
        Trust-AI-Platform/accounts/tests.py
git commit -m "Add lab management backend: 9 CRUD views, URL routes, tests"
```

---

### Task 2: Template — Labs tab + panel + JS

**Files:**
- Modify: `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`

**Interfaces:**
- Consumes: context vars `simulations`, `remote_labs`, `vr_labs`, `lab_count` from Task 1
- Consumes: URL paths (hardcoded strings, consistent with existing pattern in this file):
  - `/accounts/admin/simulations/<id>/edit/`, `/accounts/admin/simulations/create/`, etc.
- Consumes: existing JS helpers `postJSON`, `showToast`, `escHtml` (already in the template)

- [ ] **Step 1: Add Labs tab button to the nav**

Find the existing tabs nav:
```html
    <ul class="nav nav-tabs mb-0" id="adminTabs" role="tablist" style="border-bottom:2px solid #e8edf5;">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" id="tab-users-btn" ...>
```

Add a third `<li>` after the Roles tab `</li>`:
```html
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="tab-labs-btn" data-bs-toggle="tab" data-bs-target="#tab-labs" type="button" role="tab">
          <i class="bi bi-flask me-1"></i> Labs
          <span class="badge bg-secondary ms-1" style="font-size:11px;">{{ lab_count }}</span>
        </button>
      </li>
```

- [ ] **Step 2: Add Labs tab pane — three table sections**

Find the closing tag `</div><!-- /tab-roles -->` and insert the Labs tab pane immediately after it, before `</div><!-- /tab-content -->`:

```html
      <!-- Labs Tab -->
      <div class="tab-pane fade" id="tab-labs" role="tabpanel">

        <!-- Simulations -->
        <div class="d-flex align-items-center justify-content-between mb-3">
          <h5 class="mb-0" style="color:#012970;font-size:15px;font-weight:700;">
            <i class="bi bi-display me-2 text-primary"></i>Simulations
            <span class="badge bg-secondary ms-2" style="font-size:11px;">{{ simulations|length }}</span>
          </h5>
          <button class="btn btn-sm btn-primary" onclick="openLabPanel('simulation', null)">
            <i class="bi bi-plus-lg me-1"></i>Add Simulation
          </button>
        </div>
        <div class="card mb-4" style="border-radius:12px;overflow:hidden;">
          <div class="table-responsive">
            <table class="table table-hover mb-0">
              <thead style="background:#f8faff;">
                <tr>
                  <th style="font-size:12px;font-weight:600;color:#888;padding:12px 16px;">Name</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Iframe URL</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Size</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Fullscreen</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="simTableBody">
                {% for sim in simulations %}
                <tr class="sim-row"
                    data-sim-id="{{ sim.id }}"
                    data-name="{{ sim.name }}"
                    data-iframe-url="{{ sim.iframe_url }}"
                    data-width="{{ sim.width }}"
                    data-height="{{ sim.height }}"
                    data-allow-fullscreen="{{ sim.allow_fullscreen|yesno:'true,false' }}">
                  <td style="padding:12px 16px;font-weight:600;font-size:13px;color:#012970;">{{ sim.name }}</td>
                  <td><span class="text-truncate d-inline-block" style="max-width:280px;font-size:12px;color:#666;" title="{{ sim.iframe_url }}">{{ sim.iframe_url }}</span></td>
                  <td style="font-size:13px;">{{ sim.width }}&times;{{ sim.height }}px</td>
                  <td>{% if sim.allow_fullscreen %}<span class="badge bg-success">Yes</span>{% else %}<span class="badge bg-secondary">No</span>{% endif %}</td>
                  <td style="white-space:nowrap;">
                    <div class="d-flex gap-1">
                      <button class="btn btn-sm btn-outline-primary" title="Edit"
                              onclick="openLabPanel('simulation', this.closest('tr'))"><i class="bi bi-pencil"></i></button>
                      <button class="btn btn-sm btn-outline-danger" title="Delete"
                              data-lab-type="simulation" data-lab-id="{{ sim.id }}" data-lab-name="{{ sim.name }}"
                              onclick="openDeleteLabModal(this)"><i class="bi bi-trash"></i></button>
                    </div>
                  </td>
                </tr>
                {% empty %}
                <tr id="simEmptyRow"><td colspan="5" class="text-center text-muted py-4" style="font-size:13px;">No simulations yet.</td></tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Remote Labs -->
        <div class="d-flex align-items-center justify-content-between mb-3">
          <h5 class="mb-0" style="color:#012970;font-size:15px;font-weight:700;">
            <i class="bi bi-hdd-network me-2" style="color:#0891b2;"></i>Remote Labs (LabsLand)
            <span class="badge bg-secondary ms-2" style="font-size:11px;">{{ remote_labs|length }}</span>
          </h5>
          <button class="btn btn-sm" style="background:#0891b2;color:#fff;" onclick="openLabPanel('remote_lab', null)">
            <i class="bi bi-plus-lg me-1"></i>Add Remote Lab
          </button>
        </div>
        <div class="card mb-4" style="border-radius:12px;overflow:hidden;">
          <div class="table-responsive">
            <table class="table table-hover mb-0">
              <thead style="background:#f8faff;">
                <tr>
                  <th style="font-size:12px;font-weight:600;color:#888;padding:12px 16px;">Name</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Description</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Launch URL</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Consumer Key</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="remoteLabTableBody">
                {% for lab in remote_labs %}
                <tr class="remote-lab-row"
                    data-rlab-id="{{ lab.id }}"
                    data-name="{{ lab.name }}"
                    data-description="{{ lab.description }}"
                    data-launch-url="{{ lab.launch_url }}"
                    data-consumer-key="{{ lab.consumer_key }}">
                  <td style="padding:12px 16px;font-weight:600;font-size:13px;color:#012970;">{{ lab.name }}</td>
                  <td><span style="font-size:12px;color:#666;">{{ lab.description|truncatechars:60 }}</span></td>
                  <td><span class="text-truncate d-inline-block" style="max-width:200px;font-size:12px;color:#666;" title="{{ lab.launch_url }}">{{ lab.launch_url }}</span></td>
                  <td><code style="font-size:11px;">{{ lab.consumer_key }}</code></td>
                  <td style="white-space:nowrap;">
                    <div class="d-flex gap-1">
                      <button class="btn btn-sm btn-outline-primary" title="Edit"
                              onclick="openLabPanel('remote_lab', this.closest('tr'))"><i class="bi bi-pencil"></i></button>
                      <button class="btn btn-sm btn-outline-danger" title="Delete"
                              data-lab-type="remote_lab" data-lab-id="{{ lab.id }}" data-lab-name="{{ lab.name }}"
                              onclick="openDeleteLabModal(this)"><i class="bi bi-trash"></i></button>
                    </div>
                  </td>
                </tr>
                {% empty %}
                <tr id="remoteLabEmptyRow"><td colspan="5" class="text-center text-muted py-4" style="font-size:13px;">No remote labs yet.</td></tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>

        <!-- VR/AR Labs -->
        <div class="d-flex align-items-center justify-content-between mb-3">
          <h5 class="mb-0" style="color:#012970;font-size:15px;font-weight:700;">
            <i class="bi bi-badge-vr me-2 text-danger"></i>VR/AR Labs
            <span class="badge bg-secondary ms-2" style="font-size:11px;">{{ vr_labs|length }}</span>
          </h5>
          <button class="btn btn-sm btn-danger" onclick="openLabPanel('vr_lab', null)">
            <i class="bi bi-plus-lg me-1"></i>Add VR Lab
          </button>
        </div>
        <div class="card mb-4" style="border-radius:12px;overflow:hidden;">
          <div class="table-responsive">
            <table class="table table-hover mb-0">
              <thead style="background:#f8faff;">
                <tr>
                  <th style="font-size:12px;font-weight:600;color:#888;padding:12px 16px;">Name</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Description</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Launch URL</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">QR Code</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="vrLabTableBody">
                {% for vr in vr_labs %}
                <tr class="vr-lab-row"
                    data-vr-id="{{ vr.id }}"
                    data-name="{{ vr.name }}"
                    data-description="{{ vr.description }}"
                    data-launch-url="{{ vr.launch_url }}"
                    data-qr-url="{% if vr.qr_code %}{{ vr.qr_code.url }}{% endif %}">
                  <td style="padding:12px 16px;font-weight:600;font-size:13px;color:#012970;">{{ vr.name }}</td>
                  <td><span style="font-size:12px;color:#666;">{{ vr.description|truncatechars:60 }}</span></td>
                  <td><span class="text-truncate d-inline-block" style="max-width:200px;font-size:12px;color:#666;" title="{{ vr.launch_url }}">{{ vr.launch_url }}</span></td>
                  <td>
                    {% if vr.qr_code %}
                    <img src="{{ vr.qr_code.url }}" class="vr-qr-thumb" alt="QR"
                         style="width:32px;height:32px;border-radius:4px;cursor:pointer;"
                         onclick="window.open(this.src,'_blank')">
                    {% else %}
                    <span class="text-muted" style="font-size:12px;">—</span>
                    {% endif %}
                  </td>
                  <td style="white-space:nowrap;">
                    <div class="d-flex gap-1">
                      <button class="btn btn-sm btn-outline-primary" title="Edit"
                              onclick="openLabPanel('vr_lab', this.closest('tr'))"><i class="bi bi-pencil"></i></button>
                      <button class="btn btn-sm btn-outline-danger" title="Delete"
                              data-lab-type="vr_lab" data-lab-id="{{ vr.id }}" data-lab-name="{{ vr.name }}"
                              onclick="openDeleteLabModal(this)"><i class="bi bi-trash"></i></button>
                    </div>
                  </td>
                </tr>
                {% empty %}
                <tr id="vrLabEmptyRow"><td colspan="5" class="text-center text-muted py-4" style="font-size:13px;">No VR/AR labs yet.</td></tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>

      </div><!-- /tab-labs -->
```

- [ ] **Step 3: Add lab slide panel + delete modal**

Insert directly after the existing `<div id="editPanelOverlay" ...></div>` line:

```html
<!-- Lab slide panel -->
<div id="labPanel" class="edit-panel">
  <div class="edit-panel-header">
    <h5 id="labPanelTitle">Add Lab</h5>
    <button type="button" class="btn-close" onclick="closeLabPanel()"></button>
  </div>
  <div class="edit-panel-body">
    <input type="hidden" id="labPanelType">
    <input type="hidden" id="labPanelId">

    <!-- Simulation fields -->
    <div id="simFields">
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Name <span class="text-danger">*</span></label>
        <input type="text" id="simName" class="form-control form-control-sm">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Iframe URL <span class="text-danger">*</span></label>
        <input type="url" id="simIframeUrl" class="form-control form-control-sm">
      </div>
      <div class="row g-2 mb-3">
        <div class="col-6">
          <label class="form-label fw-semibold" style="font-size:13px;">Width (px)</label>
          <input type="number" id="simWidth" class="form-control form-control-sm" value="800" min="100">
        </div>
        <div class="col-6">
          <label class="form-label fw-semibold" style="font-size:13px;">Height (px)</label>
          <input type="number" id="simHeight" class="form-control form-control-sm" value="600" min="100">
        </div>
      </div>
      <div class="mb-3">
        <div class="form-check form-switch">
          <input class="form-check-input" type="checkbox" id="simAllowFullscreen" checked>
          <label class="form-check-label" for="simAllowFullscreen" style="font-size:13px;">Allow Fullscreen</label>
        </div>
      </div>
    </div>

    <!-- Remote Lab fields -->
    <div id="remoteLabFields" style="display:none;">
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Name <span class="text-danger">*</span></label>
        <input type="text" id="rlName" class="form-control form-control-sm">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Description</label>
        <textarea id="rlDescription" class="form-control form-control-sm" rows="2"></textarea>
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Launch URL <span class="text-danger">*</span></label>
        <input type="url" id="rlLaunchUrl" class="form-control form-control-sm">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Consumer Key <span class="text-danger">*</span></label>
        <input type="text" id="rlConsumerKey" class="form-control form-control-sm">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Shared Secret <span class="text-danger">*</span></label>
        <div class="input-group input-group-sm">
          <input type="password" id="rlSharedSecret" class="form-control">
          <button class="btn btn-outline-secondary" type="button" onclick="toggleSecretVisibility()">
            <i class="bi bi-eye" id="secretEyeIcon"></i>
          </button>
        </div>
        <div class="form-text" style="font-size:11px;" id="rlSecretHint"></div>
      </div>
    </div>

    <!-- VR/AR Lab fields -->
    <div id="vrLabFields" style="display:none;">
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Name <span class="text-danger">*</span></label>
        <input type="text" id="vrName" class="form-control form-control-sm">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Description</label>
        <textarea id="vrDescription" class="form-control form-control-sm" rows="2"></textarea>
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold" style="font-size:13px;">Launch URL <span class="text-danger">*</span></label>
        <input type="url" id="vrLaunchUrl" class="form-control form-control-sm">
        <div class="form-text" style="font-size:11px;">QR code regenerates automatically on save</div>
      </div>
    </div>

    <button class="btn btn-primary w-100" onclick="saveLabPanel()">
      <i class="bi bi-check-lg me-1"></i>Save
    </button>
    <div id="labPanelError" class="text-danger mt-2" style="display:none;font-size:13px;"></div>
  </div>
</div>
<div id="labPanelOverlay" class="edit-panel-overlay" onclick="closeLabPanel()"></div>

<!-- Delete Lab modal -->
<div class="modal fade" id="deleteLabModal" tabindex="-1" aria-hidden="true">
  <div class="modal-dialog modal-dialog-centered">
    <div class="modal-content" style="border-radius:12px;overflow:hidden;">
      <div class="modal-header" style="background:#fff8f8;border-bottom:1px solid #fee2e2;">
        <h5 class="modal-title" style="color:#dc2626;font-size:15px;font-weight:700;">
          <i class="bi bi-exclamation-triangle me-2"></i>Delete
        </h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body" style="font-size:14px;padding:20px 24px;">
        Delete <strong id="deleteLabName"></strong>? Activities using this will lose their experiment link.
      </div>
      <div class="modal-footer" style="border-top:1px solid #f3f4f6;padding:12px 24px;">
        <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-danger btn-sm" id="deleteLabConfirmBtn">
          <i class="bi bi-trash me-1"></i>Delete
        </button>
      </div>
    </div>
  </div>
</div>
```

- [ ] **Step 4: Add lab JavaScript**

Find the closing `</script>` tag (last line before `{% endblock atcontent %}`) and insert all lab JS immediately before it:

```javascript
// ── Lab panel ─────────────────────────────────────────────────────────────
let _labDeleteType = null, _labDeleteId = null;

function openLabPanel(type, row) {
  const isEdit = !!row;
  document.getElementById('labPanelType').value = type;

  // resolve row ID based on type-specific data attribute
  let rowId = '';
  if (isEdit) {
    if (type === 'simulation') rowId = row.dataset.simId;
    else if (type === 'remote_lab') rowId = row.dataset.rlabId;
    else rowId = row.dataset.vrId;
  }
  document.getElementById('labPanelId').value = rowId;

  document.getElementById('simFields').style.display = 'none';
  document.getElementById('remoteLabFields').style.display = 'none';
  document.getElementById('vrLabFields').style.display = 'none';

  const labels = {simulation: 'Simulation', remote_lab: 'Remote Lab', vr_lab: 'VR/AR Lab'};

  if (type === 'simulation') {
    document.getElementById('simFields').style.display = '';
    document.getElementById('simName').value = isEdit ? row.dataset.name : '';
    document.getElementById('simIframeUrl').value = isEdit ? row.dataset.iframeUrl : '';
    document.getElementById('simWidth').value = isEdit ? row.dataset.width : '800';
    document.getElementById('simHeight').value = isEdit ? row.dataset.height : '600';
    document.getElementById('simAllowFullscreen').checked = isEdit ? row.dataset.allowFullscreen === 'true' : true;
  } else if (type === 'remote_lab') {
    document.getElementById('remoteLabFields').style.display = '';
    document.getElementById('rlName').value = isEdit ? row.dataset.name : '';
    document.getElementById('rlDescription').value = isEdit ? row.dataset.description : '';
    document.getElementById('rlLaunchUrl').value = isEdit ? row.dataset.launchUrl : '';
    document.getElementById('rlConsumerKey').value = isEdit ? row.dataset.consumerKey : '';
    document.getElementById('rlSharedSecret').value = '';
    document.getElementById('rlSecretHint').textContent = isEdit ? 'Leave blank to keep current secret.' : '';
  } else {
    document.getElementById('vrLabFields').style.display = '';
    document.getElementById('vrName').value = isEdit ? row.dataset.name : '';
    document.getElementById('vrDescription').value = isEdit ? row.dataset.description : '';
    document.getElementById('vrLaunchUrl').value = isEdit ? row.dataset.launchUrl : '';
  }

  document.getElementById('labPanelTitle').textContent = (isEdit ? 'Edit ' : 'Add ') + labels[type];
  document.getElementById('labPanelError').style.display = 'none';
  document.getElementById('labPanel').classList.add('open');
  document.getElementById('labPanelOverlay').classList.add('open');
}

function closeLabPanel() {
  document.getElementById('labPanel').classList.remove('open');
  document.getElementById('labPanelOverlay').classList.remove('open');
}

function toggleSecretVisibility() {
  const inp = document.getElementById('rlSharedSecret');
  const icon = document.getElementById('secretEyeIcon');
  if (inp.type === 'password') { inp.type = 'text'; icon.className = 'bi bi-eye-slash'; }
  else { inp.type = 'password'; icon.className = 'bi bi-eye'; }
}

function saveLabPanel() {
  const type = document.getElementById('labPanelType').value;
  const id   = document.getElementById('labPanelId').value;
  let url, data;

  if (type === 'simulation') {
    url  = id ? '/accounts/admin/simulations/' + id + '/edit/' : '/accounts/admin/simulations/create/';
    data = {
      name: document.getElementById('simName').value,
      iframe_url: document.getElementById('simIframeUrl').value,
      width: document.getElementById('simWidth').value,
      height: document.getElementById('simHeight').value,
      allow_fullscreen: document.getElementById('simAllowFullscreen').checked ? 'true' : 'false',
    };
  } else if (type === 'remote_lab') {
    url  = id ? '/accounts/admin/remote_labs/' + id + '/edit/' : '/accounts/admin/remote_labs/create/';
    data = {
      name: document.getElementById('rlName').value,
      description: document.getElementById('rlDescription').value,
      launch_url: document.getElementById('rlLaunchUrl').value,
      consumer_key: document.getElementById('rlConsumerKey').value,
      shared_secret: document.getElementById('rlSharedSecret').value,
    };
  } else {
    url  = id ? '/accounts/admin/vr_labs/' + id + '/edit/' : '/accounts/admin/vr_labs/create/';
    data = {
      name: document.getElementById('vrName').value,
      description: document.getElementById('vrDescription').value,
      launch_url: document.getElementById('vrLaunchUrl').value,
    };
  }

  postJSON(url, data).then(function(res) {
    if (!res.success) {
      const err = document.getElementById('labPanelError');
      err.textContent = res.error;
      err.style.display = 'block';
      return;
    }
    showToast('Saved.', false);
    closeLabPanel();
    if (id) { updateLabRow(type, res); } else { prependLabRow(type, res); }
  });
}

function updateLabRow(type, res) {
  let row, cells;
  if (type === 'simulation') {
    row = document.querySelector('[data-sim-id="' + res.id + '"]');
    if (!row) return;
    row.dataset.name = res.name;
    row.dataset.iframeUrl = res.iframe_url;
    row.dataset.width = res.width;
    row.dataset.height = res.height;
    row.dataset.allowFullscreen = res.allow_fullscreen ? 'true' : 'false';
    cells = row.querySelectorAll('td');
    cells[0].textContent = res.name;
    cells[0].style.fontWeight = '600';
    cells[0].style.fontSize = '13px';
    cells[0].style.color = '#012970';
    const urlSpan = cells[1].querySelector('span');
    if (urlSpan) { urlSpan.textContent = res.iframe_url; urlSpan.title = res.iframe_url; }
    cells[2].textContent = res.width + '\xd7' + res.height + 'px';
    cells[3].innerHTML = res.allow_fullscreen ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>';
  } else if (type === 'remote_lab') {
    row = document.querySelector('[data-rlab-id="' + res.id + '"]');
    if (!row) return;
    row.dataset.name = res.name;
    row.dataset.description = res.description || '';
    row.dataset.launchUrl = res.launch_url;
    row.dataset.consumerKey = res.consumer_key;
    cells = row.querySelectorAll('td');
    cells[0].textContent = res.name;
    const descSpan = cells[1].querySelector('span');
    if (descSpan) descSpan.textContent = (res.description || '').substring(0, 60);
    const urlSpan2 = cells[2].querySelector('span');
    if (urlSpan2) { urlSpan2.textContent = res.launch_url; urlSpan2.title = res.launch_url; }
    const ckCode = cells[3].querySelector('code');
    if (ckCode) ckCode.textContent = res.consumer_key;
  } else {
    row = document.querySelector('[data-vr-id="' + res.id + '"]');
    if (!row) return;
    row.dataset.name = res.name;
    row.dataset.description = res.description || '';
    row.dataset.launchUrl = res.launch_url;
    if (res.qr_code_url) row.dataset.qrUrl = res.qr_code_url;
    cells = row.querySelectorAll('td');
    cells[0].textContent = res.name;
    const descSpan2 = cells[1].querySelector('span');
    if (descSpan2) descSpan2.textContent = (res.description || '').substring(0, 60);
    const urlSpan3 = cells[2].querySelector('span');
    if (urlSpan3) { urlSpan3.textContent = res.launch_url; urlSpan3.title = res.launch_url; }
    if (res.qr_code_url) {
      cells[3].innerHTML = '<img src="' + res.qr_code_url + '" class="vr-qr-thumb" alt="QR" style="width:32px;height:32px;border-radius:4px;cursor:pointer;" onclick="window.open(this.src,\'_blank\')">';
    }
  }
}

function prependLabRow(type, res) {
  let tbody, emptyId, html;

  if (type === 'simulation') {
    tbody   = document.getElementById('simTableBody');
    emptyId = 'simEmptyRow';
    html = '<tr class="sim-row" data-sim-id="' + res.id + '" data-name="' + escHtml(res.name) + '"' +
      ' data-iframe-url="' + escHtml(res.iframe_url) + '" data-width="' + res.width + '"' +
      ' data-height="' + res.height + '" data-allow-fullscreen="' + (res.allow_fullscreen ? 'true' : 'false') + '">' +
      '<td style="padding:12px 16px;font-weight:600;font-size:13px;color:#012970;">' + escHtml(res.name) + '</td>' +
      '<td><span class="text-truncate d-inline-block" style="max-width:280px;font-size:12px;color:#666;" title="' + escHtml(res.iframe_url) + '">' + escHtml(res.iframe_url) + '</span></td>' +
      '<td style="font-size:13px;">' + res.width + '\xd7' + res.height + 'px</td>' +
      '<td>' + (res.allow_fullscreen ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>') + '</td>' +
      '<td style="white-space:nowrap;"><div class="d-flex gap-1">' +
      '<button class="btn btn-sm btn-outline-primary" title="Edit" onclick="openLabPanel(\'simulation\', this.closest(\'tr\'))"><i class="bi bi-pencil"></i></button>' +
      '<button class="btn btn-sm btn-outline-danger" title="Delete" data-lab-type="simulation" data-lab-id="' + res.id + '" data-lab-name="' + escHtml(res.name) + '" onclick="openDeleteLabModal(this)"><i class="bi bi-trash"></i></button>' +
      '</div></td></tr>';

  } else if (type === 'remote_lab') {
    tbody   = document.getElementById('remoteLabTableBody');
    emptyId = 'remoteLabEmptyRow';
    const desc = (res.description || '').substring(0, 60);
    html = '<tr class="remote-lab-row" data-rlab-id="' + res.id + '" data-name="' + escHtml(res.name) + '"' +
      ' data-description="' + escHtml(res.description || '') + '" data-launch-url="' + escHtml(res.launch_url) + '"' +
      ' data-consumer-key="' + escHtml(res.consumer_key) + '">' +
      '<td style="padding:12px 16px;font-weight:600;font-size:13px;color:#012970;">' + escHtml(res.name) + '</td>' +
      '<td><span style="font-size:12px;color:#666;">' + escHtml(desc) + '</span></td>' +
      '<td><span class="text-truncate d-inline-block" style="max-width:200px;font-size:12px;color:#666;" title="' + escHtml(res.launch_url) + '">' + escHtml(res.launch_url) + '</span></td>' +
      '<td><code style="font-size:11px;">' + escHtml(res.consumer_key) + '</code></td>' +
      '<td style="white-space:nowrap;"><div class="d-flex gap-1">' +
      '<button class="btn btn-sm btn-outline-primary" title="Edit" onclick="openLabPanel(\'remote_lab\', this.closest(\'tr\'))"><i class="bi bi-pencil"></i></button>' +
      '<button class="btn btn-sm btn-outline-danger" title="Delete" data-lab-type="remote_lab" data-lab-id="' + res.id + '" data-lab-name="' + escHtml(res.name) + '" onclick="openDeleteLabModal(this)"><i class="bi bi-trash"></i></button>' +
      '</div></td></tr>';

  } else {
    tbody   = document.getElementById('vrLabTableBody');
    emptyId = 'vrLabEmptyRow';
    const vrDesc = (res.description || '').substring(0, 60);
    const qrHtml = res.qr_code_url
      ? '<img src="' + res.qr_code_url + '" class="vr-qr-thumb" alt="QR" style="width:32px;height:32px;border-radius:4px;cursor:pointer;" onclick="window.open(this.src,\'_blank\')">'
      : '<span class="text-muted" style="font-size:12px;">—</span>';
    html = '<tr class="vr-lab-row" data-vr-id="' + res.id + '" data-name="' + escHtml(res.name) + '"' +
      ' data-description="' + escHtml(res.description || '') + '" data-launch-url="' + escHtml(res.launch_url) + '"' +
      ' data-qr-url="' + (res.qr_code_url || '') + '">' +
      '<td style="padding:12px 16px;font-weight:600;font-size:13px;color:#012970;">' + escHtml(res.name) + '</td>' +
      '<td><span style="font-size:12px;color:#666;">' + escHtml(vrDesc) + '</span></td>' +
      '<td><span class="text-truncate d-inline-block" style="max-width:200px;font-size:12px;color:#666;" title="' + escHtml(res.launch_url) + '">' + escHtml(res.launch_url) + '</span></td>' +
      '<td>' + qrHtml + '</td>' +
      '<td style="white-space:nowrap;"><div class="d-flex gap-1">' +
      '<button class="btn btn-sm btn-outline-primary" title="Edit" onclick="openLabPanel(\'vr_lab\', this.closest(\'tr\'))"><i class="bi bi-pencil"></i></button>' +
      '<button class="btn btn-sm btn-outline-danger" title="Delete" data-lab-type="vr_lab" data-lab-id="' + res.id + '" data-lab-name="' + escHtml(res.name) + '" onclick="openDeleteLabModal(this)"><i class="bi bi-trash"></i></button>' +
      '</div></td></tr>';
  }

  const emptyRow = document.getElementById(emptyId);
  if (emptyRow) emptyRow.remove();
  tbody.insertAdjacentHTML('afterbegin', html);
}

function openDeleteLabModal(btn) {
  _labDeleteType = btn.dataset.labType;
  _labDeleteId   = btn.dataset.labId;
  document.getElementById('deleteLabName').textContent = btn.dataset.labName;
  bootstrap.Modal.getOrCreateInstance(document.getElementById('deleteLabModal')).show();
}

document.getElementById('deleteLabConfirmBtn').addEventListener('click', function() {
  if (!_labDeleteType || !_labDeleteId) return;
  const urlMap = {
    simulation: '/accounts/admin/simulations/' + _labDeleteId + '/delete/',
    remote_lab: '/accounts/admin/remote_labs/' + _labDeleteId + '/delete/',
    vr_lab:     '/accounts/admin/vr_labs/'     + _labDeleteId + '/delete/',
  };
  const selMap = {
    simulation: '[data-sim-id="'  + _labDeleteId + '"]',
    remote_lab: '[data-rlab-id="' + _labDeleteId + '"]',
    vr_lab:     '[data-vr-id="'   + _labDeleteId + '"]',
  };
  postJSON(urlMap[_labDeleteType], {}).then(function(data) {
    bootstrap.Modal.getInstance(document.getElementById('deleteLabModal')).hide();
    if (data.success) {
      showToast('Deleted.', false);
      const row = document.querySelector(selMap[_labDeleteType]);
      if (row) row.remove();
    } else {
      showToast(data.error, true);
    }
    _labDeleteType = null; _labDeleteId = null;
  });
});
```

- [ ] **Step 5: Run existing tests to confirm no template regression**

```
cd Trust-AI-Platform
djangofaithvenv\Scripts\python.exe manage.py test accounts.tests --settings=faithDev.settings_test -v 2
```

Expected: all 20 tests pass (9 AdminDashboardAccessTest + 11 AdminLabViewsTest).

- [ ] **Step 6: Commit**

```bash
git add Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html
git commit -m "Add Labs tab to admin dashboard: simulations, remote labs, VR/AR labs"
```

---

## Self-Review Notes

- `data-rlab-id` is used for remote lab rows (not `data-lab-id`) to avoid collision with VR row attribute naming — JS `updateLabRow` and `prependLabRow` use `[data-rlab-id=...]` as the selector.
- `shared_secret` is omitted from all AJAX responses. Edit: blank secret POST preserves existing value in DB.
- VR lab QR: edit clears old file + sets `qr_code=None` before `save()` only when `launch_url` changed, ensuring the model's `save()` regenerates correctly.
- The `_qr_url()` helper wraps `.url` access in try/except to avoid `ValueError` when MEDIA_ROOT isn't configured in tests.
- No new migrations required — no model changes.
- Remote lab rows use `data-rlab-id` (not `data-lab-id`) to avoid any selector ambiguity with future additions.
