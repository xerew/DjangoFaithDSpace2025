# Admin Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staff-only user management dashboard at `/accounts/admin/` with Users and Roles tabs, inline AJAX editing, impersonation, and a nav entry in the header dropdown.

**Architecture:** All backend logic goes into a new `accounts/admin_views.py` to keep `views.py` focused. The single template `admin_dashboard.html` follows the existing hero+breadcrumb pattern. All mutations use `fetch()` AJAX returning JSON — no page reloads except for impersonation start/exit which redirect intentionally.

**Tech Stack:** Django 4.x, Bootstrap 5, Bootstrap Icons, vanilla JS (fetch API)

## Global Constraints

- Access gate: `request.user.is_staff or request.user.is_superuser` on every admin view — non-staff gets 403
- All AJAX endpoints: `require_POST`, return `{"success": true}` or `{"success": false, "error": "..."}`
- CSRF token sent via `X-CSRFToken` header on all fetch calls
- Template extends `main.html`, uses `{% block atcontent %}`
- Follow existing pattern from `accounts/templates/accounts/profile.html` for hero/breadcrumb style
- No new pip packages

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `accounts/admin_views.py` | All 9 admin view functions |
| Create | `accounts/templates/accounts/admin_dashboard.html` | Full page template |
| Modify | `accounts/urls.py` | Register 9 new URL patterns |
| Modify | `accounts/tests.py` | Access control + endpoint tests |
| Modify | `templates/head.html` | Nav link + impersonation banner |

---

## Task 1: Backend views + URL registration

**Files:**
- Create: `Trust-AI-Platform/accounts/admin_views.py`
- Modify: `Trust-AI-Platform/accounts/urls.py`
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Produces: `admin_dashboard`, `admin_edit_user`, `admin_delete_user`, `admin_toggle_user`, `admin_create_role`, `admin_rename_role`, `admin_delete_role`, `admin_impersonate`, `admin_impersonate_exit` — all importable from `accounts.admin_views`
- URL names produced: `admin_dashboard`, `admin_edit_user`, `admin_delete_user`, `admin_toggle_user`, `admin_create_role`, `admin_rename_role`, `admin_delete_role`, `admin_impersonate`, `admin_impersonate_exit`

- [ ] **Step 1: Write failing access-control tests**

In `Trust-AI-Platform/accounts/tests.py`:

```python
from django.test import TestCase, Client
from django.contrib.auth.models import User, Group
from django.urls import reverse


class AdminDashboardAccessTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.regular = User.objects.create_user('regular', password='pass')
        self.staff = User.objects.create_user('staffuser', password='pass', is_staff=True)
        self.superuser = User.objects.create_superuser('super', password='pass')

    def test_anonymous_redirected(self):
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.status_code, 302)

    def test_regular_user_forbidden(self):
        self.client.login(username='regular', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.status_code, 403)

    def test_staff_can_access(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(r.status_code, 200)

    def test_toggle_user_requires_post(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.get(reverse('admin_toggle_user', args=[self.regular.id]))
        self.assertEqual(r.status_code, 405)

    def test_toggle_user_flips_active(self):
        self.client.login(username='staffuser', password='pass')
        self.assertTrue(self.regular.is_active)
        r = self.client.post(reverse('admin_toggle_user', args=[self.regular.id]))
        self.assertEqual(r.status_code, 200)
        self.assertJSONEqual(r.content, {'success': True, 'is_active': False})
        self.regular.refresh_from_db()
        self.assertFalse(self.regular.is_active)

    def test_delete_user(self):
        self.client.login(username='staffuser', password='pass')
        uid = self.regular.id
        r = self.client.post(reverse('admin_delete_user', args=[uid]))
        self.assertJSONEqual(r.content, {'success': True})
        self.assertFalse(User.objects.filter(id=uid).exists())

    def test_cannot_delete_self(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.post(reverse('admin_delete_user', args=[self.staff.id]))
        data = r.json()
        self.assertFalse(data['success'])

    def test_create_role(self):
        self.client.login(username='staffuser', password='pass')
        r = self.client.post(reverse('admin_create_role'), {'name': 'TestRole'})
        data = r.json()
        self.assertTrue(data['success'])
        self.assertTrue(Group.objects.filter(name='TestRole').exists())

    def test_delete_role_with_members_blocked(self):
        self.client.login(username='staffuser', password='pass')
        g = Group.objects.create(name='Occupied')
        self.regular.groups.add(g)
        r = self.client.post(reverse('admin_delete_role', args=[g.id]))
        data = r.json()
        self.assertFalse(data['success'])
        self.assertTrue(Group.objects.filter(id=g.id).exists())
```

- [ ] **Step 2: Run tests — expect failures (views don't exist yet)**

```
cd Trust-AI-Platform
python manage.py test accounts.tests.AdminDashboardAccessTest -v 2
```

Expected: multiple errors like `NoReverseMatch` or `ImportError`.

- [ ] **Step 3: Create `accounts/admin_views.py`**

```python
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User, Group
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.http import require_POST
from functools import wraps


def staff_required(view_func):
    @wraps(view_func)
    @login_required
    def _wrapped(request, *args, **kwargs):
        if not (request.user.is_staff or request.user.is_superuser):
            return HttpResponseForbidden("Access denied.")
        return view_func(request, *args, **kwargs)
    return _wrapped


@staff_required
def admin_dashboard(request):
    users = User.objects.all().prefetch_related('groups').order_by('username')
    groups = Group.objects.all().prefetch_related('user_set').order_by('name')
    stats = {
        'total': users.count(),
        'active': users.filter(is_active=True).count(),
        'staff': users.filter(is_staff=True).count(),
        'inactive': users.filter(is_active=False).count(),
    }
    is_superuser = request.user.is_superuser
    return render(request, 'accounts/admin_dashboard.html', {
        'all_users': users,
        'groups': groups,
        'stats': stats,
        'is_superuser': is_superuser,
    })


@require_POST
@staff_required
def admin_edit_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    first_name = request.POST.get('first_name', '').strip()
    last_name = request.POST.get('last_name', '').strip()
    email = request.POST.get('email', '').strip()
    is_staff = request.POST.get('is_staff') == 'true'
    is_superuser = request.POST.get('is_superuser') == 'true'
    group_ids = request.POST.getlist('groups')

    if is_superuser and not request.user.is_superuser:
        is_superuser = False

    if user == request.user and not is_staff:
        return JsonResponse({'success': False, 'error': 'Cannot remove your own staff status.'})

    user.first_name = first_name
    user.last_name = last_name
    user.email = email
    user.is_staff = is_staff
    if request.user.is_superuser:
        user.is_superuser = is_superuser
    user.save(update_fields=['first_name', 'last_name', 'email', 'is_staff', 'is_superuser'])
    user.groups.set(Group.objects.filter(id__in=group_ids))
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot delete your own account.'})
    if user.is_superuser and not request.user.is_superuser:
        return JsonResponse({'success': False, 'error': 'Cannot delete a superuser.'})
    user.delete()
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_toggle_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot suspend your own account.'})
    user.is_active = not user.is_active
    user.save(update_fields=['is_active'])
    return JsonResponse({'success': True, 'is_active': user.is_active})


@require_POST
@staff_required
def admin_create_role(request):
    name = request.POST.get('name', '').strip()
    if not name:
        return JsonResponse({'success': False, 'error': 'Role name is required.'})
    if Group.objects.filter(name__iexact=name).exists():
        return JsonResponse({'success': False, 'error': 'A role with this name already exists.'})
    group = Group.objects.create(name=name)
    return JsonResponse({'success': True, 'id': group.id, 'name': group.name})


@require_POST
@staff_required
def admin_rename_role(request, role_id):
    group = get_object_or_404(Group, id=role_id)
    name = request.POST.get('name', '').strip()
    if not name:
        return JsonResponse({'success': False, 'error': 'Role name is required.'})
    if Group.objects.filter(name__iexact=name).exclude(id=role_id).exists():
        return JsonResponse({'success': False, 'error': 'A role with this name already exists.'})
    group.name = name
    group.save()
    return JsonResponse({'success': True, 'name': group.name})


@require_POST
@staff_required
def admin_delete_role(request, role_id):
    group = get_object_or_404(Group, id=role_id)
    if group.user_set.exists():
        return JsonResponse({'success': False, 'error': 'Remove all members before deleting this role.'})
    group.delete()
    return JsonResponse({'success': True})


@require_POST
@staff_required
def admin_impersonate(request, user_id):
    target = get_object_or_404(User, id=user_id)
    if target.is_superuser:
        return JsonResponse({'success': False, 'error': 'Cannot impersonate a superuser.'})
    if target == request.user:
        return JsonResponse({'success': False, 'error': 'Cannot impersonate yourself.'})
    impersonator_id = request.user.id          # save before login() may flush session
    target.backend = 'django.contrib.auth.backends.ModelBackend'
    auth_login(request, target)                # switches session to target user
    request.session['impersonator_id'] = impersonator_id
    return JsonResponse({'success': True, 'redirect': '/'})


@require_POST
def admin_impersonate_exit(request):
    impersonator_id = request.session.get('impersonator_id')
    if not impersonator_id:
        return redirect('index')
    original = get_object_or_404(User, id=impersonator_id)
    original.backend = 'django.contrib.auth.backends.ModelBackend'
    auth_login(request, original)              # session flush clears impersonator_id
    return redirect('admin_dashboard')
```

- [ ] **Step 4: Register URLs in `accounts/urls.py`**

Add this import at the top:
```python
from . import admin_views
```

Add these paths to `urlpatterns`:
```python
    path('admin/', admin_views.admin_dashboard, name='admin_dashboard'),
    path('admin/edit_user/<int:user_id>/', admin_views.admin_edit_user, name='admin_edit_user'),
    path('admin/delete_user/<int:user_id>/', admin_views.admin_delete_user, name='admin_delete_user'),
    path('admin/toggle_user/<int:user_id>/', admin_views.admin_toggle_user, name='admin_toggle_user'),
    path('admin/create_role/', admin_views.admin_create_role, name='admin_create_role'),
    path('admin/rename_role/<int:role_id>/', admin_views.admin_rename_role, name='admin_rename_role'),
    path('admin/delete_role/<int:role_id>/', admin_views.admin_delete_role, name='admin_delete_role'),
    path('admin/impersonate/<int:user_id>/', admin_views.admin_impersonate, name='admin_impersonate'),
    path('admin/impersonate_exit/', admin_views.admin_impersonate_exit, name='admin_impersonate_exit'),
```

- [ ] **Step 5: Run tests — expect all to pass**

```
python manage.py test accounts.tests.AdminDashboardAccessTest -v 2
```

Expected: 9 tests pass. The `test_staff_can_access` test will fail until the template exists — that's OK, fix it in Task 2.

- [ ] **Step 6: Commit**

```bash
git add accounts/admin_views.py accounts/urls.py accounts/tests.py
git commit -m "Add admin dashboard backend views and URL routes"
```

---

## Task 2: Admin dashboard template — hero, stats, users tab table

**Files:**
- Create: `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`

**Interfaces:**
- Consumes: context vars `all_users`, `groups`, `stats` (keys: `total`, `active`, `staff`, `inactive`), `is_superuser` from `admin_dashboard` view
- Produces: page at `/accounts/admin/` rendering without errors; `test_staff_can_access` passes

- [ ] **Step 1: Create the template with hero + stats + users table (no JS yet)**

Create `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`:

```html
{% extends "main.html" %}
{% block page_title %}<title>Trust AI Lab — User Management</title>{% endblock %}
{% load static %}
{% block atcontent %}

<style>
  .admin-hero {
    background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
    border-radius: 14px; padding: 26px 30px 20px; color: #fff;
    margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
  }
  .admin-hero-icon {
    background: rgba(255,255,255,0.18); border-radius: 10px;
    width: 50px; height: 50px; display: flex; align-items: center;
    justify-content: center; font-size: 22px; flex-shrink: 0;
  }
  .admin-hero .breadcrumb { background: none; margin: 10px 0 0; padding: 0; font-size: 12px; }
  .admin-hero .breadcrumb-item+.breadcrumb-item::before { color: rgba(255,255,255,0.5); }
  .admin-hero .breadcrumb-item a { color: rgba(255,255,255,0.72); text-decoration: none; }
  .admin-hero .breadcrumb-item a:hover { color: #fff; }
  .admin-hero .breadcrumb-item.active { color: rgba(255,255,255,0.92); }

  .stat-card { border-radius: 10px; padding: 18px 20px; text-align: center; border: 1px solid #e8edf5; background:#fff; }
  .stat-card .stat-num { font-size: 28px; font-weight: 700; color: #012970; line-height: 1; }
  .stat-card .stat-label { font-size: 12px; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

  .user-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #4154f1, #1a56db);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; color: #fff; flex-shrink: 0;
  }
  .user-row td { vertical-align: middle; font-size: 13px; }
  .user-row.inactive-row { opacity: 0.5; }

  /* Edit slide panel */
  .edit-panel {
    position: fixed; top: 0; right: -420px; width: 400px; height: 100vh;
    background: #fff; box-shadow: -4px 0 24px rgba(0,0,0,0.12);
    z-index: 1055; transition: right 0.25s ease; overflow-y: auto;
  }
  .edit-panel.open { right: 0; }
  .edit-panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 24px 16px; border-bottom: 1px solid #e8edf5; position: sticky; top: 0; background: #fff; z-index:1;
  }
  .edit-panel-header h5 { margin: 0; font-size: 15px; font-weight: 700; color: #012970; }
  .edit-panel-body { padding: 20px 24px; }
  .edit-panel-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.3); z-index: 1054;
  }
  .edit-panel-overlay.open { display: block; }

  /* Role cards */
  .role-name-display:hover { color: #1a56db; text-decoration: underline; }
</style>

<main id="main" class="main">

  <div class="admin-hero">
    <div class="d-flex align-items-start gap-3">
      <div class="admin-hero-icon"><i class="bi bi-shield-lock-fill"></i></div>
      <div class="flex-grow-1">
        <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Administration</div>
        <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">User Management</h2>
        <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ stats.total }} user{{ stats.total|pluralize }} · {{ stats.active }} active</div>
        <nav><ol class="breadcrumb">
          <li class="breadcrumb-item active">User Management</li>
        </ol></nav>
      </div>
    </div>
  </div>

  <section class="section">

    <!-- Stats bar -->
    <div class="row g-3 mb-4">
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-num">{{ stats.total }}</div>
          <div class="stat-label">Total Users</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-num" style="color:#16a34a;">{{ stats.active }}</div>
          <div class="stat-label">Active</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-num" style="color:#d97706;">{{ stats.staff }}</div>
          <div class="stat-label">Staff / Admins</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-num" style="color:#6b7280;">{{ stats.inactive }}</div>
          <div class="stat-label">Inactive</div>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <ul class="nav nav-tabs mb-0" id="adminTabs" role="tablist" style="border-bottom:2px solid #e8edf5;">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" id="tab-users-btn" data-bs-toggle="tab" data-bs-target="#tab-users" type="button" role="tab">
          <i class="bi bi-people me-1"></i> Users
          <span class="badge bg-secondary ms-1" style="font-size:11px;">{{ stats.total }}</span>
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="tab-roles-btn" data-bs-toggle="tab" data-bs-target="#tab-roles" type="button" role="tab">
          <i class="bi bi-tag me-1"></i> Roles
          <span class="badge bg-secondary ms-1" style="font-size:11px;">{{ groups|length }}</span>
        </button>
      </li>
    </ul>

    <div class="tab-content pt-4">

      <!-- ── Users Tab ── -->
      <div class="tab-pane fade show active" id="tab-users" role="tabpanel">

        <!-- Search + filter row -->
        <div class="row g-2 mb-3 align-items-center">
          <div class="col-12 col-md-5">
            <input type="text" id="userSearch" class="form-control" placeholder="Search name, username or email…">
          </div>
          <div class="col-12 col-md-4">
            <select id="roleFilter" class="form-select">
              <option value="">All Roles</option>
              {% for group in groups %}
              <option value="{{ group.name|lower }}">{{ group.name }}</option>
              {% endfor %}
              <option value="__none__">No Role</option>
            </select>
          </div>
          <div class="col-12 col-md-3">
            <div class="form-check form-switch ms-1 mt-1">
              <input class="form-check-input" type="checkbox" id="showInactive">
              <label class="form-check-label" for="showInactive" style="font-size:13px;">Show inactive</label>
            </div>
          </div>
        </div>

        <!-- User table -->
        <div class="card" style="border-radius:12px; overflow:hidden;">
          <div class="table-responsive">
            <table class="table table-hover mb-0" id="userTable">
              <thead style="background:#f8faff;">
                <tr>
                  <th style="font-size:12px;font-weight:600;color:#888;padding:12px 16px;">User</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Email</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Roles</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Last Login</th>
                  <th style="font-size:12px;font-weight:600;color:#888;">Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {% for u in all_users %}
                <tr class="user-row {% if not u.is_active %}inactive-row{% endif %}"
                    data-user-id="{{ u.id }}"
                    data-search="{{ u.username|lower }} {{ u.get_full_name|lower }} {{ u.email|lower }}"
                    data-groups="{% for g in u.groups.all %}{{ g.name|lower }} {% endfor %}{% if u.is_staff %}__staff__ {% endif %}"
                    data-has-role="{% if u.groups.exists or u.is_staff or u.is_superuser %}yes{% else %}no{% endif %}"
                    data-is-active="{{ u.is_active|yesno:'true,false' }}"
                    {% if not u.is_active %}style="display:none;"{% endif %}>
                  <td style="padding:12px 16px;">
                    <div class="d-flex align-items-center gap-2">
                      <div class="user-avatar">{{ u.first_name|default:u.username|slice:":1"|upper }}{{ u.last_name|slice:":1"|upper }}</div>
                      <div>
                        <div style="font-weight:600;font-size:13px;color:#012970;">{{ u.get_full_name|default:u.username }}</div>
                        <div style="font-size:11px;color:#888;">@{{ u.username }}</div>
                      </div>
                    </div>
                  </td>
                  <td>{{ u.email|default:"—" }}</td>
                  <td>
                    {% if u.is_superuser %}<span class="badge bg-danger me-1">Superuser</span>{% endif %}
                    {% if u.is_staff and not u.is_superuser %}<span class="badge bg-warning text-dark me-1">Staff</span>{% endif %}
                    {% for g in u.groups.all %}<span class="badge bg-primary me-1">{{ g.name }}</span>{% endfor %}
                    {% if not u.groups.exists and not u.is_staff and not u.is_superuser %}<span style="font-size:12px;color:#aaa;">—</span>{% endif %}
                  </td>
                  <td>{% if u.last_login %}{{ u.last_login|date:"d M Y" }}{% else %}<span style="color:#aaa;">Never</span>{% endif %}</td>
                  <td>
                    {% if u.is_active %}
                    <span class="badge bg-success">Active</span>
                    {% else %}
                    <span class="badge bg-secondary">Inactive</span>
                    {% endif %}
                  </td>
                  <td style="white-space:nowrap;">
                    <div class="d-flex gap-1">
                      <button class="btn btn-sm btn-outline-primary" title="Edit"
                              onclick="openEditPanel({{ u.id }}, {{ u.first_name|default:''|escapejs|tojson }}, {{ u.last_name|default:''|escapejs|tojson }}, '{{ u.email|default:''|escapejs }}', {{ u.is_staff|yesno:'true,false' }}, {{ u.is_superuser|yesno:'true,false' }}, [{% for g in u.groups.all %}{{ g.id }}{% if not forloop.last %},{% endif %}{% endfor %}])">
                        <i class="bi bi-pencil"></i>
                      </button>
                      {% if u.id != request.user.id %}
                      <button class="btn btn-sm {% if u.is_active %}btn-outline-warning{% else %}btn-outline-success{% endif %}"
                              title="{% if u.is_active %}Suspend{% else %}Activate{% endif %}"
                              id="toggle-btn-{{ u.id }}"
                              onclick="toggleUser({{ u.id }})">
                        <i class="bi {% if u.is_active %}bi-pause-circle{% else %}bi-play-circle{% endif %}"></i>
                      </button>
                      {% if not u.is_superuser %}
                      <button class="btn btn-sm btn-outline-secondary" title="Impersonate"
                              onclick="impersonateUser({{ u.id }})">
                        <i class="bi bi-person-badge"></i>
                      </button>
                      {% endif %}
                      <button class="btn btn-sm btn-outline-danger" title="Delete"
                              onclick="openDeleteModal({{ u.id }}, '{{ u.get_full_name|default:u.username|escapejs }}')">
                        <i class="bi bi-trash"></i>
                      </button>
                      {% endif %}
                    </div>
                  </td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
      </div><!-- /tab-users -->

      <!-- ── Roles Tab ── -->
      <div class="tab-pane fade" id="tab-roles" role="tabpanel">

        <div class="d-flex gap-2 mb-4 align-items-center">
          <input type="text" id="newRoleName" class="form-control" placeholder="New role name…" style="max-width:280px;"
                 onkeydown="if(event.key==='Enter') createRole()">
          <button class="btn btn-primary" onclick="createRole()">
            <i class="bi bi-plus-lg me-1"></i>Add Role
          </button>
        </div>

        <div class="row g-3" id="roleCards">
          {% for group in groups %}
          <div class="col-12 col-md-6 col-lg-4" id="role-card-{{ group.id }}">
            <div class="card h-100" style="border-radius:12px;">
              <div class="card-body">
                <div class="d-flex align-items-start justify-content-between mb-2">
                  <span class="role-name-display fw-bold" style="cursor:pointer;font-size:15px;"
                        title="Click to rename"
                        onclick="startRenameRole({{ group.id }}, this)">{{ group.name }}</span>
                  <div class="d-flex gap-1 align-items-center">
                    <span class="badge bg-secondary">{{ group.user_set.count }}</span>
                    <button class="btn btn-sm btn-outline-danger"
                            {% if group.user_set.exists %}disabled title="Remove all members first"{% else %}onclick="deleteRole({{ group.id }})"{% endif %}>
                      <i class="bi bi-trash"></i>
                    </button>
                  </div>
                </div>
                {% if group.user_set.exists %}
                <div>
                  <button class="btn btn-link btn-sm p-0 text-muted" style="font-size:12px;"
                          data-bs-toggle="collapse" data-bs-target="#role-members-{{ group.id }}">
                    <i class="bi bi-chevron-right"></i> {{ group.user_set.count }} member{{ group.user_set.count|pluralize }}
                  </button>
                  <div class="collapse" id="role-members-{{ group.id }}">
                    <div class="mt-2 d-flex flex-column gap-1">
                      {% for member in group.user_set.all %}
                      <div style="font-size:12px;">
                        <a href="#" class="text-decoration-none"
                           onclick="switchToUserAndEdit({{ member.id }}, {{ member.first_name|default:''|escapejs|tojson }}, {{ member.last_name|default:''|escapejs|tojson }}, '{{ member.email|default:''|escapejs }}', {{ member.is_staff|yesno:'true,false' }}, {{ member.is_superuser|yesno:'true,false' }}, [{% for g in member.groups.all %}{{ g.id }}{% if not forloop.last %},{% endif %}{% endfor %}]); return false;">
                          {{ member.get_full_name|default:member.username }}
                        </a>
                      </div>
                      {% endfor %}
                    </div>
                  </div>
                </div>
                {% endif %}
              </div>
            </div>
          </div>
          {% endfor %}
        </div>

      </div><!-- /tab-roles -->

    </div><!-- /tab-content -->

  </section>

</main>

<!-- ── Edit slide panel ── -->
<div id="editPanel" class="edit-panel">
  <div class="edit-panel-header">
    <h5 id="editPanelTitle">Edit User</h5>
    <button type="button" class="btn-close" onclick="closeEditPanel()"></button>
  </div>
  <div class="edit-panel-body">
    <input type="hidden" id="editUserId">

    <div class="mb-3">
      <label class="form-label fw-semibold" style="font-size:13px;">First Name</label>
      <input type="text" id="editFirstName" class="form-control form-control-sm">
    </div>
    <div class="mb-3">
      <label class="form-label fw-semibold" style="font-size:13px;">Last Name</label>
      <input type="text" id="editLastName" class="form-control form-control-sm">
    </div>
    <div class="mb-3">
      <label class="form-label fw-semibold" style="font-size:13px;">Email</label>
      <input type="email" id="editEmail" class="form-control form-control-sm">
    </div>

    <hr>

    <div class="mb-2">
      <div class="form-check form-switch">
        <input class="form-check-input" type="checkbox" id="editIsStaff">
        <label class="form-check-label" for="editIsStaff" style="font-size:13px;">Staff (can access admin)</label>
      </div>
    </div>
    {% if is_superuser %}
    <div class="mb-3">
      <div class="form-check form-switch">
        <input class="form-check-input" type="checkbox" id="editIsSuperuser">
        <label class="form-check-label" for="editIsSuperuser" style="font-size:13px;">Superuser</label>
      </div>
    </div>
    {% endif %}

    <hr>

    <div class="mb-4">
      <label class="form-label fw-semibold" style="font-size:13px;">Roles</label>
      <div id="editGroupsContainer">
        {% for group in groups %}
        <div class="form-check">
          <input class="form-check-input edit-group-check" type="checkbox"
                 value="{{ group.id }}" id="editGroup{{ group.id }}">
          <label class="form-check-label" for="editGroup{{ group.id }}" style="font-size:13px;">{{ group.name }}</label>
        </div>
        {% endfor %}
      </div>
    </div>

    <button class="btn btn-primary w-100" onclick="saveUser()">
      <i class="bi bi-check-lg me-1"></i>Save Changes
    </button>
    <div id="editPanelError" class="text-danger mt-2" style="display:none;font-size:13px;"></div>
  </div>
</div>
<div id="editPanelOverlay" class="edit-panel-overlay" onclick="closeEditPanel()"></div>

<!-- ── Delete confirmation modal ── -->
<div class="modal fade" id="deleteUserModal" tabindex="-1" aria-hidden="true">
  <div class="modal-dialog modal-dialog-centered">
    <div class="modal-content" style="border-radius:12px;overflow:hidden;">
      <div class="modal-header" style="background:#fff8f8;border-bottom:1px solid #fee2e2;">
        <h5 class="modal-title" style="color:#dc2626;font-size:15px;font-weight:700;">
          <i class="bi bi-exclamation-triangle me-2"></i>Delete User
        </h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body" style="font-size:14px;padding:20px 24px;">
        Delete <strong id="deleteUserName"></strong>? This cannot be undone.
      </div>
      <div class="modal-footer" style="border-top:1px solid #f3f4f6;padding:12px 24px;">
        <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-danger btn-sm" id="deleteConfirmBtn">
          <i class="bi bi-trash me-1"></i>Delete
        </button>
      </div>
    </div>
  </div>
</div>

<!-- ── Toast ── -->
<div class="position-fixed bottom-0 end-0 p-3" style="z-index:9999;">
  <div id="adminToast" class="toast align-items-center text-white border-0" role="alert">
    <div class="d-flex">
      <div class="toast-body" id="adminToastMsg" style="font-size:13px;"></div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>
  </div>
</div>

<script>
const CSRF = '{{ csrf_token }}';

// ── Toast helper ──
function showToast(msg, isError) {
  const t = document.getElementById('adminToast');
  t.className = 'toast align-items-center text-white border-0 ' + (isError ? 'bg-danger' : 'bg-success');
  document.getElementById('adminToastMsg').textContent = msg;
  bootstrap.Toast.getOrCreateInstance(t, {delay: 3000}).show();
}

function postJSON(url, data) {
  const fd = new FormData();
  fd.append('csrfmiddlewaretoken', CSRF);
  for (const [k, v] of Object.entries(data)) {
    if (Array.isArray(v)) v.forEach(i => fd.append(k, i));
    else fd.append(k, v);
  }
  return fetch(url, {method: 'POST', headers: {'X-CSRFToken': CSRF}, body: fd}).then(r => r.json());
}

// ── Client-side search / filter ──
function filterUsers() {
  const q = document.getElementById('userSearch').value.toLowerCase();
  const role = document.getElementById('roleFilter').value;
  const showInactive = document.getElementById('showInactive').checked;

  document.querySelectorAll('#userTable .user-row').forEach(function(row) {
    const isActive = row.dataset.isActive === 'true';
    if (!isActive && !showInactive) { row.style.display = 'none'; return; }

    const matchSearch = !q || row.dataset.search.includes(q);
    let matchRole = true;
    if (role === '__none__') matchRole = row.dataset.hasRole === 'no';
    else if (role) matchRole = row.dataset.groups.includes(role);

    row.style.display = (matchSearch && matchRole) ? '' : 'none';
  });
}
document.getElementById('userSearch').addEventListener('input', filterUsers);
document.getElementById('roleFilter').addEventListener('change', filterUsers);
document.getElementById('showInactive').addEventListener('change', filterUsers);

// ── Edit panel ──
function openEditPanel(id, firstName, lastName, email, isStaff, isSuperuser, groupIds) {
  document.getElementById('editUserId').value = id;
  document.getElementById('editFirstName').value = firstName || '';
  document.getElementById('editLastName').value = lastName || '';
  document.getElementById('editEmail').value = email || '';
  document.getElementById('editIsStaff').checked = isStaff;
  const superEl = document.getElementById('editIsSuperuser');
  if (superEl) superEl.checked = isSuperuser;
  document.querySelectorAll('.edit-group-check').forEach(function(cb) {
    cb.checked = groupIds.includes(parseInt(cb.value));
  });
  document.getElementById('editPanelError').style.display = 'none';
  document.getElementById('editPanel').classList.add('open');
  document.getElementById('editPanelOverlay').classList.add('open');
}

function closeEditPanel() {
  document.getElementById('editPanel').classList.remove('open');
  document.getElementById('editPanelOverlay').classList.remove('open');
}

function saveUser() {
  const id = document.getElementById('editUserId').value;
  const groupIds = [...document.querySelectorAll('.edit-group-check:checked')].map(c => c.value);
  const superEl = document.getElementById('editIsSuperuser');

  postJSON('/accounts/admin/edit_user/' + id + '/', {
    first_name: document.getElementById('editFirstName').value,
    last_name: document.getElementById('editLastName').value,
    email: document.getElementById('editEmail').value,
    is_staff: document.getElementById('editIsStaff').checked ? 'true' : 'false',
    is_superuser: (superEl && superEl.checked) ? 'true' : 'false',
    groups: groupIds,
  }).then(function(data) {
    if (data.success) {
      showToast('User saved.', false);
      setTimeout(function() { location.reload(); }, 800);
    } else {
      const el = document.getElementById('editPanelError');
      el.textContent = data.error;
      el.style.display = 'block';
    }
  });
}

// ── Toggle active ──
function toggleUser(id) {
  postJSON('/accounts/admin/toggle_user/' + id + '/', {}).then(function(data) {
    if (data.success) {
      showToast(data.is_active ? 'User activated.' : 'User suspended.', false);
      setTimeout(function() { location.reload(); }, 800);
    } else {
      showToast(data.error, true);
    }
  });
}

// ── Delete ──
let _deleteId = null;
function openDeleteModal(id, name) {
  _deleteId = id;
  document.getElementById('deleteUserName').textContent = name;
  bootstrap.Modal.getOrCreateInstance(document.getElementById('deleteUserModal')).show();
}
document.getElementById('deleteConfirmBtn').addEventListener('click', function() {
  if (!_deleteId) return;
  postJSON('/accounts/admin/delete_user/' + _deleteId + '/', {}).then(function(data) {
    bootstrap.Modal.getInstance(document.getElementById('deleteUserModal')).hide();
    if (data.success) {
      showToast('User deleted.', false);
      const row = document.querySelector('[data-user-id="' + _deleteId + '"]');
      if (row) row.remove();
    } else {
      showToast(data.error, true);
    }
    _deleteId = null;
  });
});

// ── Impersonate ──
function impersonateUser(id) {
  if (!confirm('You will be logged in as this user. Continue?')) return;
  postJSON('/accounts/admin/impersonate/' + id + '/', {}).then(function(data) {
    if (data.success) window.location.href = data.redirect;
    else showToast(data.error, true);
  });
}

// ── Roles ──
function createRole() {
  const nameInput = document.getElementById('newRoleName');
  const name = nameInput.value.trim();
  if (!name) return;
  postJSON('/accounts/admin/create_role/', {name: name}).then(function(data) {
    if (data.success) {
      showToast('Role created.', false);
      nameInput.value = '';
      // Append new card
      const container = document.getElementById('roleCards');
      const col = document.createElement('div');
      col.className = 'col-12 col-md-6 col-lg-4';
      col.id = 'role-card-' + data.id;
      col.innerHTML = '<div class="card h-100" style="border-radius:12px;"><div class="card-body">' +
        '<div class="d-flex align-items-start justify-content-between mb-2">' +
        '<span class="role-name-display fw-bold" style="cursor:pointer;font-size:15px;" title="Click to rename" onclick="startRenameRole(' + data.id + ', this)">' + escHtml(data.name) + '</span>' +
        '<div class="d-flex gap-1 align-items-center"><span class="badge bg-secondary">0</span>' +
        '<button class="btn btn-sm btn-outline-danger" onclick="deleteRole(' + data.id + ')"><i class="bi bi-trash"></i></button>' +
        '</div></div></div></div>';
      container.appendChild(col);
    } else {
      showToast(data.error, true);
    }
  });
}

function deleteRole(id) {
  if (!confirm('Delete this role?')) return;
  postJSON('/accounts/admin/delete_role/' + id + '/', {}).then(function(data) {
    if (data.success) {
      showToast('Role deleted.', false);
      const card = document.getElementById('role-card-' + id);
      if (card) card.remove();
    } else {
      showToast(data.error, true);
    }
  });
}

function startRenameRole(id, el) {
  const currentName = el.textContent.trim();
  const input = document.createElement('input');
  input.type = 'text';
  input.value = currentName;
  input.className = 'form-control form-control-sm';
  input.style.maxWidth = '160px';
  el.replaceWith(input);
  input.focus();

  function finishRename() {
    const newName = input.value.trim();
    if (!newName || newName === currentName) {
      input.replaceWith(el);
      return;
    }
    postJSON('/accounts/admin/rename_role/' + id + '/', {name: newName}).then(function(data) {
      if (data.success) {
        el.textContent = data.name;
        input.replaceWith(el);
        showToast('Role renamed.', false);
      } else {
        showToast(data.error, true);
        input.replaceWith(el);
      }
    });
  }
  input.addEventListener('blur', finishRename);
  input.addEventListener('keydown', function(e) { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = currentName; input.blur(); } });
}

// ── Roles tab → user edit (from member link) ──
function switchToUserAndEdit(id, firstName, lastName, email, isStaff, isSuperuser, groupIds) {
  document.getElementById('tab-users-btn').click();
  setTimeout(function() {
    openEditPanel(id, firstName, lastName, email, isStaff, isSuperuser, groupIds);
  }, 50);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>

{% endblock atcontent %}
```

**Note:** The template uses `{{ u.first_name|default:''|escapejs|tojson }}` — Django does not have a `tojson` filter by default. Replace those two arguments passed to `openEditPanel` with data attributes on the row to avoid filter issues:

Add `data-first="{{ u.first_name|escapejs }}"` etc. on each `<tr>`, then read them in JS. The step below fixes this.

- [ ] **Step 2: Fix the data-passing approach — use data attributes on rows instead of inline JS args**

Replace the `onclick` on the Edit button for each user row with:

```html
<button class="btn btn-sm btn-outline-primary" title="Edit"
        onclick="openEditPanelFromRow(this.closest('tr'))">
  <i class="bi bi-pencil"></i>
</button>
```

Add these data attributes to each `<tr class="user-row ...">`:
```html
data-first="{{ u.first_name|escapejs }}"
data-last="{{ u.last_name|escapejs }}"
data-email="{{ u.email|escapejs }}"
data-is-staff="{{ u.is_staff|yesno:'true,false' }}"
data-is-superuser="{{ u.is_superuser|yesno:'true,false' }}"
data-group-ids="{% for g in u.groups.all %}{{ g.id }}{% if not forloop.last %},{% endif %}{% endfor %}"
```

Add JS function `openEditPanelFromRow`:
```javascript
function openEditPanelFromRow(row) {
  const groupIds = row.dataset.groupIds ? row.dataset.groupIds.split(',').map(Number).filter(Boolean) : [];
  openEditPanel(
    row.dataset.userId,
    row.dataset.first,
    row.dataset.last,
    row.dataset.email,
    row.dataset.isStaff === 'true',
    row.dataset.isSuperuser === 'true',
    groupIds
  );
}
```

Do the same for the member links in the Roles tab — add data attributes on each member `<a>` tag or make them pass `data-*` from the row they find via `document.querySelector('[data-user-id="X"]')`.

Replace `switchToUserAndEdit(...)` calls in Roles tab member links with:
```html
<a href="#" class="text-decoration-none"
   data-member-id="{{ member.id }}"
   onclick="switchToUserRow(this); return false;">
  {{ member.get_full_name|default:member.username }}
</a>
```

```javascript
function switchToUserRow(link) {
  document.getElementById('tab-users-btn').click();
  setTimeout(function() {
    const row = document.querySelector('[data-user-id="' + link.dataset.memberId + '"]');
    if (row) openEditPanelFromRow(row);
  }, 50);
}
```

- [ ] **Step 3: Run tests**

```
python manage.py test accounts.tests.AdminDashboardAccessTest.test_staff_can_access -v 2
```

Expected: PASS (template renders without errors)

- [ ] **Step 4: Manual smoke test**

Start dev server: `python manage.py runserver`
1. Log in as a staff user, visit `/accounts/admin/`
2. Verify: hero + breadcrumb renders, 4 stat cards show correct counts
3. Verify: user table shows all users, badges match roles
4. Verify: search box filters rows in real time
5. Verify: "Show inactive" toggle reveals/hides inactive users
6. Verify: role filter dropdown works

- [ ] **Step 5: Commit**

```bash
git add accounts/templates/accounts/admin_dashboard.html
git commit -m "Add admin dashboard template: hero, stats, users tab, roles tab"
```

---

## Task 3: head.html — nav link + impersonation banner

**Files:**
- Modify: `Trust-AI-Platform/templates/head.html`

**Interfaces:**
- Consumes: `request.user.is_staff`, `request.user.is_superuser`, `request.session.impersonator_id`
- Produces: "User Management" link visible in header dropdown for staff/superusers; impersonation banner visible site-wide when impersonating

- [ ] **Step 1: Add "User Management" nav link in profile dropdown**

In `templates/head.html`, locate the block:
```html
            {% if request.user|has_group:"teachers" %}
            <li>
              <a class="dropdown-item d-flex align-items-center" href="{% url 'list_organizations' %}">
```

After the closing `</li>` of the Organizations item and before `<li><hr class="dropdown-divider"></li>`, add:

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

Also add the `profile-icon-purple` CSS color (check `style.css` for the existing `profile-icon-*` classes — add `.profile-icon-purple { background: #ede9fe; color: #7c3aed; }` if it's not there, or inline `style="background:#ede9fe;color:#7c3aed;"` on the span).

- [ ] **Step 2: Add impersonation banner**

In `templates/head.html`, directly after `<body>` opens (before the `<header>` tag), add:

```html
  {% if request.session.impersonator_id %}
  <div id="impersonation-banner" style="
    position:fixed; top:0; left:0; right:0; z-index:9999;
    background:#f59e0b; color:#1c1917;
    padding:8px 20px; font-size:13px; font-weight:600;
    display:flex; align-items:center; justify-content:center; gap:12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  ">
    <i class="bi bi-eye-fill"></i>
    Viewing as <strong>@{{ request.user.username }}</strong> &mdash; not your account
    <form method="post" action="{% url 'admin_impersonate_exit' %}" style="display:inline;">
      {% csrf_token %}
      <button type="submit" style="
        background:#1c1917; color:#fff; border:none; border-radius:6px;
        padding:3px 12px; font-size:12px; font-weight:600; cursor:pointer;
      ">Exit Impersonation</button>
    </form>
  </div>
  <style>
    #header { top: 40px !important; }
    #sidebar { top: 112px !important; }
    .main { padding-top: 96px !important; }
  </style>
  {% endif %}
```

- [ ] **Step 3: Manual smoke test**

1. As a staff user, visit `/accounts/admin/`
2. Verify "User Management" appears in header profile dropdown with purple icon
3. Click Impersonate on a non-superuser; verify the amber banner appears at the top on the next page
4. Click "Exit Impersonation"; verify you're back to your original account on the admin dashboard

- [ ] **Step 4: Commit**

```bash
git add templates/head.html
git commit -m "Add User Management nav link and impersonation banner to header"
```

---

## Task 4: End-to-end verification + push

**Files:** none new — verification only

- [ ] **Step 1: Run full test suite**

```
python manage.py test accounts -v 2
```

Expected: all tests pass.

- [ ] **Step 2: Full manual walkthrough**

Log in as a superuser and verify each flow:

| Flow | Steps | Expected |
|------|-------|----------|
| User edit | Click Edit on any user row → change first name → Save | Toast "User saved", row updates after reload |
| Toggle suspend | Click pause icon on active user | Toast, badge changes to Inactive, row fades |
| Delete | Click trash → confirm | Row disappears, no page reload |
| Impersonate | Click person-badge icon → OK | Banner appears, session is target user |
| Exit impersonation | Click "Exit Impersonation" | Redirected to admin dashboard as original user |
| Create role | Roles tab → type name → Add Role | New card appears instantly |
| Rename role | Click role name → edit → Enter | Name updates in place |
| Delete empty role | Click trash on role with 0 members | Card removed |
| Delete occupied role | Click trash on role with members | Button disabled / error toast |
| Member link | Roles tab → click member name | Switches to Users tab, opens edit panel for that user |
| Search | Type partial name/email | Table filters instantly |
| Role filter | Select "Teacher" from dropdown | Only teachers visible |
| Inactive toggle | Enable "Show inactive" | Inactive rows appear |

- [ ] **Step 3: Push**

```bash
git push
```
