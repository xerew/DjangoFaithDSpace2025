# Teacher Home Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new `home` Django app that serves a welcoming, informational landing page for teachers, becoming their post-login destination instead of the analytics dashboard.

**Architecture:** New `home` app with a single `teacher_home` view that queries scenario/group stats and latest public scenarios, rendered in a full-page template extending `main.html`. Routing changes touch `faithDev/urls.py`, `accounts/views.py`, and `templates/head.html`.

**Tech Stack:** Django 5.1, Bootstrap 5, Bootstrap Icons, existing `main.html` base template, `authoringtool` and `usergroups` models.

## Global Constraints

- Access control: `@group_required('teachers')` from `accounts.views` — same decorator used across the whole codebase
- Template base: extends `main.html`, uses the same hero pattern (`dash-hero` div) already established in `index.html` and `scenarios.html`
- Hero colour: `linear-gradient(135deg,#1a56db 0%,#1e3a8a 100%)` — matches every other page hero
- Model fields: `Scenario.created_on` (not `created_at`), `Scenario.visibility_status='public'`, `UserGroup.created_by` for teacher ownership, members via `UserGroupMembership`
- URL name for the new view: `teacher_home`
- Test settings module: `faithDev.settings_test` (SQLite in-memory, avoids Postgres-specific fields)
- Run tests with: `python manage.py test home --settings=faithDev.settings_test -v 2`
- No new dependencies — everything already installed
- No migrations needed — `home` app has no models

---

### Task 1: `home` app — scaffold, view, URL, settings, tests

**Files:**
- Create: `Trust-AI-Platform/home/__init__.py`
- Create: `Trust-AI-Platform/home/apps.py`
- Create: `Trust-AI-Platform/home/views.py`
- Create: `Trust-AI-Platform/home/urls.py`
- Modify: `Trust-AI-Platform/faithDev/settings.py` — add `'home'` to `INSTALLED_APPS`
- Modify: `Trust-AI-Platform/faithDev/urls.py` — add `home` include, update root redirect
- Create: `Trust-AI-Platform/home/tests.py`

**Interfaces:**
- Produces: `teacher_home` view at `GET /home/`, context keys: `my_scenario_count` (int), `my_group_count` (int), `total_students` (int), `latest_public` (queryset of up to 5 Scenario objects), `show_get_started` (bool)
- Consumes: `Scenario` from `authoringtool.models`, `UserGroup`/`UserGroupMembership` from `usergroups.models`, `group_required` from `accounts.views`

- [ ] **Step 1: Write failing tests**

Create `Trust-AI-Platform/home/tests.py`:

```python
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User, Group
from authoringtool.models import Scenario
from usergroups.models import UserGroup


def make_teacher(username='teacher1', password='pass'):
    user = User.objects.create_user(username, password=password)
    group = Group.objects.get_or_create(name='teachers')[0]
    user.groups.add(group)
    return user


class TeacherHomeAccessTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.teacher = make_teacher()
        self.regular = User.objects.create_user('regular', password='pass')

    def test_anonymous_redirected(self):
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.status_code, 302)
        self.assertIn('/login', r['Location'])

    def test_non_teacher_forbidden(self):
        self.client.login(username='regular', password='pass')
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.status_code, 403)

    def test_teacher_gets_200(self):
        self.client.login(username='teacher1', password='pass')
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.status_code, 200)


class TeacherHomeContextTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.teacher = make_teacher()
        self.client.login(username='teacher1', password='pass')

    def test_context_has_stat_keys(self):
        r = self.client.get(reverse('teacher_home'))
        for key in ('my_scenario_count', 'my_group_count', 'total_students', 'latest_public', 'show_get_started'):
            self.assertIn(key, r.context, msg=f'Missing context key: {key}')

    def test_show_get_started_true_when_no_scenarios(self):
        r = self.client.get(reverse('teacher_home'))
        self.assertTrue(r.context['show_get_started'])

    def test_show_get_started_false_when_has_scenario(self):
        Scenario.objects.create(
            name='My Scenario', created_by=self.teacher, updated_by=self.teacher
        )
        r = self.client.get(reverse('teacher_home'))
        self.assertFalse(r.context['show_get_started'])

    def test_my_scenario_count(self):
        Scenario.objects.create(name='S1', created_by=self.teacher, updated_by=self.teacher)
        Scenario.objects.create(name='S2', created_by=self.teacher, updated_by=self.teacher)
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.context['my_scenario_count'], 2)

    def test_latest_public_max_5(self):
        for i in range(7):
            Scenario.objects.create(
                name=f'Pub{i}', created_by=self.teacher, updated_by=self.teacher,
                visibility_status='public'
            )
        r = self.client.get(reverse('teacher_home'))
        self.assertLessEqual(len(r.context['latest_public']), 5)

    def test_latest_public_only_public(self):
        Scenario.objects.create(
            name='Private', created_by=self.teacher, updated_by=self.teacher,
            visibility_status='private'
        )
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(len(r.context['latest_public']), 0)
```

- [ ] **Step 2: Run tests — expect ImportError or URL resolution failure**

```
cd Trust-AI-Platform
python manage.py test home --settings=faithDev.settings_test -v 2
```

Expected: error (app not registered or URL not found).

- [ ] **Step 3: Create app scaffold**

`Trust-AI-Platform/home/__init__.py` — empty file.

`Trust-AI-Platform/home/apps.py`:
```python
from django.apps import AppConfig

class HomeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'home'
```

- [ ] **Step 4: Write the view**

`Trust-AI-Platform/home/views.py`:
```python
from django.db.models import Count
from authoringtool.models import Scenario
from usergroups.models import UserGroup, UserGroupMembership
from accounts.views import group_required


@group_required('teachers')
def teacher_home(request):
    from django.shortcuts import render
    user = request.user
    my_scenario_count = Scenario.objects.filter(created_by=user).count()
    my_group_count = UserGroup.objects.filter(created_by=user).count()
    total_students = UserGroupMembership.objects.filter(group__created_by=user).count()
    latest_public = Scenario.objects.filter(
        visibility_status='public'
    ).order_by('-created_on')[:5]

    return render(request, 'home/home.html', {
        'my_scenario_count': my_scenario_count,
        'my_group_count': my_group_count,
        'total_students': total_students,
        'latest_public': latest_public,
        'show_get_started': my_scenario_count == 0,
    })
```

- [ ] **Step 5: Write the URL config**

`Trust-AI-Platform/home/urls.py`:
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.teacher_home, name='teacher_home'),
]
```

- [ ] **Step 6: Register app and wire URL**

In `Trust-AI-Platform/faithDev/settings.py`, add `'home'` to `INSTALLED_APPS`:
```python
INSTALLED_APPS = [
    ...
    'home',
    ...
]
```

In `Trust-AI-Platform/faithDev/urls.py`, replace:
```python
path('', lambda request: redirect('/authoringtool/', permanent=True)),
```
with:
```python
path('', lambda request: redirect('/home/', permanent=True)),
path('home/', include('home.urls')),
```
Add `include` to the import if not already present: `from django.urls import path, include`.

- [ ] **Step 7: Run tests — all must pass**

```
cd Trust-AI-Platform
python manage.py test home --settings=faithDev.settings_test -v 2
```

Expected: `Ran 9 tests … OK`

- [ ] **Step 8: Commit**

```
git add home/ Trust-AI-Platform/faithDev/settings.py Trust-AI-Platform/faithDev/urls.py
git commit -m "feat: add home app with teacher_home view and URL"
```

---

### Task 2: Template, routing, sidebar, placeholder image

**Files:**
- Create: `Trust-AI-Platform/home/templates/home/home.html`
- Create: `Trust-AI-Platform/static/img/home_ibl.jpg` (placeholder — a small solid-colour JPEG or copy of an existing image)
- Modify: `Trust-AI-Platform/accounts/views.py` — teacher login redirect → `teacher_home`
- Modify: `Trust-AI-Platform/templates/head.html` — add Home sidebar item

**Interfaces:**
- Consumes: context keys from Task 1 (`my_scenario_count`, `my_group_count`, `total_students`, `latest_public`, `show_get_started`)
- Produces: fully rendered HTML page, updated login redirect, updated sidebar

- [ ] **Step 1: Write the template**

Create directory `Trust-AI-Platform/home/templates/home/` then write `home.html`:

```html
{% extends "main.html" %}
{% load static %}
{% block page_title %}<title>Trust AI Lab — Home</title>{% endblock %}

{% block atcontent %}
<main id="main" class="main">

  <!-- ── Hero ── -->
  <div class="dash-hero" style="background:linear-gradient(135deg,#1a56db 0%,#1e3a8a 100%);border-radius:14px;padding:26px 30px 20px;color:#fff;margin-bottom:26px;box-shadow:0 4px 20px rgba(26,86,219,0.18);">
    <div class="d-flex align-items-start gap-3">
      <div style="background:rgba(255,255,255,0.18);border-radius:10px;width:50px;height:50px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">
        <i class="bi bi-house-heart-fill"></i>
      </div>
      <div class="flex-grow-1">
        <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Trust AI Lab</div>
        <h2 style="margin:0;font-size:22px;font-weight:700;line-height:1.2;">Welcome back, {{ user.get_full_name|default:user.username }}!</h2>
        <div style="font-size:12.5px;opacity:0.75;margin-top:4px;">{{ today|date:"l, j F Y" }}</div>
      </div>
    </div>
  </div>

  <!-- ── Get Started Strip (only when 0 scenarios) ── -->
  {% if show_get_started %}
  <div style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:12px;padding:18px 24px;margin-bottom:24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:14px;font-weight:700;color:#1e3a8a;margin-bottom:2px;"><i class="bi bi-rocket-takeoff me-2"></i>Get started</div>
      <div style="font-size:13px;color:#374151;">You haven't created a scenario yet — start building your first one.</div>
    </div>
    <a href="{% url 'createScenario' %}" style="background:#1a56db;color:#fff;border-radius:8px;padding:8px 20px;font-size:13px;font-weight:600;text-decoration:none;white-space:nowrap;">Create your first scenario</a>
  </div>
  {% endif %}

  <!-- ── Quick Stats ── -->
  <div class="row g-3 mb-4">
    <div class="col-12 col-sm-4">
      <div style="background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.04);text-align:center;">
        <div style="font-size:32px;font-weight:800;color:#1a56db;">{{ my_scenario_count }}</div>
        <div style="font-size:12.5px;color:#6b7280;margin-top:2px;"><i class="bi bi-collection me-1"></i>Your Scenarios</div>
      </div>
    </div>
    <div class="col-12 col-sm-4">
      <div style="background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.04);text-align:center;">
        <div style="font-size:32px;font-weight:800;color:#1a56db;">{{ my_group_count }}</div>
        <div style="font-size:12.5px;color:#6b7280;margin-top:2px;"><i class="bi bi-people me-1"></i>Student Groups</div>
      </div>
    </div>
    <div class="col-12 col-sm-4">
      <div style="background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.04);text-align:center;">
        <div style="font-size:32px;font-weight:800;color:#1a56db;">{{ total_students }}</div>
        <div style="font-size:12.5px;color:#6b7280;margin-top:2px;"><i class="bi bi-mortarboard me-1"></i>Total Students</div>
      </div>
    </div>
  </div>

  <!-- ── About the Platform / IBL ── -->
  <div style="background:#fff;border-radius:14px;border:1px solid #e8edf5;box-shadow:0 1px 4px rgba(0,0,0,0.04);padding:32px;margin-bottom:28px;">
    <div class="row align-items-center g-4">
      <div class="col-12 col-md-7">
        <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;color:#1a56db;font-weight:700;margin-bottom:8px;">About the Platform</div>
        <h3 style="font-size:20px;font-weight:700;color:#1e3a8a;margin-bottom:12px;">Inquiry-Based Learning with Trust AI Lab</h3>
        <p style="font-size:13.5px;color:#374151;line-height:1.7;margin-bottom:12px;">
          Trust AI Lab supports <strong>Inquiry-Based Learning (IBL)</strong> — a student-centred pedagogical approach where learners explore, question, and discover knowledge through guided investigation rather than passive instruction.
        </p>
        <p style="font-size:13.5px;color:#374151;line-height:1.7;margin-bottom:12px;">
          As a teacher, you design <strong>scenarios</strong> — interactive digital experiments that guide students step by step through a scientific inquiry cycle. Each scenario can include simulations, remote labs, VR/AR experiences, adaptive evaluation, and branching paths based on student performance.
        </p>
        <p style="font-size:13.5px;color:#374151;line-height:1.7;">
          This platform was developed as part of the <strong>FAITH</strong> and <strong>DSpace</strong> projects, funded by the European Union under grant agreements No. 101135932 and No. 101086701.
        </p>
      </div>
      <div class="col-12 col-md-5 text-center">
        <img src="{% static 'img/home_ibl.jpg' %}" alt="Teachers in classroom"
             style="width:100%;max-width:420px;border-radius:12px;object-fit:cover;aspect-ratio:4/3;box-shadow:0 4px 18px rgba(0,0,0,0.1);">
      </div>
    </div>
  </div>

  <!-- ── Latest Public Scenarios ── -->
  <div style="margin-bottom:28px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <h4 style="font-size:16px;font-weight:700;color:#1e3a8a;margin:0;"><i class="bi bi-globe me-2" style="color:#1a56db;"></i>Latest Public Scenarios</h4>
      <a href="{% url 'scenarios' %}" style="font-size:12.5px;color:#1a56db;text-decoration:none;font-weight:600;">View all <i class="bi bi-arrow-right"></i></a>
    </div>
    {% if latest_public %}
    <div class="row g-3">
      {% for scenario in latest_public %}
      <div class="col-12 col-sm-6 col-lg-4">
        <div style="background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);height:100%;display:flex;flex-direction:column;justify-content:space-between;">
          <div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:6px;">{{ scenario.name }}</div>
            {% if scenario.subject_domains %}
            <div style="font-size:11.5px;color:#6b7280;margin-bottom:4px;"><i class="bi bi-tag me-1"></i>{{ scenario.subject_domains }}</div>
            {% endif %}
            {% if scenario.age_of_students %}
            <div style="font-size:11.5px;color:#6b7280;"><i class="bi bi-person me-1"></i>Ages {{ scenario.age_of_students }}</div>
            {% endif %}
          </div>
          <div style="margin-top:14px;">
            <a href="{% url 'viewScenario' scenario.id %}" style="display:inline-block;background:#eff6ff;color:#1a56db;border-radius:7px;padding:6px 14px;font-size:12.5px;font-weight:600;text-decoration:none;">View <i class="bi bi-arrow-right-short"></i></a>
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
    {% else %}
    <div style="background:#f9fafb;border-radius:12px;border:1px dashed #d1d9e0;padding:28px;text-align:center;color:#9ca3af;font-size:13px;">
      No public scenarios yet.
    </div>
    {% endif %}
  </div>

  <!-- ── Quick Actions ── -->
  <div>
    <h4 style="font-size:16px;font-weight:700;color:#1e3a8a;margin-bottom:14px;"><i class="bi bi-lightning-charge me-2" style="color:#1a56db;"></i>Quick Actions</h4>
    <div class="row g-3">
      <div class="col-12 col-sm-4">
        <a href="{% url 'createScenario' %}" style="display:block;background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:22px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);text-decoration:none;transition:box-shadow 0.15s;" onmouseover="this.style.boxShadow='0 4px 16px rgba(26,86,219,0.12)'" onmouseout="this.style.boxShadow='0 1px 4px rgba(0,0,0,0.04)'">
          <div style="font-size:26px;color:#1a56db;margin-bottom:8px;"><i class="bi bi-plus-circle-fill"></i></div>
          <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:4px;">Create Scenario</div>
          <div style="font-size:12px;color:#6b7280;">Build a new IBL scenario for your students</div>
        </a>
      </div>
      <div class="col-12 col-sm-4">
        <a href="{% url 'list_groups' %}" style="display:block;background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:22px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);text-decoration:none;transition:box-shadow 0.15s;" onmouseover="this.style.boxShadow='0 4px 16px rgba(26,86,219,0.12)'" onmouseout="this.style.boxShadow='0 1px 4px rgba(0,0,0,0.04)'">
          <div style="font-size:26px;color:#1a56db;margin-bottom:8px;"><i class="bi bi-people-fill"></i></div>
          <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:4px;">Student Groups</div>
          <div style="font-size:12px;color:#6b7280;">Manage your student groups and credentials</div>
        </a>
      </div>
      <div class="col-12 col-sm-4">
        <a href="{% url 'index' %}" style="display:block;background:#fff;border-radius:12px;border:1px solid #e8edf5;padding:22px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);text-decoration:none;transition:box-shadow 0.15s;" onmouseover="this.style.boxShadow='0 4px 16px rgba(26,86,219,0.12)'" onmouseout="this.style.boxShadow='0 1px 4px rgba(0,0,0,0.04)'">
          <div style="font-size:26px;color:#1a56db;margin-bottom:8px;"><i class="bi bi-bar-chart-line-fill"></i></div>
          <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:4px;">Analytics Dashboard</div>
          <div style="font-size:12px;color:#6b7280;">View student performance and scenario analytics</div>
        </a>
      </div>
    </div>
  </div>

</main>
{% endblock %}
```

- [ ] **Step 2: Add `today` to the view context**

In `Trust-AI-Platform/home/views.py`, add `from datetime import date` at the top and add `'today': date.today()` to the context dict.

- [ ] **Step 3: Create placeholder image**

Copy any existing image from `static/img/` and rename it to `home_ibl.jpg` as a placeholder. The teacher replaces it with a real classroom photo.

```bash
# From Trust-AI-Platform/
cp static/img/logo.png static/img/home_ibl.jpg
```

(Any image works as placeholder — it will be replaced.)

- [ ] **Step 4: Update teacher login redirect**

In `Trust-AI-Platform/accounts/views.py`, find:

```python
if user.groups.filter(name='teachers').exists():
    return redirect('index')  # Redirect to the teachers dashboard
```

Change to:

```python
if user.groups.filter(name='teachers').exists():
    return redirect('teacher_home')
```

- [ ] **Step 5: Add Home item to sidebar**

In `Trust-AI-Platform/templates/head.html`, find the first teacher-only sidebar block:

```html
{% if request.user|has_group:"teachers" %}
      <li class="nav-item">
        <a class="nav-link collapsed" href="{% url 'index' %}">
          <i class="bi bi-grid"></i>
          <span>Dashboard</span>
        </a>
      </li><!-- End Dashboard Nav -->
```

Add a Home item immediately before the Dashboard item:

```html
{% if request.user|has_group:"teachers" %}
      <li class="nav-item">
        <a class="nav-link collapsed" href="{% url 'teacher_home' %}">
          <i class="bi bi-house-heart-fill"></i>
          <span>Home</span>
        </a>
      </li>

      <li class="nav-item">
        <a class="nav-link collapsed" href="{% url 'index' %}">
          <i class="bi bi-grid"></i>
          <span>Dashboard</span>
        </a>
      </li><!-- End Dashboard Nav -->
```

- [ ] **Step 6: Run the full test suite**

```
cd Trust-AI-Platform
python manage.py test home authoringtool accounts --settings=faithDev.settings_test -v 2
```

Expected: all tests pass (including the 9 new home tests and the 51 existing tests).

- [ ] **Step 7: Commit**

```
git add home/templates/ static/img/home_ibl.jpg accounts/views.py templates/head.html
git commit -m "feat: teacher home page template, routing, and sidebar item"
```
