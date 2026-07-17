# Admin User List and Form Editor Tweaks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide no-role accounts (students) from the User Management table, make the feedback form editor's "Assign to all scenarios" checkbox sync every scenario checkbox in both directions, and add a search box to the scenario assignment list.

**Architecture:** Two independent tasks. Task 1 is server-side + template cleanup in `accounts` (queryset filter, dead filter-option removal). Task 2 is client-side JS + markup in the `feedback` form editor (checkbox sync else-branch, search input + filter loop).

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Bootstrap 5 · SQLite (dev/test)

## Global Constraints

- **User list exclusion rule:** a user is listed iff they have at least one group OR `is_staff` OR `is_superuser`. The queryset MUST carry `.distinct()` — the groups join duplicates users in multiple groups.
- **Stats cards unchanged:** "Total"/"Active"/"Students" keep counting ALL accounts platform-wide (the `agg` aggregate block in `admin_dashboard` is untouched). Only the table's queryset changes.
- **Scenario search must never change checked state:** filtering only toggles row `display`; hidden checked rows still submit (HTML forms post hidden inputs; only `disabled` would drop them — never set it).
- **Assign-to-all sync is all-or-nothing regardless of an active search filter:** the master toggle hits every `.scenario-cb` via `querySelectorAll`, including rows currently hidden by the search.
- Responsive per this branch's conventions: the new search input is a full-width `form-control` (no fixed pixel widths); everything else touched is existing-markup surgery.
- Client-side-only behaviors (checkbox sync, live filtering) cannot be unit-tested server-side — render-level tests assert the markup/elements exist; the interactive behavior is flagged for manual verification, not assumed.

---

### Task 1: Exclude no-role accounts from the User Management table

**Files:**
- Modify: `Trust-AI-Platform/accounts/admin_views.py`
- Modify: `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Produces: no new interfaces — `admin_dashboard`'s `all_users` context changes contents only.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class AdminDashboardUserListTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.staff = User.objects.create_user('list_staff', password='pass', is_staff=True)
          self.client.login(username='list_staff', password='pass')

      def test_no_role_user_not_listed(self):
          User.objects.create_user('list_student', password='pass')
          r = self.client.get(reverse('admin_dashboard'))
          usernames = [u.username for u in r.context['all_users']]
          self.assertNotIn('list_student', usernames)

      def test_grouped_user_listed(self):
          from django.contrib.auth.models import Group
          teachers, _ = Group.objects.get_or_create(name='teachers')
          teacher = User.objects.create_user('list_teacher', password='pass')
          teacher.groups.add(teachers)
          r = self.client.get(reverse('admin_dashboard'))
          usernames = [u.username for u in r.context['all_users']]
          self.assertIn('list_teacher', usernames)

      def test_staff_without_group_listed(self):
          r = self.client.get(reverse('admin_dashboard'))
          usernames = [u.username for u in r.context['all_users']]
          self.assertIn('list_staff', usernames)

      def test_multi_group_user_listed_once(self):
          from django.contrib.auth.models import Group
          g1, _ = Group.objects.get_or_create(name='teachers')
          g2, _ = Group.objects.get_or_create(name='dspace_partners')
          multi = User.objects.create_user('list_multi', password='pass')
          multi.groups.add(g1, g2)
          r = self.client.get(reverse('admin_dashboard'))
          usernames = [u.username for u in r.context['all_users']]
          self.assertEqual(usernames.count('list_multi'), 1)

      def test_stats_still_count_all_accounts(self):
          User.objects.create_user('list_student2', password='pass')
          r = self.client.get(reverse('admin_dashboard'))
          self.assertEqual(r.context['stats']['total'], User.objects.count())
          self.assertEqual(r.context['stats']['no_role'], 1)

      def test_no_role_filter_option_removed(self):
          r = self.client.get(reverse('admin_dashboard'))
          self.assertNotContains(r, 'id="rfNone"')
  ```

  Note: check the top of `accounts/tests.py` for what's already imported (`User`, `Client`, `TestCase`, `reverse` are — `Group` may not be at module level, hence the function-level imports above; if `Group` IS already imported at module level, drop the local imports and use it directly).

- [ ] **Step 2: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.AdminDashboardUserListTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `test_no_role_user_not_listed` and `test_no_role_filter_option_removed` fail against the current everyone-listed behavior; the others may pass already.

- [ ] **Step 3: Filter the queryset**

  In `Trust-AI-Platform/accounts/admin_views.py`, replace:
  ```python
      users = User.objects.all().prefetch_related('groups').order_by('username')
  ```
  with:
  ```python
      users = User.objects.filter(
          Q(groups__isnull=False) | Q(is_staff=True) | Q(is_superuser=True)
      ).distinct().prefetch_related('groups').order_by('username')
  ```
  Note: `Q` is already imported at the top of this file (`from django.db.models import Count, Q`) — no import change.

- [ ] **Step 4: Remove the dead "No Role" filter option**

  In `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`, replace:
  ```html
                  {% endif %}
                  <li><hr class="dropdown-divider my-1"></li>
                  <li>
                    <div class="form-check mb-1">
                      <input class="form-check-input role-filter-check" type="checkbox" value="__none__" id="rfNone">
                      <label class="form-check-label" for="rfNone" style="font-size:13px;color:#888;">No Role</label>
                    </div>
                  </li>
                  <li><hr class="dropdown-divider my-1"></li>
  ```
  with:
  ```html
                  {% endif %}
                  <li><hr class="dropdown-divider my-1"></li>
  ```
  (The template's actual indentation may differ slightly from the above — anchor on the `value="__none__"`/`id="rfNone"` block, which appears exactly once, and remove that `<li>` plus ONE of its adjacent divider `<li>`s so a single divider remains between the group list and the Clear-filter button.)

  Then, in the same file's `filterUsers()` JS function, replace:
  ```javascript
        matchRole = checkedRoles.some(function(role) {
          if (role === '__none__') return userGroups.length === 0;
          return userGroups.includes(role);
        });
  ```
  with:
  ```javascript
        matchRole = checkedRoles.some(function(role) {
          return userGroups.includes(role);
        });
  ```

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.AdminDashboardUserListTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (6 tests).

  Then the full accounts suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK`, zero failures. (If any pre-existing dashboard test asserted a no-role user in the list, it will surface here — fix the expectation, not the queryset.)

- [ ] **Step 6: Commit**

  ```bash
  git add Trust-AI-Platform/accounts/admin_views.py Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html Trust-AI-Platform/accounts/tests.py
  git commit -m "Hide no-role accounts from the User Management table"
  ```

---

### Task 2: Two-way assign-all sync + scenario search in the form editor

**Files:**
- Modify: `Trust-AI-Platform/feedback/templates/feedback/form_edit.html`
- Modify: `Trust-AI-Platform/feedback/tests.py`

**Interfaces:**
- Consumes: nothing new. Produces: no new interfaces.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/feedback/tests.py`:
  ```python
  class FormEditorScenarioControlsTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.staff = User.objects.create_user('editor_staff', password='pass', is_staff=True)
          Scenario.objects.create(name='Editor Scenario', created_by=self.staff, updated_by=self.staff)
          self.client.login(username='editor_staff', password='pass')

      def test_scenario_search_input_rendered(self):
          r = self.client.get(reverse('feedback_form_create'))
          self.assertContains(r, 'id="scenarioSearch"')

      def test_no_match_hint_rendered(self):
          r = self.client.get(reverse('feedback_form_create'))
          self.assertContains(r, 'id="scenarioNoMatch"')
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.FormEditorScenarioControlsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — neither element exists yet.

- [ ] **Step 3: Add the search input and no-match hint to the template**

  In `Trust-AI-Platform/feedback/templates/feedback/form_edit.html`, replace:
  ```html
            <div class="text-muted small mb-2" id="scenarioHint"></div>
            <div class="scenario-box">
  ```
  with:
  ```html
            <div class="text-muted small mb-2" id="scenarioHint"></div>
            <input type="text" class="form-control form-control-sm mb-2" id="scenarioSearch" placeholder="Search scenarios…" autocomplete="off">
            <div class="scenario-box">
  ```

  Then, in the same file, replace:
  ```html
              {% empty %}
              <div class="text-muted small">No scenarios exist yet.</div>
              {% endfor %}
            </div>
  ```
  with:
  ```html
              {% empty %}
              <div class="text-muted small">No scenarios exist yet.</div>
              {% endfor %}
              <div class="text-muted small" id="scenarioNoMatch" style="display:none;">No scenarios match.</div>
            </div>
  ```

- [ ] **Step 4: Add the sync else-branch and the filter JS**

  In the same file's script block, replace:
  ```javascript
    assignAll.addEventListener('change', function () {
      updateHint();
      if (assignAll.checked) {
        document.querySelectorAll('.scenario-cb').forEach(function (cb) { cb.checked = true; });
      }
    });
    updateHint();
  ```
  with:
  ```javascript
    assignAll.addEventListener('change', function () {
      updateHint();
      const checkAll = assignAll.checked;
      document.querySelectorAll('.scenario-cb').forEach(function (cb) { cb.checked = checkAll; });
    });
    updateHint();

    const scenarioSearch = document.getElementById('scenarioSearch');
    if (scenarioSearch) {
      scenarioSearch.addEventListener('input', function () {
        const q = scenarioSearch.value.trim().toLowerCase();
        let anyVisible = false;
        document.querySelectorAll('.scenario-box .form-check').forEach(function (row) {
          const label = row.querySelector('.form-check-label');
          if (!label) return;
          const match = !q || label.textContent.toLowerCase().includes(q);
          row.style.display = match ? '' : 'none';
          if (match) anyVisible = true;
        });
        const noMatch = document.getElementById('scenarioNoMatch');
        if (noMatch) noMatch.style.display = anyVisible ? 'none' : '';
      });
    }
  ```

  Notes: the filter loop targets `.scenario-box .form-check` rows only — the `{% empty %}` div and the `#scenarioNoMatch` hint have no `form-check` class, so they're untouched by it. Filtering only sets `row.style.display`; it never touches `checked` or `disabled`, so hidden checked rows still submit. The assign-all sync intentionally uses `querySelectorAll('.scenario-cb')` (all rows, hidden or not) — the master toggle is all-or-nothing regardless of an active search, per the spec.

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback.tests.FormEditorScenarioControlsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (2 tests).

  Then the full feedback suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test feedback -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK`, zero failures.

- [ ] **Step 6: Manually verify (if a real dev environment is available)**

  Open the form editor. Type in the search box — rows filter live, clearing restores all, a nonsense query shows "No scenarios match." Check a filtered-out scenario, search for something else, save — confirm the hidden checked scenario still submitted. Toggle "Assign to all scenarios" off — every box unchecks (including hidden ones); toggle on — every box checks.

  If unavailable, Step 5's render tests are the load-bearing automated verification; the interactive behavior must be flagged as unverified, not assumed.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/feedback/templates/feedback/form_edit.html Trust-AI-Platform/feedback/tests.py
  git commit -m "Add two-way assign-all sync and scenario search to form editor"
  ```
