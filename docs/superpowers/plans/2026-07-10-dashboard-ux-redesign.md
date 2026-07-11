# Dashboard UX Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the missing empty/prompt state on the analytics dashboard, add visual hierarchy via section group headers, and move the URL from `/authoringtool/` to `/authoringtool/dashboard/`.

**Architecture:** Two sequential tasks — URL rename first (one file, trivial), then the template overhaul (one file, all HTML/CSS/JS changes). No backend view changes. No new API endpoints.

**Tech Stack:** Django 5.1 · Bootstrap 5 · Bootstrap Icons · vanilla JS · ECharts (unchanged)

## Global Constraints

- The page must remain fully responsive on phones (≥320 px) and tablets (≥768 px).
- No new fixed pixel widths on outer containers — use Bootstrap grid classes and relative units only.
- Do NOT touch chart rendering logic, Celery polling, or the filter card internals.
- Do NOT change any template other than `index.html` and `urls.py`.

---

### Task 1: Rename URL to `/authoringtool/dashboard/`

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/urls.py`

**Interfaces:**
- Produces: named URL `index` now resolves to `/authoringtool/dashboard/` — all four existing `{% url 'index' %}` references in other templates continue to work without changes.

- [ ] **Step 1: Open urls.py and read its current first two lines**

  ```
  Trust-AI-Platform/authoringtool/urls.py  lines 1-6
  ```

  Current state:
  ```python
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.index, name='index'),
      path('scenarios/', views.scenarios, name='scenarios'),
  ```

- [ ] **Step 2: Edit urls.py — add RedirectView import and new entries**

  Replace the import block and first urlpattern entry with:

  ```python
  from django.urls import path
  from django.views.generic import RedirectView
  from . import views

  urlpatterns = [
      path('', RedirectView.as_view(pattern_name='index', permanent=False)),
      path('dashboard/', views.index, name='index'),
      path('scenarios/', views.scenarios, name='scenarios'),
  ```

  `permanent=False` → HTTP 302 (safe to change later). `pattern_name='index'` means the redirect target is resolved by name, not hardcoded string.

- [ ] **Step 3: Verify no other file needs changing**

  Run:
  ```
  grep -rn "url 'index'\|reverse('index')" Trust-AI-Platform/ --include="*.html" --include="*.py"
  ```

  Expected: 4 hits, all `{% url 'index' %}` in templates — none need editing because `name='index'` is preserved on the new `dashboard/` path.

- [ ] **Step 4: Start the dev server and manually verify**

  ```bash
  python manage.py runserver
  ```

  Check:
  - `GET /authoringtool/` → HTTP 302 → `/authoringtool/dashboard/`
  - `GET /authoringtool/dashboard/` → HTTP 200, page title "Trust AI Lab — Dashboard"
  - The sidebar/nav link (`{% url 'index' %}`) in `head.html` resolves to `/authoringtool/dashboard/`

- [ ] **Step 5: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/urls.py
  git commit -m "Move authoringtool root URL to /authoringtool/dashboard/"
  ```

---

### Task 2: Empty state, section headers, and show/hide logic

**Files:**
- Modify: `Trust-AI-Platform/authoringtool/templates/authoringtool/index.html`

**Interfaces:**
- Consumes: named URL `index` resolves to `/authoringtool/dashboard/` (from Task 1)
- Produces: page shows empty state prompt when no scenario is selected; shows grouped charts with section headers when a scenario is applied

**Overview of HTML changes needed:**

The current structure inside `<section class="section dashboard">` is:

```
<div class="row">
  [col-12: Filter card]         ← keep
  [col-12: Sankey card]         ← move into wrapper
  [col-12: Waterfall card]      ← move into wrapper
</div>
<div class="row">
  [col-12 col-lg-6: Activity Answers + Performance Overview]  ← move into wrapper
  [col-12 col-lg-6: Time Spent + Detailed Phase Scores]       ← move into wrapper
  [col-12 col-lg-6: Performers]                               ← move into wrapper
  [col-12 col-lg-6: Phase Time]                               ← move into wrapper
</div>
```

Target structure:

```
<div class="row">
  [col-12: Filter card]         ← unchanged
</div>

<div id="dash-empty-state">     ← NEW, visible by default
  ...prompt text...
</div>

<div id="dash-charts-wrapper" style="display:none;">   ← NEW wrapper, hidden by default
  [Flow section header]
  <div class="row">
    [col-12: Sankey]
    [col-12: Waterfall]
  </div>

  [Performance section header]
  <div class="row">
    [col-12 col-lg-6: Activity Answers + Performance Overview]
    [col-12 col-lg-6: Detailed Phase Scores + Performers]
  </div>

  [Engagement section header]
  <div class="row">
    [col-12 col-lg-6: Time Spent]
    [col-12 col-lg-6: Phase Time]
  </div>
</div>
```

Note the Performance row reorganises which charts are in the right column (Detailed Phase Scores + Performers, instead of Time Spent + Detailed Phase Scores). This puts all performance-related charts together and all engagement/time charts together.

- [ ] **Step 1: Add CSS for `.dash-section-header` and `#dash-empty-state` inside the existing `<style>` block**

  Append inside the `<style>` tag (before its closing `</style>`):

  ```css
  /* ── Section headers ── */
  .dash-section-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; font-weight: 700; color: #1e3a8a;
    border-bottom: 2px solid #e8edf5;
    padding-bottom: 8px; margin: 28px 0 16px;
    text-transform: uppercase; letter-spacing: 0.6px;
  }
  .dash-section-header i { font-size: 15px; color: #1a56db; }

  /* ── Empty state ── */
  #dash-empty-state {
    text-align: center; padding: 60px 20px 40px;
  }
  #dash-empty-state .dash-empty-icon {
    font-size: 3rem; color: #d1d9e0; display: block; margin-bottom: 16px;
  }
  #dash-empty-state h5 {
    color: #374151; font-weight: 700; font-size: 16px; margin-bottom: 8px;
  }
  #dash-empty-state p {
    color: #9ca3af; font-size: 14px; margin: 0;
  }
  ```

- [ ] **Step 2: Close the first `.row` after the filter card col, and add the empty state block**

  The filter card col ends at approximately line 382–383 (the `</div></div>` that closes the filter card's inner content and then the `col-12` div). After that closing `</div>` (the col-12), insert:

  ```html
  </div><!-- /row (filter only) -->

  <!-- ── Empty state ── -->
  <div id="dash-empty-state">
    <i class="bi bi-bar-chart-line dash-empty-icon"></i>
    <h5>Select a scenario to view analytics</h5>
    <p>Use the filter above to choose a scenario, then click Apply.</p>
  </div>

  <!-- ── Charts wrapper (hidden until scenario applied) ── -->
  <div id="dash-charts-wrapper" style="display:none;">
  ```

  At the very end of `</section>` (after all chart rows), close the wrapper:

  ```html
  </div><!-- /dash-charts-wrapper -->
  </section>
  ```

- [ ] **Step 3: Add Flow section header inside the wrapper, before the Sankey card**

  Inside `#dash-charts-wrapper`, open a new row for the Flow group and its header:

  ```html
  <!-- ── Flow ── -->
  <div class="row">
    <div class="col-12">
      <div class="dash-section-header">
        <i class="bi bi-diagram-3"></i><span>Flow</span>
      </div>
    </div>
    <!-- Sankey card col -->
    <div class="col-12">
      <div class="card">
        ... (existing Sankey card body, unchanged) ...
      </div>
    </div>
    <!-- Waterfall card col -->
    <div class="col-12">
      <div class="card">
        ... (existing Waterfall card body, unchanged) ...
      </div>
    </div>
  </div><!-- /row Flow -->
  ```

- [ ] **Step 4: Add Performance section header and reorganise the second row**

  After the Flow row, add:

  ```html
  <!-- ── Performance ── -->
  <div class="row">
    <div class="col-12">
      <div class="dash-section-header">
        <i class="bi bi-graph-up"></i><span>Performance</span>
      </div>
    </div>
  </div>
  <div class="row">
    <!-- Left column: Activity Answers + Performance Overview -->
    <div class="col-12 col-lg-6">
      <div class="row">
        <div class="col-12">
          ... Activity Answers card (unchanged) ...
        </div>
        <div class="col-12">
          ... Performance Overview card (unchanged) ...
        </div>
      </div>
    </div>
    <!-- Right column: Detailed Phase Scores + Performers per Phase -->
    <div class="col-12 col-lg-6">
      ... Detailed Phase Scores card (unchanged) ...
      ... Number of Performers per Phase card (unchanged) ...
    </div>
  </div><!-- /row Performance -->
  ```

- [ ] **Step 5: Add Engagement section header and reorganise the time charts**

  After the Performance row:

  ```html
  <!-- ── Engagement ── -->
  <div class="row">
    <div class="col-12">
      <div class="dash-section-header">
        <i class="bi bi-clock-history"></i><span>Engagement</span>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="col-12 col-lg-6">
      ... Average Time Spent per Activity/Phase card (unchanged) ...
    </div>
    <div class="col-12 col-lg-6">
      ... Time Spent in Each Phase by Performer Type card (unchanged) ...
    </div>
  </div><!-- /row Engagement -->
  ```

- [ ] **Step 6: Add the central show/hide JavaScript**

  Add the following `<script>` block immediately after the closing `</style>` tag and before `<main id="main" ...>`. This script must run before the chart-specific `DOMContentLoaded` handlers:

  ```html
  <script>
  (function () {
    function showCharts() {
      document.getElementById('dash-empty-state').style.display = 'none';
      document.getElementById('dash-charts-wrapper').style.display = '';
    }
    function showEmpty() {
      document.getElementById('dash-empty-state').style.display = '';
      document.getElementById('dash-charts-wrapper').style.display = 'none';
    }

    document.addEventListener('DOMContentLoaded', function () {
      var scenarioSelect = document.getElementById('scenarioSelect');
      var filterForm     = document.getElementById('filter-form');
      var resetBtn       = document.getElementById('reset-button');

      // Initial state
      showEmpty();

      // Apply button
      filterForm.addEventListener('submit', function (e) {
        if (scenarioSelect.value) {
          showCharts();
        } else {
          showEmpty();
        }
        // Do NOT call e.preventDefault() — chart handlers below need the event too
      });

      // Reset button — clear picker and return to empty state
      if (resetBtn) {
        resetBtn.addEventListener('click', function () {
          showEmpty();
        });
      }
    });
  }());
  </script>
  ```

  **Important:** The existing chart-specific `filterForm.addEventListener('submit', ...)` handlers each call `event.preventDefault()`. This is fine — the show/hide handler does NOT call `preventDefault`, so those handlers still fire and prevent a real form submission. The order of execution is: show/hide handler fires first (no `preventDefault`), then each chart's handler fires and calls `preventDefault`.

- [ ] **Step 7: Manually verify on desktop**

  Start dev server. Navigate to `/authoringtool/dashboard/`.

  - Page loads → empty state visible, charts hidden ✓
  - Select a scenario, click Apply → empty state hides, all three section groups appear with headers ✓
  - Click Reset → empty state reappears, charts hidden ✓
  - Apply again with no scenario selected → empty state stays ✓
  - Section headers render correctly: Flow / Performance / Engagement with icons ✓

- [ ] **Step 8: Manually verify responsive layout**

  Open browser DevTools, switch to mobile (375 px) and tablet (768 px) viewport.

  - Empty state: text centered, not clipped ✓
  - Section headers: text readable, no overflow ✓
  - Charts: each card stacks full-width on mobile (`col-12` kicks in), two-column on desktop (`col-lg-6`) ✓
  - No horizontal scroll on any viewport ✓

- [ ] **Step 9: Commit**

  ```bash
  git add Trust-AI-Platform/authoringtool/templates/authoringtool/index.html
  git commit -m "Dashboard: empty state, section group headers, show/hide on scenario apply"
  ```
