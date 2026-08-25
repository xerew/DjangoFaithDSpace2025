# Dashboard UX Redesign — Design Spec

## Goal

Improve the analytics dashboard at `/authoringtool/` by fixing the missing empty state, adding visual hierarchy to the chart wall, and moving the URL to `/authoringtool/dashboard/`.

## Global Constraints

- **Responsive & mobile-first:** every change must work correctly on phones (≥320px) and tablets (≥768px). No new fixed widths. The empty state, section headers, and chart containers must all reflow gracefully on small screens. Use Bootstrap's responsive grid (`col-12`, `col-lg-6`, etc.) and relative units — no `px`-based widths on outer containers.

## Scope

- `Trust-AI-Platform/authoringtool/urls.py` — URL path change + redirect
- `Trust-AI-Platform/authoringtool/templates/authoringtool/index.html` — empty state + chart grouping
- No backend view changes required; no new API endpoints; no KPI cards

---

## 1. URL Change

| Before | After |
|---|---|
| `path('', views.index, name='index')` | `path('dashboard/', views.index, name='dashboard')` |
| — | `path('', RedirectView.as_view(url='/authoringtool/dashboard/'), name='index')` |

- The named URL `index` is used in the breadcrumb inside `index.html` (`{% url 'index' %}`). It must be updated to `dashboard` (or the redirect alias kept as `index`).
- No other templates use `{% url 'index' %}` — confirm with grep before changing.
- The `main.html` navbar/sidebar likely links to `{% url 'index' %}` — all such references must be updated to point to `dashboard`.

---

## 2. Empty State

**When shown:** the page loads with no scenario selected (i.e. `scenarioSelect.value === ''`).

**What is shown:**
- The Analytics Filters card at the top (unchanged).
- A centered empty-state block below the filter card:
  - Bootstrap icon `bi-bar-chart-line`, size `3rem`, colour `#d1d9e0`
  - `<h5>` heading: "Select a scenario to view analytics"
  - `<p>` subtext: "Use the filter above to choose a scenario, then click Apply."
  - Styling matches the existing `.sc-empty` pattern used in `scenarios.html`

**What is hidden:** every chart section (`sankeyChart`, `waterfallChart`, `activityChart`, `performanceChart`, `timeSpentChart`, `detailedPhaseScoresChart`, `performersChart`, `phaseTimeChart`) and their loading divs, plus the three section-group headers.

**Trigger to reveal charts:** the filter form `submit` event with a non-empty `scenarioSelect.value`. If the user resets and re-submits with no scenario, return to the empty state.

---

## 3. Chart Grouping (section headers)

When a scenario is selected and Apply is clicked, charts appear grouped under three section dividers inserted in the HTML above the relevant chart cards:

| Group | Charts included |
|---|---|
| **Flow** | Student Performance (Sankey) · Final Student Performance (Waterfall) |
| **Performance** | Activity Answers Report · Performance Overview · Detailed Phase Scores Overview · Number of Performers per Phase |
| **Engagement** | Average Time Spent per Activity/Phase · Time Spent in Each Phase by Performer Type |

**Section divider markup** (one per group):

```html
<div class="dash-section-header">
  <i class="bi bi-{icon}"></i>
  <span>{Group Name}</span>
</div>
```

Icons: Flow → `bi-diagram-3`, Performance → `bi-graph-up`, Engagement → `bi-clock-history`

**CSS for `.dash-section-header`:**
```css
.dash-section-header {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 700; color: #1e3a8a;
  border-bottom: 2px solid #e8edf5;
  padding-bottom: 8px; margin: 28px 0 16px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.dash-section-header i { font-size: 15px; color: #1a56db; }
```

Section headers are hidden alongside charts on the empty state, revealed on Apply.

---

## 4. Behaviour Details

- On page load with a pre-selected scenario (the current code pre-selects "Photoelectric Effect" if it exists), the empty state must NOT show — charts load immediately as they do today. The pre-selection logic already fires `scenarioSelect.dispatchEvent(new Event('change'))` on DOMContentLoaded; the "Apply" button click (`filterForm submit`) is what triggers chart rendering. So the empty state is shown by default, and the first Apply (or an auto-submit triggered when a pre-selected scenario fires) hides it.
- The CSV "Generate & Download" button remains in the filter card. It already guards against no scenario with `alert("Please select a scenario.")`.
- Loading spinners (`⏳ Generating …`) inside each chart's loading div remain unchanged in markup; they are simply inside hidden sections until Apply is clicked.

---

## 5. What does NOT change

- Chart rendering logic (ECharts, Celery polling, fetch calls) — untouched
- Filter card layout (scenario picker, group select, date range, Apply, Reset, CSV) — untouched
- Hero banner — untouched (title "Dashboard", breadcrumb "Home → Dashboard")
- Admin dashboard (`admin_dashboard.html`) — not in scope
- Scenarios page (`scenarios.html`) — not in scope
