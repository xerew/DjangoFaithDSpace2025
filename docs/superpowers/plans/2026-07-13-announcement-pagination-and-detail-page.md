# Announcement Pagination and Detail Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Announcements card on the organization detail page into a paginated (5/page), truncated-preview list where each announcement links to a dedicated detail page showing the full content.

**Architecture:** Two sequential tasks. Task 1 builds the destination first — a standalone `announcement_detail` view, URL, and template, reachable directly by URL even before anything links to it. Task 2 paginates and truncates the existing list and wires each row to link to that detail page — mirroring this branch's established ordering (build the destination before wiring the entry point, as in the Team Chat plan's Task 2-then-Task 3 split).

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Bootstrap 5 · SQLite (dev/test)

## Global Constraints

- No model or migration changes. `Announcement.plain_text` already exists and is already auto-synced from `body` via `strip_html_tags` on every create/edit (`organization/views.py`'s `create_announcement`/`edit_announcement`) — it is the truncation source, nothing new to populate it.
- **Reuse the existing site-wide pagination convention**, don't invent a new one: `django.core.paginator.Paginator`, a local `_smart_page_range(page_obj)` helper (ellipsis'd page numbers — keeps page 1, the last page, and a window around the current page), and the matching Bootstrap `.pagination` markup, all already established in `authoringtool/views.py:261-277` and `authoringtool/templates/authoringtool/scenarios.html:317-351`. Duplicate `_smart_page_range` locally into `organization/views.py` rather than importing it cross-app — matches this app's established per-file-duplication convention (`group_required`, `strip_html_tags`, `_is_org_admin` are all local duplicates, not cross-app imports).
- Unlike `scenarios.html`'s AJAX-swapped pagination, this list uses a **plain full-page reload** (`?page=N`) — no AJAX infrastructure needed for a 5-per-page list.
- Preview truncation: `{{ announcement.plain_text|truncatechars:250 }}` — Django's built-in filter, rendered as plain auto-escaped text (no `|safe`), since `plain_text` has no HTML in it by construction.
- The detail page's full body still renders with `|safe` — same trust model as the existing list did, already decided (ship as-is) in the original Announcements plan. Not revisited here.
- `announcement_detail` is `@login_required` only, **no membership gate** — matches `organization_detail`'s own existing pattern (any logged-in user can view the org page and its announcements; membership/admin status only changes which buttons are visible).
- Admin Edit/Delete controls appear in **both** places: on each row of the paginated list (unchanged from today) and on the detail page (new).
- Cross-org IDOR safety: `announcement_detail`'s `Announcement` lookup must be scoped by `organization=organization` (404 if the announcement belongs to a different org), matching the exact pattern `edit_announcement`/`delete_announcement` already use.
- Responsive & mobile-first: the Bootstrap `.pagination` component and truncated preview text wrap naturally at any width; no new fixed-pixel widths on outer containers. The new detail template reuses the `@media (max-width: 575.98px)` breakpoint pattern already established for `.org-hero` in `create_announcement.html`/`edit_announcement.html`.

---

### Task 1: `announcement_detail` view, URL, and template

**Files:**
- Modify: `Trust-AI-Platform/organization/views.py`
- Modify: `Trust-AI-Platform/organization/urls.py`
- Modify: `Trust-AI-Platform/organization/tests.py`
- Create: `Trust-AI-Platform/organization/templates/organization/announcement_detail.html`

**Interfaces:**
- Consumes: `Announcement`, `_is_org_admin` (already in `organization/views.py`)
- Produces: URL name `announcement_detail` (`organization/<int:org_id>/announcements/<int:announcement_id>/`) — consumed by Task 2's list-row links.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  class AnnouncementDetailViewTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.admin = User.objects.create_user('detail_admin', password='pass')
          self.member = User.objects.create_user('detail_member', password='pass')
          self.outsider = User.objects.create_user('detail_outsider', password='pass')
          self.org = Organization.objects.create(
              name='Detail Org', short_name='DO', created_by=self.admin,
          )
          self.org.admins.add(self.admin)
          self.org.members.add(self.admin, self.member)
          self.announcement = Announcement.objects.create(
              organization=self.org, title='Full Detail', body='<p>Full <b>content</b> here.</p>',
              plain_text='Full content here.', created_by=self.admin,
          )
          self.other_org = Organization.objects.create(
              name='Other Org', short_name='OO', created_by=self.outsider,
          )

      def test_member_can_view_full_announcement(self):
          self.client.login(username='detail_member', password='pass')
          r = self.client.get(reverse('announcement_detail', args=[self.org.id, self.announcement.id]))
          self.assertEqual(r.status_code, 200)
          self.assertContains(r, 'Full Detail')
          self.assertContains(r, 'Full <b>content</b> here.')

      def test_non_member_can_still_view(self):
          self.client.login(username='detail_outsider', password='pass')
          r = self.client.get(reverse('announcement_detail', args=[self.org.id, self.announcement.id]))
          self.assertEqual(r.status_code, 200)

      def test_login_required(self):
          r = self.client.get(reverse('announcement_detail', args=[self.org.id, self.announcement.id]))
          self.assertEqual(r.status_code, 302)

      def test_admin_sees_edit_delete_controls(self):
          self.client.login(username='detail_admin', password='pass')
          r = self.client.get(reverse('announcement_detail', args=[self.org.id, self.announcement.id]))
          self.assertContains(r, reverse('edit_announcement', args=[self.org.id, self.announcement.id]))
          self.assertContains(r, reverse('delete_announcement', args=[self.org.id, self.announcement.id]))

      def test_non_admin_does_not_see_edit_delete_controls(self):
          self.client.login(username='detail_member', password='pass')
          r = self.client.get(reverse('announcement_detail', args=[self.org.id, self.announcement.id]))
          self.assertNotContains(r, reverse('edit_announcement', args=[self.org.id, self.announcement.id]))
          self.assertNotContains(r, reverse('delete_announcement', args=[self.org.id, self.announcement.id]))

      def test_announcement_from_wrong_org_404s(self):
          self.client.login(username='detail_member', password='pass')
          r = self.client.get(reverse('announcement_detail', args=[self.other_org.id, self.announcement.id]))
          self.assertEqual(r.status_code, 404)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementDetailViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` (the `announcement_detail` URL name doesn't exist yet).

- [ ] **Step 3: Add the view**

  In `Trust-AI-Platform/organization/views.py`, append at the end of the file:
  ```python

  @login_required
  def announcement_detail(request, org_id, announcement_id):
      organization = get_object_or_404(Organization, id=org_id)
      announcement = get_object_or_404(Announcement, id=announcement_id, organization=organization)
      is_admin = _is_org_admin(request.user, organization)

      return render(request, 'organization/announcement_detail.html', {
          'organization': organization,
          'announcement': announcement,
          'is_admin': is_admin,
      })
  ```

- [ ] **Step 4: Add the URL**

  In `Trust-AI-Platform/organization/urls.py`, replace:
  ```python
      path('organization/<int:org_id>/announcements/create/', views.create_announcement, name='create_announcement'),
      path('organization/<int:org_id>/announcements/<int:announcement_id>/edit/', views.edit_announcement, name='edit_announcement'),
  ```
  with:
  ```python
      path('organization/<int:org_id>/announcements/create/', views.create_announcement, name='create_announcement'),
      path('organization/<int:org_id>/announcements/<int:announcement_id>/', views.announcement_detail, name='announcement_detail'),
      path('organization/<int:org_id>/announcements/<int:announcement_id>/edit/', views.edit_announcement, name='edit_announcement'),
  ```

  Note: no route-collision risk — Django's `<int:...>` path converter only matches digit strings, so it can never accidentally match the literal `create` segment, and the `/edit/`/`/delete/` routes have an extra trailing segment this pattern doesn't, so they remain distinct regardless of declaration order.

- [ ] **Step 5: Create the detail template**

  Create `Trust-AI-Platform/organization/templates/organization/announcement_detail.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — {{ announcement.title }}</title>{% endblock %}
  {% block atcontent %}

  <style>
    .org-hero {
      background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
      border-radius: 14px; padding: 26px 30px 20px; color: #fff;
      margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18);
    }
    .org-hero-icon {
      background: rgba(255,255,255,0.18); border-radius: 10px;
      width: 50px; height: 50px; display: flex; align-items: center;
      justify-content: center; font-size: 22px; flex-shrink: 0;
    }
    .org-hero .breadcrumb { background: none; margin: 10px 0 0; padding: 0; font-size: 12px; }
    .org-hero .breadcrumb-item+.breadcrumb-item::before { color: rgba(255,255,255,0.5); }
    .org-hero .breadcrumb-item a { color: rgba(255,255,255,0.72); text-decoration: none; }
    .org-hero .breadcrumb-item a:hover { color: #fff; }
    .org-hero .breadcrumb-item.active { color: rgba(255,255,255,0.92); }
    @media (max-width: 575.98px) {
      .org-hero { padding: 14px 16px 12px; }
      .org-hero > .d-flex { flex-wrap: wrap; }
      .org-hero-icon { display: none; }
      .org-hero .d-flex.flex-shrink-0 { flex-shrink: 1 !important; width: 100%; justify-content: flex-start !important; margin-top: 10px; }
      .org-hero h2 { font-size: 15px !important; }
    }
    .hero-btn-ghost {
      background: rgba(255,255,255,0.15); color: #fff; border: 1.5px solid rgba(255,255,255,0.4);
      font-weight: 600; font-size: 13.5px; border-radius: 8px;
      padding: 7px 18px; display: inline-flex; align-items: center; gap: 6px;
      text-decoration: none; transition: background 0.15s; white-space: nowrap;
    }
    .hero-btn-ghost:hover { background: rgba(255,255,255,0.25); color: #fff; }
    .action-form { display: inline; }
    .action-btn { background: none; border: none; padding: 4px 6px; border-radius: 4px; cursor: pointer; font-size: 15px; line-height: 1; transition: background 0.15s; color: #333; }
    .action-btn:hover { background: #f0f4ff; }
    .action-btn.remove { color: #888; }
    .action-btn.remove:hover { color: #c62828; background: #ffebee; }
    .announcement-body img { max-width: 100%; height: auto; }
    .form-card { max-width: 760px; margin: 0 auto; }
    .member-meta { font-size: 12px; color: #888; }
  </style>

  <main id="main" class="main">
    <div class="org-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="org-hero-icon"><i class="bi bi-megaphone-fill"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Organizations</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">{{ announcement.title }}</h2>
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ organization.name }}</div>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'list_organizations' %}">Organizations</a></li>
            <li class="breadcrumb-item"><a href="{% url 'organization_detail' organization.id %}">{{ organization.short_name }}</a></li>
            <li class="breadcrumb-item active">{{ announcement.title }}</li>
          </ol></nav>
        </div>
        <div class="flex-shrink-0 d-flex gap-2 align-items-start" style="padding-top:4px;">
          <a href="{% url 'organization_detail' organization.id %}" class="hero-btn-ghost">
            <i class="bi bi-arrow-left"></i> Back
          </a>
        </div>
      </div>
    </div>

    <section class="section">
      <div class="card form-card">
        <div class="card-body p-4">
          <div class="d-flex align-items-start justify-content-between gap-2 flex-wrap mb-2">
            <div class="member-meta">
              {{ announcement.created_by.get_full_name|default:announcement.created_by.username|default:"Unknown" }}
              · {{ announcement.created_on|date:"d M Y, H:i" }}
            </div>
            {% if is_admin %}
            <div class="d-flex gap-1 flex-shrink-0">
              <a href="{% url 'edit_announcement' organization.id announcement.id %}" class="action-btn" title="Edit">
                <i class="bi bi-pencil"></i>
              </a>
              <form method="post" action="{% url 'delete_announcement' organization.id announcement.id %}" class="action-form" onsubmit="return confirm('Delete this announcement?');">
                {% csrf_token %}
                <button type="submit" class="action-btn remove" title="Delete">
                  <i class="bi bi-trash"></i>
                </button>
              </form>
            </div>
            {% endif %}
          </div>
          <div class="announcement-body" style="font-size:14px; color:#333; line-height:1.6; word-wrap:break-word;">{{ announcement.body|safe }}</div>
        </div>
      </div>
    </section>
  </main>
  {% endblock %}
  ```

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementDetailViewTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (6 tests).

  Then the full organization suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 7: Manually verify (if a real dev environment is available)**

  Same caveat as prior features on this branch — needs a live Postgres-backed dev environment. If available:
  ```bash
  python manage.py runserver
  ```
  Navigate directly to `/organization/<id>/announcements/<id>/` for an existing announcement. Confirm the full body renders (including any embedded images, bold/lists from TinyMCE), admin Edit/Delete buttons work, and a non-admin member sees the content with no controls.

  If unavailable, Step 6's automated tests are the load-bearing verification.

- [ ] **Step 8: Commit**

  ```bash
  git add Trust-AI-Platform/organization/views.py Trust-AI-Platform/organization/urls.py Trust-AI-Platform/organization/tests.py Trust-AI-Platform/organization/templates/organization/announcement_detail.html
  git commit -m "Add announcement detail page"
  ```

---

### Task 2: Paginate and truncate the announcements list, link each row to the detail page

**Files:**
- Modify: `Trust-AI-Platform/organization/views.py`
- Modify: `Trust-AI-Platform/organization/templates/organization/organization_detail.html`
- Modify: `Trust-AI-Platform/organization/tests.py`

**Interfaces:**
- Consumes: `announcement_detail` URL (Task 1)
- Produces: no new interfaces — `organization_detail` view context changes `announcements` from a plain queryset to a `Page` object, and adds `announcement_page_range`.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  class AnnouncementPaginationTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.admin = User.objects.create_user('page_admin', password='pass')
          self.org = Organization.objects.create(
              name='Pagination Org', short_name='PO', created_by=self.admin,
          )
          self.org.admins.add(self.admin)
          self.org.members.add(self.admin)
          for i in range(7):
              Announcement.objects.create(
                  organization=self.org, title=f'Announcement {i}', body=f'<p>Body {i}</p>',
                  plain_text=f'Body {i}', created_by=self.admin,
              )
          self.client.login(username='page_admin', password='pass')

      def test_first_page_shows_five_newest(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertEqual(len(r.context['announcements']), 5)
          self.assertContains(r, 'Announcement 6')
          self.assertNotContains(r, 'Announcement 1')

      def test_second_page_shows_remaining_two(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]), {'page': 2})
          self.assertEqual(len(r.context['announcements']), 2)
          self.assertContains(r, 'Announcement 1')
          self.assertContains(r, 'Announcement 0')

      def test_pagination_controls_hidden_for_single_page(self):
          Announcement.objects.filter(organization=self.org).delete()
          Announcement.objects.create(
              organization=self.org, title='Only One', body='<p>x</p>',
              plain_text='x', created_by=self.admin,
          )
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertNotContains(r, 'pagination')

      def test_pagination_controls_shown_for_multiple_pages(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, 'pagination')


  class AnnouncementPreviewLinkTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.admin = User.objects.create_user('link_admin', password='pass')
          self.org = Organization.objects.create(
              name='Link Org', short_name='LO', created_by=self.admin,
          )
          self.org.admins.add(self.admin)
          self.org.members.add(self.admin)
          self.announcement = Announcement.objects.create(
              organization=self.org, title='Link Me',
              body='<p>' + ('word ' * 100) + '</p>',
              plain_text='word ' * 100, created_by=self.admin,
          )
          self.client.login(username='link_admin', password='pass')

      def test_title_links_to_detail_page(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, reverse('announcement_detail', args=[self.org.id, self.announcement.id]))

      def test_preview_is_truncated(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          full_plain_text = 'word ' * 100
          self.assertNotContains(r, full_plain_text)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementPaginationTests organization.tests.AnnouncementPreviewLinkTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `organization_detail` currently returns all announcements unpaginated with no truncation and no link to a detail page.

- [ ] **Step 3: Add the `_smart_page_range` helper and paginate the view**

  In `Trust-AI-Platform/organization/views.py`, replace the import block:
  ```python
  from django.shortcuts import render, redirect, get_object_or_404
  from django.contrib.auth.decorators import login_required
  from django.core.exceptions import PermissionDenied
  from django.http import HttpResponseForbidden, JsonResponse
  from django.views.decorators.http import require_POST, require_GET
  from django.contrib import messages
  from functools import wraps
  from django.utils.html import strip_tags
  from html import unescape
  from .models import Organization, JoinRequest, Announcement, OrgChatMessage
  from django.contrib.auth.models import User
  from .forms import OrganizationForm, AnnouncementForm
  from authoringtool.models import Language
  ```
  with:
  ```python
  from django.shortcuts import render, redirect, get_object_or_404
  from django.contrib.auth.decorators import login_required
  from django.core.exceptions import PermissionDenied
  from django.core.paginator import Paginator
  from django.http import HttpResponseForbidden, JsonResponse
  from django.views.decorators.http import require_POST, require_GET
  from django.contrib import messages
  from functools import wraps
  from django.utils.html import strip_tags
  from html import unescape
  from .models import Organization, JoinRequest, Announcement, OrgChatMessage
  from django.contrib.auth.models import User
  from .forms import OrganizationForm, AnnouncementForm
  from authoringtool.models import Language


  def _smart_page_range(page_obj):
      """Returns a list of page numbers with None representing an ellipsis."""
      current = page_obj.number
      total = page_obj.paginator.num_pages
      pages = set()
      pages.add(1)
      pages.add(total)
      for i in range(max(1, current - 2), min(total, current + 2) + 1):
          pages.add(i)
      result = []
      prev = None
      for p in sorted(pages):
          if prev is not None and p - prev > 1:
              result.append(None)  # ellipsis
          result.append(p)
          prev = p
      return result
  ```

  Note: this is a verbatim duplicate of `authoringtool/views.py:261-277`'s existing helper — matches this app's established per-file-duplication convention, not a new pattern.

  Then in `organization_detail`, replace:
  ```python
      return render(request, 'organization/organization_detail.html', {
          'organization': organization,
          'is_member': is_member,
          'is_admin': is_admin,
          'join_request': join_request,
          'pending_requests': pending_requests,
          'announcements': organization.announcements.select_related('created_by'),
      })
  ```
  with:
  ```python
      announcements_qs = organization.announcements.select_related('created_by')
      announcements_paginator = Paginator(announcements_qs, 5)
      announcements_page = announcements_paginator.get_page(request.GET.get('page'))
      announcement_page_range = _smart_page_range(announcements_page)

      return render(request, 'organization/organization_detail.html', {
          'organization': organization,
          'is_member': is_member,
          'is_admin': is_admin,
          'join_request': join_request,
          'pending_requests': pending_requests,
          'announcements': announcements_page,
          'announcement_page_range': announcement_page_range,
      })
  ```

  Note: `announcements` stays the template's iteration variable — a `Page` object is iterable exactly like the queryset it replaces, so `{% for announcement in announcements %}` in the template needs no change to its loop line, only to what's rendered inside the loop (Step 4) plus new pagination controls below it.

- [ ] **Step 4: Update the Announcements card — truncated, linked preview rows plus pagination controls**

  In `Trust-AI-Platform/organization/templates/organization/organization_detail.html`, replace:
  ```html
              {% for announcement in announcements %}
              <div class="px-4 py-3 border-bottom">
                <div class="d-flex align-items-start justify-content-between gap-2 flex-wrap">
                  <h6 style="color:#012970; font-weight:700; margin:0;">{{ announcement.title }}</h6>
                  {% if is_admin %}
                  <div class="d-flex gap-1 flex-shrink-0">
                    <a href="{% url 'edit_announcement' organization.id announcement.id %}" class="action-btn" title="Edit">
                      <i class="bi bi-pencil"></i>
                    </a>
                    <form method="post" action="{% url 'delete_announcement' organization.id announcement.id %}" class="action-form" onsubmit="return confirm('Delete this announcement?');">
                      {% csrf_token %}
                      <button type="submit" class="action-btn remove" title="Delete">
                        <i class="bi bi-trash"></i>
                      </button>
                    </form>
                  </div>
                  {% endif %}
                </div>
                <div class="member-meta mb-2">
                  {{ announcement.created_by.get_full_name|default:announcement.created_by.username|default:"Unknown" }}
                  · {{ announcement.created_on|date:"d M Y, H:i" }}
                </div>
                <div class="announcement-body" style="font-size:14px; color:#333; line-height:1.6; word-wrap:break-word;">{{ announcement.body|safe }}</div>
              </div>
              {% empty %}
              <div class="px-4 py-4 text-center text-muted" style="font-size:14px;">
                <i class="bi bi-megaphone" style="font-size:28px; display:block; margin-bottom:8px;"></i>
                No announcements yet.
              </div>
              {% endfor %}
            </div>
          </div>
        </div>
      </div>
    </div><!-- /row Announcements -->
  ```
  with:
  ```html
              {% for announcement in announcements %}
              <div class="px-4 py-3 border-bottom">
                <div class="d-flex align-items-start justify-content-between gap-2 flex-wrap">
                  <a href="{% url 'announcement_detail' organization.id announcement.id %}" style="text-decoration:none;">
                    <h6 style="color:#012970; font-weight:700; margin:0;">{{ announcement.title }}</h6>
                  </a>
                  {% if is_admin %}
                  <div class="d-flex gap-1 flex-shrink-0">
                    <a href="{% url 'edit_announcement' organization.id announcement.id %}" class="action-btn" title="Edit">
                      <i class="bi bi-pencil"></i>
                    </a>
                    <form method="post" action="{% url 'delete_announcement' organization.id announcement.id %}" class="action-form" onsubmit="return confirm('Delete this announcement?');">
                      {% csrf_token %}
                      <button type="submit" class="action-btn remove" title="Delete">
                        <i class="bi bi-trash"></i>
                      </button>
                    </form>
                  </div>
                  {% endif %}
                </div>
                <div class="member-meta mb-2">
                  {{ announcement.created_by.get_full_name|default:announcement.created_by.username|default:"Unknown" }}
                  · {{ announcement.created_on|date:"d M Y, H:i" }}
                </div>
                <a href="{% url 'announcement_detail' organization.id announcement.id %}" style="text-decoration:none; color:#333; display:block;">
                  <div style="font-size:14px; line-height:1.6; word-wrap:break-word;">{{ announcement.plain_text|truncatechars:250 }}</div>
                </a>
              </div>
              {% empty %}
              <div class="px-4 py-4 text-center text-muted" style="font-size:14px;">
                <i class="bi bi-megaphone" style="font-size:28px; display:block; margin-bottom:8px;"></i>
                No announcements yet.
              </div>
              {% endfor %}
            </div>
            {% if announcements.paginator.num_pages > 1 %}
            <nav class="px-4 py-3 border-top" aria-label="Announcement pages">
              <ul class="pagination pagination-sm justify-content-center flex-wrap mb-0">
                {% if announcements.has_previous %}
                  <li class="page-item">
                    <a class="page-link" href="?page={{ announcements.previous_page_number }}">&laquo;</a>
                  </li>
                {% else %}
                  <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                {% endif %}

                {% for p in announcement_page_range %}
                  {% if p is None %}
                    <li class="page-item disabled"><span class="page-link">&hellip;</span></li>
                  {% elif p == announcements.number %}
                    <li class="page-item active"><span class="page-link">{{ p }}</span></li>
                  {% else %}
                    <li class="page-item"><a class="page-link" href="?page={{ p }}">{{ p }}</a></li>
                  {% endif %}
                {% endfor %}

                {% if announcements.has_next %}
                  <li class="page-item">
                    <a class="page-link" href="?page={{ announcements.next_page_number }}">&raquo;</a>
                  </li>
                {% else %}
                  <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                {% endif %}
              </ul>
            </nav>
            {% endif %}
          </div>
        </div>
      </div>
    </div><!-- /row Announcements -->
  ```

  Note: the title and the truncated preview are each wrapped in their own `<a>` tag rather than wrapping the entire row (which also contains the admin Edit/Delete `<form>`/`<button>` controls) in one big `<a>`. Nesting interactive elements (`<form>`, `<button>`, another `<a>`) inside a single wrapping `<a>` is invalid HTML5 and causes unpredictable browser click/reparse behavior — two sibling links inside the row avoids that entirely while still satisfying "click the announcement to go inside" for both the title and the preview text. The `.announcement-body img { max-width: 100%; height: auto; }` rule (already in this file's `<style>` block from the earlier responsive fix) no longer applies to anything in this card, since the preview no longer renders raw HTML — that's expected and fine, the rule still applies wherever else `.announcement-body` is used (the new `announcement_detail.html` template from Task 1 reuses the same class name for its own full-body render).

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementPaginationTests organization.tests.AnnouncementPreviewLinkTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (6 tests).

  Then the full organization suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization -v 2 --settings=faithDev.settings_test
  ```

  Also re-run the pre-existing `AnnouncementCardTests` specifically, since this task changes the exact template block those tests assert against:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementCardTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (3 tests, unchanged — those tests check for title text and the presence/absence of the edit/delete/create URLs, all of which are still present in the new markup).

- [ ] **Step 6: Manually verify responsive layout (if a real dev environment is available)**

  Same caveat as Task 1. If available, open an organization with more than 5 announcements at 375px and 768px viewport widths. Confirm the pagination controls wrap/center cleanly and don't cause horizontal scroll, and that clicking a page number or Prev/Next reloads the page showing the correct 5 (or fewer, on the last page) announcements.

  If unavailable, Step 5's automated tests are the load-bearing verification; the responsive visual check specifically cannot be automated and should be flagged as unverified rather than assumed correct.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/organization/views.py Trust-AI-Platform/organization/templates/organization/organization_detail.html Trust-AI-Platform/organization/tests.py
  git commit -m "Paginate announcements list and link previews to the detail page"
  ```
