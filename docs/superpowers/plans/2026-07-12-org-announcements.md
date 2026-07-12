# Organization Announcements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let organization admins post, edit, and delete rich-text announcements on their org's page, using the same TinyMCE pattern Activities already use.

**Architecture:** Three sequential tasks. Task 1 lays the data foundation (model, form, migration, admin). Task 2 builds the create/edit/delete views, URLs, and the TinyMCE-editor templates. Task 3 wires the announcements list into the org detail page with admin controls.

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Bootstrap 5 · TinyMCE 6 (CDN-loaded) · SQLite (dev/test)

## Global Constraints

- Announcement create/edit/delete is admin-only: `organization.admins.filter(id=request.user.id).exists() or request.user.is_staff or request.user.is_superuser` — the exact check already used by `edit_organization`/`delete_organization` (`organization/views.py:264-266, 205-206`).
- The TinyMCE editor config and image-upload wiring must match `authoringtool/templates/authoringtool/createActivity.html:780-812` exactly (same CDN script, same `tinymce.init` options, same `images_upload_handler` posting to the existing `/authoringtool/tinymce/upload/` endpoint) — do not build a second upload endpoint.
- `plain_text` generation reuses the same two-line pattern as `Activity.plain_text` (`strip_tags` + `unescape`, from `authoringtool/views.py:229-231`) — duplicated locally in `organization/views.py`, not imported cross-app, matching this codebase's established per-app-duplication convention (same as `group_required`).
- **Responsive & mobile-first:** every new/changed template must work on phones (≥320px) and tablets (≥768px) — no new fixed pixel widths on outer containers. `organization_detail.html` already has an established `@media (max-width: 575.98px)` breakpoint pattern — follow it, don't invent a new one.
- No admin-side bulk/broadcast tooling, no pagination on the announcements list, no soft-delete/archive — plain hard delete, matching how member removal already works in this app.

---

### Task 1: `Announcement` model, form, migration, admin

**Files:**
- Modify: `Trust-AI-Platform/organization/models.py`
- Modify: `Trust-AI-Platform/organization/forms.py`
- Modify: `Trust-AI-Platform/organization/admin.py`
- Modify: `Trust-AI-Platform/organization/tests.py`
- Create (via `makemigrations`): `Trust-AI-Platform/organization/migrations/000X_announcement.py`

**Interfaces:**
- Produces: `Announcement` model (`organization`, `title`, `body`, `plain_text`, `created_by`, `created_on`, `updated_on`), ordered newest-first. `AnnouncementForm` (ModelForm, fields `['title', 'body']`). Consumed by Task 2 (views) and Task 3 (template context).

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  from .models import Announcement


  class AnnouncementModelTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('announce_owner', password='pass')
          self.org = Organization.objects.create(
              name='Announce Org', short_name='AO', created_by=self.user,
          )

      def test_create_announcement(self):
          a = Announcement.objects.create(
              organization=self.org, title='Welcome', body='<p>Hello <b>team</b></p>',
              plain_text='Hello team', created_by=self.user,
          )
          self.assertEqual(a.organization, self.org)
          self.assertIn(a, self.org.announcements.all())

      def test_announcements_ordered_newest_first(self):
          older = Announcement.objects.create(
              organization=self.org, title='Older', body='<p>a</p>', created_by=self.user,
          )
          newer = Announcement.objects.create(
              organization=self.org, title='Newer', body='<p>b</p>', created_by=self.user,
          )
          titles = list(self.org.announcements.values_list('title', flat=True))
          self.assertEqual(titles, ['Newer', 'Older'])

      def test_announcement_survives_creator_deletion(self):
          a = Announcement.objects.create(
              organization=self.org, title='Orphan-safe', body='<p>x</p>', created_by=self.user,
          )
          self.user.delete()
          a.refresh_from_db()
          self.assertIsNone(a.created_by)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementModelTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `ImportError: cannot import name 'Announcement'`.

- [ ] **Step 3: Add the model**

  In `Trust-AI-Platform/organization/models.py`, add at the end of the file (after the `Organization` class):
  ```python
  class Announcement(models.Model):
      organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='announcements')
      title = models.CharField(max_length=255)
      body = models.TextField()
      plain_text = models.TextField(blank=True)
      created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='org_announcements')
      created_on = models.DateTimeField(auto_now_add=True)
      updated_on = models.DateTimeField(auto_now=True)

      class Meta:
          ordering = ['-created_on']

      def __str__(self):
          return f"{self.title} ({self.organization.short_name})"
  ```

- [ ] **Step 4: Generate and apply the migration**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py makemigrations organization --settings=faithDev.settings_test
  "../djangofaithvenv/Scripts/python.exe" manage.py migrate organization --settings=faithDev.settings_test
  ```
  Expected: `Migrations for 'organization': ... Create model Announcement`, applied cleanly.

- [ ] **Step 5: Add the form**

  In `Trust-AI-Platform/organization/forms.py`, replace:
  ```python
  from django import forms
  from .models import Organization

  class OrganizationForm(forms.ModelForm):
      class Meta:
          model = Organization
          fields = ['name', 'short_name', 'description', 'country', 'language', 'picture']
  ```
  with:
  ```python
  from django import forms
  from .models import Organization, Announcement

  class OrganizationForm(forms.ModelForm):
      class Meta:
          model = Organization
          fields = ['name', 'short_name', 'description', 'country', 'language', 'picture']


  class AnnouncementForm(forms.ModelForm):
      class Meta:
          model = Announcement
          fields = ['title', 'body']
  ```

- [ ] **Step 6: Register in admin**

  In `Trust-AI-Platform/organization/admin.py`, replace:
  ```python
  from django.contrib import admin
  from django.utils.html import format_html
  from .models import Organization
  ```
  with:
  ```python
  from django.contrib import admin
  from django.utils.html import format_html
  from .models import Organization, Announcement
  ```

  Then append at the end of the file:
  ```python

  @admin.register(Announcement)
  class AnnouncementAdmin(admin.ModelAdmin):
      list_display = ('id', 'title', 'organization', 'created_by', 'created_on')
      list_filter = ('created_on',)
      search_fields = ('title', 'organization__name', 'organization__short_name')
      raw_id_fields = ('organization', 'created_by')
      readonly_fields = ('created_on', 'updated_on')
      date_hierarchy = 'created_on'
  ```

- [ ] **Step 7: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementModelTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (3 tests).

- [ ] **Step 8: Commit**

  ```bash
  git status --short -- Trust-AI-Platform/organization/migrations/
  ```
  Confirm only the new `Announcement` migration is untracked (no other stray untracked migrations should be swept in).
  ```bash
  git add Trust-AI-Platform/organization/models.py Trust-AI-Platform/organization/forms.py Trust-AI-Platform/organization/admin.py Trust-AI-Platform/organization/tests.py Trust-AI-Platform/organization/migrations/
  git commit -m "Add Announcement model, form, and admin registration"
  ```

---

### Task 2: Create/edit/delete views, URLs, TinyMCE templates

**Files:**
- Modify: `Trust-AI-Platform/organization/views.py`
- Modify: `Trust-AI-Platform/organization/urls.py`
- Modify: `Trust-AI-Platform/organization/tests.py`
- Create: `Trust-AI-Platform/organization/templates/organization/create_announcement.html`
- Create: `Trust-AI-Platform/organization/templates/organization/edit_announcement.html`

**Interfaces:**
- Consumes: `Announcement`, `AnnouncementForm` (Task 1)
- Produces: URL names `create_announcement` (`organization/<int:org_id>/announcements/create/`), `edit_announcement` (`organization/<int:org_id>/announcements/<int:announcement_id>/edit/`), `delete_announcement` (`organization/<int:org_id>/announcements/<int:announcement_id>/delete/`) — consumed by Task 3's template links.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  class AnnouncementViewsTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.admin = User.objects.create_user('announce_admin', password='pass')
          self.member = User.objects.create_user('announce_member', password='pass')
          self.org = Organization.objects.create(
              name='Views Org', short_name='VO', created_by=self.admin,
          )
          self.org.admins.add(self.admin)
          self.org.members.add(self.admin, self.member)

      def test_admin_can_create_announcement(self):
          self.client.login(username='announce_admin', password='pass')
          r = self.client.post(
              reverse('create_announcement', args=[self.org.id]),
              {'title': 'New Policy', 'body': '<p>Please <b>read</b> this.</p>'},
          )
          self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))
          a = Announcement.objects.get(organization=self.org, title='New Policy')
          self.assertEqual(a.created_by, self.admin)
          self.assertEqual(a.plain_text, 'Please read this.')

      def test_member_cannot_create_announcement(self):
          self.client.login(username='announce_member', password='pass')
          r = self.client.post(
              reverse('create_announcement', args=[self.org.id]),
              {'title': 'Nope', 'body': '<p>x</p>'},
          )
          self.assertFalse(Announcement.objects.filter(title='Nope').exists())
          self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))

      def test_admin_can_edit_announcement(self):
          a = Announcement.objects.create(
              organization=self.org, title='Old Title', body='<p>old</p>', created_by=self.admin,
          )
          self.client.login(username='announce_admin', password='pass')
          r = self.client.post(
              reverse('edit_announcement', args=[self.org.id, a.id]),
              {'title': 'Updated Title', 'body': '<p>updated</p>'},
          )
          self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))
          a.refresh_from_db()
          self.assertEqual(a.title, 'Updated Title')
          self.assertEqual(a.plain_text, 'updated')

      def test_member_cannot_edit_announcement(self):
          a = Announcement.objects.create(
              organization=self.org, title='Untouched', body='<p>x</p>', created_by=self.admin,
          )
          self.client.login(username='announce_member', password='pass')
          self.client.post(
              reverse('edit_announcement', args=[self.org.id, a.id]),
              {'title': 'Hacked', 'body': '<p>x</p>'},
          )
          a.refresh_from_db()
          self.assertEqual(a.title, 'Untouched')

      def test_admin_can_delete_announcement(self):
          a = Announcement.objects.create(
              organization=self.org, title='Delete Me', body='<p>x</p>', created_by=self.admin,
          )
          self.client.login(username='announce_admin', password='pass')
          r = self.client.post(reverse('delete_announcement', args=[self.org.id, a.id]))
          self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))
          self.assertFalse(Announcement.objects.filter(id=a.id).exists())

      def test_member_cannot_delete_announcement(self):
          a = Announcement.objects.create(
              organization=self.org, title='Stays', body='<p>x</p>', created_by=self.admin,
          )
          self.client.login(username='announce_member', password='pass')
          self.client.post(reverse('delete_announcement', args=[self.org.id, a.id]))
          self.assertTrue(Announcement.objects.filter(id=a.id).exists())

      def test_delete_requires_post(self):
          a = Announcement.objects.create(
              organization=self.org, title='Get-safe', body='<p>x</p>', created_by=self.admin,
          )
          self.client.login(username='announce_admin', password='pass')
          r = self.client.get(reverse('delete_announcement', args=[self.org.id, a.id]))
          self.assertEqual(r.status_code, 405)
          self.assertTrue(Announcement.objects.filter(id=a.id).exists())
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementViewsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` (the URL names don't exist yet).

- [ ] **Step 3: Add the `strip_html_tags` helper and the three views**

  In `Trust-AI-Platform/organization/views.py`, replace the import block:
  ```python
  from django.shortcuts import render, redirect, get_object_or_404
  from django.contrib.auth.decorators import login_required
  from django.core.exceptions import PermissionDenied
  from django.http import HttpResponseForbidden
  from django.views.decorators.http import require_POST
  from django.contrib import messages
  from functools import wraps
  from .models import Organization, JoinRequest
  from django.contrib.auth.models import User
  from .forms import OrganizationForm
  from authoringtool.models import Language
  ```
  with:
  ```python
  from django.shortcuts import render, redirect, get_object_or_404
  from django.contrib.auth.decorators import login_required
  from django.core.exceptions import PermissionDenied
  from django.http import HttpResponseForbidden
  from django.views.decorators.http import require_POST
  from django.contrib import messages
  from functools import wraps
  from django.utils.html import strip_tags
  from html import unescape
  from .models import Organization, JoinRequest, Announcement
  from django.contrib.auth.models import User
  from .forms import OrganizationForm, AnnouncementForm
  from authoringtool.models import Language


  def strip_html_tags(html_content):
      text = strip_tags(html_content)
      return unescape(text)
  ```

  Then append at the end of the file:
  ```python

  def _is_org_admin(user, organization):
      is_site_admin = user.is_staff or user.is_superuser
      return organization.admins.filter(id=user.id).exists() or is_site_admin


  @login_required
  def create_announcement(request, org_id):
      organization = get_object_or_404(Organization, id=org_id)
      if not _is_org_admin(request.user, organization):
          return redirect('organization_detail', org_id=org_id)

      if request.method == 'POST':
          form = AnnouncementForm(request.POST)
          if form.is_valid():
              announcement = form.save(commit=False)
              announcement.organization = organization
              announcement.created_by = request.user
              announcement.plain_text = strip_html_tags(announcement.body)
              announcement.save()
              messages.success(request, "Announcement posted.")
              return redirect('organization_detail', org_id=org_id)
      else:
          form = AnnouncementForm()

      return render(request, 'organization/create_announcement.html', {
          'form': form,
          'organization': organization,
      })


  @login_required
  def edit_announcement(request, org_id, announcement_id):
      organization = get_object_or_404(Organization, id=org_id)
      announcement = get_object_or_404(Announcement, id=announcement_id, organization=organization)
      if not _is_org_admin(request.user, organization):
          return redirect('organization_detail', org_id=org_id)

      if request.method == 'POST':
          form = AnnouncementForm(request.POST, instance=announcement)
          if form.is_valid():
              updated = form.save(commit=False)
              updated.plain_text = strip_html_tags(updated.body)
              updated.save()
              messages.success(request, "Announcement updated.")
              return redirect('organization_detail', org_id=org_id)
      else:
          form = AnnouncementForm(instance=announcement)

      return render(request, 'organization/edit_announcement.html', {
          'form': form,
          'organization': organization,
          'announcement': announcement,
      })


  @require_POST
  @login_required
  def delete_announcement(request, org_id, announcement_id):
      organization = get_object_or_404(Organization, id=org_id)
      announcement = get_object_or_404(Announcement, id=announcement_id, organization=organization)
      if _is_org_admin(request.user, organization):
          announcement.delete()
          messages.success(request, "Announcement deleted.")
      return redirect('organization_detail', org_id=org_id)
  ```

  Note: `_is_org_admin` is a small new helper, not a duplicate of anything — the two-line `is_site_admin`/`.admins.filter(...).exists()` check was previously inlined separately in `edit_organization` and `delete_organization`; factoring it out here avoids a third near-identical inline copy for these three new views. Do not refactor the pre-existing inline copies in `edit_organization`/`delete_organization` to use it — that's out of scope for this task.

- [ ] **Step 4: Add the URLs**

  In `Trust-AI-Platform/organization/urls.py`, replace:
  ```python
      path('join_request/<int:request_id>/approve/', views.approve_join_request, name='approve_join_request'),
      path('join_request/<int:request_id>/reject/', views.reject_join_request, name='reject_join_request'),
  ]
  ```
  with:
  ```python
      path('join_request/<int:request_id>/approve/', views.approve_join_request, name='approve_join_request'),
      path('join_request/<int:request_id>/reject/', views.reject_join_request, name='reject_join_request'),
      path('organization/<int:org_id>/announcements/create/', views.create_announcement, name='create_announcement'),
      path('organization/<int:org_id>/announcements/<int:announcement_id>/edit/', views.edit_announcement, name='edit_announcement'),
      path('organization/<int:org_id>/announcements/<int:announcement_id>/delete/', views.delete_announcement, name='delete_announcement'),
  ]
  ```

- [ ] **Step 5: Add a `form-control` widget to the title field**

  Django's default `CharField` widget doesn't get Bootstrap's `.form-control` class automatically — the templates in the next two steps render `{{ form.title }}` directly and expect it already styled. In `Trust-AI-Platform/organization/forms.py`, replace:
  ```python
  class AnnouncementForm(forms.ModelForm):
      class Meta:
          model = Announcement
          fields = ['title', 'body']
  ```
  with:
  ```python
  class AnnouncementForm(forms.ModelForm):
      class Meta:
          model = Announcement
          fields = ['title', 'body']
          widgets = {
              'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Announcement title'}),
          }
  ```

- [ ] **Step 6: Create the "New Announcement" template**

  Create `Trust-AI-Platform/organization/templates/organization/create_announcement.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — New Announcement</title>{% endblock %}
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
    .hero-btn-solid {
      background: #fff; color: #1a56db; border: none;
      font-weight: 600; font-size: 13.5px; border-radius: 8px;
      padding: 8px 20px; display: inline-flex; align-items: center; gap: 6px;
      text-decoration: none; transition: background 0.15s, box-shadow 0.15s;
      cursor: pointer; white-space: nowrap;
    }
    .hero-btn-solid:hover { background: #eef3ff; color: #1a56db; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
    .form-card { max-width: 760px; margin: 0 auto; }
    .field-label { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 5px; }
  </style>

  <main id="main" class="main">
    <div class="org-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="org-hero-icon"><i class="bi bi-megaphone-fill"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Organizations</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">New Announcement</h2>
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ organization.name }}</div>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'list_organizations' %}">Organizations</a></li>
            <li class="breadcrumb-item"><a href="{% url 'organization_detail' organization.id %}">{{ organization.short_name }}</a></li>
            <li class="breadcrumb-item active">New Announcement</li>
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
          <form method="post">
            {% csrf_token %}
            <div class="mb-3">
              <label class="field-label" for="id_title">Title</label>
              {{ form.title }}
            </div>
            <div class="mb-3">
              <label class="field-label" for="id_body">Announcement</label>
              <textarea class="tinymce-editor" name="body" id="id_body">{{ form.body.value|default:'' }}</textarea>
            </div>
            <div class="text-end">
              <button type="submit" class="hero-btn-solid" style="color:#1a56db;">
                <i class="bi bi-send"></i> Post Announcement
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/tinymce@6/tinymce.min.js" referrerpolicy="origin"></script>
  <script>
  tinymce.init({
    selector: '.tinymce-editor',
    height: 320,
    menubar: false,
    plugins: 'lists link image table code',
    toolbar: 'undo redo | bold italic underline | bullist numlist | link image table | code',
    content_style: 'body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size: 14px; }',
    automatic_uploads: true,
    images_upload_handler: function (blobInfo) {
      return new Promise(function (resolve, reject) {
        var formData = new FormData();
        formData.append('file', blobInfo.blob(), blobInfo.filename());
        var csrfMatch = document.cookie.match(/csrftoken=([^;]+)/);
        fetch('/authoringtool/tinymce/upload/', {
          method: 'POST',
          headers: { 'X-CSRFToken': csrfMatch ? csrfMatch[1] : '' },
          body: formData,
          credentials: 'same-origin'
        })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { reject(d.error || 'Upload failed'); });
          return r.json();
        })
        .then(function (data) { if (data && data.location) resolve(data.location); })
        .catch(function () { reject('Image upload failed. Check file size (max 8 MB).'); });
      });
    },
    setup: function (editor) {
      editor.on('change', function () { editor.save(); });
    }
  });
  </script>
  {% endblock %}
  ```

- [ ] **Step 7: Create the "Edit Announcement" template**

  Create `Trust-AI-Platform/organization/templates/organization/edit_announcement.html`. This is Step 6's `create_announcement.html` with four differences: the page title, the hero heading, the final breadcrumb crumb, and the submit button — `{{ form.title }}` and the body textarea need no changes since Django's bound-form rendering already fills in the existing values when `form = AnnouncementForm(instance=announcement)`. Full file:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — Edit Announcement</title>{% endblock %}
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
    .hero-btn-solid {
      background: #fff; color: #1a56db; border: none;
      font-weight: 600; font-size: 13.5px; border-radius: 8px;
      padding: 8px 20px; display: inline-flex; align-items: center; gap: 6px;
      text-decoration: none; transition: background 0.15s, box-shadow 0.15s;
      cursor: pointer; white-space: nowrap;
    }
    .hero-btn-solid:hover { background: #eef3ff; color: #1a56db; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
    .form-card { max-width: 760px; margin: 0 auto; }
    .field-label { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 5px; }
  </style>

  <main id="main" class="main">
    <div class="org-hero">
      <div class="d-flex align-items-start gap-3">
        <div class="org-hero-icon"><i class="bi bi-megaphone-fill"></i></div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Organizations</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">Edit Announcement</h2>
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ organization.name }}</div>
          <nav><ol class="breadcrumb">
            <li class="breadcrumb-item"><a href="{% url 'list_organizations' %}">Organizations</a></li>
            <li class="breadcrumb-item"><a href="{% url 'organization_detail' organization.id %}">{{ organization.short_name }}</a></li>
            <li class="breadcrumb-item active">Edit Announcement</li>
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
          <form method="post">
            {% csrf_token %}
            <div class="mb-3">
              <label class="field-label" for="id_title">Title</label>
              {{ form.title }}
            </div>
            <div class="mb-3">
              <label class="field-label" for="id_body">Announcement</label>
              <textarea class="tinymce-editor" name="body" id="id_body">{{ form.body.value|default:'' }}</textarea>
            </div>
            <div class="text-end">
              <button type="submit" class="hero-btn-solid" style="color:#1a56db;">
                <i class="bi bi-check-lg"></i> Save Changes
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/tinymce@6/tinymce.min.js" referrerpolicy="origin"></script>
  <script>
  tinymce.init({
    selector: '.tinymce-editor',
    height: 320,
    menubar: false,
    plugins: 'lists link image table code',
    toolbar: 'undo redo | bold italic underline | bullist numlist | link image table | code',
    content_style: 'body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size: 14px; }',
    automatic_uploads: true,
    images_upload_handler: function (blobInfo) {
      return new Promise(function (resolve, reject) {
        var formData = new FormData();
        formData.append('file', blobInfo.blob(), blobInfo.filename());
        var csrfMatch = document.cookie.match(/csrftoken=([^;]+)/);
        fetch('/authoringtool/tinymce/upload/', {
          method: 'POST',
          headers: { 'X-CSRFToken': csrfMatch ? csrfMatch[1] : '' },
          body: formData,
          credentials: 'same-origin'
        })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { reject(d.error || 'Upload failed'); });
          return r.json();
        })
        .then(function (data) { if (data && data.location) resolve(data.location); })
        .catch(function () { reject('Image upload failed. Check file size (max 8 MB).'); });
      });
    },
    setup: function (editor) {
      editor.on('change', function () { editor.save(); });
    }
  });
  </script>
  {% endblock %}
  ```

- [ ] **Step 8: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementViewsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (7 tests).

  Then the full organization suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 9: Manually verify (if a real dev environment is available)**

  Same caveat as prior features on this branch — needs a live Postgres-backed dev environment. If available:
  ```bash
  python manage.py runserver
  ```
  As an org admin, create an announcement with a title and formatted body (bold text, a bullet list, an inline image upload), confirm it saves and redirects. Edit it, confirm the TinyMCE editor pre-loads the existing content. Delete it. As a non-admin member, confirm you cannot reach `/organization/<id>/announcements/create/` (redirects away) and see no create/edit/delete affordances.

  If unavailable, Step 8's automated tests are the load-bearing verification.

- [ ] **Step 10: Commit**

  ```bash
  git add Trust-AI-Platform/organization/views.py Trust-AI-Platform/organization/urls.py Trust-AI-Platform/organization/forms.py Trust-AI-Platform/organization/tests.py Trust-AI-Platform/organization/templates/organization/create_announcement.html Trust-AI-Platform/organization/templates/organization/edit_announcement.html
  git commit -m "Add announcement create/edit/delete views and TinyMCE templates"
  ```

---

### Task 3: Announcements card on the organization detail page

**Files:**
- Modify: `Trust-AI-Platform/organization/views.py`
- Modify: `Trust-AI-Platform/organization/templates/organization/organization_detail.html`
- Modify: `Trust-AI-Platform/organization/tests.py`

**Interfaces:**
- Consumes: `Announcement` (Task 1), `create_announcement`/`edit_announcement`/`delete_announcement` URLs (Task 2)
- Produces: no new interfaces — `organization_detail` view context gains `announcements`.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  class AnnouncementCardTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.admin = User.objects.create_user('card_admin', password='pass')
          self.member = User.objects.create_user('card_member', password='pass')
          self.org = Organization.objects.create(
              name='Card Org', short_name='CO', created_by=self.admin,
          )
          self.org.admins.add(self.admin)
          self.org.members.add(self.admin, self.member)
          self.announcement = Announcement.objects.create(
              organization=self.org, title='Kickoff Meeting',
              body='<p>Join us <b>Monday</b> at 10am.</p>',
              plain_text='Join us Monday at 10am.', created_by=self.admin,
          )

      def test_announcement_visible_to_admin_with_controls(self):
          self.client.login(username='card_admin', password='pass')
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, 'Kickoff Meeting')
          self.assertContains(r, reverse('edit_announcement', args=[self.org.id, self.announcement.id]))
          self.assertContains(r, reverse('delete_announcement', args=[self.org.id, self.announcement.id]))
          self.assertContains(r, reverse('create_announcement', args=[self.org.id]))

      def test_announcement_visible_to_member_without_controls(self):
          self.client.login(username='card_member', password='pass')
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, 'Kickoff Meeting')
          self.assertNotContains(r, reverse('edit_announcement', args=[self.org.id, self.announcement.id]))
          self.assertNotContains(r, reverse('delete_announcement', args=[self.org.id, self.announcement.id]))
          self.assertNotContains(r, reverse('create_announcement', args=[self.org.id]))

      def test_no_announcements_shows_empty_state(self):
          empty_org = Organization.objects.create(
              name='Empty Org', short_name='EO', created_by=self.admin,
          )
          empty_org.admins.add(self.admin)
          empty_org.members.add(self.admin)
          self.client.login(username='card_admin', password='pass')
          r = self.client.get(reverse('organization_detail', args=[empty_org.id]))
          self.assertContains(r, 'No announcements yet')
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementCardTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — the org detail page doesn't render any announcement content yet.

- [ ] **Step 3: Add `announcements` to the view context**

  In `Trust-AI-Platform/organization/views.py`, replace:
  ```python
      return render(request, 'organization/organization_detail.html', {
          'organization': organization,
          'is_member': is_member,
          'is_admin': is_admin,
          'join_request': join_request,
          'pending_requests': pending_requests,
      })
  ```
  with:
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

- [ ] **Step 4: Add the Announcements card**

  In `Trust-AI-Platform/organization/templates/organization/organization_detail.html`, replace:
  ```html
      </div><!-- /row -->

      {% if is_admin and pending_requests %}
  ```
  with:
  ```html
      </div><!-- /row -->

      <div class="row g-3 mt-2">
        <div class="col-12">
          <div class="card" style="border-radius:12px;">
            <div class="card-body p-0">
              <div class="d-flex align-items-center justify-content-between gap-2 px-4 pt-4 pb-3 border-bottom flex-wrap">
                <h5 style="color:#012970; font-weight:700; margin:0;">
                  <i class="bi bi-megaphone-fill me-2"></i>Announcements
                </h5>
                {% if is_admin %}
                <a href="{% url 'create_announcement' organization.id %}" class="hero-btn-solid" style="background:#1a56db;color:#fff;">
                  <i class="bi bi-plus-lg"></i> New Announcement
                </a>
                {% endif %}
              </div>
              <div>
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
                  <div style="font-size:14px; color:#333; line-height:1.6; word-wrap:break-word;">{{ announcement.body|safe }}</div>
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

      {% if is_admin and pending_requests %}
  ```

  Note: `.hero-btn-solid` is defined in this file's own `<style>` block (`organization_detail.html:36-42`) with `color: #1a56db` baked in for the white-background hero context — the inline `style="background:#1a56db;color:#fff;"` override here is intentional so the button reads correctly on the white card background instead of the hero's dark gradient.

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.AnnouncementCardTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (3 tests).

  Then the full organization suite one more time:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 6: Manually verify responsive layout (if a real dev environment is available)**

  Same caveat as Task 2. If available, open the org detail page at 375px and 768px viewport widths — confirm the Announcements card header (title + "New Announcement" button) wraps instead of overflowing on phone width (the `flex-wrap` classes already added should handle this), announcement bodies with long unbroken text or images don't cause horizontal scroll (`word-wrap:break-word` handles text; TinyMCE-inserted images should be checked to confirm they don't render wider than the card — if they do, a follow-up `max-width:100%` rule on rendered announcement images may be needed, note this as a finding rather than guessing a fix blind).

  If unavailable, Step 5's automated tests are the load-bearing verification for structural correctness; the responsive visual check specifically cannot be automated and should be flagged as unverified rather than assumed correct.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/organization/views.py Trust-AI-Platform/organization/templates/organization/organization_detail.html Trust-AI-Platform/organization/tests.py
  git commit -m "Add announcements card to organization detail page"
  ```
