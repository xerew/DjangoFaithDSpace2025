# Organization Team Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let all members of an organization chat together in a shared team chat room reachable from the org page, and let any teacher reach the org chats they belong to from the Messages page.

**Architecture:** Three sequential tasks. Task 1 lays the data foundation (`OrgChatMessage` model, admin, migration). Task 2 builds the chat room view, the send/poll JSON endpoints, URLs, and the chat-room template with its own polling loop. Task 3 wires up both entry points: a "Team Chat" button on the organization detail page, and an "Organization Chats" section on the Messages page.

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Bootstrap 5 · SQLite (dev/test) · `setInterval`+`fetch` polling (no Channels/WebSockets in this codebase)

## Global Constraints

- Chat room access, sending, and polling are all member-gated using the exact existing idiom: `organization.members.filter(id=request.user.id).exists()` — the same check `organization_detail` (`organization/views.py:58`) already uses. No admin-only restriction — any member can read and post.
- **`OrgChatMessage.body` is rendered without `|safe`.** Unlike `Announcement.body` (a trusted TinyMCE rich-text field intentionally rendered with `|safe`), chat messages come from a plain `<input type="text">` compose box — Django's default auto-escaping must apply. Do not copy the `|safe` pattern from announcements into chat.
- `OrgChatMessage.Meta.ordering` is `['created_at', 'pk']`, not the spec's literal `['created_at']` alone. This preempts the exact `auto_now_add`-tie flakiness already found and fixed twice on this branch (`Message.Meta.ordering` in Teacher Messaging, `Announcement.Meta.ordering` in Organization Announcements) — both independently verified safe against SQLite `AUTOINCREMENT` (no rowid reuse) by a task reviewer. Apply the same fix proactively rather than waiting for a flaky test run to rediscover it.
- No read-receipts, no per-member unread tracking, no unread badge for team chat — explicitly out of scope (spec Part B).
- No pagination on chat history — full history renders on every page load, matching the existing 1:1 `thread` view's approach.
- The chat room's live-update polling interval is `20000` ms, matching the existing sidebar unread-badge poller's cadence (`Trust-AI-Platform/templates/main.html:82`). This is a **new, independent** polling loop scoped to the chat page's own `<script>` block — it does not touch or reuse `main.html`'s poller.
- **Responsive & mobile-first:** every new/changed template must work on phones (≥320px) and tablets (≥768px) — no new fixed pixel widths on outer containers. The chat room template reuses the `max-width: 576px` breakpoint pattern already established in `messaging/templates/messaging/thread.html` (its closest sibling in the UI). The Messages-page changes need no new breakpoints — they reuse the existing unconstrained-width `.thread-row`/`.thread-avatar` classes as-is.
- New models live in `organization`, not `messaging` — `messaging`'s own design is explicitly 1:1-only. `OrgChatMessage` belongs with `Organization` (cascade-deleted with it).
- Cross-app import (`from organization.models import Organization` inside `messaging/views.py`'s `message_threads`) follows the exact precedent already used in `accounts/views.py`'s `profile_view`/`view_profile` — a local import inside the function body, not a module-level import.

---

### Task 1: `OrgChatMessage` model, admin, migration

**Files:**
- Modify: `Trust-AI-Platform/organization/models.py`
- Modify: `Trust-AI-Platform/organization/admin.py`
- Modify: `Trust-AI-Platform/organization/tests.py`
- Create (via `makemigrations`): `Trust-AI-Platform/organization/migrations/0004_orgchatmessage.py`

**Interfaces:**
- Produces: `OrgChatMessage` model (`organization`, `sender`, `body`, `created_at`), ordered oldest-first, accessible via `organization.chat_messages.all()`. Consumed by Task 2 (views) and Task 3 (no new context needed there, view already exists).

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  from .models import OrgChatMessage


  class OrgChatMessageModelTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('chat_owner', password='pass')
          self.org = Organization.objects.create(
              name='Chat Org', short_name='CHO', created_by=self.user,
          )

      def test_create_message(self):
          m = OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='Hello team')
          self.assertEqual(m.organization, self.org)
          self.assertIn(m, self.org.chat_messages.all())

      def test_messages_ordered_oldest_first(self):
          OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='First')
          OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='Second')
          bodies = list(self.org.chat_messages.values_list('body', flat=True))
          self.assertEqual(bodies, ['First', 'Second'])

      def test_message_deleted_with_sender(self):
          m = OrgChatMessage.objects.create(organization=self.org, sender=self.user, body='Bye')
          self.user.delete()
          self.assertFalse(OrgChatMessage.objects.filter(id=m.id).exists())
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrgChatMessageModelTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `ImportError: cannot import name 'OrgChatMessage'`.

- [ ] **Step 3: Add the model**

  In `Trust-AI-Platform/organization/models.py`, add at the end of the file (after the `Announcement` class):
  ```python
  class OrgChatMessage(models.Model):
      organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='chat_messages')
      sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='org_chat_messages')
      body = models.TextField()
      created_at = models.DateTimeField(auto_now_add=True)

      class Meta:
          ordering = ['created_at', 'pk']

      def __str__(self):
          return f"{self.sender} @ {self.organization.short_name}: {self.body[:30]}"
  ```

- [ ] **Step 4: Generate and apply the migration**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py makemigrations organization --settings=faithDev.settings_test
  "../djangofaithvenv/Scripts/python.exe" manage.py migrate organization --settings=faithDev.settings_test
  ```
  Expected: `Migrations for 'organization': ... Create model OrgChatMessage`, applied cleanly.

- [ ] **Step 5: Register in admin**

  In `Trust-AI-Platform/organization/admin.py`, replace:
  ```python
  from django.contrib import admin
  from django.utils.html import format_html
  from .models import Organization, Announcement
  ```
  with:
  ```python
  from django.contrib import admin
  from django.utils.html import format_html
  from .models import Organization, Announcement, OrgChatMessage
  ```

  Then append at the end of the file:
  ```python

  @admin.register(OrgChatMessage)
  class OrgChatMessageAdmin(admin.ModelAdmin):
      list_display = ('id', 'organization', 'sender', 'created_at')
      list_filter = ('created_at',)
      search_fields = ('organization__name', 'organization__short_name', 'sender__username', 'body')
      raw_id_fields = ('organization', 'sender')
      readonly_fields = ('created_at',)
      date_hierarchy = 'created_at'
  ```

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrgChatMessageModelTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (3 tests).

- [ ] **Step 7: Commit**

  ```bash
  git status --short -- Trust-AI-Platform/organization/migrations/
  ```
  Confirm only the new `OrgChatMessage` migration is untracked (no other stray untracked migrations should be swept in).
  ```bash
  git add Trust-AI-Platform/organization/models.py Trust-AI-Platform/organization/admin.py Trust-AI-Platform/organization/tests.py Trust-AI-Platform/organization/migrations/
  git commit -m "Add OrgChatMessage model and admin registration"
  ```

---

### Task 2: Chat room view, send/poll endpoints, URLs, template

**Files:**
- Modify: `Trust-AI-Platform/organization/views.py`
- Modify: `Trust-AI-Platform/organization/urls.py`
- Modify: `Trust-AI-Platform/organization/tests.py`
- Create: `Trust-AI-Platform/organization/templates/organization/org_chat.html`

**Interfaces:**
- Consumes: `OrgChatMessage` (Task 1)
- Produces: URL names `org_chat` (`organization/<int:org_id>/chat/`), `send_org_chat_message` (`organization/<int:org_id>/chat/send/`), `org_chat_poll` (`organization/<int:org_id>/chat/poll/`) — consumed by Task 3's "Team Chat" button and Messages-page links.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  class OrgChatViewsTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.member1 = User.objects.create_user('chat_member1', password='pass')
          self.member2 = User.objects.create_user('chat_member2', password='pass')
          self.outsider = User.objects.create_user('chat_outsider', password='pass')
          self.org = Organization.objects.create(
              name='Chat Views Org', short_name='CVO', created_by=self.member1,
          )
          self.org.admins.add(self.member1)
          self.org.members.add(self.member1, self.member2)

      def test_member_can_view_chat_room(self):
          self.client.login(username='chat_member1', password='pass')
          r = self.client.get(reverse('org_chat', args=[self.org.id]))
          self.assertEqual(r.status_code, 200)

      def test_non_member_redirected_from_chat_room(self):
          self.client.login(username='chat_outsider', password='pass')
          r = self.client.get(reverse('org_chat', args=[self.org.id]))
          self.assertRedirects(r, reverse('organization_detail', args=[self.org.id]))

      def test_member_can_send_message(self):
          self.client.login(username='chat_member1', password='pass')
          r = self.client.post(reverse('send_org_chat_message', args=[self.org.id]), {'body': 'Hi team'})
          data = r.json()
          self.assertTrue(data['success'])
          self.assertEqual(OrgChatMessage.objects.filter(organization=self.org, body='Hi team').count(), 1)

      def test_non_member_cannot_send_message(self):
          self.client.login(username='chat_outsider', password='pass')
          r = self.client.post(reverse('send_org_chat_message', args=[self.org.id]), {'body': 'Sneaky'})
          self.assertEqual(r.status_code, 403)
          self.assertFalse(OrgChatMessage.objects.filter(body='Sneaky').exists())

      def test_send_rejects_empty_body(self):
          self.client.login(username='chat_member1', password='pass')
          r = self.client.post(reverse('send_org_chat_message', args=[self.org.id]), {'body': '   '})
          data = r.json()
          self.assertFalse(data['success'])
          self.assertEqual(OrgChatMessage.objects.count(), 0)

      def test_send_requires_post(self):
          self.client.login(username='chat_member1', password='pass')
          r = self.client.get(reverse('send_org_chat_message', args=[self.org.id]))
          self.assertEqual(r.status_code, 405)

      def test_poll_returns_only_messages_after_since_id(self):
          m1 = OrgChatMessage.objects.create(organization=self.org, sender=self.member1, body='First')
          OrgChatMessage.objects.create(organization=self.org, sender=self.member2, body='Second')
          self.client.login(username='chat_member1', password='pass')
          r = self.client.get(reverse('org_chat_poll', args=[self.org.id]), {'since_id': m1.id})
          data = r.json()
          self.assertEqual(len(data['messages']), 1)
          self.assertEqual(data['messages'][0]['body'], 'Second')

      def test_poll_blocks_non_member(self):
          self.client.login(username='chat_outsider', password='pass')
          r = self.client.get(reverse('org_chat_poll', args=[self.org.id]))
          self.assertEqual(r.status_code, 403)
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrgChatViewsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `NoReverseMatch` (the URL names don't exist yet).

- [ ] **Step 3: Add the three views**

  In `Trust-AI-Platform/organization/views.py`, replace the import block:
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
  ```
  with:
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

  Then append at the end of the file:
  ```python

  @login_required
  def org_chat(request, org_id):
      organization = get_object_or_404(Organization, id=org_id)
      if not organization.members.filter(id=request.user.id).exists():
          return redirect('organization_detail', org_id=org_id)

      chat_messages = organization.chat_messages.select_related('sender')
      return render(request, 'organization/org_chat.html', {
          'organization': organization,
          'chat_messages': chat_messages,
      })


  @require_POST
  @login_required
  def send_org_chat_message(request, org_id):
      organization = get_object_or_404(Organization, id=org_id)
      if not organization.members.filter(id=request.user.id).exists():
          return JsonResponse({'success': False, 'error': 'Not a member.'}, status=403)

      body = (request.POST.get('body') or '').strip()
      if not body:
          return JsonResponse({'success': False, 'error': 'Message cannot be empty.'}, status=400)

      msg = OrgChatMessage.objects.create(organization=organization, sender=request.user, body=body)
      return JsonResponse({
          'success': True,
          'message': {
              'id': msg.id,
              'body': msg.body,
              'created_at': msg.created_at.strftime('%d %b %Y, %H:%M'),
              'sender_id': msg.sender_id,
              'sender_name': msg.sender.get_full_name() or msg.sender.username,
          },
      })


  @require_GET
  @login_required
  def org_chat_poll(request, org_id):
      organization = get_object_or_404(Organization, id=org_id)
      if not organization.members.filter(id=request.user.id).exists():
          return JsonResponse({'success': False, 'error': 'Not a member.'}, status=403)

      try:
          since_id = int(request.GET.get('since_id', '0'))
      except ValueError:
          since_id = 0

      new_messages = organization.chat_messages.select_related('sender').filter(id__gt=since_id)
      return JsonResponse({
          'success': True,
          'messages': [
              {
                  'id': m.id,
                  'body': m.body,
                  'created_at': m.created_at.strftime('%d %b %Y, %H:%M'),
                  'sender_id': m.sender_id,
                  'sender_name': m.sender.get_full_name() or m.sender.username,
              }
              for m in new_messages
          ],
      })
  ```

  Note: `send_org_chat_message` returns `403` (not a redirect) because it's a JSON endpoint hit by `fetch()`, not a normal form submission — matching the JSON-error-response pattern `messaging`'s own `send_message` uses for its invalid-recipient case. `org_chat` itself (a normal page view) redirects rather than 403ing, matching `organization_detail`'s own pattern for non-member access elsewhere in this app.

- [ ] **Step 4: Add the URLs**

  In `Trust-AI-Platform/organization/urls.py`, replace:
  ```python
      path('organization/<int:org_id>/announcements/<int:announcement_id>/delete/', views.delete_announcement, name='delete_announcement'),
  ]
  ```
  with:
  ```python
      path('organization/<int:org_id>/announcements/<int:announcement_id>/delete/', views.delete_announcement, name='delete_announcement'),
      path('organization/<int:org_id>/chat/', views.org_chat, name='org_chat'),
      path('organization/<int:org_id>/chat/send/', views.send_org_chat_message, name='send_org_chat_message'),
      path('organization/<int:org_id>/chat/poll/', views.org_chat_poll, name='org_chat_poll'),
  ]
  ```

- [ ] **Step 5: Create the chat room template**

  Create `Trust-AI-Platform/organization/templates/organization/org_chat.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — {{ organization.name }} Team Chat</title>{% endblock %}
  {% block atcontent %}

  <style>
    .chat-hero { background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%); border-radius: 14px; padding: 20px 30px; color: #fff; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(26,86,219,0.18); display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .chat-hero-icon { background: rgba(255,255,255,0.18); border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
    .chat-hero-back { background: rgba(255,255,255,0.15); color: #fff; border: 1.5px solid rgba(255,255,255,0.4); font-weight: 600; font-size: 13px; border-radius: 8px; padding: 6px 14px; display: inline-flex; align-items: center; gap: 6px; text-decoration: none; white-space: nowrap; }
    .chat-hero-back:hover { background: rgba(255,255,255,0.25); color: #fff; }
    .chat-body { height: 55vh; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
    .chat-msg { max-width: 70%; }
    .chat-msg.mine { align-self: flex-end; }
    .chat-msg.theirs { align-self: flex-start; }
    .chat-msg-sender { font-size: 11.5px; font-weight: 600; color: #4154f1; margin-bottom: 3px; padding: 0 4px; }
    .chat-msg.mine .chat-msg-sender { text-align: right; }
    .chat-bubble { padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.4; word-wrap: break-word; }
    .chat-msg.mine .chat-bubble { background: #1a56db; color: #fff; border-bottom-right-radius: 4px; }
    .chat-msg.theirs .chat-bubble { background: #f0f4ff; color: #1e293b; border-bottom-left-radius: 4px; }
    .chat-bubble .chat-time { font-size: 10.5px; opacity: 0.7; margin-top: 4px; display: block; }
    @media (max-width: 576px) {
      .chat-hero { padding: 16px 20px; }
      .chat-body { padding: 14px; }
      .chat-msg { max-width: 88%; }
    }
  </style>

  <main id="main" class="main">
    <div class="chat-hero">
      <div class="chat-hero-icon"><i class="bi bi-people-fill"></i></div>
      <div class="flex-grow-1">
        <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Team Chat</div>
        <h2 style="margin:0;font-size:18px;font-weight:700;">{{ organization.name }}</h2>
      </div>
      <a href="{% url 'organization_detail' organization.id %}" class="chat-hero-back">
        <i class="bi bi-arrow-left"></i> Back
      </a>
    </div>

    <section class="section">
      <div class="card">
        <div class="chat-body" id="chatBody">
          {% for m in chat_messages %}
          <div class="chat-msg {% if m.sender_id == request.user.id %}mine{% else %}theirs{% endif %}" data-msg-id="{{ m.id }}">
            <div class="chat-msg-sender">{{ m.sender.get_full_name|default:m.sender.username }}</div>
            <div class="chat-bubble">
              {{ m.body }}
              <span class="chat-time">{{ m.created_at|date:"d M, H:i" }}</span>
            </div>
          </div>
          {% empty %}
          <div class="text-center text-muted py-5">
            <i class="bi bi-chat-dots" style="font-size:2rem;color:#d1d9e0;"></i>
            <p class="mt-2 mb-0">No messages yet. Say hello to the team!</p>
          </div>
          {% endfor %}
        </div>
        <div class="card-body border-top pt-3">
          <form id="composeForm" class="d-flex gap-2">
            {% csrf_token %}
            <input type="text" class="form-control flex-grow-1" id="composeBody" placeholder="Message the team…" autocomplete="off" required>
            <button type="submit" class="btn btn-primary flex-shrink-0"><i class="bi bi-send"></i></button>
          </form>
        </div>
      </div>
    </section>
  </main>

  <script>
  document.addEventListener('DOMContentLoaded', function () {
    const chatBody = document.getElementById('chatBody');
    const form = document.getElementById('composeForm');
    const input = document.getElementById('composeBody');
    const myUserId = {{ request.user.id }};
    let lastId = 0;
    chatBody.querySelectorAll('[data-msg-id]').forEach(function (el) {
      const id = parseInt(el.getAttribute('data-msg-id'), 10);
      if (id > lastId) lastId = id;
    });
    chatBody.scrollTop = chatBody.scrollHeight;

    function appendMessage(m) {
      const wrap = document.createElement('div');
      wrap.className = 'chat-msg ' + (m.sender_id === myUserId ? 'mine' : 'theirs');
      wrap.setAttribute('data-msg-id', m.id);

      const senderEl = document.createElement('div');
      senderEl.className = 'chat-msg-sender';
      senderEl.textContent = m.sender_name;
      wrap.appendChild(senderEl);

      const bubble = document.createElement('div');
      bubble.className = 'chat-bubble';
      bubble.appendChild(document.createTextNode(m.body));
      bubble.appendChild(document.createElement('br'));
      const timeSpan = document.createElement('span');
      timeSpan.className = 'chat-time';
      timeSpan.textContent = m.created_at;
      bubble.appendChild(timeSpan);
      wrap.appendChild(bubble);

      chatBody.appendChild(wrap);
      if (m.id > lastId) lastId = m.id;
    }

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const body = input.value.trim();
      if (!body) return;
      const csrfToken = form.querySelector('[name=csrfmiddlewaretoken]').value;

      fetch('{% url "send_org_chat_message" organization.id %}', {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken, 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'body=' + encodeURIComponent(body),
      })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (!res.success) { return; }
        appendMessage(res.message);
        chatBody.scrollTop = chatBody.scrollHeight;
        input.value = '';
      });
    });

    function poll() {
      fetch('{% url "org_chat_poll" organization.id %}?since_id=' + lastId)
        .then(function (r) { return r.json(); })
        .then(function (res) {
          if (!res.success || !res.messages.length) return;
          res.messages.forEach(appendMessage);
          chatBody.scrollTop = chatBody.scrollHeight;
        });
    }
    setInterval(poll, 20000);
  });
  </script>
  {% endblock %}
  ```

  Note: `lastId` is seeded from the highest `data-msg-id` already rendered server-side (or `0` if the room is empty), so the first poll only asks for messages newer than what's on screen — it never re-fetches history. Sending a message updates `lastId` from the send response too, so the next poll can't re-append your own just-sent message.

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrgChatViewsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (8 tests).

  Then the full organization suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 7: Manually verify (if a real dev environment is available)**

  Same caveat as prior features on this branch — needs a live Postgres-backed dev environment. If available:
  ```bash
  python manage.py runserver
  ```
  As two different members of the same org (two browser sessions), open `/organization/<id>/chat/` for both. Send a message from one session, confirm it appears in the other session within ~20 seconds without a manual refresh. Confirm a non-member gets redirected away from the chat URL. Confirm the compose box rejects an empty/whitespace-only message client-side (the `required` attribute) and server-side.

  If unavailable, Step 6's automated tests are the load-bearing verification.

- [ ] **Step 8: Commit**

  ```bash
  git add Trust-AI-Platform/organization/views.py Trust-AI-Platform/organization/urls.py Trust-AI-Platform/organization/tests.py Trust-AI-Platform/organization/templates/organization/org_chat.html
  git commit -m "Add organization team chat room, send/poll endpoints, and URLs"
  ```

---

### Task 3: Team Chat entry points — org page button and Messages page section

**Files:**
- Modify: `Trust-AI-Platform/organization/templates/organization/organization_detail.html`
- Modify: `Trust-AI-Platform/organization/tests.py`
- Modify: `Trust-AI-Platform/messaging/views.py`
- Modify: `Trust-AI-Platform/messaging/templates/messaging/thread_list.html`
- Modify: `Trust-AI-Platform/messaging/tests.py`

**Interfaces:**
- Consumes: `org_chat` URL (Task 2)
- Produces: no new interfaces — `organization_detail.html` gains a visible link; `message_threads` view context gains `organizations`.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/organization/tests.py`:
  ```python
  class OrganizationDetailTeamChatButtonTests(TestCase):
      def setUp(self):
          self.client = Client()
          self.member = User.objects.create_user('chatbtn_member', password='pass')
          self.outsider = User.objects.create_user('chatbtn_outsider', password='pass')
          self.org = Organization.objects.create(
              name='Chat Button Org', short_name='CBO', created_by=self.member,
          )
          self.org.members.add(self.member)

      def test_member_sees_team_chat_button(self):
          self.client.login(username='chatbtn_member', password='pass')
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, reverse('org_chat', args=[self.org.id]))

      def test_non_member_does_not_see_team_chat_button(self):
          self.client.login(username='chatbtn_outsider', password='pass')
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertNotContains(r, reverse('org_chat', args=[self.org.id]))
  ```

  Append to `Trust-AI-Platform/messaging/tests.py`:
  ```python
  from organization.models import Organization


  class MessageThreadsOrganizationChatsTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.alice = User.objects.create_user('alice_orgchat', password='pass')
          self.alice.groups.add(teachers)
          self.org = Organization.objects.create(name='Chat Link Org', short_name='CLO', created_by=self.alice)
          self.org.members.add(self.alice)
          Organization.objects.create(name='Not Mine Org', short_name='NMO', created_by=self.alice)
          self.client.login(username='alice_orgchat', password='pass')

      def test_message_threads_lists_my_organizations(self):
          r = self.client.get(reverse('message_threads'))
          orgs = list(r.context['organizations'])
          self.assertEqual(orgs, [self.org])

      def test_message_threads_links_to_org_chat(self):
          r = self.client.get(reverse('message_threads'))
          self.assertContains(r, reverse('org_chat', args=[self.org.id]))

      def test_message_threads_shows_empty_state_for_no_orgs(self):
          self.org.members.remove(self.alice)
          r = self.client.get(reverse('message_threads'))
          self.assertContains(r, 'not part of any organization')
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrganizationDetailTeamChatButtonTests -v 2 --settings=faithDev.settings_test
  "../djangofaithvenv/Scripts/python.exe" manage.py test messaging.tests.MessageThreadsOrganizationChatsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: both FAIL — the button doesn't exist yet, and `organizations` isn't in the `message_threads` context yet.

- [ ] **Step 3: Add the Team Chat button to the org detail page**

  In `Trust-AI-Platform/organization/templates/organization/organization_detail.html`, replace:
  ```html
        <a href="{% url 'list_organizations' %}" class="hero-btn-ghost">
          <i class="bi bi-arrow-left"></i> Back
        </a>
        {% if is_admin %}
  ```
  with:
  ```html
        <a href="{% url 'list_organizations' %}" class="hero-btn-ghost">
          <i class="bi bi-arrow-left"></i> Back
        </a>
        {% if is_member %}
        <a href="{% url 'org_chat' organization.id %}" class="hero-btn-ghost">
          <i class="bi bi-chat-dots"></i> Team Chat
        </a>
        {% endif %}
        {% if is_admin %}
  ```

  Note: gated on `is_member`, not `is_admin` — every member sees this button, including admins (admins are also members). This is a separate `{% if %}` block from the `{% if is_admin %}...{% elif not is_member %}...{% endif %}` chain right after it, so it doesn't interfere with that existing branching.

- [ ] **Step 4: Add the `organizations` query to `message_threads`**

  In `Trust-AI-Platform/messaging/views.py`, replace:
  ```python
  @group_required('teachers')
  def message_threads(request):
      me = request.user
      sent_to = Message.objects.filter(sender=me).values_list('recipient_id', flat=True)
      received_from = Message.objects.filter(recipient=me).values_list('sender_id', flat=True)
      partner_ids = set(sent_to) | set(received_from)

      threads = []
      for partner in User.objects.filter(id__in=partner_ids):
          latest = Message.objects.filter(
              Q(sender=me, recipient=partner) | Q(sender=partner, recipient=me)
          ).order_by('-created_at').first()
          unread = Message.objects.filter(sender=partner, recipient=me, read_at__isnull=True).count()
          threads.append({'partner': partner, 'latest': latest, 'unread': unread})
      threads.sort(key=lambda t: (t['latest'].created_at, t['latest'].id), reverse=True)

      return render(request, 'messaging/thread_list.html', {'threads': threads})
  ```
  with:
  ```python
  @group_required('teachers')
  def message_threads(request):
      from organization.models import Organization

      me = request.user
      sent_to = Message.objects.filter(sender=me).values_list('recipient_id', flat=True)
      received_from = Message.objects.filter(recipient=me).values_list('sender_id', flat=True)
      partner_ids = set(sent_to) | set(received_from)

      threads = []
      for partner in User.objects.filter(id__in=partner_ids):
          latest = Message.objects.filter(
              Q(sender=me, recipient=partner) | Q(sender=partner, recipient=me)
          ).order_by('-created_at').first()
          unread = Message.objects.filter(sender=partner, recipient=me, read_at__isnull=True).count()
          threads.append({'partner': partner, 'latest': latest, 'unread': unread})
      threads.sort(key=lambda t: (t['latest'].created_at, t['latest'].id), reverse=True)

      organizations = Organization.objects.filter(members=me).order_by('name')

      return render(request, 'messaging/thread_list.html', {'threads': threads, 'organizations': organizations})
  ```

- [ ] **Step 5: Add the Organization Chats section to the Messages page**

  In `Trust-AI-Platform/messaging/templates/messaging/thread_list.html`, replace:
  ```html
    <div class="card">
      <div class="card-body p-0">
        {% for t in threads %}
  ```
  with:
  ```html
    <div class="card mb-3">
      <div class="card-body p-0">
        <div class="px-4 pt-3 pb-2" style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Organization Chats</div>
        {% for org in organizations %}
        <a href="{% url 'org_chat' org.id %}" class="thread-row">
          <div class="thread-avatar">{{ org.short_name|default:org.name|slice:":2"|upper }}</div>
          <div class="flex-grow-1 min-width-0">
            <div class="thread-name">{{ org.name }}</div>
            <div class="thread-snippet">Team chat</div>
          </div>
        </a>
        {% empty %}
        <div class="text-center text-muted py-4">
          <i class="bi bi-people" style="font-size:2rem;color:#d1d9e0;"></i>
          <p class="mt-2 mb-0 small">You're not part of any organization yet.</p>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="card">
      <div class="card-body p-0">
        <div class="px-4 pt-3 pb-2" style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Messages</div>
        {% for t in threads %}
  ```

  Note: this leaves the existing `{% for t in threads %}...{% endfor %}` block and its `{% empty %}` case completely unchanged below the insertion point — only the opening `<div class="card">` tag and the text immediately after it are touched. The `.thread-row`/`.thread-avatar`/`.thread-name`/`.thread-snippet` classes are the same ones the existing 1:1 list already uses; no new CSS is needed for either section.

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrganizationDetailTeamChatButtonTests -v 2 --settings=faithDev.settings_test
  "../djangofaithvenv/Scripts/python.exe" manage.py test messaging.tests.MessageThreadsOrganizationChatsTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` for both (2 and 3 tests respectively).

  Then the full organization and messaging suites:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization messaging -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 7: Manually verify responsive layout (if a real dev environment is available)**

  Same caveat as Task 2. If available, open the Messages page and the org detail page at 375px and 768px viewport widths. Confirm the org hero's button row (Back / Team Chat / Add Member / Edit / Delete) wraps instead of overflowing on phone width (the row already has `flex-wrap` from the pre-existing Announcements work). Confirm the Organization Chats and Messages cards on the Messages page stack cleanly with no horizontal scroll.

  If unavailable, Step 6's automated tests are the load-bearing verification; the responsive visual check specifically cannot be automated and should be flagged as unverified rather than assumed correct.

- [ ] **Step 8: Commit**

  ```bash
  git add Trust-AI-Platform/organization/templates/organization/organization_detail.html Trust-AI-Platform/organization/tests.py Trust-AI-Platform/messaging/views.py Trust-AI-Platform/messaging/templates/messaging/thread_list.html Trust-AI-Platform/messaging/tests.py
  git commit -m "Add Team Chat entry points: org page button and Messages page section"
  ```
