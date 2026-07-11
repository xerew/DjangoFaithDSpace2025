# Teacher Profiles & Messaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let teachers view each other's profiles and exchange direct messages, with a polling-driven notification toast and sidebar unread badge, following the approved design spec.

**Architecture:** Three sequential tasks. Task 1 builds a self-contained new `messaging` Django app (model, views, templates, tests) reachable directly by URL. Task 2 adds cross-user profile viewing to the existing `accounts` app, with a "Message" button linking into Task 1's thread view. Task 3 wires discovery and notifications into the rest of the UI (Organization member list, sidebar nav, base-template polling/toast) — pure template/JS integration, no new backend logic.

**Tech Stack:** Django 5.1 (this checkout reports Django 5.2.16 at runtime — same APIs used) · Bootstrap 5 · Bootstrap Icons · vanilla JS (`setInterval`+`fetch`, matching the existing dashboard polling pattern) · SQLite (dev/test)

## Global Constraints

- Every messaging/profile-viewing view is gated to the `teachers` group (or staff/superuser) using the exact `group_required` decorator pattern already duplicated per-app in this codebase (`organization/views.py:13-23`, `usergroups/views.py`, `authoringtool/views.py:49`) — duplicate it again in `messaging/views.py` rather than importing across apps, matching existing convention.
- No new "browse people" directory page. The only discovery surface is the existing Organization member list (`organization_detail.html`).
- 1:1 threads only, no group messaging, no separate `Conversation` table — a thread is computed from `Message` rows filtered by sender/recipient pair.
- Notification detection is polling (`setInterval`+`fetch`), not WebSockets — no new infrastructure (no Django Channels, no Redis pub/sub, no ASGI changes).
- Any teacher/staff user may message any other teacher/staff user platform-wide — not scoped to shared Organization/UserGroup.
- A target user who is not in the `teachers` group and not staff/superuser must 404 on profile view and be rejected (400/redirect) as a message recipient — this is what keeps students out of the entire feature even via a guessed URL.

---

### Task 1: `messaging` app — model, views, templates, tests

**Files:**
- Modify: `Trust-AI-Platform/faithDev/settings.py`
- Modify: `Trust-AI-Platform/faithDev/urls.py`
- Create: `Trust-AI-Platform/messaging/__init__.py`
- Create: `Trust-AI-Platform/messaging/apps.py`
- Create: `Trust-AI-Platform/messaging/models.py`
- Create: `Trust-AI-Platform/messaging/admin.py`
- Create: `Trust-AI-Platform/messaging/urls.py`
- Create: `Trust-AI-Platform/messaging/views.py`
- Create: `Trust-AI-Platform/messaging/tests.py`
- Create: `Trust-AI-Platform/messaging/templates/messaging/thread_list.html`
- Create: `Trust-AI-Platform/messaging/templates/messaging/thread.html`
- Create: `Trust-AI-Platform/messaging/migrations/__init__.py`
- Create (via `makemigrations`): `Trust-AI-Platform/messaging/migrations/0001_initial.py`

**Interfaces:**
- Produces: URL names `message_threads` (`/messaging/`), `thread` (`/messaging/<int:user_id>/`), `send_message` (`/messaging/send/`), `unread_status` (`/messaging/unread_status/`) — all consumed by Task 2 (profile "Message" button) and Task 3 (sidebar link, org member list button, polling script).
- Produces: `Message` model (`sender`, `recipient`, `body`, `created_at`, `read_at`).

- [ ] **Step 1: Register the app**

  In `Trust-AI-Platform/faithDev/settings.py`, replace:
  ```python
      'django_celery_results',
      'django_redis',
      'home',
  ]
  ```
  with:
  ```python
      'django_celery_results',
      'django_redis',
      'home',
      'messaging',
  ]
  ```

  In `Trust-AI-Platform/faithDev/urls.py`, replace:
  ```python
      path('usergroups/', include('usergroups.urls')),
  ```
  with:
  ```python
      path('usergroups/', include('usergroups.urls')),
      path('messaging/', include('messaging.urls')),
  ```

- [ ] **Step 2: Create the app skeleton**

  Create `Trust-AI-Platform/messaging/__init__.py` (empty file).

  Create `Trust-AI-Platform/messaging/migrations/__init__.py` (empty file).

  Create `Trust-AI-Platform/messaging/apps.py`:
  ```python
  from django.apps import AppConfig


  class MessagingConfig(AppConfig):
      default_auto_field = 'django.db.models.BigAutoField'
      name = 'messaging'
  ```

- [ ] **Step 3: Write the model**

  Create `Trust-AI-Platform/messaging/models.py`:
  ```python
  from django.contrib.auth.models import User
  from django.db import models


  class Message(models.Model):
      sender = models.ForeignKey(User, related_name='sent_messages', on_delete=models.CASCADE)
      recipient = models.ForeignKey(User, related_name='received_messages', on_delete=models.CASCADE)
      body = models.TextField()
      created_at = models.DateTimeField(auto_now_add=True)
      read_at = models.DateTimeField(null=True, blank=True)

      class Meta:
          ordering = ['created_at']

      def __str__(self):
          return f"{self.sender} -> {self.recipient}: {self.body[:30]}"
  ```

- [ ] **Step 4: Generate and apply the migration**

  From `Trust-AI-Platform/`:
  ```bash
  python manage.py makemigrations messaging
  python manage.py migrate messaging
  ```
  Expected: `Migrations for 'messaging': ... Create model Message`, then `Applying messaging.0001_initial... OK`.

- [ ] **Step 5: Register in admin**

  Create `Trust-AI-Platform/messaging/admin.py`:
  ```python
  from django.contrib import admin

  from .models import Message


  @admin.register(Message)
  class MessageAdmin(admin.ModelAdmin):
      list_display = ('id', 'sender', 'recipient', 'created_at', 'read_at')
      list_filter = ('created_at',)
      search_fields = ('sender__username', 'recipient__username', 'body')
      raw_id_fields = ('sender', 'recipient')
      readonly_fields = ('created_at',)
      date_hierarchy = 'created_at'
  ```

- [ ] **Step 6: Write the views**

  Create `Trust-AI-Platform/messaging/views.py`:
  ```python
  from functools import wraps

  from django.contrib.auth.decorators import login_required
  from django.contrib.auth.models import User
  from django.core.exceptions import PermissionDenied
  from django.db.models import Q
  from django.http import JsonResponse
  from django.shortcuts import get_object_or_404, redirect, render
  from django.utils import timezone
  from django.views.decorators.http import require_GET, require_POST

  from .models import Message


  def group_required(group_name):
      def decorator(view_func):
          @wraps(view_func)
          @login_required
          def _wrapped_view(request, *args, **kwargs):
              if request.user.groups.filter(name=group_name).exists():
                  return view_func(request, *args, **kwargs)
              else:
                  raise PermissionDenied
          return _wrapped_view
      return decorator


  def _is_valid_target(user):
      return user.is_staff or user.is_superuser or user.groups.filter(name='teachers').exists()


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
      threads.sort(key=lambda t: t['latest'].created_at, reverse=True)

      return render(request, 'messaging/thread_list.html', {'threads': threads})


  @group_required('teachers')
  def thread(request, user_id):
      partner = get_object_or_404(User, pk=user_id)
      if partner == request.user or not _is_valid_target(partner):
          return redirect('message_threads')

      Message.objects.filter(
          sender=partner, recipient=request.user, read_at__isnull=True
      ).update(read_at=timezone.now())

      thread_messages = Message.objects.filter(
          Q(sender=request.user, recipient=partner) | Q(sender=partner, recipient=request.user)
      )
      return render(request, 'messaging/thread.html', {'partner': partner, 'thread_messages': thread_messages})


  @require_POST
  @group_required('teachers')
  def send_message(request):
      recipient_id = request.POST.get('recipient_id')
      body = (request.POST.get('body') or '').strip()

      if not body:
          return JsonResponse({'success': False, 'error': 'Message cannot be empty.'}, status=400)

      recipient = get_object_or_404(User, pk=recipient_id)
      if recipient == request.user or not _is_valid_target(recipient):
          return JsonResponse({'success': False, 'error': 'Invalid recipient.'}, status=400)

      msg = Message.objects.create(sender=request.user, recipient=recipient, body=body)
      return JsonResponse({
          'success': True,
          'message': {
              'id': msg.id,
              'body': msg.body,
              'created_at': msg.created_at.strftime('%d %b %Y, %H:%M'),
              'sender_id': msg.sender_id,
          },
      })


  @require_GET
  @group_required('teachers')
  def unread_status(request):
      unread_qs = Message.objects.filter(recipient=request.user, read_at__isnull=True)
      latest = unread_qs.order_by('-created_at').first()

      data = {'unread_count': unread_qs.count(), 'latest': None}
      if latest:
          data['latest'] = {
              'id': latest.id,
              'sender_id': latest.sender_id,
              'sender_name': latest.sender.get_full_name() or latest.sender.username,
              'snippet': latest.body[:80],
              'created_at': latest.created_at.isoformat(),
          }
      return JsonResponse(data)
  ```

- [ ] **Step 7: Write the URLs**

  Create `Trust-AI-Platform/messaging/urls.py`:
  ```python
  from django.urls import path

  from . import views

  urlpatterns = [
      path('', views.message_threads, name='message_threads'),
      path('send/', views.send_message, name='send_message'),
      path('unread_status/', views.unread_status, name='unread_status'),
      path('<int:user_id>/', views.thread, name='thread'),
  ]
  ```
  Note: `<int:user_id>/` is listed last so it doesn't shadow the more specific `send/` and `unread_status/` paths.

- [ ] **Step 8: Write the thread-list template**

  Create `Trust-AI-Platform/messaging/templates/messaging/thread_list.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — Messages</title>{% endblock %}
  {% block atcontent %}

  <style>
    .thread-hero { background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%); border-radius: 14px; padding: 26px 30px 20px; color: #fff; margin-bottom: 26px; box-shadow: 0 4px 20px rgba(26,86,219,0.18); }
    .thread-row { display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-bottom: 1px solid #f0f4ff; text-decoration: none; color: inherit; transition: background 0.15s; }
    .thread-row:hover { background: #f8faff; }
    .thread-row:last-child { border-bottom: none; }
    .thread-avatar { width: 42px; height: 42px; border-radius: 50%; background: linear-gradient(135deg, #4154f1, #1a56db); display: flex; align-items: center; justify-content: center; font-size: 15px; font-weight: 700; color: #fff; flex-shrink: 0; }
    .thread-name { font-weight: 600; font-size: 14px; color: #012970; }
    .thread-snippet { font-size: 13px; color: #888; }
    .thread-unread-dot { width: 10px; height: 10px; border-radius: 50%; background: #dc3545; margin-left: auto; }
  </style>

  <main id="main" class="main">
    <div class="thread-hero">
      <div class="d-flex align-items-start gap-3">
        <div style="background:rgba(255,255,255,0.18);border-radius:10px;width:50px;height:50px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">
          <i class="bi bi-chat-dots-fill"></i>
        </div>
        <div class="flex-grow-1">
          <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-bottom:2px;">Account</div>
          <h2 style="margin:0;font-size:20px;font-weight:700;line-height:1.2;">Messages</h2>
        </div>
      </div>
    </div>

    <section class="section">
      <div class="card">
        <div class="card-body p-0">
          {% for t in threads %}
          <a href="{% url 'thread' t.partner.id %}" class="thread-row">
            <div class="thread-avatar">{{ t.partner.first_name|default:t.partner.username|slice:":1"|upper }}{{ t.partner.last_name|slice:":1"|upper }}</div>
            <div class="flex-grow-1 min-width-0">
              <div class="thread-name">{{ t.partner.get_full_name|default:t.partner.username }}</div>
              <div class="thread-snippet">{{ t.latest.body|truncatechars:60 }}</div>
            </div>
            <div class="text-end flex-shrink-0 d-flex flex-column align-items-end gap-1">
              <div class="text-muted small">{{ t.latest.created_at|date:"d M, H:i" }}</div>
              {% if t.unread %}<div class="thread-unread-dot"></div>{% endif %}
            </div>
          </a>
          {% empty %}
          <div class="text-center text-muted py-5">
            <i class="bi bi-chat-dots" style="font-size:2.5rem;color:#d1d9e0;"></i>
            <p class="mt-2 mb-0">No conversations yet.</p>
            <p class="small">Message a colleague from their profile or an organization's member list.</p>
          </div>
          {% endfor %}
        </div>
      </div>
    </section>
  </main>
  {% endblock %}
  ```

- [ ] **Step 9: Write the thread-detail template**

  Create `Trust-AI-Platform/messaging/templates/messaging/thread.html`:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — {{ partner.get_full_name|default:partner.username }}</title>{% endblock %}
  {% block atcontent %}

  <style>
    .chat-hero { background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%); border-radius: 14px; padding: 20px 30px; color: #fff; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(26,86,219,0.18); display: flex; align-items: center; gap: 14px; }
    .chat-body { height: 55vh; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
    .chat-bubble { max-width: 65%; padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.4; }
    .chat-bubble.mine { align-self: flex-end; background: #1a56db; color: #fff; border-bottom-right-radius: 4px; }
    .chat-bubble.theirs { align-self: flex-start; background: #f0f4ff; color: #1e293b; border-bottom-left-radius: 4px; }
    .chat-bubble .chat-time { font-size: 10.5px; opacity: 0.7; margin-top: 4px; display: block; }
  </style>

  <main id="main" class="main">
    <div class="chat-hero">
      <div style="background:rgba(255,255,255,0.18);border-radius:50%;width:44px;height:44px;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;flex-shrink:0;">
        {{ partner.first_name|default:partner.username|slice:":1"|upper }}{{ partner.last_name|slice:":1"|upper }}
      </div>
      <div>
        <h2 style="margin:0;font-size:18px;font-weight:700;">{{ partner.get_full_name|default:partner.username }}</h2>
      </div>
    </div>

    <section class="section">
      <div class="card">
        <div class="chat-body" id="chatBody">
          {% for m in thread_messages %}
          <div class="chat-bubble {% if m.sender_id == request.user.id %}mine{% else %}theirs{% endif %}">
            {{ m.body }}
            <span class="chat-time">{{ m.created_at|date:"d M, H:i" }}</span>
          </div>
          {% empty %}
          <div class="text-center text-muted py-5">
            <i class="bi bi-chat-dots" style="font-size:2rem;color:#d1d9e0;"></i>
            <p class="mt-2 mb-0">No messages yet. Say hello!</p>
          </div>
          {% endfor %}
        </div>
        <div class="card-body border-top pt-3">
          <form id="composeForm" class="d-flex gap-2">
            {% csrf_token %}
            <input type="text" class="form-control" id="composeBody" placeholder="Type a message…" autocomplete="off" required>
            <button type="submit" class="btn btn-primary"><i class="bi bi-send"></i></button>
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
    chatBody.scrollTop = chatBody.scrollHeight;

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const body = input.value.trim();
      if (!body) return;
      const csrfToken = form.querySelector('[name=csrfmiddlewaretoken]').value;

      fetch('{% url "send_message" %}', {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken, 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'recipient_id={{ partner.id }}&body=' + encodeURIComponent(body),
      })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (!res.success) { return; }
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble mine';
        bubble.appendChild(document.createTextNode(res.message.body));
        bubble.appendChild(document.createElement('br'));
        const timeSpan = document.createElement('span');
        timeSpan.className = 'chat-time';
        timeSpan.textContent = res.message.created_at;
        bubble.appendChild(timeSpan);
        chatBody.appendChild(bubble);
        chatBody.scrollTop = chatBody.scrollHeight;
        input.value = '';
      });
    });
  });
  </script>
  {% endblock %}
  ```

  Note: this template deliberately does NOT link to a profile page yet — the `view_profile` URL name doesn't exist until Task 2, and Task 1's own tests render this template, so referencing an undefined URL name here would break Task 1's tests before Task 2 ever runs. Task 2 Step 5 adds the "View profile" link back into this exact template once `view_profile` exists.

- [ ] **Step 10: Write the tests**

  Create `Trust-AI-Platform/messaging/tests.py`:
  ```python
  from django.contrib.auth.models import Group, User
  from django.test import Client, TestCase
  from django.urls import reverse

  from .models import Message


  class MessagingViewsTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.alice = User.objects.create_user('alice_msg', password='pass', first_name='Alice', last_name='A')
          self.bob = User.objects.create_user('bob_msg', password='pass', first_name='Bob', last_name='B')
          self.alice.groups.add(teachers)
          self.bob.groups.add(teachers)
          self.student = User.objects.create_user('stu_msg', password='pass')
          self.client.login(username='alice_msg', password='pass')

      def test_send_message_creates_message(self):
          r = self.client.post(reverse('send_message'), {'recipient_id': self.bob.id, 'body': 'Hi Bob'})
          data = r.json()
          self.assertTrue(data['success'])
          self.assertEqual(Message.objects.filter(sender=self.alice, recipient=self.bob).count(), 1)

      def test_send_message_rejects_empty_body(self):
          r = self.client.post(reverse('send_message'), {'recipient_id': self.bob.id, 'body': '   '})
          data = r.json()
          self.assertFalse(data['success'])
          self.assertEqual(Message.objects.count(), 0)

      def test_send_message_rejects_non_teacher_recipient(self):
          r = self.client.post(reverse('send_message'), {'recipient_id': self.student.id, 'body': 'Hi'})
          data = r.json()
          self.assertFalse(data['success'])
          self.assertEqual(Message.objects.count(), 0)

      def test_send_message_requires_post(self):
          r = self.client.get(reverse('send_message'))
          self.assertEqual(r.status_code, 405)

      def test_thread_marks_messages_as_read(self):
          Message.objects.create(sender=self.bob, recipient=self.alice, body='Hello Alice')
          r = self.client.get(reverse('thread', args=[self.bob.id]))
          self.assertEqual(r.status_code, 200)
          msg = Message.objects.get(sender=self.bob, recipient=self.alice)
          self.assertIsNotNone(msg.read_at)

      def test_thread_shows_messages_both_directions(self):
          Message.objects.create(sender=self.bob, recipient=self.alice, body='From Bob')
          Message.objects.create(sender=self.alice, recipient=self.bob, body='From Alice')
          r = self.client.get(reverse('thread', args=[self.bob.id]))
          self.assertContains(r, 'From Bob')
          self.assertContains(r, 'From Alice')

      def test_thread_blocks_non_teacher_target(self):
          r = self.client.get(reverse('thread', args=[self.student.id]))
          self.assertRedirects(r, reverse('message_threads'))

      def test_message_threads_lists_partner_with_unread_count(self):
          Message.objects.create(sender=self.bob, recipient=self.alice, body='Unread 1')
          Message.objects.create(sender=self.bob, recipient=self.alice, body='Unread 2')
          r = self.client.get(reverse('message_threads'))
          threads = r.context['threads']
          self.assertEqual(len(threads), 1)
          self.assertEqual(threads[0]['partner'], self.bob)
          self.assertEqual(threads[0]['unread'], 2)

      def test_message_threads_ordered_by_latest_message(self):
          teachers = Group.objects.get(name='teachers')
          carol = User.objects.create_user('carol_msg', password='pass')
          carol.groups.add(teachers)
          Message.objects.create(sender=self.bob, recipient=self.alice, body='Older')
          Message.objects.create(sender=carol, recipient=self.alice, body='Newer')
          r = self.client.get(reverse('message_threads'))
          threads = r.context['threads']
          self.assertEqual(threads[0]['partner'], carol)
          self.assertEqual(threads[1]['partner'], self.bob)

      def test_unread_status_returns_latest_and_count(self):
          Message.objects.create(sender=self.bob, recipient=self.alice, body='Ping')
          r = self.client.get(reverse('unread_status'))
          data = r.json()
          self.assertEqual(data['unread_count'], 1)
          self.assertEqual(data['latest']['sender_id'], self.bob.id)
          self.assertEqual(data['latest']['snippet'], 'Ping')

      def test_unread_status_empty_when_no_messages(self):
          r = self.client.get(reverse('unread_status'))
          data = r.json()
          self.assertEqual(data['unread_count'], 0)
          self.assertIsNone(data['latest'])

      def test_non_teacher_cannot_access_message_threads(self):
          self.client.logout()
          self.client.login(username='stu_msg', password='pass')
          r = self.client.get(reverse('message_threads'))
          self.assertEqual(r.status_code, 403)
  ```

- [ ] **Step 11: Run the tests**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test messaging -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (12 tests).

- [ ] **Step 12: Manually verify (if a real dev environment is available)**

  The project's default `DATABASES` setting (`faithDev/settings.py`) requires a live PostgreSQL instance reachable via `POSTGRES_*` env vars, and `faithDev.settings_test` uses a throwaway in-memory SQLite DB with no seed data — neither is suitable for browser click-through in an environment without a running Postgres instance and existing teacher accounts (the same constraint hit in this project's prior work). If such an environment is available:
  ```bash
  python manage.py runserver
  ```
  As a teacher user, navigate directly to `/messaging/` (empty state), then to `/messaging/<other-teacher-id>/`, send a message, confirm it appears as a right-aligned bubble without a page reload. Confirm `/messaging/<student-id>/` redirects to `/messaging/`.

  If no such environment is available, Step 11's automated test suite is the load-bearing verification for this task — note the skip and why, rather than fabricating results.

- [ ] **Step 13: Commit**

  ```bash
  git add Trust-AI-Platform/faithDev/settings.py Trust-AI-Platform/faithDev/urls.py Trust-AI-Platform/messaging/
  git commit -m "Add messaging app: 1:1 teacher direct messages"
  ```

---

### Task 2: Cross-user profile viewing (`accounts` app)

**Files:**
- Modify: `Trust-AI-Platform/accounts/urls.py`
- Modify: `Trust-AI-Platform/accounts/views.py`
- Modify: `Trust-AI-Platform/accounts/templates/accounts/profile.html`
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Consumes: `thread` URL name (`/messaging/<int:user_id>/`) from Task 1, for the "Message" button.
- Produces: URL name `view_profile` (`/accounts/profile/<int:user_id>/`) — consumed by Task 3 (Organization member list link) and by Task 1's `thread.html` ("View profile" link, already written to reference it in Task 1 Step 9).

- [ ] **Step 1: Add the URL**

  In `Trust-AI-Platform/accounts/urls.py`, replace:
  ```python
      path('profile/', views.profile_view, name='profile'),
  ```
  with:
  ```python
      path('profile/', views.profile_view, name='profile'),
      path('profile/<int:user_id>/', views.view_profile, name='view_profile'),
  ```

- [ ] **Step 2: Update the import line for `get_object_or_404`**

  In `Trust-AI-Platform/accounts/views.py`, replace:
  ```python
  from django.shortcuts import render
  ```
  with:
  ```python
  from django.shortcuts import render, get_object_or_404
  ```

- [ ] **Step 3: Make `profile_view` pass the shared context keys**

  In `Trust-AI-Platform/accounts/views.py`, replace:
  ```python
      context = {
          'admin_orgs':      list(admin_orgs),
          'member_orgs':     list(member_orgs),
          'roles':           roles,
          'profile':         profile,
          'country_choices': COUNTRY_CHOICES,
      }
      return render(request, 'accounts/profile.html', context)
  ```
  with:
  ```python
      context = {
          'admin_orgs':      list(admin_orgs),
          'member_orgs':     list(member_orgs),
          'roles':           roles,
          'profile':         profile,
          'country_choices': COUNTRY_CHOICES,
          'profile_user':    user,
          'is_own_profile':  True,
      }
      return render(request, 'accounts/profile.html', context)


  def _is_valid_target(user):
      return user.is_staff or user.is_superuser or user.groups.filter(name='teachers').exists()


  @group_required('teachers')
  def view_profile(request, user_id):
      from organization.models import Organization

      target = get_object_or_404(User, pk=user_id)
      if target == request.user:
          return redirect('profile')
      if not _is_valid_target(target):
          from django.http import Http404
          raise Http404("No such profile.")

      admin_orgs = Organization.objects.filter(admins=target).values('id', 'name', 'short_name', 'country')
      member_orgs = Organization.objects.filter(members=target).exclude(admins=target).values('id', 'name', 'short_name', 'country')

      BADGE_MAP = {
          'teachers':       ('Teacher',        'primary'),
          'dspace_partners':('DSpace Partner',  'info'),
      }
      roles = []
      if target.is_superuser:
          roles.append({'label': 'Superuser', 'color': 'danger'})
      if target.is_staff and not target.is_superuser:
          roles.append({'label': 'Admin', 'color': 'warning'})
      for group in target.groups.all():
          badge = BADGE_MAP.get(group.name, (group.name.replace('_', ' ').title(), 'secondary'))
          roles.append({'label': badge[0], 'color': badge[1]})

      profile, _ = UserProfile.objects.get_or_create(user=target)

      context = {
          'admin_orgs':      list(admin_orgs),
          'member_orgs':     list(member_orgs),
          'roles':           roles,
          'profile':         profile,
          'country_choices': COUNTRY_CHOICES,
          'profile_user':    target,
          'is_own_profile':  False,
      }
      return render(request, 'accounts/profile.html', context)
  ```

  This duplicates `_is_valid_target` from `messaging/views.py` (Task 1) deliberately — matches the existing per-app duplication convention for `group_required` already established in this codebase, rather than introducing a cross-app import.

- [ ] **Step 4: Update the template's display fields to use `profile_user`**

  In `Trust-AI-Platform/accounts/templates/accounts/profile.html`, make these six replacements (each is a distinct, uniquely-matchable line):

  Replace:
  ```html
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ user.get_full_name|default:user.username }}</div>
  ```
  with:
  ```html
          <div style="font-size:13px;opacity:0.82;margin-top:2px;">{{ profile_user.get_full_name|default:profile_user.username }}</div>
  ```

  Replace:
  ```html
            <h5 class="card-title">{{ user.first_name }} {{ user.last_name }}</h5>
            <h6 class="text-muted small">@{{ user.username }}</h6>
  ```
  with:
  ```html
            <h5 class="card-title">{{ profile_user.first_name }} {{ profile_user.last_name }}</h5>
            <h6 class="text-muted small">@{{ profile_user.username }}</h6>
  ```

  Replace:
  ```html
                <span class="fw-semibold small">{{ user.username }}</span>
  ```
  with:
  ```html
                <span class="fw-semibold small">{{ profile_user.username }}</span>
  ```

  Replace:
  ```html
                <span class="fw-semibold small">{{ user.date_joined|date:"d M Y" }}</span>
  ```
  with:
  ```html
                <span class="fw-semibold small">{{ profile_user.date_joined|date:"d M Y" }}</span>
  ```

  Replace:
  ```html
                  {% if user.last_login %}{{ user.last_login|date:"d M Y, H:i" }}{% else %}—{% endif %}
  ```
  with:
  ```html
                  {% if profile_user.last_login %}{{ profile_user.last_login|date:"d M Y, H:i" }}{% else %}—{% endif %}
  ```

  Do NOT change the four `user.*` references inside the personal-info edit form (`first_name`/`last_name`/`email`/`username` input `value=`, currently around lines 181-205) — that form only ever renders when `is_own_profile` is true (Step 5 below), where `profile_user == request.user == user`, so leaving them as `user.` is correct and avoids unnecessary churn.

- [ ] **Step 5: Gate the edit-form column behind `is_own_profile`, add a Message CTA for other profiles**

  Replace:
  ```html
      <!-- Right column: forms -->
      <div class="col-12 col-md-7 col-xl-8">

        <!-- Personal info form -->
        <div class="card">
  ```
  with:
  ```html
      <!-- Right column: forms -->
      <div class="col-12 col-md-7 col-xl-8">
      {% if is_own_profile %}

        <!-- Personal info form -->
        <div class="card">
  ```

  Replace:
  ```html
            </div><!-- End Tab Content -->
          </div>
        </div><!-- End Forms Card -->

      </div><!-- End Right Column -->
  ```
  with:
  ```html
            </div><!-- End Tab Content -->
          </div>
        </div><!-- End Forms Card -->

      {% else %}
        <div class="card">
          <div class="card-body text-center py-5">
            <i class="bi bi-chat-dots" style="font-size:2.5rem;color:#94a3b8;"></i>
            <h5 class="mt-3 mb-1">Send {{ profile_user.first_name|default:profile_user.username }} a message</h5>
            <p class="text-muted small mb-4">Start a conversation with this teacher.</p>
            <a href="{% url 'thread' profile_user.id %}" class="btn btn-primary">
              <i class="bi bi-send me-1"></i> Message
            </a>
          </div>
        </div>
      {% endif %}

      </div><!-- End Right Column -->
  ```

- [ ] **Step 6: Gate the form-only JS behind `is_own_profile`**

  The trailing `<script>` block (password toggle, strength meter, info-form and password-form AJAX submit handlers) only targets elements that exist inside the form gated in Step 5, and will throw on `getElementById(...).addEventListener` (null reference) if that form isn't in the DOM. Wrap the whole block.

  Replace:
  ```html
  <script>
  // --- Toggle password visibility ---
  ```
  with:
  ```html
  {% if is_own_profile %}
  <script>
  // --- Toggle password visibility ---
  ```

  Replace:
  ```html
  });
  </script>
  {% endblock %}
  ```
  with:
  ```html
  });
  </script>
  {% endif %}
  {% endblock %}
  ```

- [ ] **Step 7: Add the "View profile" link into Task 1's thread template**

  Now that `view_profile` exists, wire it into the thread header Task 1 deliberately left unlinked. In `Trust-AI-Platform/messaging/templates/messaging/thread.html`, replace:
  ```html
      <div>
        <h2 style="margin:0;font-size:18px;font-weight:700;">{{ partner.get_full_name|default:partner.username }}</h2>
      </div>
    </div>
  ```
  with:
  ```html
      <div>
        <h2 style="margin:0;font-size:18px;font-weight:700;">{{ partner.get_full_name|default:partner.username }}</h2>
        <a href="{% url 'view_profile' partner.id %}" style="font-size:12px;color:rgba(255,255,255,0.75);">View profile</a>
      </div>
    </div>
  ```

- [ ] **Step 8: Write the tests**

  Append to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class ViewProfileTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.alice = User.objects.create_user('alice_prof', password='pass', first_name='Alice', last_name='A')
          self.bob = User.objects.create_user('bob_prof', password='pass', first_name='Bob', last_name='B')
          self.alice.groups.add(teachers)
          self.bob.groups.add(teachers)
          self.student = User.objects.create_user('stu_prof', password='pass')
          self.client.login(username='alice_prof', password='pass')

      def test_view_other_teacher_profile(self):
          r = self.client.get(reverse('view_profile', args=[self.bob.id]))
          self.assertEqual(r.status_code, 200)
          self.assertFalse(r.context['is_own_profile'])
          self.assertEqual(r.context['profile_user'], self.bob)

      def test_view_own_profile_via_view_profile_redirects(self):
          r = self.client.get(reverse('view_profile', args=[self.alice.id]))
          self.assertRedirects(r, reverse('profile'))

      def test_view_student_profile_404s(self):
          r = self.client.get(reverse('view_profile', args=[self.student.id]))
          self.assertEqual(r.status_code, 404)

      def test_own_profile_page_marks_is_own_profile_true(self):
          r = self.client.get(reverse('profile'))
          self.assertTrue(r.context['is_own_profile'])
          self.assertEqual(r.context['profile_user'], self.alice)

      def test_student_cannot_view_other_profile(self):
          self.client.logout()
          self.client.login(username='stu_prof', password='pass')
          r = self.client.get(reverse('view_profile', args=[self.bob.id]))
          self.assertEqual(r.status_code, 403)

      def test_other_profile_shows_message_button_not_edit_form(self):
          r = self.client.get(reverse('view_profile', args=[self.bob.id]))
          self.assertContains(r, 'Send Bob a message')
          self.assertNotContains(r, 'id="infoForm"')
  ```

- [ ] **Step 8: Run the tests**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.ViewProfileTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (6 tests).

  Then run the full `accounts` suite to confirm no regression to the existing self-profile flow:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 9: Manually verify (if a real dev environment is available)**

  Same caveat as Task 1 Step 12 — this needs a live Postgres-backed dev environment with existing teacher accounts, not the throwaway `settings_test` DB. If available:
  ```bash
  python manage.py runserver
  ```
  As a teacher, visit `/accounts/profile/<other-teacher-id>/` — confirm the edit form/password tabs are gone, a "Message" button appears, and clicking it opens the Task 1 thread view. Visit your own `/accounts/profile/` and confirm nothing changed (edit form still works, password change still works).

  If no such environment is available, Step 8's automated test suite is the load-bearing verification for this task.

- [ ] **Step 10: Commit**

  ```bash
  git add Trust-AI-Platform/accounts/ Trust-AI-Platform/messaging/templates/messaging/thread.html
  git commit -m "Add cross-user profile viewing, gated to teachers"
  ```

---

### Task 3: Discovery and notification integration

**Files:**
- Modify: `Trust-AI-Platform/organization/templates/organization/organization_detail.html`
- Modify: `Trust-AI-Platform/templates/head.html`
- Modify: `Trust-AI-Platform/templates/main.html`
- Create/Modify: `Trust-AI-Platform/organization/tests.py` (currently an empty scaffold)
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Consumes: `view_profile` URL (Task 2), `message_threads`/`thread`/`unread_status` URLs (Task 1).
- Produces: no new interfaces — this is the final integration layer, pure template/JS plus response-content tests.

- [ ] **Step 1: Link member names to profiles and add a Message button in the Organization member list**

  In `Trust-AI-Platform/organization/templates/organization/organization_detail.html`, this template has no `{% load %}` tags today — the new markup below needs the `has_group` filter, so add it. Replace:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — {{ organization.name }}</title>{% endblock %}
  {% block atcontent %}
  ```
  with:
  ```html
  {% extends "main.html" %}
  {% load group_tags %}
  {% block page_title %}<title>Trust AI Lab — {{ organization.name }}</title>{% endblock %}
  {% block atcontent %}
  ```

  Then replace:
  ```html
                <div class="member-name">
                    {% if member.get_full_name %}
                      {{ member.get_full_name }}
                      <span class="member-meta ms-1">@{{ member.username }}</span>
                    {% else %}
                      {{ member.username }}
                    {% endif %}
                  </div>
  ```
  with:
  ```html
                <div class="member-name">
                    <a href="{% url 'view_profile' member.id %}" class="text-decoration-none" style="color:inherit;">
                    {% if member.get_full_name %}
                      {{ member.get_full_name }}
                      <span class="member-meta ms-1">@{{ member.username }}</span>
                    {% else %}
                      {{ member.username }}
                    {% endif %}
                    </a>
                  </div>
  ```

  Then, replace:
  ```html
                <!-- Actions (admin/staff only, not self) -->
                {% if member != request.user %}
                {% if request.user in organization.admins.all or request.user.is_staff or request.user.is_superuser %}
                <div class="d-flex gap-1 flex-shrink-0">
  ```
  with:
  ```html
                <!-- Message (any teacher, not self) -->
                {% if member != request.user and request.user|has_group:"teachers" %}
                <a href="{% url 'thread' member.id %}" class="action-btn" title="Message" style="flex-shrink:0;">
                  <i class="bi bi-chat-dots"></i>
                </a>
                {% endif %}

                <!-- Actions (admin/staff only, not self) -->
                {% if member != request.user %}
                {% if request.user in organization.admins.all or request.user.is_staff or request.user.is_superuser %}
                <div class="d-flex gap-1 flex-shrink-0">
  ```

  This requires `{% load group_tags %}` at the top of the file — check first: if the `{% load %}` line isn't already present, add `{% load group_tags %}` on its own line immediately after the existing `{% extends %}`/`{% load static %}` lines at the top of the file (follow whatever load-tag pattern the file already uses for other tags at the top).

- [ ] **Step 2: Add the "Messages" sidebar nav item with an unread badge**

  In `Trust-AI-Platform/templates/head.html`, replace:
  ```html
      <li class="nav-item">
        <a class="nav-link collapsed" href="{% url 'list_groups' %}">
          <i class="bi bi-people-fill"></i>
          <span>Student Groups</span>
        </a>
      </li>
  	{% endif %}
  ```
  with:
  ```html
      <li class="nav-item">
        <a class="nav-link collapsed" href="{% url 'list_groups' %}">
          <i class="bi bi-people-fill"></i>
          <span>Student Groups</span>
        </a>
      </li>

      <li class="nav-item">
        <a class="nav-link collapsed d-flex align-items-center" href="{% url 'message_threads' %}">
          <i class="bi bi-chat-dots-fill"></i>
          <span>Messages</span>
          <span id="sidebar-unread-badge" class="badge bg-danger ms-auto me-2" style="display:none;"></span>
        </a>
      </li>
  	{% endif %}
  ```

- [ ] **Step 3: Add the polling script and toast container to the base template**

  In `Trust-AI-Platform/templates/main.html`, replace the entire file content:
  ```html
  <!DOCTYPE html>
  {% block page_title %}<title>Trust AI Lab</title>{% endblock %}
  {% include 'head.html' %}

  {% if messages %}
  <div style="position:fixed;top:76px;right:16px;z-index:9999;min-width:300px;max-width:420px;">
    {% for message in messages %}
    <div class="alert alert-{{ message.tags }} alert-dismissible fade show shadow-sm" role="alert">
      {{ message }}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% block atcontent %}

  {% endblock %}

  {% load group_tags %}
  {% if request.user.is_authenticated and request.user|has_group:"teachers" %}
  <div id="new-message-toast-container" style="position:fixed;bottom:20px;right:16px;z-index:9999;min-width:300px;max-width:420px;"></div>
  <script>
  (function () {
    let lastSeenMessageId = 0;

    function renderToast(latest) {
      const container = document.getElementById('new-message-toast-container');
      if (!container) return;
      const toast = document.createElement('div');
      toast.className = 'alert alert-info alert-dismissible fade show shadow-sm';
      toast.setAttribute('role', 'alert');
      toast.style.cursor = 'pointer';

      const strong = document.createElement('strong');
      strong.textContent = 'New message from ' + latest.sender_name;
      toast.appendChild(strong);
      toast.appendChild(document.createElement('br'));
      toast.appendChild(document.createTextNode(latest.snippet));

      const closeBtn = document.createElement('button');
      closeBtn.type = 'button';
      closeBtn.className = 'btn-close';
      closeBtn.setAttribute('data-bs-dismiss', 'alert');
      closeBtn.setAttribute('aria-label', 'Close');
      toast.appendChild(closeBtn);

      toast.addEventListener('click', function (e) {
        if (e.target === closeBtn) return;
        window.location.href = '/messaging/' + latest.sender_id + '/';
      });
      container.appendChild(toast);
    }

    function updateBadge(count) {
      const badge = document.getElementById('sidebar-unread-badge');
      if (!badge) return;
      if (count > 0) {
        badge.textContent = count > 99 ? '99+' : count;
        badge.style.display = '';
      } else {
        badge.style.display = 'none';
      }
    }

    function poll() {
      fetch('/messaging/unread_status/')
        .then(function (r) { return r.json(); })
        .then(function (data) {
          updateBadge(data.unread_count);
          if (data.latest && data.latest.id !== lastSeenMessageId) {
            if (lastSeenMessageId !== 0) {
              renderToast(data.latest);
            }
            lastSeenMessageId = data.latest.id;
          }
        })
        .catch(function () {});
    }

    poll();
    setInterval(poll, 20000);
  }());
  </script>
  {% endif %}

  {% include 'footer.html' %}
  ```

  Note on behavior: `lastSeenMessageId` starts at `0` and the first poll only establishes the baseline silently (no toast on page load for a pre-existing unread message) — a toast only fires when a *newer* unread message is detected on a later poll than the one that established the baseline. This avoids re-showing a toast for the same already-known unread message on every page navigation.

- [ ] **Step 4: Write tests for the template wiring**

  This task has no view/logic changes, but the template edits are worth verifying with the Django test client (response-content assertions) rather than relying solely on manual browser verification, which this sandboxed environment mostly can't do (see Task 1/2's manual-verify caveats).

  Create `Trust-AI-Platform/organization/tests.py` (replacing its current empty scaffold):
  ```python
  from django.contrib.auth.models import Group, User
  from django.test import Client, TestCase
  from django.urls import reverse

  from .models import Organization


  class OrganizationDetailMessagingLinksTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.alice = User.objects.create_user('alice_org', password='pass')
          self.bob = User.objects.create_user('bob_org', password='pass', first_name='Bob', last_name='B')
          self.alice.groups.add(teachers)
          self.bob.groups.add(teachers)
          self.org = Organization.objects.create(name='Test Org', short_name='TO', created_by=self.alice)
          self.org.members.add(self.alice, self.bob)
          self.client.login(username='alice_org', password='pass')

      def test_member_name_links_to_profile(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, reverse('view_profile', args=[self.bob.id]))

      def test_message_button_links_to_thread(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertContains(r, reverse('thread', args=[self.bob.id]))

      def test_no_message_button_for_self(self):
          r = self.client.get(reverse('organization_detail', args=[self.org.id]))
          self.assertNotContains(r, reverse('thread', args=[self.alice.id]))
  ```

  Append to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class SidebarMessagingIntegrationTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.teacher = User.objects.create_user('sidebar_teacher', password='pass')
          self.teacher.groups.add(teachers)
          self.student = User.objects.create_user('sidebar_student', password='pass')

      def test_teacher_sees_messages_nav_and_polling_script(self):
          self.client.login(username='sidebar_teacher', password='pass')
          r = self.client.get(reverse('profile'))
          self.assertContains(r, reverse('message_threads'))
          self.assertContains(r, 'sidebar-unread-badge')
          self.assertContains(r, 'new-message-toast-container')

      def test_student_does_not_see_messages_nav_or_polling_script(self):
          self.client.login(username='sidebar_student', password='pass')
          r = self.client.get(reverse('studentScenarios'))
          self.assertNotContains(r, 'sidebar-unread-badge')
          self.assertNotContains(r, 'new-message-toast-container')
  ```

- [ ] **Step 5: Run the tests**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test organization.tests.OrganizationDetailMessagingLinksTests accounts.tests.SidebarMessagingIntegrationTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (5 tests).

  Then run the full suite one more time to confirm nothing else regressed:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test messaging accounts organization -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 6: Manually verify end-to-end (if a real dev environment is available)**

  Same caveat as Task 1/2 — needs a live Postgres-backed dev environment with existing teacher accounts. If available:
  ```bash
  python manage.py runserver
  ```
  With two teacher accounts in two browser sessions (or one incognito): from the Organization member list, click a colleague's name (goes to their profile), click Message (goes to the thread), send a message. In the other account's session, wait up to ~20s and confirm the sidebar "Messages" badge updates and a toast appears bottom-right; click the toast and confirm it navigates to the thread. Confirm a student account never sees the "Messages" sidebar item or the toast script, and gets 403 on any `/messaging/` or `/accounts/profile/<id>/` URL.

  If no such environment is available, Step 5's automated tests are the load-bearing verification for this task.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/organization/templates/organization/organization_detail.html Trust-AI-Platform/organization/tests.py Trust-AI-Platform/templates/head.html Trust-AI-Platform/templates/main.html Trust-AI-Platform/accounts/tests.py
  git commit -m "Wire messaging discovery, sidebar badge, and notification toast into base templates"
  ```
