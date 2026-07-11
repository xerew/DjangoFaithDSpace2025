# Teacher Profiles & Messaging — Design Spec

## Goal

Let teachers view each other's profiles and send each other direct messages, with a notification banner when a new message arrives. Today neither exists: `profile/` only ever renders `request.user`'s own profile, there is no messaging model or UI anywhere in the codebase, and the only "notification"-looking markup in the templates (`navbar.html`'s `notificationDropdown`/`messageDropdown`) is dead, unused demo code with hard-coded fake data.

## Scope

- `Trust-AI-Platform/accounts/` — add read-only "view another teacher's profile" mode
- `Trust-AI-Platform/organization/templates/organization/organization_detail.html` — add profile link + Message button to the member list (the sole discovery surface — no new directory page)
- New Django app `Trust-AI-Platform/messaging/` — `Message` model, thread-list view, thread-detail view, send endpoint, unread-status polling endpoint
- `Trust-AI-Platform/templates/head.html` — new sidebar "Messages" nav item with unread badge
- `Trust-AI-Platform/templates/main.html` — polling script + toast banner for new messages
- `Trust-AI-Platform/faithDev/settings.py` and root `urls.py` — register the new app

## Global Constraints

- **Audience:** messaging and profile-viewing are restricted to users in the `teachers` group (plus staff/superusers). Students never see or reach any of this — every view is gated with the existing `group_required('teachers')` decorator pattern (`accounts/views.py:25-35`).
- **No new discovery UI:** no "browse teachers" directory page. The only place a teacher finds another teacher to view/message is the existing Organization member list (`organization_detail.html`). Once a teacher has a target user's profile page open (reached via that list, or via a message thread they're already in), they can message from there too.
- **1:1 threads only**, no group messaging. A "thread" is computed as the ordered set of `Message` rows between two specific users — no separate `Conversation`/`Thread` table.
- **Polling, not WebSockets.** New-message detection reuses the existing Celery-task-status polling pattern already used throughout the analytics dashboard (`authoringtool/templates/authoringtool/index.html`, e.g. `pollActivityTask`) — `setInterval` + `fetch` against a small JSON status endpoint. No Django Channels, no Redis pub/sub, no ASGI changes.
- Any teacher/staff user can message any other teacher/staff user platform-wide (not scoped to shared Organization/UserGroup) — permission is platform-wide even though the discovery UI is organization-based.

---

## 1. Profile Viewing

**URL:** add `path('profile/<int:user_id>/', views.view_profile, name='view_profile')` in `accounts/urls.py`, alongside the existing `path('profile/', views.profile_view, name='profile')` (accounts/urls.py:27) which continues to mean "my own profile."

**View (`accounts/views.py`, new function `view_profile`):**
- Gated with `@group_required('teachers')`.
- `target = get_object_or_404(User, pk=user_id)`; if `target == request.user`, redirect to the existing `profile` URL (no separate "view my own profile read-only" mode needed).
- If `target` is not in the `teachers` group and is not staff/superuser, return 404 (students are not viewable even by direct URL guess).
- Builds the same `roles`/`admin_orgs`/`member_orgs`/`profile` context `profile_view` already builds (accounts/views.py:234-258), but keyed off `target` instead of `request.user`.
- Renders `accounts/profile.html` with an added `profile_user` (the target) and `is_own_profile=False` in context.

**Template (`accounts/templates/accounts/profile.html`):**
- Every place the template currently reads `user.*` for display (first/last name, username, date_joined, last_login — lines 32, 50-51, 63-73) must read `profile_user.*` instead, so the same template serves both "my profile" (where the view passes `profile_user=request.user, is_own_profile=True`) and "their profile."
- The right-hand tabbed edit form (personal info + change password) is wrapped in `{% if is_own_profile %}...{% endif %}` — only ever shown for your own profile.
- When `is_own_profile` is `False`, a "Message" button appears near the name/role-badges area, linking to the new thread view (`{% url 'thread' profile_user.id %}`).
- `profile_view` (the existing self-profile view) is updated to also pass `profile_user=user` and `is_own_profile=True` so the template only needs one code path.

## 2. Organization Member List — Discovery Surface

In `organization_detail.html`, inside the existing member loop (lines 187-209):
- The member name (currently plain text, lines 198-204) becomes a link: `<a href="{% url 'view_profile' member.id %}">`.
- A small message icon button is added next to the existing role badge/action buttons (following the same `action-btn`-style pattern already used for promote/demote/remove, lines 220-245): `<a href="{% url 'thread' member.id %}" class="action-btn" title="Message"><i class="bi bi-chat-dots"></i></a>` — shown whenever `member != request.user` and `member` is in the `teachers` group (mirrors the platform-wide teacher-to-teacher permission, not org-admin-only).

## 3. Data Model — new `messaging` app

```python
# Trust-AI-Platform/messaging/models.py
from django.db import models
from django.contrib.auth.models import User

class Message(models.Model):
    sender = models.ForeignKey(User, related_name='sent_messages', on_delete=models.CASCADE)
    recipient = models.ForeignKey(User, related_name='received_messages', on_delete=models.CASCADE)
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['created_at']
```

A "thread" between users A and B is `Message.objects.filter(Q(sender=A, recipient=B) | Q(sender=B, recipient=A)).order_by('created_at')` — computed at query time, no denormalized thread table. "Unread count for user X" is `Message.objects.filter(recipient=X, read_at__isnull=True).count()`.

## 4. Messaging Views & URLs

New app `Trust-AI-Platform/messaging/`, registered in `INSTALLED_APPS` and included in the root URLconf under `path('messaging/', include('messaging.urls'))`.

| URL | Name | Purpose |
|---|---|---|
| `messages/` | `message_threads` | Thread list: one row per distinct person the user has exchanged messages with, ordered by latest message time, showing name/avatar-initial, latest message snippet, unread indicator. |
| `messages/<int:user_id>/` | `thread` | Chat-style thread view with the given user: all messages between the two users, chronological, chat-bubble styling (own messages right-aligned, theirs left-aligned — same visual language as nothing else in the app currently, so this is new but self-contained CSS scoped to this template), a compose box at the bottom that POSTs a new message via AJAX and appends it without a full page reload. Opening this view also marks all unread messages from that other user as read (`read_at = now()`). |
| `messaging/send/` | `send_message` | POST-only endpoint (`@require_POST`, `@group_required('teachers')`) — `recipient_id`, `body` → creates a `Message` row, returns the created message as JSON for the calling page to append to the thread. |
| `messaging/unread_status/` | `unread_status` | GET-only, returns `{"unread_count": N, "latest": {"id": ..., "sender_name": ..., "sender_id": ..., "snippet": ..., "created_at": ...} | null}` for the polling script — `latest` is the most recent unread message across all threads, or `null` if none. `sender_name` is `sender.get_full_name() or sender.username`, matching the fallback pattern already used elsewhere (e.g. `templates/head.html:80`). `snippet` is `body` truncated to ~80 chars. |

All four views gated `@group_required('teachers')`; `thread` and `send_message` additionally validate the other party is a valid teacher/staff target (same check as `view_profile`) before allowing a message to be sent, so a crafted `user_id` can't be used to message a student.

## 5. Notification Banner

- `templates/main.html` gets a small polling `<script>` block (inline or a new static JS file), active on every page since it lives in the base template — but only rendered `{% if request.user|has_group:"teachers" %}` so students never load it.
- Every ~20 seconds, `fetch('/messaging/unread_status/')`. The script tracks the last-seen message ID in a JS variable (reset on page load). If the returned `latest.id` is newer than the last one seen, it shows a dismissible toast — reusing the same visual pattern as the existing Django-messages toast block (`main.html:5-14`, fixed top-right, Bootstrap `alert alert-info alert-dismissible`), reading "New message from {sender_name}: {snippet}", clicking it navigates to `{% url 'thread' sender_id %}`.
- This is a genuinely new polling loop (not reusing the dashboard's per-chart interval machinery, which is scoped to that page/its `dashboardState`), but follows the identical `setInterval`+`fetch` shape already established there.

## 6. Sidebar Nav Integration

In `templates/head.html`, inside the existing teachers-only `sidebar-nav` block (lines 147-175), add one new `<li class="nav-item">` (e.g. after "Student Groups", before the closing `{% endif %}` at line 175):

```html
<li class="nav-item">
  <a class="nav-link collapsed" href="{% url 'message_threads' %}">
    <i class="bi bi-chat-dots-fill"></i>
    <span>Messages</span>
    <span id="sidebar-unread-badge" class="badge bg-danger ms-auto" style="display:none;"></span>
  </a>
</li>
```

The same polling script (§5) updates `#sidebar-unread-badge`'s text/visibility from the `unread_count` field of the same `unread_status` response — one fetch call drives both the badge and the toast, not two separate polls.

## 7. What Does NOT Change

- `accounts/views.py:profile_view` keeps its existing URL (`profile/`) and behavior for self-editing; it's only extended to pass `profile_user`/`is_own_profile` into the same template `view_profile` also uses.
- No changes to `organization` app's membership/admin logic, promote/demote/remove flows — only the member-row template markup gains a link + a button.
- No changes to `UserProfile` model (country/institution/bio) — messaging doesn't touch it.
- The dead `navbar.html`/`sidebar.html` demo templates are left alone (unused, out of scope) — not cleaned up as part of this feature.
- No email notifications, no push notifications — the toast + badge are the only notification surfaces.
- No message editing/deletion, no read-receipts beyond the single `read_at` timestamp, no attachments — plain text `body` only.
