# Organization Announcements & Team Chat — Design Spec

## Goal

Let organization admins post rich-text announcements on their org's page (reusing the TinyMCE pattern Activities already use), and let all members of an organization chat together in a shared team chat room, reachable from both the org page and the existing Messages page. Everything must be responsive on phones and tablets.

## Scope

- `Trust-AI-Platform/organization/models.py` — new `Announcement` and `OrgChatMessage` models
- `Trust-AI-Platform/organization/views.py` — announcement CRUD views, chat room/send/poll views
- `Trust-AI-Platform/organization/urls.py` — new routes for both features
- `Trust-AI-Platform/organization/admin.py` — register both new models
- `Trust-AI-Platform/organization/templates/organization/organization_detail.html` — Announcements card, "Team Chat" hero button
- New templates: announcement create/edit form, org chat room page
- `Trust-AI-Platform/messaging/views.py` and `messaging/templates/messaging/thread_list.html` — "Organization Chats" section added to the existing Messages page
- **Two separate implementation plans**, per your sequencing choice: Plan 1 (Announcements) ships first, Plan 2 (Team Chat + Messages-page integration) ships second. This spec covers both since they share context, but each gets its own plan/review cycle.

## Global Constraints

- **Responsive & mobile-first:** every new template must work on phones (≥320px) and tablets (≥768px) — no new fixed pixel widths on outer containers, Bootstrap grid/relative units only. `organization_detail.html` already has an established mobile breakpoint pattern (`@media (max-width: 575.98px)`) — new sections follow that same convention.
- **Permission pattern:** reuse the exact existing idiom — `organization.members.filter(id=request.user.id).exists()` for membership, `organization.admins.filter(id=request.user.id).exists() or request.user.is_staff or request.user.is_superuser` for admin/moderation actions. No new permission abstraction.
- **No new infrastructure:** no WebSockets/Channels — this codebase's only live-update mechanism is `setInterval`+`fetch` polling (confirmed, no Channels/Daphne/ASGI-consumer setup exists). The chat room needs its own new polling loop (unlike the 1:1 messaging thread view, which doesn't poll at all today — it only appends the locally-sent message).
- **Reuse, don't duplicate:** the announcement image uploader reuses the existing `/authoringtool/tinymce/upload/` endpoint (`@login_required`, generic, no activity-specific coupling) rather than building a second one.
- **New models live in `organization`**, not `messaging` — the `messaging` app's own design explicitly scoped itself to "1:1 threads only, no group messaging" as a hard constraint; bolting a group-chat model onto it would violate that. `OrgChatMessage` is conceptually an organization concern (deleted when the org is deleted, scoped by org membership) and belongs in the `organization` app.
- Match existing per-app conventions already established this session: local `group_required`/permission-check duplication over cross-app imports where the codebase already does this; cross-app model imports (e.g. `from organization.models import Organization` inside `messaging/views.py`) follow the same pattern already used in `accounts/views.py`'s `profile_view`.

---

## Part A — Announcements

### Model

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
```

`plain_text` mirrors `Activity.plain_text` exactly — generated server-side via the same 2-line helper `authoringtool/views.py:229-231` already uses (`strip_tags` + `unescape`), duplicated locally in `organization/views.py` per this codebase's established per-app-duplication convention (matching `group_required`'s precedent) rather than importing from `authoringtool`.

### Views

- `create_announcement(org_id)` — admin-only (org admin or site staff/superuser, same check as `organization_detail`). GET renders a form (title + TinyMCE body), POST creates the `Announcement`, strips the body for `plain_text`, redirects back to `organization_detail`.
- `edit_announcement(org_id, announcement_id)` — same permission gate, pre-fills the form, updates in place.
- `delete_announcement(org_id, announcement_id)` — POST-only, same permission gate, hard delete (matching how member removal already works — no soft-delete/archive concept elsewhere in this app).

### TinyMCE integration

Identical pattern to `createActivity.html` (`authoringtool/templates/authoringtool/createActivity.html:780-812`):
- CDN script: `<script src="https://cdn.jsdelivr.net/npm/tinymce@6/tinymce.min.js" referrerpolicy="origin"></script>`
- Textarea selected by class (`.tinymce-editor`), not by field-name magic.
- Same `tinymce.init({...})` config (height 320, `menubar: false`, `plugins: 'lists link image table code'`, same toolbar), with `images_upload_handler` posting to `/authoringtool/tinymce/upload/` — the exact same endpoint, unmodified, reused across apps.
- `editor.save()` on change syncs content back into the underlying textarea for normal form submission (not AJAX) — same as Activities.

### UI on `organization_detail.html`

New card inserted between the existing Details/Members row (closes at line 274, `</div><!-- /row -->`) and the Pending Join Requests block (starts at line 276) — same `<div class="row g-3 mt-2"><div class="col-12"><div class="card">...</div></div></div>` wrapper pattern already used for Pending Join Requests, so it inherits the same responsive behavior with zero new CSS needed for the outer structure.

Each announcement: title, rendered `body` (safe HTML, author, relative date). Admins see Edit/Delete icon buttons per item (reusing the `.action-btn` class already defined in this template's `<style>` block) and a "New Announcement" button in the card header. Non-admin members see read-only content, no controls.

---

## Part B — Team Chat

### Model

```python
class OrgChatMessage(models.Model):
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='chat_messages')
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='org_chat_messages')
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
```

No read-receipts, no per-member unread tracking — not requested, and group read-tracking is materially more complex than the 1:1 `read_at` timestamp the `messaging` app uses. Out of scope unless asked for later.

### Views

- `org_chat(org_id)` — membership-gated (`organization.members.filter(...)`, same as `organization_detail`). Renders full message history (chronological, no pagination — matching the existing 1:1 `thread` view's approach of rendering the whole history) plus a compose box.
- `send_org_chat_message(org_id)` — POST-only, membership-gated, creates an `OrgChatMessage`, returns the created message as JSON (id, sender name, body, timestamp) for the sending client to append locally — same contract shape as `messaging`'s `send_message` endpoint.
- `org_chat_poll(org_id)` — GET, membership-gated, accepts a `since_id` query param, returns any `OrgChatMessage` rows with `id > since_id` as JSON. This is new — there's no existing "poll a room for new messages since X" endpoint anywhere in the codebase to copy verbatim, unlike the 1:1 messaging feature's `unread_status` endpoint (which reports a single unread count/latest message, not a stream of new messages in an open room).

### Live update in the open chat room

New polling loop scoped to the chat page's own `<script>` block (not the site-wide `main.html` poller, which only handles the unread badge/toast for 1:1 messages) — `setInterval` calling `org_chat_poll` every few seconds while the room is open, appending any new messages to the DOM. Distinct from the existing 1:1 `thread.html`, which has no live-refresh at all today (it only appends the message the current user just sent).

### UI

Dedicated full-page chat view (`org_chat.html`), same chat-bubble visual language as `messaging/templates/messaging/thread.html` (`.chat-body`, `.chat-bubble`, mobile media query bumping bubble width on narrow screens), adapted for multiple participants: since there's more than one "other" person, every message — including your own — shows the sender's name/avatar-initial above or beside the bubble (the existing 1:1 view can get away with an unlabeled "mine vs. theirs" split because there's only ever one other person; a group room can't).

A "Team Chat" button appears in the org page's hero action-button row (`organization_detail.html:101-130`, alongside Back/Add Member/Edit), visible to any member (not just admins) — placed using the existing `.hero-btn-ghost` class for visual consistency, gated by `is_member` (not `is_admin`).

---

## Part C — Reaching Org Chat from the Messages Page

The existing Messages page (`messaging/templates/messaging/thread_list.html`) currently shows only 1:1 conversation threads. Add a new "Organization Chats" section **above** the existing "Messages" list (per your ordering choice), listing every organization the current user is a member of (`request.user.member_of_organizations.all()`), each row linking to that org's `org_chat` view.

Implementation note: `messaging/views.py`'s `message_threads` view gains a query for the user's organizations (cross-app import of `Organization` from `organization.models`, following the exact precedent already established in `accounts/views.py`'s `profile_view`/`view_profile`). The template gets a second card, visually matching the existing `.thread-row` list style so both sections feel like one cohesive page, not two bolted-together halves.

Empty state: if the user isn't a member of any organization, the section either shows a brief "You're not part of any organization yet" message or (simpler) is omitted entirely when the queryset is empty — matching the existing empty-state pattern already used for the 1:1 thread list (`{% empty %}` block with an icon + message).

---

## What Does NOT Change

- The existing 1:1 `messaging` app's `Message` model, views, and templates — untouched, no group concept added to it.
- No admin-side bulk announcement/broadcast tooling.
- No read-receipts or unread-count badges for team chat.
- No pagination on chat history or announcement lists in this first pass — if either grows large enough to matter, that's a natural fast-follow, not blocking this feature.
- `organization_detail.html`'s existing Details/Members/Pending-Requests sections and their permission logic — untouched, only new sections added alongside them.
