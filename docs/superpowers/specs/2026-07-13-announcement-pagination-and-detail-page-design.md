# Announcement Pagination and Detail Page — Design Spec

## Goal

The Announcements card on the organization detail page currently renders every announcement's full rendered body inline, with no limit. Replace that with a paginated (5 per page), truncated-preview list where each announcement links to a dedicated detail page showing the full content.

## Scope

- `Trust-AI-Platform/organization/views.py` — paginate `organization_detail`'s announcements query; add a new `announcement_detail` view; add a local `_smart_page_range` helper.
- `Trust-AI-Platform/organization/urls.py` — one new route.
- `Trust-AI-Platform/organization/templates/organization/organization_detail.html` — Announcements card becomes a truncated, clickable, paginated list.
- New template: `Trust-AI-Platform/organization/templates/organization/announcement_detail.html`.

No model or migration changes — `Announcement.plain_text` (added in the original Announcements plan, auto-synced from `body` via `strip_html_tags` on every create/edit) already provides exactly the plain-text source a truncated preview needs.

## Global Constraints

- **Reuse the existing pagination convention**, don't invent a new one. `authoringtool/views.py` already establishes the pattern used site-wide: `django.core.paginator.Paginator`, a `_smart_page_range(page_obj)` helper (ellipsis'd page numbers, keeps first/last plus a window around the current page), and matching Bootstrap `.pagination` markup in `authoringtool/templates/authoringtool/scenarios.html:317-351`. Duplicate `_smart_page_range` locally into `organization/views.py` (this app's own established per-file-duplication convention — same precedent as `group_required`, `strip_html_tags`) rather than importing it cross-app. Unlike `scenarios.html`'s AJAX-swapped pagination, this list does a plain full-page reload (`?page=N`) — no AJAX infrastructure needed for a 5-per-page list.
- **Preview truncation:** `{{ announcement.plain_text|truncatechars:250 }}` — Django's built-in filter, no custom code. Rendered as plain auto-escaped text (no `|safe`), since `plain_text` has no HTML in it by construction.
- **Detail page body still uses `|safe`** — same trust model as today (admin-authored rich text, already decided and shipped as-is in the original Announcements plan). Do not add sanitization as part of this change; that was already surfaced to and decided by the user separately.
- **Visibility:** `announcement_detail` is `@login_required` only, no membership gate — matches `organization_detail`'s own existing pattern (any logged-in user can view the org page and its announcements; membership/admin status only changes which buttons are visible).
- **Admin Edit/Delete controls appear in both places** — on each row of the paginated list (as today) AND on the detail page. Both use the existing `_is_org_admin` check, unchanged.
- **No new pagination query-param collisions:** `organization_detail.html` has no other paginated section today, so a plain `page` GET param is unambiguous. If a second paginated section is ever added to this page, that's a future problem, not this one (YAGNI).
- Responsive & mobile-first, consistent with every other change on this branch: the Bootstrap `.pagination` component and truncated preview text already wrap naturally at any width; no new fixed-pixel widths.

## Behavior

### Paginated list (`organization_detail.html`, Announcements card)

- Query: `organization.announcements.select_related('created_by')`, paginated 5 per page via `Paginator`, current page from `request.GET.get('page')`.
- Each row: title (now a link to `announcement_detail`), truncated preview (`plain_text|truncatechars:250`), author + date meta (unchanged), admin Edit/Delete icon buttons (unchanged, same permission gate).
- Pagination controls below the list: Prev/Next + smart ellipsis'd page numbers, shown only when `page_obj.paginator.num_pages > 1` — same conditional and markup shape as `scenarios.html`.
- Empty state unchanged (no announcements at all).
- A page number beyond the last page (or a non-numeric `page` value) falls back to the last page / page 1 respectively — this is `Paginator.get_page()`'s built-in behavior, not custom code.

### Detail page (`announcement_detail`, new)

- URL: `organization/<int:org_id>/announcements/<int:announcement_id>/`, name `announcement_detail`.
- View: `@login_required`, fetches the `Organization` and the `Announcement` (404 if the announcement doesn't belong to that org — same IDOR-safe lookup pattern `edit_announcement`/`delete_announcement` already use).
- Template: full title, full `{{ announcement.body|safe }}`, author, timestamp, admin Edit/Delete buttons (same permission gate as the list), a "Back to Organization" link.
- Same hero/breadcrumb visual pattern as `create_announcement.html`/`edit_announcement.html` (already established in this app for announcement-related pages).

## What Does NOT Change

- `Announcement` model — no new fields, no migration.
- `create_announcement`/`edit_announcement`/`delete_announcement` views — unchanged, still redirect to `organization_detail` (page 1) after create/edit/delete.
- The `|safe` rendering trust model for the full body — already decided (ship as-is) in the original Announcements plan; not revisited here.
- No sanitization, no read receipts, no other unrelated scope.
