# Profile Gender & Avatar — Design Spec

## Goal

Add a gender field to user profiles (settable at self-registration and editable later on the profile page) and let users upload a custom profile picture, with gender-based default images shown until they upload one of their own. Today `UserProfile` has no gender or image field at all, avatar rendering everywhere in the app is initials-only or a generic icon, and two gender-default image files (`profile_d_man.webp`, `profile_d_woman.jpg`) already sit unused in `static/img/`.

## Scope

- `Trust-AI-Platform/accounts/models.py` — `UserProfile` gains `gender` and `picture` fields
- `Trust-AI-Platform/accounts/views.py` — `registerAccount` (gender at signup) and `profile_view`'s `update_info` action (gender + picture edit)
- `Trust-AI-Platform/accounts/templates/accounts/register.html` — add gender field to the signup form
- `Trust-AI-Platform/accounts/templates/accounts/profile.html` — add gender + picture upload to the edit form, real avatar in the left card
- `Trust-AI-Platform/templates/head.html` — real avatar in the two nav-dropdown spots
- `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html` — real avatar in the user list
- New `Trust-AI-Platform/templatetags/profile_tags.py` — shared avatar-resolution template filter
- `Trust-AI-Platform/faithDev/settings.py` — register the new template library (same pattern as `group_tags`)
- One migration
- **Out of scope (per explicit decision):** no new admin-side "create user" flow — gender is added only to the existing self-registration form. No custom image validators beyond Django's default `ImageField` behavior (no other `ImageField` in this codebase has custom validation either).

## 1. Model Changes

`Trust-AI-Platform/accounts/models.py` — add to `UserProfile` (after `bio`, before `__str__`), following the exact style of the existing `COUNTRY_CHOICES` list:

```python
GENDER_CHOICES = [
    ('', '— Prefer not to say —'),
    ('male', 'Male'),
    ('female', 'Female'),
]
```

Placed at module level near `COUNTRY_CHOICES` (both are lookup tables for `UserProfile` fields).

```python
    gender  = models.CharField(max_length=10, blank=True, choices=GENDER_CHOICES)
    picture = models.ImageField(upload_to='profile_pictures', null=True, blank=True)
```

`gender=''` is the explicit "Prefer not to say" state (matches the existing `country=''` "— Select country —" pattern already used on this model) rather than a separate `'unspecified'` value — one blank/empty convention for both optional-choice fields on this model, consistent with existing code.

`picture` follows the identical pattern to `Organization.picture` (`organization/models.py:34`): `ImageField(upload_to=..., null=True, blank=True)`, no custom validators, relying on Django's default Pillow-backed `ImageField` validation (matches every other `ImageField` in this codebase — none have custom size/type validators).

## 2. Avatar Resolution — Shared Template Filter

New file `Trust-AI-Platform/templatetags/profile_tags.py`, following the exact registration pattern already used by `templatetags/group_tags.py` (a plain package at the repo root, manually registered as a template library — not app-based auto-discovery):

```python
from django import template
from django.templatetags.static import static

register = template.Library()

@register.filter
def avatar_url(user):
    profile = getattr(user, 'profile', None)
    if not profile:
        return ''
    if profile.picture:
        return profile.picture.url
    if profile.gender == 'male':
        return static('img/profile_d_man.webp')
    if profile.gender == 'female':
        return static('img/profile_d_woman.jpg')
    return ''
```

`getattr(user, 'profile', None)` guards against a user who has never had a `UserProfile` row created (today `UserProfile` rows are only lazily `get_or_create`'d when `profile_view`/`view_profile` is visited — `head.html` renders on every page for every logged-in user, including ones who've never opened their profile page, so this filter must not assume a profile row exists, and must not perform a DB write as a side effect of rendering the nav bar). An empty string return means "no photo, no gender default — fall back to the existing initials/icon placeholder," which every call site already renders today.

Register in `Trust-AI-Platform/faithDev/settings.py`, extending the existing `libraries` dict (`faithDev/settings.py:100-102`):

```python
            'libraries': {
                'group_tags': 'templatetags.group_tags',
                'profile_tags': 'templatetags.profile_tags',
            }
```

**Template usage pattern** (identical at all four call sites, using `{% with %}` so the filter runs once per render rather than being called twice for the `{% if %}` and the `{% endif %}` branch):

```html
{% load profile_tags %}
{% with avatar=user|avatar_url %}
{% if avatar %}
<img src="{{ avatar }}" alt="" style="...same box-size as the element it replaces...">
{% else %}
...existing initials/icon markup, unchanged...
{% endif %}
{% endwith %}
```

## 3. Self-Registration

`accounts/templates/accounts/register.html` — add a gender field to the existing `#registerForm` (`register.html:90-169`), placed after the Last Name field (`register.html:98-101`), before Email:

```html
<div class="col-12">
  <label class="form-label">Gender</label>
  <select class="form-select" name="gender" id="gender">
    <option value="">Prefer not to say</option>
    <option value="male">Male</option>
    <option value="female">Female</option>
  </select>
</div>
```

No `required` attribute — matches the field's `blank=True` model definition and the "Prefer not to say" default being a legitimate, explicit choice, not a validation failure. No JS changes needed: the form already submits via `new FormData(this)` (`register.html:288`), which automatically includes any named field — the existing `fetch('/accounts/register/', {..., body: new FormData(this)})` call requires no modification.

No picture upload at registration — per the approved design, picture is set later via the profile page.

`accounts/views.py:registerAccount` (currently creates the `User` but never a `UserProfile` — `views.py:130`) — read `gender` from `request.POST` and create the `UserProfile` alongside the `User`:

```python
gender = request.POST.get('gender', '').strip()
...
user = User.objects.create(username=username, first_name=first_name, last_name=last_name, email=email, password=hashed_password)
UserProfile.objects.create(user=user, gender=gender)
```

(`UserProfile` needs importing into `accounts/views.py` if not already — confirm at implementation time; it's already imported for `profile_view`'s use at `views.py:5`.)

## 4. Profile Edit Page

`accounts/templates/accounts/profile.html` — two changes:

**a) Real avatar in the left card**, replacing the static icon (`profile.html:47-49`):
```html
<div class="profile-avatar mb-3">
  {% load profile_tags %}
  {% with avatar=profile_user|avatar_url %}
  {% if avatar %}
  <img src="{{ avatar }}" alt="" style="width:5rem;height:5rem;border-radius:50%;object-fit:cover;">
  {% else %}
  <i class="bi bi-person-circle" style="font-size: 5rem; color: #012970;"></i>
  {% endif %}
  {% endwith %}
</div>
```
Note this uses `profile_user` (the template variable already established by the earlier profile-viewing work — self and other-profile both resolve through it), not `user`.

**b) Gender select + picture upload added to the "Personal Info" tab**, inside `#infoForm` (`profile.html:175-244`), after the Country field (`profile.html:211-220`):

```html
<div class="row mb-3">
  <label class="col-md-4 col-form-label">Gender</label>
  <div class="col-md-8">
    <select class="form-select" name="gender" id="gender">
      <option value="" {% if not profile.gender %}selected{% endif %}>Prefer not to say</option>
      <option value="male" {% if profile.gender == 'male' %}selected{% endif %}>Male</option>
      <option value="female" {% if profile.gender == 'female' %}selected{% endif %}>Female</option>
    </select>
  </div>
</div>

<div class="row mb-3">
  <label class="col-md-4 col-form-label">Profile Picture</label>
  <div class="col-md-8">
    <input type="file" class="form-control" name="picture" id="picture" accept="image/*">
    <div class="form-text text-muted">Leave blank to keep your current picture (or gender default).</div>
  </div>
</div>
```

This form is only ever rendered when `is_own_profile` is true (existing gating from the earlier profile-viewing work), so no separate permission check is needed here — you can only edit your own gender/picture, matching every other field on this form.

**No JS/submission changes needed.** The existing `infoForm` submit handler (`profile.html`, JS block) already builds `const data = new FormData(this)` and sends it as `fetch(url, {..., body: data})` — `FormData` natively serializes file inputs as multipart, and `fetch` sets the correct `Content-Type: multipart/form-data` boundary automatically whenever the body is a `FormData` object, file input present or not. Adding `name="picture"` to a file input is sufficient; the transport layer needs no changes. (This corrects an earlier assumption during design discussion that the form would need "converting" to multipart — inspection of the actual JS showed it already uses `FormData` + `fetch(body: data)`, which already supports file uploads as-is.)

`accounts/views.py:profile_view`'s `update_info` branch (`views.py:169-204`) — read `gender` from `request.POST` and `picture` from `request.FILES`:

```python
gender = request.POST.get('gender', '').strip()
picture = request.FILES.get('picture')
...
profile.country     = country
profile.institution = institution
profile.bio         = bio
profile.gender       = gender
update_fields = ['country', 'institution', 'bio', 'gender']
if picture:
    profile.picture = picture
    update_fields.append('picture')
profile.save(update_fields=update_fields)
```

`picture` is only touched (and only added to `update_fields`) when a new file was actually submitted — leaving the field untouched on saves where the user didn't pick a new file (matching the existing "leave blank to keep current" UX already used for `change_password`'s optional-field handling elsewhere on this same view).

No explicit "reset to default" control — clearing back to a gender-based default happens by simply not uploading a custom picture in the first place (or, if the model instance's `picture` were later cleared through some other means, the avatar filter already falls back to the gender default automatically since the fallback logic is computed at render time, not baked into stored data).

## 5. Other Avatar Render Sites

**`templates/head.html`** — two spots, both driven by `user` (the request user, via the always-present auth context processor — this template renders on every page):

Nav dropdown avatar (`head.html:79-82`):
```html
<a class="nav-link nav-profile d-flex align-items-center pe-0 gap-2" href="#" data-bs-toggle="dropdown">
  {% load profile_tags %}
  {% with avatar=user|avatar_url %}
  {% if avatar %}
  <img src="{{ avatar }}" alt="" style="width:36px;height:36px;border-radius:50%;object-fit:cover;">
  {% else %}
  <span class="nav-profile-avatar">{{ user.get_full_name|default:user.username|slice:":1"|upper }}</span>
  {% endif %}
  {% endwith %}
  <span class="d-none d-md-block dropdown-toggle nav-profile-name">{{ user.get_full_name|default:user.username }}</span>
</a>
```
(exact pixel size for the `<img>` to be confirmed against the existing `.nav-profile-avatar` CSS class's computed size at implementation time, so the image doesn't visually jump relative to the initials circle it replaces.)

Dropdown header avatar (`head.html:87-90`) — same `{% if avatar %}...{% else %}...{% endif %}` swap around the existing `.profile-avatar-initial` div.

**`accounts/templates/accounts/admin_dashboard.html`** — the user-list avatar (`admin_dashboard.html:259`), same pattern, swapping the two-letter-initials `.user-avatar` div for an `<img>` when `u|avatar_url` is non-empty (note: loop variable here is `u`, not `user`).

## 6. Migration

One migration, auto-generatable via `makemigrations`: `AddField` for `UserProfile.gender` (default `''`) and `UserProfile.picture` (default `None`/null). No data migration needed — existing rows get the blank/null defaults, which resolve through the avatar filter exactly like a user who has no `UserProfile` at all (empty string → existing placeholder rendering, unchanged).

## 7. What Does NOT Change

- No new admin-side user-creation flow (confirmed out of scope).
- No changes to `Organization.picture` or any other existing `ImageField` usage.
- No custom image validation (size/dimension/type limits) — matches existing codebase convention of relying on Django/Pillow defaults only.
- No "remove picture" / explicit reset-to-default button — omitting a new upload is sufficient since defaults are computed at render time, not stored.
- `registerAccount`'s existing AJAX/`FormData` submission mechanism, and `profile_view`'s existing `infoForm` AJAX/`FormData` submission mechanism, are both already multipart-capable and need no transport-layer changes — only new named fields.
