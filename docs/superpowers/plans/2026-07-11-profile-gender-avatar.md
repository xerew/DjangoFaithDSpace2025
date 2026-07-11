# Profile Gender & Avatar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gender field to user profiles (self-registration + profile edit) and let users upload a custom profile picture, with gender-based default images shown until they do, replacing the current initials/icon placeholders everywhere an avatar renders.

**Architecture:** Four sequential tasks. Task 1 lays the foundation (model fields, migration, and a shared avatar-resolution template filter used by every later task). Task 2 adds gender to self-registration. Task 3 adds gender + picture upload to the profile edit page (including the profile page's own avatar). Task 4 wires the filter into the three remaining render sites (nav dropdown ×2, admin user list) — pure template wiring, no new logic.

**Tech Stack:** Django 5.1 (runtime reports 5.2.16) · Bootstrap 5 · Pillow-backed `ImageField` (already a dependency, used by `Organization.picture` and others) · SQLite (dev/test)

## Global Constraints

- No new admin-side "create user" flow — gender is added only to the existing self-registration form (`/register/`), per explicit scope decision.
- No custom image validation (size/dimensions/type) beyond Django's default `ImageField`/Pillow behavior — matches every other `ImageField` in this codebase.
- `gender=''` is the explicit "Prefer not to say" state — one blank-string convention, matching the existing `country=''` pattern on the same model. Do not introduce a separate `'unspecified'` value.
- The avatar-resolution filter must never perform a DB write (no `get_or_create`) since it's called from `head.html`, which renders on every page for every logged-in user, including ones who've never opened their own profile page.
- Both existing AJAX forms (`registerAccount`, `profile_view`'s `update_info`) already submit via `FormData` + `fetch(body: data)`, which is multipart-capable as-is — do not add `enctype` attributes, do not change the JS submission mechanism. Only add new named form fields.

---

### Task 1: Model fields, migration, shared avatar filter

**Files:**
- Modify: `Trust-AI-Platform/accounts/models.py`
- Modify: `Trust-AI-Platform/faithDev/settings.py`
- Create: `Trust-AI-Platform/templatetags/profile_tags.py`
- Create: `Trust-AI-Platform/accounts/tests.py` additions
- Create (via `makemigrations`): `Trust-AI-Platform/accounts/migrations/0002_userprofile_gender_userprofile_picture.py` (exact filename may differ — accept whatever Django's autodetector names it)

**Interfaces:**
- Produces: `UserProfile.gender` (CharField, choices `''`/`'male'`/`'female'`), `UserProfile.picture` (ImageField, `upload_to='profile_pictures'`, nullable)
- Produces: `{% load profile_tags %}` / `{{ user|avatar_url }}` filter — returns a URL string (custom picture, or gender-default static path) or `''` if neither applies. Consumed by Tasks 3 and 4.

- [ ] **Step 1: Write the failing tests**

  Add this import to the top of `Trust-AI-Platform/accounts/tests.py` (`templatetags/profile_tags.py` is a plain top-level package at the repo root, importable as `templatetags.profile_tags` — matching how `templatetags/group_tags.py` already works, not nested under `accounts/`):
  ```python
  from templatetags.profile_tags import avatar_url as avatar_url_filter
  ```

  Append this test class to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class AvatarUrlFilterTests(TestCase):
      def setUp(self):
          self.user = User.objects.create_user('avatar_user', password='pass')

      def test_no_profile_returns_empty_string(self):
          self.assertEqual(avatar_url_filter(self.user), '')

      def test_gender_male_returns_default_static_path(self):
          UserProfile.objects.create(user=self.user, gender='male')
          self.assertEqual(avatar_url_filter(self.user), '/static/img/profile_d_man.webp')

      def test_gender_female_returns_default_static_path(self):
          UserProfile.objects.create(user=self.user, gender='female')
          self.assertEqual(avatar_url_filter(self.user), '/static/img/profile_d_woman.jpg')

      def test_blank_gender_returns_empty_string(self):
          UserProfile.objects.create(user=self.user, gender='')
          self.assertEqual(avatar_url_filter(self.user), '')

      def test_custom_picture_takes_priority_over_gender(self):
          from django.core.files.uploadedfile import SimpleUploadedFile
          tiny_gif = (
              b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
              b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
          )
          profile = UserProfile.objects.create(
              user=self.user, gender='female',
              picture=SimpleUploadedFile('test.gif', tiny_gif, content_type='image/gif'),
          )
          url = avatar_url_filter(self.user)
          self.assertTrue(url.startswith('/media/profile_pictures/test'))
          self.assertNotIn('profile_d_woman', url)
  ```

  The top of `Trust-AI-Platform/accounts/tests.py` currently imports `User, Group` from `django.contrib.auth.models` but does not import `UserProfile`. Add this import alongside it:
  ```python
  from accounts.models import UserProfile
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.AvatarUrlFilterTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL with `ModuleNotFoundError: No module named 'templatetags.profile_tags'` (or `AttributeError` on `UserProfile.gender`/`.picture` if the import is adjusted manually first).

- [ ] **Step 3: Add the model fields**

  In `Trust-AI-Platform/accounts/models.py`, replace:
  ```python
  class UserProfile(models.Model):
      user        = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
      country     = models.CharField(max_length=100, blank=True, choices=COUNTRY_CHOICES)
      institution = models.CharField(max_length=255, blank=True, help_text="School or university you work at")
      bio         = models.TextField(max_length=500, blank=True, help_text="A short bio (max 500 characters)")

      def __str__(self):
          return f"Profile of {self.user.username}"
  ```
  with:
  ```python
  GENDER_CHOICES = [
      ('', '— Prefer not to say —'),
      ('male', 'Male'),
      ('female', 'Female'),
  ]


  class UserProfile(models.Model):
      user        = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
      country     = models.CharField(max_length=100, blank=True, choices=COUNTRY_CHOICES)
      institution = models.CharField(max_length=255, blank=True, help_text="School or university you work at")
      bio         = models.TextField(max_length=500, blank=True, help_text="A short bio (max 500 characters)")
      gender      = models.CharField(max_length=10, blank=True, choices=GENDER_CHOICES)
      picture     = models.ImageField(upload_to='profile_pictures', null=True, blank=True)

      def __str__(self):
          return f"Profile of {self.user.username}"
  ```

- [ ] **Step 4: Generate and apply the migration**

  From `Trust-AI-Platform/`:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py makemigrations accounts --settings=faithDev.settings_test
  "../djangofaithvenv/Scripts/python.exe" manage.py migrate accounts --settings=faithDev.settings_test
  ```
  Expected: a new migration adding `gender` and `picture` fields to `UserProfile`, applied cleanly.

  Note: `accounts/migrations/0001_userprofile.py` currently exists on disk but has never been committed to git (confirmed in a prior session's work on this branch) — this new migration will depend on it. Both must be committed together in Step 8 or the migration chain will be broken for anyone else pulling this branch (this is the same class of issue documented in the "Proposal Edit Tracking" plan earlier on this branch — check `git status` before committing and include `0001_userprofile.py` if it's still untracked).

- [ ] **Step 5: Write the avatar filter**

  Create `Trust-AI-Platform/templatetags/profile_tags.py`:
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

- [ ] **Step 6: Register the template library**

  In `Trust-AI-Platform/faithDev/settings.py`, replace:
  ```python
              'libraries': {
                  'group_tags': 'templatetags.group_tags',
              }
  ```
  with:
  ```python
              'libraries': {
                  'group_tags': 'templatetags.group_tags',
                  'profile_tags': 'templatetags.profile_tags',
              }
  ```

- [ ] **Step 7: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.AvatarUrlFilterTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (5 tests).

- [ ] **Step 8: Commit**

  ```bash
  git status --short -- Trust-AI-Platform/accounts/migrations/
  ```
  If `0001_userprofile.py` shows as untracked (`??`), include it in the add below.
  ```bash
  git add Trust-AI-Platform/accounts/models.py Trust-AI-Platform/accounts/tests.py Trust-AI-Platform/accounts/migrations/ Trust-AI-Platform/faithDev/settings.py Trust-AI-Platform/templatetags/profile_tags.py
  git commit -m "Add UserProfile gender/picture fields and shared avatar_url filter"
  ```

---

### Task 2: Gender field on self-registration

**Files:**
- Modify: `Trust-AI-Platform/accounts/views.py`
- Modify: `Trust-AI-Platform/accounts/templates/accounts/register.html`
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Consumes: `UserProfile` model (Task 1)
- Produces: no new interfaces — `registerAccount` now creates a `UserProfile` row at signup time (previously it created none at all; `UserProfile` was only ever lazily `get_or_create`'d on first profile-page visit).

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class RegisterAccountGenderTests(TestCase):
      # accounts/views.py:14 imports TEACHER_ACCESS_CODE_HASHED at module load time
      # (`from faithDev.settings import TEACHER_ACCESS_CODE_HASHED`), so patching
      # django.conf.settings at test time would NOT affect the check at views.py:118 —
      # that name is a plain module attribute, not looked up dynamically. Simpler and
      # robust: settings.py:18 hashes this literal plaintext default whenever the
      # TEACHER_ACCESS_CODE_HASHED env var isn't set (the normal case in this test
      # environment), so just submit the real default plaintext code directly.
      VALID_ACCESS_CODE = r"}{80s%3B\x/+"

      def setUp(self):
          self.client = Client()

      def _register(self, **overrides):
          data = {
              'first_name': 'Test',
              'last_name': 'User',
              'email': 'testuser@example.com',
              'username': 'testuser_gender',
              'password': 'SuperSecret123!',
              'access_code': self.VALID_ACCESS_CODE,
              'gender': 'female',
          }
          data.update(overrides)
          return self.client.post(
              reverse('register'), data,
              HTTP_X_REQUESTED_WITH='XMLHttpRequest',
          )

      def test_register_creates_userprofile_with_gender(self):
          r = self._register()
          data = r.json()
          self.assertTrue(data['success'])
          user = User.objects.get(username='testuser_gender')
          self.assertEqual(user.profile.gender, 'female')

      def test_register_without_gender_defaults_to_blank(self):
          r = self._register(gender='', username='testuser_nogender')
          data = r.json()
          self.assertTrue(data['success'])
          user = User.objects.get(username='testuser_nogender')
          self.assertEqual(user.profile.gender, '')
  ```

  If `TEACHER_ACCESS_CODE_HASHED` env var happens to be set in the actual test-running environment (overriding the plaintext default above), these two tests will fail on the access-code check rather than the gender logic — if that happens, use `unittest.mock.patch('accounts.views.TEACHER_ACCESS_CODE_HASHED', make_password(self.VALID_ACCESS_CODE))` (patching the name in `accounts.views`, not `django.conf.settings`, since that's the actual module-level name the view reads) instead of relying on the environment default.

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.RegisterAccountGenderTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `user.profile` raises `RelatedObjectDoesNotExist` since `registerAccount` doesn't create a `UserProfile` yet.

- [ ] **Step 3: Add gender handling to `registerAccount`**

  In `Trust-AI-Platform/accounts/views.py`, replace:
  ```python
              first_name = request.POST.get('first_name')
              last_name = request.POST.get('last_name')
              email = request.POST.get('email')
              username = request.POST.get('username')
              password = request.POST.get('password')
              access_code = request.POST.get('access_code')  # New field for access code
  ```
  with:
  ```python
              first_name = request.POST.get('first_name')
              last_name = request.POST.get('last_name')
              email = request.POST.get('email')
              username = request.POST.get('username')
              password = request.POST.get('password')
              access_code = request.POST.get('access_code')  # New field for access code
              gender = request.POST.get('gender', '').strip()
  ```

  Then replace:
  ```python
              # Create a new user with the assigned role
              user = User.objects.create(username=username, first_name=first_name, last_name=last_name, email=email, password=hashed_password)

              # Add user to "Teacher" group
              teacher_group = Group.objects.get_or_create(name="teachers")[0]
              user.groups.add(teacher_group)
  ```
  with:
  ```python
              # Create a new user with the assigned role
              user = User.objects.create(username=username, first_name=first_name, last_name=last_name, email=email, password=hashed_password)
              UserProfile.objects.create(user=user, gender=gender)

              # Add user to "Teacher" group
              teacher_group = Group.objects.get_or_create(name="teachers")[0]
              user.groups.add(teacher_group)
  ```

  `UserProfile` is already imported at the top of this file (`from .models import UserProfile, COUNTRY_CHOICES`, `accounts/views.py:5`) — no new import needed.

- [ ] **Step 4: Add the gender field to the registration form**

  In `Trust-AI-Platform/accounts/templates/accounts/register.html`, replace:
  ```html
                    <div class="col-md-6">
                      <label for="yourLastName" class="form-label">Last Name</label>
                      <input type="text" name="last_name" class="form-control" id="yourLastName" required>
                    </div>

                    <div class="col-12">
                      <label for="yourEmail" class="form-label">Email</label>
  ```
  with:
  ```html
                    <div class="col-md-6">
                      <label for="yourLastName" class="form-label">Last Name</label>
                      <input type="text" name="last_name" class="form-control" id="yourLastName" required>
                    </div>

                    <div class="col-12">
                      <label for="gender" class="form-label">Gender</label>
                      <select class="form-select" name="gender" id="gender">
                        <option value="">Prefer not to say</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                      </select>
                    </div>

                    <div class="col-12">
                      <label for="yourEmail" class="form-label">Email</label>
  ```

  No JS changes needed — the form already submits via `new FormData(this)` (`register.html`, submit handler), which automatically includes the new named field.

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.RegisterAccountGenderTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (2 tests).

  Then run the full accounts suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add Trust-AI-Platform/accounts/views.py Trust-AI-Platform/accounts/templates/accounts/register.html Trust-AI-Platform/accounts/tests.py
  git commit -m "Add gender field to self-registration"
  ```

---

### Task 3: Gender + picture upload on the profile edit page

**Files:**
- Modify: `Trust-AI-Platform/accounts/views.py`
- Modify: `Trust-AI-Platform/accounts/templates/accounts/profile.html`
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Consumes: `UserProfile.gender`/`.picture` (Task 1), `avatar_url` filter (Task 1)
- Produces: no new interfaces — `update_info` now also accepts `gender` and `picture`.

- [ ] **Step 1: Write the failing tests**

  Append to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class ProfileEditGenderAvatarTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.user = User.objects.create_user(
              'profile_edit_user', password='pass', first_name='Pat', last_name='Doe',
              email='pat@example.com',
          )
          self.user.groups.add(teachers)
          self.client.login(username='profile_edit_user', password='pass')

      def _update_info(self, **overrides):
          data = {
              'action': 'update_info',
              'first_name': 'Pat',
              'last_name': 'Doe',
              'email': 'pat@example.com',
              'country': '',
              'institution': '',
              'bio': '',
              'gender': 'male',
          }
          data.update(overrides)
          return self.client.post(
              reverse('profile'), data,
              HTTP_X_REQUESTED_WITH='XMLHttpRequest',
          )

      def test_update_info_sets_gender(self):
          r = self._update_info(gender='female')
          self.assertTrue(r.json()['success'])
          profile = UserProfile.objects.get(user=self.user)
          self.assertEqual(profile.gender, 'female')

      def test_update_info_uploads_picture(self):
          from django.core.files.uploadedfile import SimpleUploadedFile
          tiny_gif = (
              b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
              b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
          )
          upload = SimpleUploadedFile('avatar.gif', tiny_gif, content_type='image/gif')
          r = self._update_info(picture=upload)
          self.assertTrue(r.json()['success'])
          profile = UserProfile.objects.get(user=self.user)
          self.assertTrue(profile.picture.name)

      def test_update_info_without_new_picture_keeps_existing(self):
          from django.core.files.uploadedfile import SimpleUploadedFile
          tiny_gif = (
              b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
              b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
          )
          profile = UserProfile.objects.create(
              user=self.user,
              picture=SimpleUploadedFile('original.gif', tiny_gif, content_type='image/gif'),
          )
          original_name = profile.picture.name
          r = self._update_info()
          self.assertTrue(r.json()['success'])
          profile.refresh_from_db()
          self.assertEqual(profile.picture.name, original_name)

      def test_profile_page_renders_avatar_img_when_picture_set(self):
          from django.core.files.uploadedfile import SimpleUploadedFile
          tiny_gif = (
              b'GIF87a\x01\x00\x01\x00\x80\x01\x00\x00\x00\x00ccc,\x00'
              b'\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
          )
          UserProfile.objects.create(
              user=self.user,
              picture=SimpleUploadedFile('portrait.gif', tiny_gif, content_type='image/gif'),
          )
          r = self.client.get(reverse('profile'))
          self.assertContains(r, '<img src="/media/profile_pictures/portrait')

      def test_profile_page_falls_back_to_icon_without_picture_or_gender(self):
          UserProfile.objects.create(user=self.user)
          r = self.client.get(reverse('profile'))
          self.assertContains(r, 'bi-person-circle')
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.ProfileEditGenderAvatarTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: FAIL — `gender`/`picture` aren't read or saved yet, and `profile.html` doesn't render an `<img>` yet.

- [ ] **Step 3: Add gender + picture handling to `update_info`**

  In `Trust-AI-Platform/accounts/views.py`, replace:
  ```python
          if action == 'update_info':
              first_name  = request.POST.get('first_name', '').strip()
              last_name   = request.POST.get('last_name', '').strip()
              email       = request.POST.get('email', '').strip()
              country     = request.POST.get('country', '').strip()
              institution = request.POST.get('institution', '').strip()
              bio         = request.POST.get('bio', '').strip()

              errors = {}
              if not first_name:
                  errors['first_name'] = 'First name is required.'
              if not last_name:
                  errors['last_name'] = 'Last name is required.'
              if not email:
                  errors['email'] = 'Email is required.'
              elif User.objects.filter(email=email).exclude(pk=user.pk).exists():
                  errors['email'] = 'This email is already in use by another account.'
              if len(bio) > 500:
                  errors['bio'] = 'Bio must be 500 characters or fewer.'

              if errors:
                  return JsonResponse({'success': False, 'errors': errors})

              user.first_name = first_name
              user.last_name  = last_name
              user.email      = email
              user.save(update_fields=['first_name', 'last_name', 'email'])

              profile, _ = UserProfile.objects.get_or_create(user=user)
              profile.country     = country
              profile.institution = institution
              profile.bio         = bio
              profile.save(update_fields=['country', 'institution', 'bio'])

              return JsonResponse({'success': True, 'message': 'Profile updated successfully.',
                                   'country': country, 'institution': institution})
  ```
  with:
  ```python
          if action == 'update_info':
              first_name  = request.POST.get('first_name', '').strip()
              last_name   = request.POST.get('last_name', '').strip()
              email       = request.POST.get('email', '').strip()
              country     = request.POST.get('country', '').strip()
              institution = request.POST.get('institution', '').strip()
              bio         = request.POST.get('bio', '').strip()
              gender      = request.POST.get('gender', '').strip()
              picture     = request.FILES.get('picture')

              errors = {}
              if not first_name:
                  errors['first_name'] = 'First name is required.'
              if not last_name:
                  errors['last_name'] = 'Last name is required.'
              if not email:
                  errors['email'] = 'Email is required.'
              elif User.objects.filter(email=email).exclude(pk=user.pk).exists():
                  errors['email'] = 'This email is already in use by another account.'
              if len(bio) > 500:
                  errors['bio'] = 'Bio must be 500 characters or fewer.'

              if errors:
                  return JsonResponse({'success': False, 'errors': errors})

              user.first_name = first_name
              user.last_name  = last_name
              user.email      = email
              user.save(update_fields=['first_name', 'last_name', 'email'])

              profile, _ = UserProfile.objects.get_or_create(user=user)
              profile.country     = country
              profile.institution = institution
              profile.bio         = bio
              profile.gender      = gender
              update_fields = ['country', 'institution', 'bio', 'gender']
              if picture:
                  profile.picture = picture
                  update_fields.append('picture')
              profile.save(update_fields=update_fields)

              return JsonResponse({'success': True, 'message': 'Profile updated successfully.',
                                   'country': country, 'institution': institution})
  ```

- [ ] **Step 4: Real avatar in the profile page's left card**

  In `Trust-AI-Platform/accounts/templates/accounts/profile.html`, replace:
  ```html
              <div class="profile-avatar mb-3">
                <i class="bi bi-person-circle" style="font-size: 5rem; color: #012970;"></i>
              </div>
  ```
  with:
  ```html
              {% load profile_tags %}
              <div class="profile-avatar mb-3">
                {% with avatar=profile_user|avatar_url %}
                {% if avatar %}
                <img src="{{ avatar }}" alt="" style="width:5rem;height:5rem;border-radius:50%;object-fit:cover;">
                {% else %}
                <i class="bi bi-person-circle" style="font-size: 5rem; color: #012970;"></i>
                {% endif %}
                {% endwith %}
              </div>
  ```

- [ ] **Step 5: Add gender select and picture upload to the edit form**

  In `Trust-AI-Platform/accounts/templates/accounts/profile.html`, replace:
  ```html
                  <div class="row mb-3">
                    <label class="col-md-4 col-form-label">Country</label>
                    <div class="col-md-8">
                      <select class="form-select" name="country" id="country">
                        {% for value, label in country_choices %}
                        <option value="{{ value }}" {% if profile.country == value %}selected{% endif %}>{{ label }}</option>
                        {% endfor %}
                      </select>
                    </div>
                  </div>
  ```
  with:
  ```html
                  <div class="row mb-3">
                    <label class="col-md-4 col-form-label">Country</label>
                    <div class="col-md-8">
                      <select class="form-select" name="country" id="country">
                        {% for value, label in country_choices %}
                        <option value="{{ value }}" {% if profile.country == value %}selected{% endif %}>{{ label }}</option>
                        {% endfor %}
                      </select>
                    </div>
                  </div>

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

  No JS/submission changes needed — `infoForm`'s existing submit handler already builds `new FormData(this)` and posts it via `fetch(url, {..., body: data})`, which is multipart-capable as-is; the new `<input type="file">` is picked up automatically.

- [ ] **Step 6: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.ProfileEditGenderAvatarTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (5 tests).

  Then the full accounts suite:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 7: Manually verify (if a real dev environment is available)**

  Same caveat as prior features on this branch — needs a live Postgres-backed dev environment. If available:
  ```bash
  python manage.py runserver
  ```
  Edit your profile, select a gender, save, confirm the correct default avatar (`profile_d_man.webp` or `profile_d_woman.jpg`) appears in the left card. Upload a custom picture, save, confirm it now shows instead of the gender default. Check at 375px/768px viewport widths that the new form rows and avatar don't cause any layout overflow.

  If unavailable, Step 6's automated tests are the load-bearing verification.

- [ ] **Step 8: Commit**

  ```bash
  git add Trust-AI-Platform/accounts/views.py Trust-AI-Platform/accounts/templates/accounts/profile.html Trust-AI-Platform/accounts/tests.py
  git commit -m "Add gender and picture upload to profile edit page"
  ```

---

### Task 4: Wire the avatar filter into the remaining render sites

**Files:**
- Modify: `Trust-AI-Platform/templates/head.html`
- Modify: `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`
- Modify: `Trust-AI-Platform/accounts/tests.py`

**Interfaces:**
- Consumes: `avatar_url` filter (Task 1)
- Produces: no new interfaces — pure template wiring, no view/model changes.

- [ ] **Step 1: Nav dropdown avatar (top-right header)**

  In `Trust-AI-Platform/templates/head.html`, replace:
  ```html
            <a class="nav-link nav-profile d-flex align-items-center pe-0 gap-2" href="#" data-bs-toggle="dropdown">
              <span class="nav-profile-avatar">{{ user.get_full_name|default:user.username|slice:":1"|upper }}</span>
              <span class="d-none d-md-block dropdown-toggle nav-profile-name">{{ user.get_full_name|default:user.username }}</span>
            </a><!-- End Profile Image Icon -->
  ```
  with:
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
            </a><!-- End Profile Image Icon -->
  ```
  (`.nav-profile-avatar` is 36×36px per `static/css/style.css:493-497` — the `<img>` matches that exact size.)

- [ ] **Step 2: Dropdown header avatar**

  In the same file, replace:
  ```html
            <li class="dropdown-header">
              <div class="profile-avatar-initial">{{ user.get_full_name|default:user.username|slice:":1"|upper }}</div>
              <h6>{{ user.get_full_name|default:user.username }}</h6>
              <span class="profile-role-badge"><span class="profile-role-dot"></span>Active</span>
            </li>
  ```
  with:
  ```html
            <li class="dropdown-header">
              {% with avatar=user|avatar_url %}
              {% if avatar %}
              <img src="{{ avatar }}" alt="" style="width:48px;height:48px;border-radius:50%;object-fit:cover;">
              {% else %}
              <div class="profile-avatar-initial">{{ user.get_full_name|default:user.username|slice:":1"|upper }}</div>
              {% endif %}
              {% endwith %}
              <h6>{{ user.get_full_name|default:user.username }}</h6>
              <span class="profile-role-badge"><span class="profile-role-dot"></span>Active</span>
            </li>
  ```
  (`.profile-avatar-initial` is 48×48px per `static/css/style.css:612-616`.)

- [ ] **Step 3: Admin dashboard user list avatar**

  In `Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html`, replace:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — User Management</title>{% endblock %}
  {% load static %}
  {% block atcontent %}
  ```
  with:
  ```html
  {% extends "main.html" %}
  {% block page_title %}<title>Trust AI Lab — User Management</title>{% endblock %}
  {% load static %}
  {% load profile_tags %}
  {% block atcontent %}
  ```

  Then replace:
  ```html
                      <div class="user-avatar">{{ u.first_name|default:u.username|slice:":1"|upper }}{{ u.last_name|slice:":1"|upper }}</div>
  ```
  with:
  ```html
                      {% with avatar=u|avatar_url %}
                      {% if avatar %}
                      <img src="{{ avatar }}" alt="" style="width:36px;height:36px;border-radius:50%;object-fit:cover;">
                      {% else %}
                      <div class="user-avatar">{{ u.first_name|default:u.username|slice:":1"|upper }}{{ u.last_name|slice:":1"|upper }}</div>
                      {% endif %}
                      {% endwith %}
  ```
  (`.user-avatar` in this file is 36×36px per `accounts/templates/accounts/admin_dashboard.html:27-31`.)

- [ ] **Step 4: Write tests for the template wiring**

  Append to `Trust-AI-Platform/accounts/tests.py`:
  ```python
  class AvatarRenderSitesTests(TestCase):
      def setUp(self):
          self.client = Client()
          teachers, _ = Group.objects.get_or_create(name='teachers')
          self.staff = User.objects.create_user('avatar_sites_staff', password='pass', is_staff=True)
          self.staff.groups.add(teachers)
          UserProfile.objects.create(user=self.staff, gender='male')
          self.client.login(username='avatar_sites_staff', password='pass')

      def test_head_nav_renders_gender_default_avatar_img(self):
          r = self.client.get(reverse('profile'))
          self.assertContains(r, '/static/img/profile_d_man.webp')

      def test_admin_dashboard_renders_gender_default_avatar_img(self):
          r = self.client.get(reverse('admin_dashboard'))
          self.assertContains(r, '/static/img/profile_d_man.webp')

      def test_user_without_gender_or_picture_falls_back_to_initials_in_nav(self):
          plain_user = User.objects.create_user('avatar_plain', password='pass')
          plain_user.groups.add(Group.objects.get(name='teachers'))
          self.client.logout()
          self.client.login(username='avatar_plain', password='pass')
          r = self.client.get(reverse('profile'))
          self.assertNotContains(r, '/static/img/profile_d_')
          self.assertContains(r, 'nav-profile-avatar')
  ```

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts.tests.AvatarRenderSitesTests -v 2 --settings=faithDev.settings_test
  ```
  Expected: `OK` (3 tests).

  Then run the full accounts suite one more time:
  ```bash
  "../djangofaithvenv/Scripts/python.exe" manage.py test accounts -v 2 --settings=faithDev.settings_test
  ```

- [ ] **Step 6: Manually verify (if a real dev environment is available)**

  Same caveat as Task 3. If available, log in as a user with a gender set and confirm the nav dropdown avatar (top-right) and the admin dashboard user list (if staff) both show the correct gender-default image, and that a user with no gender/picture set still shows initials/icon as before. Check the nav dropdown avatar doesn't visually break the header layout at 375px width (the header already has a mobile burger-collapse behavior — confirm the new `<img>` doesn't interfere with it).

  If unavailable, Step 5's automated tests are the load-bearing verification.

- [ ] **Step 7: Commit**

  ```bash
  git add Trust-AI-Platform/templates/head.html Trust-AI-Platform/accounts/templates/accounts/admin_dashboard.html Trust-AI-Platform/accounts/tests.py
  git commit -m "Wire avatar_url filter into nav dropdown and admin dashboard"
  ```
