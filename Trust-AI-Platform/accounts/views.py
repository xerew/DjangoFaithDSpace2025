from django.shortcuts import render, get_object_or_404
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib.auth.models import User
from .models import UserProfile, COUNTRY_CHOICES
from django.contrib.auth.hashers import make_password, check_password
from django.http import JsonResponse
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.views import PasswordResetView
from django.conf import settings
from faithDev.settings import TEACHER_ACCESS_CODE_HASHED
from django.contrib.auth.models import Group
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from functools import wraps

# Create your views here.
class CustomPasswordResetView(PasswordResetView):
    email_template_name = 'registration/password_reset_email.html'  # Plain text fallback
    html_email_template_name = 'registration/password_reset_email.html'  # HTML version of the email

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
    
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember')  # Checkbox input

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            
            if remember_me:
                request.session.set_expiry(1209600)  # 2 weeks
            else:
                request.session.set_expiry(0)  # Browser close

            if user.groups.filter(name='teachers').exists():
                return redirect('teacher_home')  # Redirect to the teachers home page
            else:
                return redirect('studentScenarios')  # Redirect to the default page
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('login')  # Redirect back to the login page with an error message

    return render(request, 'accounts/login.html')  # Render the login page for GET requests


@login_required
def userData(request):
    # Access the logged-in user from the request object
    user = request.user

    # Check if the user is authenticated
    if user.is_authenticated:
        # Now you can use user's attributes
        username = user.username
        email = user.email
        first_name = user.first_name
        last_name = user.last_name
        # etc.

        # You can pass user information to your template
        context = {'username': username, 'email': email, 'first_name': first_name, 'last_name': last_name}
        return render(request, 'header.html', context)
    
def logoutView(request):
    # Perform any pre-logout actions here

    # Logout the user
    logout(request)

    # Redirect to desired page after logout
    return redirect('/login/')

def registerAccount(request):
    print("Request received:", request.method)
    if request.method == 'POST':
        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            print("Handling AJAX request")
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            email = request.POST.get('email')
            username = request.POST.get('username')
            password = request.POST.get('password')
            access_code = request.POST.get('access_code')  # New field for access code

            errors = {}
            if User.objects.filter(username=username).exists():
                errors['username'] = 'Username already taken.'

            if User.objects.filter(email=email).exists():
                errors['email'] = 'Email already in use.'
            
            # Validate password
            try:
                validate_password(password)
            except ValidationError as e:
                errors['password'] = list(e.messages)

            # Validate access code for Teacher role
            if access_code:
                if not check_password(access_code, TEACHER_ACCESS_CODE_HASHED):
                    errors['access_code'] = 'Invalid access code for Teacher registration.'
            else:
                errors['access_code'] = 'Access code is required for Teacher registration.'

            if errors:
                return JsonResponse({'success': False, 'errors': errors})

            # Hash the password
            hashed_password = make_password(password)

            # Create a new user with the assigned role
            user = User.objects.create(username=username, first_name=first_name, last_name=last_name, email=email, password=hashed_password)

            # Add user to "Teacher" group
            teacher_group = Group.objects.get_or_create(name="teachers")[0]
            user.groups.add(teacher_group)

            messages.success(request, 'Account created successfully. Please log in.')

            return JsonResponse({'success': True})

        else:
            return JsonResponse({'success': False, 'error': 'AJAX required'}, status=400)

    # For GET requests, render the registration form
    return render(request, 'accounts/register.html')

def view404(request, exception):
    return render(request, '404.html', {}, status=404)

@group_required('teachers')
def documentation_view(request):
    is_dspace_partner = request.user.groups.filter(name="dspace_partners").exists()
    return render(request, 'accounts/documentation.html', {'is_dspace_partner': is_dspace_partner})

def tos_view(request):
    if request.user.is_authenticated:
        return render(request, 'accounts/tos.html')
    else:
        return render(request, 'accounts/tos_public.html')

@group_required('teachers')
def profile_view(request):
    from organization.models import Organization

    user = request.user

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        action = request.POST.get('action')

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

        elif action == 'change_password':
            current_password = request.POST.get('current_password', '')
            new_password = request.POST.get('new_password', '')
            confirm_password = request.POST.get('confirm_password', '')

            errors = {}
            if not user.check_password(current_password):
                errors['current_password'] = 'Current password is incorrect.'
            if new_password != confirm_password:
                errors['confirm_password'] = 'Passwords do not match.'
            if new_password:
                try:
                    validate_password(new_password, user)
                except ValidationError as e:
                    errors['new_password'] = list(e.messages)

            if errors:
                return JsonResponse({'success': False, 'errors': errors})

            user.set_password(new_password)
            user.save()
            # Re-authenticate to keep the session alive
            from django.contrib.auth import update_session_auth_hash
            update_session_auth_hash(request, user)
            return JsonResponse({'success': True, 'message': 'Password changed successfully.'})

        return JsonResponse({'success': False, 'error': 'Unknown action.'}, status=400)

    admin_orgs = Organization.objects.filter(admins=user).values('id', 'name', 'short_name', 'country')
    member_orgs = Organization.objects.filter(members=user).exclude(admins=user).values('id', 'name', 'short_name', 'country')

    # Build role badges
    BADGE_MAP = {
        'teachers':       ('Teacher',        'primary'),
        'dspace_partners':('DSpace Partner',  'info'),
    }
    roles = []
    if user.is_superuser:
        roles.append({'label': 'Superuser', 'color': 'danger'})
    if user.is_staff and not user.is_superuser:
        roles.append({'label': 'Admin', 'color': 'warning'})
    for group in user.groups.all():
        badge = BADGE_MAP.get(group.name, (group.name.replace('_', ' ').title(), 'secondary'))
        roles.append({'label': badge[0], 'color': badge[1]})

    profile, _ = UserProfile.objects.get_or_create(user=user)

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