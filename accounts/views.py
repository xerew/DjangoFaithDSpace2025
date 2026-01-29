from django.shortcuts import render
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib.auth.models import User
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
                return redirect('index')  # Redirect to the teachers dashboard
            else:
                return redirect('studentScenarios')  # Redirect to the default page
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('login')  # Redirect back to the login page with an error message

    return render(request, 'accounts/login.html')  # Render the login page for GET requests


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
            teacher_group = Group.objects.get(name="teachers")
            user.groups.add(teacher_group)

            messages.success(request, 'Account created successfully. Please log in.')

            return JsonResponse({'success': True})

    # For non-AJAX POST requests or GET requests, render the registration form
    return render(request, 'accounts/register.html')

def view404(request, exception):
    return render(request, '404.html', {}, status=404)

@group_required('teachers')
def documentation_view(request):
    is_dspace_partner = request.user.groups.filter(name="dspace_partners").exists()
    return render(request, 'accounts/documentation.html', {'is_dspace_partner': is_dspace_partner})

def tos_view(request):
    return render(request, 'accounts/tos.html')

def tos_view(request):
    if request.user.is_authenticated:
        return render(request, 'accounts/tos.html')
    else:
        return render(request, 'accounts/tos_public.html')