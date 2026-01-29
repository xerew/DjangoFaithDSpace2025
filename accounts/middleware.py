from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages

class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        excluded_paths = ['/accounts/', '/static/', '/media/']
        
        if not request.user.is_authenticated and not any(request.path.startswith(path) for path in excluded_paths):
            # Redirect to the login page if the user is not authenticated
            messages.error(request, 'You need to login in order to access the pages')
            return redirect(reverse('login'))  # Adjust the URL name if needed
        return self.get_response(request)
