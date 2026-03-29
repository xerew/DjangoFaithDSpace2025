from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages

class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        excluded_paths = [
            '/accounts/', '/static/', '/media/', '/admin/',
        ]

        if not request.user.is_authenticated and not any(request.path.startswith(path) for path in excluded_paths):
            return redirect(reverse('login'))
        return self.get_response(request)
