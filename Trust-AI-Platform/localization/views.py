from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from .languages import supported_codes


@require_POST
def switch_language(request):
    code = request.POST.get('language', '')
    if code not in supported_codes():
        return HttpResponseBadRequest('Unknown language')

    target = request.POST.get('next', '')
    if not url_has_allowed_host_and_scheme(
            target, allowed_hosts={request.get_host()},
            require_https=request.is_secure()):
        target = '/'
    response = redirect(target)
    response.set_cookie(
        settings.LANGUAGE_COOKIE_NAME, code,
        max_age=settings.LANGUAGE_COOKIE_AGE,
        path=settings.LANGUAGE_COOKIE_PATH,
        domain=settings.LANGUAGE_COOKIE_DOMAIN,
        secure=settings.LANGUAGE_COOKIE_SECURE or None,
        httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
        samesite=settings.LANGUAGE_COOKIE_SAMESITE,
    )

    if request.user.is_authenticated:
        from accounts.models import UserProfile
        profile, _created = UserProfile.objects.get_or_create(user=request.user)
        profile.language = code
        profile.save(update_fields=['language'])
    return response
