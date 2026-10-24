from django.conf import settings
from django.utils import translation

from .languages import supported_codes


class UserLanguageMiddleware:
    """Apply a signed-in user's saved language; remember it on first visit."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        saved_language = None
        user = getattr(request, 'user', None)
        if user is not None and user.is_authenticated:
            from accounts.models import UserProfile
            profile = UserProfile.objects.filter(user=user).first()
            saved = profile.language if profile else ''
            if saved in supported_codes():
                saved_language = saved
                translation.activate(saved)
                request.LANGUAGE_CODE = saved
            else:
                current = translation.get_language() or settings.LANGUAGE_CODE
                current = current.split('-')[0]
                if current not in supported_codes():
                    current = 'en'
                if profile is None:
                    profile = UserProfile(user=user)
                profile.language = current
                profile.save()

        response = self.get_response(request)

        if saved_language:
            cookie = request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME)
            if cookie != saved_language:
                response.set_cookie(
                    settings.LANGUAGE_COOKIE_NAME, saved_language,
                    max_age=settings.LANGUAGE_COOKIE_AGE,
                    path=settings.LANGUAGE_COOKIE_PATH,
                    domain=settings.LANGUAGE_COOKIE_DOMAIN,
                    secure=settings.LANGUAGE_COOKIE_SECURE or None,
                    httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
                    samesite=settings.LANGUAGE_COOKIE_SAMESITE,
                )
        return response


class TranslationOverrideMiddleware:
    """Make reviewers' corrections for the request's language current in this process.

    Runs after UserLanguageMiddleware so the user's saved language is active.
    Costs one cache read per request; see localization.overrides.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from .overrides import ensure_overrides
        ensure_overrides(translation.get_language())
        return self.get_response(request)
