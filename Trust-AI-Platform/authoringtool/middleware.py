from django.contrib import messages
from django.urls import reverse
from django.utils import timezone
from django.utils.html import format_html
from django.utils.translation import gettext as _

from .models import ScenarioTranslationJob


class MachineTranslationNoticeMiddleware:
    """Tell the requester, on their next page, that a translation finished."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, 'user', None)
        if (
            request.method == 'GET'
            and user is not None
            and user.is_authenticated
            and request.headers.get('x-requested-with') != 'XMLHttpRequest'
        ):
            self.add_notices(request, user)
        return self.get_response(request)

    def add_notices(self, request, user):
        jobs = list(
            ScenarioTranslationJob.objects.filter(
                requested_by=user,
                notified_at__isnull=True,
                status__in=[
                    ScenarioTranslationJob.STATUS_DONE,
                    ScenarioTranslationJob.STATUS_FAILED,
                ],
            ).select_related('target_scenario')
        )
        for job in jobs:
            name = job.target_scenario.name
            if job.status == ScenarioTranslationJob.STATUS_DONE:
                messages.success(request, format_html(
                    _('Your {language} translation of <a href="{url}">{name}</a> is ready.'),
                    language=job.target_language,
                    url=reverse('viewScenario', args=[job.target_scenario_id]),
                    name=name,
                ))
            else:
                messages.error(
                    request,
                    _('Your %(language)s translation of %(name)s failed. Open your scenarios to retry.')
                    % {'language': job.target_language, 'name': name},
                )
        if jobs:
            ScenarioTranslationJob.objects.filter(
                pk__in=[job.pk for job in jobs],
            ).update(notified_at=timezone.now())
