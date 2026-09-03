from django.utils import timezone

from .models import MaintenanceNotice


def maintenance_notices(request):
    """Expose every currently active maintenance notice to rendered pages."""
    notices = list(
        MaintenanceNotice.objects.active(timezone.now())
        .only('id', 'reason', 'starts_at', 'ends_at')
        .order_by('ends_at', 'id')
    )
    return {'active_maintenance_notices': notices}
