from django.db import transaction
from django.db.models import ProtectedError


def remove_student_account(user):
    """Delete an account, or deactivate it if it holds student work.

    Student evidence protects its user at the database level, so a delete
    that would lose answers, scores or attempts is refused; the account is
    then deactivated (cannot log in) and everything is kept.
    Returns 'deleted' or 'deactivated'.
    """
    try:
        with transaction.atomic():
            user.delete()
        return 'deleted'
    except ProtectedError:
        user.is_active = False
        user.save(update_fields=['is_active'])
        return 'deactivated'
