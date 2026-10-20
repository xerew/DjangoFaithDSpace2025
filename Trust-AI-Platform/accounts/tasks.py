"""Asynchronous tasks owned by the accounts application."""

from celery import shared_task

from .outbox import drain_outbox


@shared_task
def drain_email_outbox():
    """Run by Celery beat every 5 minutes; sends within the hourly cap."""
    return drain_outbox()
