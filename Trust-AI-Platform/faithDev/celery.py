from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set default Django settings module for Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faithDev.settings')

app = Celery('faithDev')

# Using a string here means the worker doesn't need to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Look for task modules in Django apps
app.autodiscover_tasks()