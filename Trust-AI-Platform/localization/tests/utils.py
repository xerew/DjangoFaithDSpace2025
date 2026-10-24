from django.core.mail.backends.locmem import EmailBackend
from django.utils import translation


class LanguageRecordingBackend(EmailBackend):
    """In-memory backend that notes the active language at send time."""

    def send_messages(self, messages):
        for message in messages:
            message.recorded_language = translation.get_language()
        return super().send_messages(messages)
