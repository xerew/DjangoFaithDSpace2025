"""Encrypt the initial passwords that teachers download for student groups.

Stored values look like ``fernet:<token>``. Values saved before encryption
existed are plaintext and still read correctly until the
``reencrypt_student_credentials`` command encrypts them.

The key is ``STUDENT_CREDENTIALS_KEY`` (a Fernet key). Without it a key
derived from SECRET_KEY is used, and a startup warning says so: values
encrypted that way become unreadable if SECRET_KEY changes, so set the key
and run the command before rotating SECRET_KEY.
"""
import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from django.conf import settings

ENCRYPTED_PREFIX = 'fernet:'

logger = logging.getLogger(__name__)


def _derived_key(secret_key=None):
    digest = hashlib.sha256(
        ('student-credentials:' + (secret_key or settings.SECRET_KEY)).encode()
    ).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet():
    """The configured key encrypts; it and the derived key both decrypt."""
    keys = []
    configured = getattr(settings, 'STUDENT_CREDENTIALS_KEY', '')
    if configured:
        keys.append(Fernet(configured))
    keys.append(Fernet(_derived_key()))
    return MultiFernet(keys)


def encrypt_initial_password(plain):
    if not plain:
        return ''
    return ENCRYPTED_PREFIX + _fernet().encrypt(plain.encode()).decode()


def reveal_initial_password(stored):
    """Return the readable password (legacy plaintext is returned as is)."""
    if not stored:
        return ''
    if not stored.startswith(ENCRYPTED_PREFIX):
        return stored
    try:
        return _fernet().decrypt(stored[len(ENCRYPTED_PREFIX):].encode()).decode()
    except InvalidToken:
        logger.error('An initial student password could not be decrypted; '
                     'was STUDENT_CREDENTIALS_KEY or SECRET_KEY changed?')
        return ''


def reencrypt(stored):
    """Encrypt plaintext, or re-encrypt with the current key. Returns the new value."""
    if not stored:
        return stored
    if not stored.startswith(ENCRYPTED_PREFIX):
        return encrypt_initial_password(stored)
    token = stored[len(ENCRYPTED_PREFIX):].encode()
    return ENCRYPTED_PREFIX + _fernet().rotate(token).decode()


def old_key_fernet(secret_keys=(), credential_keys=()):
    """Decrypt with keys that are no longer configured, to recover old values.

    credential_keys are former STUDENT_CREDENTIALS_KEY values; secret_keys are
    former SECRET_KEY values, used when no credentials key was set. A malformed
    credentials key raises ValueError.
    """
    keys = [Fernet(key) for key in credential_keys]
    keys += [Fernet(_derived_key(secret)) for secret in secret_keys]
    return MultiFernet(keys)


def recover(stored, old_fernet):
    """Return (state, value) for one stored password.

    state is 'readable' (the current keys read it; value unchanged),
    'recovered' (only an old key reads it; value re-encrypted with the current
    key) or 'unreadable' (no key reads it; value unchanged).
    """
    if not stored or not stored.startswith(ENCRYPTED_PREFIX):
        return 'readable', stored
    token = stored[len(ENCRYPTED_PREFIX):].encode()
    try:
        _fernet().decrypt(token)
        return 'readable', stored
    except InvalidToken:
        pass
    try:
        plain = old_fernet.decrypt(token).decode()
    except InvalidToken:
        return 'unreadable', stored
    return 'recovered', encrypt_initial_password(plain)
