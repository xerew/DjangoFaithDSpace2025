"""Which NTUA address the Ollama SSH tunnel uses.

The `ollama-tunnel` container (deploy/ollama-tunnel) and the web container
share a small folder, OLLAMA_TUNNEL_CONTROL_DIR:

- `status`, written by the tunnel every few seconds as `key=value` lines:
  state (connecting / connected / waiting), host, mode, hosts (the addresses
  the server allows, from OLLAMA_SSH_HOSTS), since, updated (Unix seconds)
  and message (the last SSH error).
- `target`, written here: the first line is `auto` or one of those
  addresses; the next lines say who chose it and when. The tunnel reads only
  the first line, and ignores an address it does not list itself.
"""

import os
import tempfile
from datetime import datetime, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.utils import timezone

AUTO = 'auto'
# The tunnel rewrites its status every few seconds; older than this means
# the container is stopped or stuck.
STALE_AFTER_SECONDS = 60


def control_dir():
    return Path(settings.OLLAMA_TUNNEL_CONTROL_DIR)


def _read_pairs(path):
    try:
        text = path.read_text(encoding='utf-8')
    except (FileNotFoundError, OSError):
        return {}
    pairs = {}
    for line in text.splitlines():
        key, sep, value = line.partition('=')
        if sep:
            pairs[key.strip()] = value.strip()
    return pairs


def _when(value):
    try:
        return datetime.fromtimestamp(int(value), tz=dt_timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def read_status(now=None):
    """What the tunnel last reported, and whether that report is recent."""
    pairs = _read_pairs(control_dir() / 'status')
    now = now or timezone.now()
    updated = _when(pairs.get('updated'))
    return {
        'reporting': bool(
            updated
            and (now - updated).total_seconds() <= STALE_AFTER_SECONDS
        ),
        'state': pairs.get('state', ''),
        'host': pairs.get('host', ''),
        'mode': pairs.get('mode', ''),
        'hosts': [
            host.strip()
            for host in pairs.get('hosts', '').split(',')
            if host.strip()
        ],
        'since': _when(pairs.get('since')),
        'updated': updated,
        'message': pairs.get('message', ''),
    }


def read_target():
    """The address the admin chose (`auto` when nothing was chosen)."""
    try:
        lines = (control_dir() / 'target').read_text(
            encoding='utf-8'
        ).splitlines()
    except (FileNotFoundError, OSError):
        lines = []
    target = lines[0].strip() if lines and lines[0].strip() else AUTO
    details = {}
    for line in lines[1:]:
        key, sep, value = line.partition('=')
        if sep:
            details[key.strip()] = value.strip()
    return {
        'target': target,
        'changed_by': details.get('by', ''),
        'changed_at': _when(details.get('at')),
    }


def _one_line(value):
    return ' '.join(str(value).split())


def set_target(target, username):
    """Save the admin's choice for the tunnel. Raises ValueError for an
    address the tunnel does not list as allowed."""
    target = (target or '').strip()
    if target != AUTO and target not in read_status()['hosts']:
        raise ValueError(f'{target!r} is not an allowed address')
    folder = control_dir()
    folder.mkdir(parents=True, exist_ok=True)
    content = (
        f'{target}\n'
        f'by={_one_line(username)}\n'
        f'at={int(timezone.now().timestamp())}\n'
    )
    # Write then rename, so the tunnel never reads a half-written file.
    handle, temporary = tempfile.mkstemp(dir=folder, prefix='.target-')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as fh:
            fh.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, folder / 'target')
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
