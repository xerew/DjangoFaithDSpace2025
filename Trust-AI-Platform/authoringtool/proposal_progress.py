"""Progress of a scenario's proposal job, kept in the shared cache.

The Celery job writes it (JobProgress); the Metrics & AI page reads it by
scenario (job_status), so a reload or another teacher sees the same progress
and a second job cannot start while one runs (reserve).
"""
from datetime import datetime, timedelta

from django.core.cache import cache
from django.utils import timezone

# key, label while running, label when done, share of the job's time (sums to 100)
STAGES = [
    ('check', 'Checking the AI server', 'Checked the AI server', 2),
    ('pdfs', 'Reading the PDFs', 'Read the PDFs', 4),
    ('activities', 'Summarising activities', 'Summarised activities', 25),
    ('overview', 'Summarising the scenario and its phases', 'Summarised the scenario and its phases', 6),
    ('paths', 'Analysing learning paths', 'Analysed learning paths', 15),
    ('risk', 'Finding activities at risk', 'Found activities at risk', 4),
    ('proposals', 'Writing proposals', 'Wrote proposals', 40),
    ('saving', 'Saving the new proposals', 'Saved the new proposals', 4),
]
STAGE_KEYS = [key for key, *_ in STAGES]

# Kept long enough for any run; a silent job stops locking much sooner.
KEEP_SECONDS = 6 * 3600
STUCK_AFTER = timedelta(minutes=30)


def _key(scenario_id):
    return f'authoringtool:proposal-job:{scenario_id}'


def _new_record(started_by):
    now = timezone.now().isoformat()
    return {
        'task_id': None,
        'started_by': started_by or '',
        'started_at': now,
        'updated_at': now,
        'state': 'running',
        'stage': None,
        'done': 0,
        'total': 0,
        'detail': '',
        'summaries': {},
        'message': '',
    }


def _is_running(record, now):
    updated_at = datetime.fromisoformat(record['updated_at'])
    return record['state'] == 'running' and now - updated_at < STUCK_AFTER


def reserve(scenario_id, started_by):
    """Claim the scenario for a new job; False while another one runs."""
    record = cache.get(_key(scenario_id))
    if record and _is_running(record, timezone.now()):
        return False
    cache.set(_key(scenario_id), _new_record(started_by), KEEP_SECONDS)
    return True


def attach_task(scenario_id, task_id):
    record = cache.get(_key(scenario_id))
    if record:
        record['task_id'] = str(task_id)
        cache.set(_key(scenario_id), record, KEEP_SECONDS)


def job_status(scenario_id, now=None):
    """The job's record with 'running', 'percent' and 'stages', or None."""
    record = cache.get(_key(scenario_id))
    if not record:
        return None
    now = now or timezone.now()
    status = dict(record)
    status['running'] = _is_running(record, now)
    if record['state'] == 'running' and not status['running']:
        status['state'] = 'stuck'

    current = record['stage']
    current_index = STAGE_KEYS.index(current) if current in STAGE_KEYS else -1
    stages, percent = [], 0.0
    for index, (key, running_label, done_label, weight) in enumerate(STAGES):
        if record['state'] == 'done' or index < current_index:
            state = 'done'
            percent += weight
        elif index == current_index:
            state = {'running': 'active', 'failed': 'failed'}.get(status['state'], 'stopped')
            if record['total']:
                percent += weight * min(record['done'], record['total']) / record['total']
        else:
            state = 'pending'
        stages.append({
            'key': key,
            'label': running_label,
            'status': state,
            'summary': record['summaries'].get(key, done_label),
            'done': record['done'] if index == current_index else 0,
            'total': record['total'] if index == current_index else 0,
        })
    status['stages'] = stages
    status['percent'] = 100 if record['state'] == 'done' else int(percent)
    return status


class JobProgress:
    """What the running job reports; every call is written to the cache."""

    def __init__(self, scenario_id, started_by=''):
        self.scenario_id = scenario_id
        record = cache.get(_key(scenario_id))
        if record and record['state'] == 'running':
            self.record = record  # reserved by the view: keep its starter
        else:
            self.record = _new_record(started_by)
        self._save()

    def _save(self):
        self.record['updated_at'] = timezone.now().isoformat()
        cache.set(_key(self.scenario_id), self.record, KEEP_SECONDS)

    def _close_stage(self):
        key = self.record['stage']
        if not key or key in self.record['summaries']:
            return
        done_label = STAGES[STAGE_KEYS.index(key)][2]
        total = self.record['total']
        if self._counted and not total:
            text = f'{done_label} (already up to date)'
        elif total:
            text = f"{done_label} ({min(self.record['done'], total)} of {total})"
        else:
            text = done_label
        self.record['summaries'][key] = text

    def stage(self, key, total=None):
        """Start a stage. Pass total for one that counts items (0: nothing to do)."""
        self._close_stage()
        self._counted = total is not None
        self.record.update(stage=key, done=0, total=total or 0, detail='')
        self._save()

    def step(self, done, detail=None):
        self.record['done'] = done
        if detail is not None:
            self.record['detail'] = detail
        self._save()

    def detail(self, text):
        self.record['detail'] = text
        self._save()

    def summary(self, text):
        """What the current stage's tick says once it is done."""
        if self.record['stage']:
            self.record['summaries'][self.record['stage']] = text
            self._save()

    def finish(self, ok, message):
        if ok:
            self._close_stage()
        self.record.update(state='done' if ok else 'failed', message=message, detail='')
        self._save()

    _counted = False
