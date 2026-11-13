"""Chatbot intent evaluation worker (the `rasa-eval` compose service).

Runs in the rasa/rasa image, next to the live chatbot but separate from it.
Admins queue an evaluation on the Student categorization page; this loop
claims it from the database, runs

    rasa test nlu --cross-validation --folds N --successes --no-plot

on a copy of the chatbot's data/nlu.yml and config.yml, and writes Rasa's
result files back into the same row. Django builds the tables, charts and
Excel file from them (categorization/chatbot_report.py).

Only the standard library, psycopg2 and PyYAML are used; all three are in
the rasa image. The table is categorization.models.ChatbotEvaluation; tests
check that the columns written here exist there and that the time limits
agree with its STALE_AFTER.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from importlib import metadata
from pathlib import Path

TABLE = 'categorization_chatbotevaluation'
RESULT_COLUMNS = (
    'status', 'progress', 'error', 'started_at', 'finished_at', 'nlu_text', 'nlu_sha256',
    'rasa_version', 'fallback_threshold', 'report', 'errors', 'successes', 'log_tail',
)
SOURCE = Path(os.environ.get('RASA_SOURCE', '/rasa-src'))
POLL_SECONDS = int(os.environ.get('POLL_SECONDS', '10'))
# The page gives up on a job after this long (ChatbotEvaluation.STALE_AFTER):
# a queued job older than that is not started, and Rasa is stopped a little
# before it, so the worker records its own clear error first.
STALE_HOURS = 3
TIME_LIMIT_SECONDS = STALE_HOURS * 3600 - 300
LOG_LINES = 300
# tqdm epoch bars ("Epochs:  45%|####5  | 45/100 [...]") say nothing after the fact.
PROGRESS_BAR = re.compile(r'\d+%\|')
DEFAULT_FALLBACK = 0.3
RESTARTED_ERROR = 'The chatbot evaluation worker restarted during the run.'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('rasa-eval')


class JobGone(Exception):
    """The job is no longer running in the database (expired on the page): stop."""


def fallback_threshold(config):
    """The FallbackClassifier threshold in config.yml, or Rasa's default."""
    for component in (config or {}).get('pipeline') or []:
        if isinstance(component, dict) and component.get('name') == 'FallbackClassifier':
            return float(component.get('threshold', DEFAULT_FALLBACK))
    return DEFAULT_FALLBACK


class Progress:
    """Turns Rasa's console lines into a short progress text.

    Rasa trains a DIETClassifier once per fold, then writes the
    accumulated results. Returns the new text, or None when unchanged.
    """

    def __init__(self, folds):
        self.folds = folds
        self.fold = 0

    def read(self, line):
        if "Starting to train component 'DIETClassifier'" in line:
            self.fold += 1
            return f'Fold {self.fold} / {self.folds}: training'
        if 'Accumulated test folds intent evaluation results' in line:
            return 'Writing the results'
        return None


def rasa_command(folds):
    return [
        'rasa', 'test', 'nlu',
        '--nlu', 'data/nlu.yml',
        '--config', 'config.yml',
        '--cross-validation', '--folds', str(folds),
        '--successes',
        '--no-plot',
        '--out', 'out',
    ]


def collect_results(out):
    def read(name, empty):
        path = Path(out) / name
        if not path.exists():
            return empty
        return json.loads(path.read_text(encoding='utf-8'))

    return {
        'report': read('intent_report.json', {}),
        'errors': read('intent_errors.json', []),
        'successes': read('intent_successes.json', []),
    }


def rasa_version():
    try:
        return metadata.version('rasa')
    except metadata.PackageNotFoundError:
        return ''


# ---- database ---------------------------------------------------------------

def connect():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get('DB_HOST', 'db'),
        port=os.environ.get('DB_PORT', '5432'),
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
    )


def execute(connection, sql, params=()):
    """Run one statement and commit. Returns (rows or None, rowcount)."""
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall() if cursor.description else None
        count = cursor.rowcount
    connection.commit()
    return rows, count


def fail_interrupted(connection):
    """Runs left 'running' by an earlier worker can never finish: only one worker exists."""
    execute(connection, f"""
        UPDATE {TABLE} SET status = 'failed', error = %s, finished_at = now()
        WHERE status = 'running'
    """, (RESTARTED_ERROR,))


def claim(connection):
    """The oldest queued job the page still waits for, now marked running."""
    rows, _ = execute(connection, f"""
        UPDATE {TABLE} SET status = 'running', started_at = now(), progress = 'Starting Rasa'
        WHERE id = (
            SELECT id FROM {TABLE}
            WHERE status = 'queued' AND created_at > now() - make_interval(hours => %s)
            ORDER BY created_at, id FOR UPDATE SKIP LOCKED LIMIT 1
        )
        RETURNING id, folds
    """, (STALE_HOURS,))
    return rows[0] if rows else None


def update(connection, job_id, **values):
    """Write to a running job only; raise JobGone when it is not running any more."""
    from psycopg2.extras import Json

    assert set(values) <= set(RESULT_COLUMNS), values
    columns = ', '.join(f'{name} = %s' for name in values)
    params = [Json(v) if isinstance(v, (dict, list)) else v for v in values.values()]
    _, count = execute(
        connection, f"UPDATE {TABLE} SET {columns} WHERE id = %s AND status = 'running'",
        (*params, job_id),
    )
    if count == 0:
        raise JobGone(job_id)


def _now(connection):
    rows, _ = execute(connection, 'SELECT now()')
    return rows[0][0]


# ---- one evaluation -----------------------------------------------------------

def _record_failure(connection, job_id, error, tail):
    """Best effort: a dead connection must not stop the clean-up after it."""
    try:
        connection.rollback()
        update(connection, job_id, status='failed', error=str(error)[:1000] or type(error).__name__,
               log_tail='\n'.join(tail), finished_at=_now(connection))
    except JobGone:
        pass
    except Exception:  # noqa: BLE001
        log.exception('Could not record the failure of evaluation %s.', job_id)


def evaluate(connection, job_id, folds):
    import yaml

    work = Path(tempfile.mkdtemp(prefix='rasa-eval-'))
    tail = deque(maxlen=LOG_LINES)
    process = None
    timed_out = threading.Event()
    timer = None
    try:
        (work / 'data').mkdir()
        shutil.copy(SOURCE / 'data' / 'nlu.yml', work / 'data' / 'nlu.yml')
        shutil.copy(SOURCE / 'config.yml', work / 'config.yml')
        nlu_text = (work / 'data' / 'nlu.yml').read_text(encoding='utf-8')
        config = yaml.safe_load((work / 'config.yml').read_text(encoding='utf-8'))
        update(
            connection, job_id,
            nlu_text=nlu_text,
            nlu_sha256=hashlib.sha256(nlu_text.encode('utf-8')).hexdigest(),
            rasa_version=rasa_version(),
            fallback_threshold=fallback_threshold(config),
        )

        progress = Progress(folds)
        process = subprocess.Popen(
            rasa_command(folds), cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
        )

        def stop_for_time():
            timed_out.set()
            process.kill()

        timer = threading.Timer(TIME_LIMIT_SECONDS, stop_for_time)
        timer.daemon = True
        timer.start()
        for line in process.stdout:
            line = line.rstrip()
            if not PROGRESS_BAR.search(line):
                tail.append(line)
            text = progress.read(line)
            if text:
                log.info('Evaluation %s: %s', job_id, text)
                update(connection, job_id, progress=text)
        code = process.wait()
        if timed_out.is_set():
            raise RuntimeError(f'Rasa did not finish within {TIME_LIMIT_SECONDS // 60} minutes and was stopped.')
        if code != 0:
            raise RuntimeError(f'Rasa stopped with exit code {code}.')

        results = collect_results(work / 'out')
        if not results['report']:
            raise RuntimeError('Rasa wrote no intent report.')
        update(connection, job_id, **results, log_tail='\n'.join(tail), progress='',
               finished_at=_now(connection), status='finished')
        log.info('Evaluation %s finished.', job_id)
    except JobGone:
        log.warning('Evaluation %s is no longer running on the page; Rasa stopped.', job_id)
    except Exception as error:  # noqa: BLE001 - every failure must reach the page
        log.exception('Evaluation %s failed.', job_id)
        _record_failure(connection, job_id, error, tail)
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
        shutil.rmtree(work, ignore_errors=True)


def main():
    log.info('Chatbot evaluation worker started; reading %s', SOURCE)
    connection = None
    while True:
        try:
            if connection is None or connection.closed:
                connection = connect()
                fail_interrupted(connection)
            job = claim(connection)
            if job:
                log.info('Evaluation %s claimed (%s folds).', *job)
                evaluate(connection, *job)
                continue
        except Exception:  # noqa: BLE001 - keep the worker alive (database restarts)
            log.exception('Worker loop error; retrying.')
            if connection is not None:
                connection.close()
            connection = None
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
