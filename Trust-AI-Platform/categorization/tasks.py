"""The training job: started by an admin, run on the Celery worker."""

import logging
import os
from collections import defaultdict

import sklearn
import xgboost
from celery import shared_task
from django.conf import settings
from django.utils import timezone

from . import training
from .dataset import LAST_PHASE, ROUTE_END, UNFINISHED, load_runs
from .features import (
    FAST_ANSWER_RULE,
    build_norms,
    checkpoint_table,
    final_table,
    input_columns,
)
from .models import TrainingRun

logger = logging.getLogger(__name__)


def run_counts(runs):
    labelled = [run for run in runs if run.label is not None]
    scenarios = {}
    per_scenario = defaultdict(lambda: {'runs': 0, 'labelled': 0})
    for run in runs:
        scenarios[run.scenario_id] = (run.scenario_name, run.language or 'Unspecified')
        per_scenario[run.scenario_id]['runs'] += 1
        per_scenario[run.scenario_id]['labelled'] += run.label is not None
    return {
        'students': len({run.user_id for run in labelled}),
        'runs_total': len(runs),
        'labelled': len(labelled),
        'unfinished': sum(run.finished_rule == UNFINISHED for run in runs),
        'route_end': sum(run.finished_rule == ROUTE_END for run in labelled),
        'last_phase': sum(run.finished_rule == LAST_PHASE for run in labelled),
        'current_version': sum(run.is_current_version for run in labelled),
        'older_version': sum(not run.is_current_version for run in labelled),
        'scenarios': [
            {'id': scenario_id, 'name': name, 'language': language, **per_scenario[scenario_id]}
            for scenario_id, (name, language) in sorted(scenarios.items())
        ],
    }


def _save_model(booster, run, name):
    if booster is None:
        return ''
    os.makedirs(settings.CATEGORIZATION_MODEL_DIR, exist_ok=True)
    path = os.path.join(settings.CATEGORIZATION_MODEL_DIR, f'run_{run.id}_{name}.json')
    booster.save_model(path)
    return path


def run_training(run_id):
    run = TrainingRun.objects.get(pk=run_id)
    run.status = 'running'
    run.started_at = timezone.now()
    run.progress = 'Building inputs…'
    run.save(update_fields=['status', 'started_at', 'progress'])
    try:
        runs = load_runs(progress=run.set_progress)
        run.counts = run_counts(runs)
        run.set_progress('Building inputs…')
        norms = build_norms(runs)
        final = final_table(runs, norms)
        checkpoints = checkpoint_table(runs, norms)
        run.versions = {'xgboost': xgboost.__version__, 'scikit-learn': sklearn.__version__}
        run.parameters = {
            'xgboost': training.PARAMS,
            'num_boost_round': training.NUM_BOOST_ROUND,
            'early_stopping_rounds': training.EARLY_STOPPING_ROUNDS,
            'folds': training.FOLDS,
            'loso_min_runs': training.LOSO_MIN_RUNS,
            'final_inputs': input_columns(final) if len(final) else [],
            'early_inputs': input_columns(checkpoints) if len(checkpoints) else [],
            'fast_answer_rule': FAST_ANSWER_RULE,
        }
        run.save(update_fields=['counts', 'versions', 'parameters'])
        results, boosters = training.train_all(final, checkpoints, run.set_progress)
        run.final_model_path = _save_model(boosters['final'], run, 'final')
        run.early_model_path = _save_model(boosters['early'], run, 'early')
        run.results = results
        run.status = 'finished'
        run.progress = ''
    except training.NotEnoughData as error:
        run.status = 'failed'
        run.error = str(error)
    except Exception as error:  # shown to the admin; the traceback goes to the log
        logger.exception('Categorization training run %s failed', run_id)
        run.status = 'failed'
        run.error = f'{type(error).__name__}: {error}'
    run.finished_at = timezone.now()
    run.save()


@shared_task
def train_models_task(run_id):
    run_training(run_id)
