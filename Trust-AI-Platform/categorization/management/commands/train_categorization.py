import json

from django.core.management.base import BaseCommand

from categorization.models import TrainingRun
from categorization.tasks import run_training


class Command(BaseCommand):
    help = 'Train the categorization models now, without Celery, and print a summary.'

    def handle(self, *args, **options):
        run = TrainingRun.start(None)
        run_training(run.id)
        run.refresh_from_db()
        self.stdout.write(f'Run {run.id}: {run.status}')
        if run.error:
            self.stdout.write(run.error)
        counts = {key: value for key, value in run.counts.items() if key != 'scenarios'}
        self.stdout.write(json.dumps(counts, indent=2))
        if run.status == 'finished':
            final = run.results['final']['overall']
            self.stdout.write(f"Final model: agreement {final['accuracy']:.3f}, macro F1 {final['macro_f1']:.3f}")
            for row in run.results['early']['checkpoints']:
                self.stdout.write(
                    f"After phase {row['checkpoint']}: XGBoost {row['xgb']['accuracy']:.3f} "
                    f"vs formula so far {row['baseline']['accuracy']:.3f} ({row['xgb']['runs']} runs)"
                )
            self.stdout.write(f"Seconds: {json.dumps(run.results['seconds'])}")
