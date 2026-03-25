from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.cache import cache
from .models import UserAnswer

@receiver(post_save, sender=UserAnswer)
def invalidate_cache_on_user_answer_save(sender, instance, **kwargs):
    scenario_id = instance.activity.phase.scenario.id
    # NOTE: Cache keys include date-range parameters that are only known at query
    # time, so we cannot construct the exact keys here.  Delete the
    # scenario-level wildcard keys instead.  If the cache backend supports
    # delete_pattern() (e.g. django-redis), switch to that for precision.
    cache.delete(f'sankey_data_{scenario_id}')
    cache.delete(f'activity_answers_data_{scenario_id}_activities')
    cache.delete(f'activity_answers_data_{scenario_id}_phases')
    cache.delete(f'performance_data_{scenario_id}')
    cache.delete(f'time_spent_data_{scenario_id}_activities')
    cache.delete(f'time_spent_data_{scenario_id}_phases')
    cache.delete(f'detailed_phase_scores_data_{scenario_id}')
    cache.delete(f'performers_data_{scenario_id}')
    cache.delete(f'time_spent_by_performer_type_{scenario_id}')
    cache.delete(f'performance_by_department_{scenario_id}')
    cache.delete(f'final_performance_data_{scenario_id}')
