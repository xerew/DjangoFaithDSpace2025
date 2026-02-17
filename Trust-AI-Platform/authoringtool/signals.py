from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.cache import cache
from .models import UserAnswer

@receiver(post_save, sender=UserAnswer)
def invalidate_cache_on_user_answer_save(sender, instance, **kwargs):
    scenario_id = instance.activity.phase.scenario.id
    cache.delete(f'sankey_data_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'sankey_data_{scenario_id}')
    cache.delete(f'activity_answers_data_{scenario_id}_activities_{start_date}_{end_date}') # cache.delete(f'activity_answers_data_{scenario_id}_activities')
    cache.delete(f'activity_answers_data_{scenario_id}_phases_{start_date}_{end_date}')
    cache.delete(f'performance_data_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'performance_data_{scenario_id}')
    cache.delete(f'time_spent_data_{scenario_id}_activities_{start_date}_{end_date}') # cache.delete(f'time_spent_data_{scenario_id}_activities')
    cache.delete(f'time_spent_data_{scenario_id}_phases_{start_date}_{end_date}') # cache.delete(f'time_spent_data_{scenario_id}_phases')
    cache.delete(f'detailed_phase_scores_data_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'detailed_phase_scores_data_{scenario_id}')
    cache.delete(f'performers_data_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'performers_data_{scenario_id}')
    cache.delete(f'time_spent_by_performer_type_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'time_spent_by_performer_type_{scenario_id}')
    cache.delete(f'performance_by_department_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'performance_by_department_{scenario_id}')
    cache.delete(f'final_performance_data_{scenario_id}_{start_date}_{end_date}') # cache.delete(f'final_performance_data_{scenario_id}')