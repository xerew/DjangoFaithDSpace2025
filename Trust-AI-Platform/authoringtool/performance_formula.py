"""The PISA-aligned student categorization, in one place.

Used by the waterfall chart (`compute_final_performance`), the student CSV
(`compute_student_performance_metrics`) and the categorization models, so
all three categorize a run the same way. Pure functions: no queries.

- Only the last answer per activity counts.
- Phase % = sum of chosen answer weights / sum of each scored activity's
  highest weight. Scored activities are the primary evaluatable ones plus
  the other activities of their QuestionBunch.
- Final % = phase % weighted by phase position; phases with nothing scored
  are skipped and the remaining weights rescaled.
"""

HIGH_THRESHOLD = 83.3
MODERATE_THRESHOLD = 49.7
CATEGORIES = ('Low', 'Moderate', 'High')

_WEIGHTS_UP_TO_4 = [0.2, 0.2, 0.45, 0.15]
_WEIGHTS_5_OR_MORE = [0.2, 0.2, 0.3, 0.15, 0.15]


def phase_weights(phase_count):
    """Weight of each phase position; positions past the list weigh 0."""
    return list(_WEIGHTS_5_OR_MORE if phase_count > 4 else _WEIGHTS_UP_TO_4)


def categorize(percentage):
    if percentage >= HIGH_THRESHOLD:
        return 'High'
    if percentage >= MODERATE_THRESHOLD:
        return 'Moderate'
    return 'Low'


def scored_activity_ids(primary_ids, bunch_map, known_ids=None):
    """Scored activities of one phase, in order, each once.

    A primary with a QuestionBunch contributes the bunch's activities,
    otherwise itself. With `known_ids`, ids outside it are dropped.
    """
    seen = set()
    ordered = []
    for primary_id in primary_ids:
        for activity_id in bunch_map.get(primary_id, [primary_id]):
            if activity_id in seen:
                continue
            if known_ids is not None and activity_id not in known_ids:
                continue
            seen.add(activity_id)
            ordered.append(activity_id)
    return ordered


def phase_score(scored_ids, chosen_weight, max_weight):
    """(total, maximum) of one phase. Unanswered activities add 0."""
    total = 0
    maximum = 0
    for activity_id in scored_ids:
        total += chosen_weight.get(activity_id) or 0
        maximum += max_weight.get(activity_id) or 0
    return total, maximum


def weighted_percentage(phase_scores, weights):
    """Final % from per-position (total, maximum); None when nothing counts."""
    weighted = 0
    weight_sum = 0
    for index, (total, maximum) in enumerate(phase_scores):
        if maximum <= 0:
            continue
        weight = weights[index] if index < len(weights) else 0
        weighted += (total / maximum) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return weighted / weight_sum * 100
