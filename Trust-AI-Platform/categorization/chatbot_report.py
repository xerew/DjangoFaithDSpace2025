"""What the Chatbot intents tab and its Excel file show, from Rasa's own files.

The logic follows RasaFaith/RasaFaith/tools/export_intent_report.py, the
script used for the first, hand-run evaluation.
"""

import re
from collections import Counter

# Keys of intent_report.json that are totals, not intents.
SUMMARY_KEYS = {'accuracy', 'macro avg', 'micro avg', 'weighted avg'}
# Shown when Rasa predicted no intent at all.
NO_INTENT = '(no intent)'
DEFAULT_FALLBACK = 0.3
BUCKETS = 10
FOLD_LINE = re.compile(r'(train|test) (Accuracy|F1-score|Precision): ([\d.]+) \(([\d.]+)\)')


def per_fold(log_tail):
    """The intent 'test Accuracy: 0.776 (0.019)' lines: mean and std across folds.

    Rasa prints the same kind of lines for entities and response selection
    after the intent block; only the intent block is read.
    """
    text = log_tail or ''
    start = text.find('Intent evaluation results')
    if start >= 0:
        text = text[start + len('Intent evaluation results'):]
        end = text.find('evaluation results')
        if end >= 0:
            text = text[:end]
    found = {}
    for split, metric, mean, std in FOLD_LINE.findall(text):
        found.setdefault(f'{split} {metric}', (float(mean), float(std)))
    return found


def _predicted(item):
    return (item.get('intent_prediction') or {}).get('name') or NO_INTENT


def _confidence(item):
    return float((item.get('intent_prediction') or {}).get('confidence') or 0.0)


def build(evaluation):
    report = evaluation.report or {}
    errors = evaluation.errors or []
    successes = evaluation.successes or []
    fallback = evaluation.fallback_threshold
    if fallback is None:
        fallback = DEFAULT_FALLBACK

    wrong_by_intent = Counter(item.get('intent') for item in errors)
    per_intent = []
    for intent, scores in report.items():
        if intent in SUMMARY_KEYS or not isinstance(scores, dict):
            continue
        per_intent.append({
            'intent': intent,
            'examples': int(scores.get('support', 0)),
            'precision': scores.get('precision', 0.0),
            'recall': scores.get('recall', 0.0),
            'f1': scores.get('f1-score', 0.0),
            'wrong': wrong_by_intent.get(intent, 0),
        })
    per_intent.sort(key=lambda row: (row['f1'], row['intent']))

    predictions = []
    for item, correct in [(s, True) for s in successes] + [(e, False) for e in errors]:
        confidence = _confidence(item)
        predictions.append({
            'text': item.get('text', ''),
            'intent': item.get('intent', ''),
            'predicted': _predicted(item),
            'confidence': confidence,
            'correct': correct,
            'below_fallback': confidence < fallback,
        })
    wrong = sorted((row for row in predictions if not row['correct']),
                   key=lambda row: -row['confidence'])

    intents = sorted(row['intent'] for row in per_intent)
    columns = intents + sorted({_predicted(item) for item in errors} - set(intents))
    pairs = Counter()
    for row in per_intent:
        pairs[(row['intent'], row['intent'])] = row['examples'] - row['wrong']
    for item in errors:
        pairs[(item.get('intent'), _predicted(item))] += 1
    confusion = {
        'rows': intents,
        'columns': columns,
        'matrix': [[pairs[(true, predicted)] for predicted in columns] for true in intents],
    }

    right_buckets = [0] * BUCKETS
    wrong_buckets = [0] * BUCKETS
    for row in predictions:
        index = min(int(row['confidence'] * BUCKETS), BUCKETS - 1)
        (right_buckets if row['correct'] else wrong_buckets)[index] += 1

    macro = report.get('macro avg') or {}
    weighted = report.get('weighted avg') or {}
    summary = {
        'examples': sum(row['examples'] for row in per_intent),
        'intents': len(per_intent),
        'folds': evaluation.folds,
        'accuracy': report.get('accuracy'),
        'macro_f1': macro.get('f1-score'),
        'weighted_f1': weighted.get('f1-score'),
        'wrong': len(wrong),
        'wrong_above_fallback': sum(1 for row in wrong if not row['below_fallback']),
        'fallback': fallback,
        'per_fold': per_fold(evaluation.log_tail),
    }
    return {
        'summary': summary,
        'per_intent': per_intent,
        'predictions': predictions,
        'wrong': wrong,
        'confusion': confusion,
        'confidence': {
            'labels': [f'{i / BUCKETS:.1f}–{(i + 1) / BUCKETS:.1f}' for i in range(BUCKETS)],
            'right': right_buckets,
            'wrong': wrong_buckets,
            'fallback': fallback,
        },
    }
