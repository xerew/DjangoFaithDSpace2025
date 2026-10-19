"""Show what a revise proposal changes: the current activity next to it."""
from difflib import SequenceMatcher

from django.utils.html import strip_tags


def _clean(text):
    return ' '.join(strip_tags(text or '').split())


def word_diff(old, new):
    """[(op, text)] turning old into new, op being 'same', 'del' or 'add'."""
    old_words, new_words = _clean(old).split(), _clean(new).split()
    parts = []
    matcher = SequenceMatcher(None, old_words, new_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            parts.append(('same', ' '.join(old_words[i1:i2])))
            continue
        if tag in ('replace', 'delete'):
            parts.append(('del', ' '.join(old_words[i1:i2])))
        if tag in ('replace', 'insert'):
            parts.append(('add', ' '.join(new_words[j1:j2])))
    return parts


def _answer(value):
    if isinstance(value, dict):
        return _clean(value.get('text')), bool(value.get('is_correct'))
    return _clean(value), False


def revision_changes(activity, proposed):
    """Rows comparing the activity with a proposed revision of it.

    Each row has the current and proposed value and whether it changed; text
    rows also carry a word diff. Answers are compared position by position.
    """
    proposed = proposed or {}
    rows = []
    for label, current, new in (
        ('Name', activity.name, proposed.get('activity_name')),
        ('Content', activity.plain_text or activity.text, proposed.get('content')),
    ):
        current, new = _clean(current), _clean(new) or _clean(current)
        rows.append({
            'label': label,
            'current': current,
            'proposed': new,
            'changed': current != new,
            'diff': word_diff(current, new),
        })

    current_answers = [
        (_clean(answer.text), answer.is_correct)
        for answer in activity.answers.order_by('id')
    ]
    proposed_answers = [_answer(answer) for answer in proposed.get('answers') or []]
    if current_answers or proposed_answers:
        answers = []
        for index in range(max(len(current_answers), len(proposed_answers))):
            current = current_answers[index] if index < len(current_answers) else ('', False)
            new = proposed_answers[index] if index < len(proposed_answers) else ('', False)
            answers.append({
                'current': current[0],
                'current_correct': current[1],
                'proposed': new[0],
                'proposed_correct': new[1],
                'changed': current != new,
            })
        rows.append({
            'label': 'Answers',
            'answers': answers,
            'changed': any(answer['changed'] for answer in answers),
        })
    return rows
