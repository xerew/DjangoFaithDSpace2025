"""Prompts that make an LLM answer an activity as a student of a given ability."""
import json
import random

# Change whenever the wording or the reply format changes, so runs made with
# different prompts are never mixed in one analysis.
PROMPT_VERSION = 'ss-2026-10-20'

LETTERS = 'ABCDEFGHIJ'

# Described qualitatively on purpose: giving the model a target accuracy would
# make it reproduce the number instead of student reasoning.
PERSONAS = {
    'low': (
        'You find science difficult. You often rely on everyday intuition instead '
        'of what was taught, you sometimes misread details, and you sometimes guess.'
    ),
    'mid': (
        'You are an average student. You understand the basic ideas but make '
        'mistakes on tricky or unfamiliar questions.'
    ),
    'high': (
        'You are a strong student. You read carefully, remember what was taught '
        'and apply it correctly.'
    ),
}

CONTEXTS = ('none', 'material')


def option_order(item, seed):
    """A reproducible shuffle of the option positions, so position bias averages out."""
    order = list(range(len(item['options'])))
    random.Random(seed).shuffle(order)
    return order


def image_paths(item):
    """Images in the order the prompt refers to them: question images, then option images."""
    return list(item.get('images') or []) + [
        option['image'] for option in item['options'] if option.get('image')
    ]


def _age(item):
    low, high = item.get('age_min'), item.get('age_max')
    if low and high:
        return f'a {low}-{high} year old'
    return 'a secondary-school'


def build_prompt(item, persona, context, order):
    """Return the prompt and the mapping from offered letters to option ids."""
    language = item.get('language') or 'English'
    letters, lines = {}, []
    image_number = len(item.get('images') or [])
    image_notes = []
    for position, option_index in enumerate(order):
        option = item['options'][option_index]
        letter = LETTERS[position]
        letters[letter] = option['id']
        lines.append(f'{letter}. {option["text"]}')
        if option.get('image'):
            image_number += 1
            image_notes.append(f'Image {image_number} shows option {letter}.')

    parts = [
        f'You are role-playing {_age(item)} school student whose first language is {language}.',
        PERSONAS[persona],
        'Answer the question below exactly as this student would, not as an expert. '
        'Students like you do sometimes choose a wrong answer.',
    ]
    material = (item.get('preceding_material') or '').strip()
    if context == 'material' and material:
        parts.append(f'Earlier in the lesson you read:\n"""\n{material}\n"""')
    parts.append(f'Question (in {language}):\n{item["text"]}')
    parts.append('Options:\n' + '\n'.join(lines))
    if image_notes:
        parts.append(' '.join(image_notes))
    parts.append(
        'Reply in JSON with the letter you choose, how confident you are (0-100) '
        "and one short sentence in the student's own words explaining why."
    )
    return '\n\n'.join(parts), letters


def reply_schema(option_count):
    return {
        'type': 'object',
        'properties': {
            'choice': {'type': 'string', 'enum': list(LETTERS[:option_count])},
            'confidence': {'type': 'integer', 'minimum': 0, 'maximum': 100},
            'reason': {'type': 'string'},
        },
        'required': ['choice', 'confidence', 'reason'],
    }


def parse_reply(text, letters):
    """Return (option_id, letter, confidence, reason), or None if unusable."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    letter = str(data.get('choice', '')).strip().upper()[:1]
    if letter not in letters:
        return None
    confidence = data.get('confidence')
    return letters[letter], letter, confidence if isinstance(confidence, int) else None, str(data.get('reason', ''))
