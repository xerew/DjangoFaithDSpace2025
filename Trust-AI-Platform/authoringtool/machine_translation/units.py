"""Pieces of a scenario that are translated in one request each.

A unit is the scenario header, one phase, or one activity with its answers.
Text is read from the target (copied) rows; units already applied are
listed in ``job.completed_units`` and skipped.
"""

import re
from dataclasses import dataclass, field
from html import unescape

from django.db import transaction
from django.utils.html import strip_tags

from authoringtool.models import Activity, Answer, Phase, Scenario

_LETTER = re.compile(r'[^\W\d_]')


@dataclass
class Unit:
    key: str
    label: str
    fields: dict = field(default_factory=dict)


def strip_html(html):
    """Plain text of activity HTML, as the editor stores it in plain_text."""
    return unescape(strip_tags(html or ''))


def _has_words(value):
    return bool(_LETTER.search(strip_html(value)))


def _fields(pairs):
    return {key: value for key, value in pairs if value and _has_words(value)}


def _ordered_activities(scenario):
    return (
        Activity.objects.filter(scenario=scenario)
        .select_related('phase')
        .prefetch_related('answers')
        .order_by('phase__order', 'phase_id', 'order', 'id')
    )


def all_units(job):
    target = job.target_scenario
    units = [Unit('scenario', 'Scenario details', _fields([
        ('name', target.name),
        ('description', target.description),
        ('learning_goals', target.learning_goals),
    ]))]
    for phase in target.phases.order_by('order', 'created_on', 'id'):
        units.append(Unit(f'phase:{phase.id}', f'Phase "{phase.name}"', _fields([
            ('name', phase.name),
            ('description', phase.description),
        ])))
    for activity in _ordered_activities(target):
        pairs = [
            ('name', activity.name),
            ('text', activity.text),
            ('helper', activity.helper),
        ]
        pairs += [
            (f'answer:{answer.id}', answer.text)
            for answer in sorted(activity.answers.all(), key=lambda a: a.id)
        ]
        units.append(Unit(
            f'activity:{activity.id}', f'Activity "{activity.name}"', _fields(pairs),
        ))
    return units


def pending_units(job):
    done = set(job.completed_units)
    return [unit for unit in all_units(job) if unit.key not in done]


def _free_name(name, scenario):
    candidate, counter = name, 2
    while Scenario.objects.filter(
        created_by_id=scenario.created_by_id, name=candidate,
    ).exclude(pk=scenario.pk).exists():
        candidate = f'{name} ({counter})'
        counter += 1
    return candidate


@transaction.atomic
def apply_unit(job, unit, translated):
    """Write a validated unit to the target rows and mark it done."""
    kind, _, row_id = unit.key.partition(':')
    if kind == 'scenario':
        scenario = Scenario.objects.get(pk=job.target_scenario_id)
        for name in ('description', 'learning_goals'):
            if name in translated:
                setattr(scenario, name, translated[name])
        if 'name' in translated:
            scenario.name = _free_name(translated['name'], scenario)
        scenario.save(update_fields=['name', 'description', 'learning_goals'])
    elif kind == 'phase':
        phase = Phase.objects.get(pk=int(row_id), scenario_id=job.target_scenario_id)
        for name in ('name', 'description'):
            if name in translated:
                setattr(phase, name, translated[name])
        phase.save(update_fields=['name', 'description'])
    else:
        activity = Activity.objects.get(pk=int(row_id), scenario_id=job.target_scenario_id)
        for name in ('name', 'text', 'helper'):
            if name in translated:
                setattr(activity, name, translated[name])
        activity.plain_text = strip_html(activity.text)
        activity.save(update_fields=['name', 'text', 'helper', 'plain_text'])
        for key, value in translated.items():
            if key.startswith('answer:'):
                Answer.objects.filter(
                    pk=int(key.split(':', 1)[1]), activity=activity,
                ).update(text=value)
    job.completed_units = list(job.completed_units) + [unit.key]
    job.save(update_fields=['completed_units'])
