"""Move proposal runs from a local copy of the platform to the server.

Proposals can be generated on a PC that runs the platform from a database
dump and its own Ollama, then copied to the server as one JSON file:

    local:  python manage.py export_proposals --output proposals.json
    server: python manage.py import_proposals proposals.json --dry-run
            python manage.py import_proposals proposals.json

A dump keeps the server's ids, so scenarios, phases and activities are found
by id and checked by name. Importing a run does what a finished Generate
does: the run becomes the scenario's current one and the old run moves to
Proposal History. Proposals arrive unreviewed ('new').
"""

from dataclasses import dataclass, field

from django.contrib.auth.models import User
from django.db import transaction
from django.utils.dateparse import parse_datetime

from .models import (
    Activity,
    ActivityFlag,
    ActivityProposal,
    CategoryTag,
    ProposalGenerationRun,
    ProposalStructuralFailure,
    Scenario,
    ScenarioVersion,
)
from .proposal_progress import job_status

FORMAT = 'faith-proposal-runs/1'

_PROPOSAL_FIELDS = (
    'proposal_type', 'suggested_action', 'translated_action', 'json_action',
    'json_translated_action', 'llm_model', 'prompt_version', 'selection_policy',
    'selection_probability', 'action_probabilities', 'selection_contexts',
)
_FLAG_FIELDS = (
    'category', 'flag_type', 'flag_reason', 'is_at_risk', 'value_at_risk',
    'threshold_used', 'auto_flagged', 'evidence_scope', 'evidence_signature',
    'evidence_sources',
)


def _when(value):
    return value.isoformat() if value else None


def export_runs(runs):
    """The runs, their proposals, flags, risk categories and structural
    failures, as plain JSON data."""
    items = []
    for run in runs:
        proposals = (
            run.proposals.select_related('activity')
            .prefetch_related('flag', 'categories_in_risk').order_by('id')
        )
        items.append({
            'id': run.id,
            'scenario_id': run.scenario_id,
            'scenario_name': run.scenario.name,
            'scenario_version_id': run.scenario_version_id,
            'created_by_id': run.created_by_id,
            'created_at': _when(run.created_at),
            'evidence_scope': run.evidence_scope,
            'evidence_version_ids': run.evidence_version_ids,
            'evidence_summary': run.evidence_summary,
            'proposals': [
                {
                    'id': proposal.id,
                    'activity_id': proposal.activity_id,
                    'activity_name': proposal.activity.name if proposal.activity else '',
                    'phase_id': proposal.phase_id,
                    'created_at': _when(proposal.created_at),
                    'categories': sorted(t.name for t in proposal.categories_in_risk.all()),
                    'flags': [
                        {
                            **{name: getattr(flag, name) for name in _FLAG_FIELDS},
                            'flagged_on': _when(flag.flagged_on),
                        }
                        for flag in proposal.flag.all()
                    ],
                    **{name: getattr(proposal, name) for name in _PROPOSAL_FIELDS},
                }
                for proposal in proposals
            ],
            'failures': [
                {
                    'activity_id': failure.activity_id,
                    'proposal_id': failure.proposal_id,
                    'selected_action': failure.selected_action,
                    'stage': failure.stage,
                    'errors': failure.errors,
                    'raw_output': failure.raw_output,
                    'created_at': _when(failure.created_at),
                }
                for failure in run.structural_failures.order_by('id')
            ],
        })
    return {'format': FORMAT, 'runs': items}


@dataclass
class ImportReport:
    imported: int = 0
    lines: list = field(default_factory=list)

    def say(self, text):
        self.lines.append(text)


def import_runs(data, dry_run=False, force=False):
    """Import the runs in an export file. Returns an ImportReport; raises
    ValueError for a file that is not an export."""
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise ValueError(f'This is not a proposals export (expected format {FORMAT}).')
    report = ImportReport()
    for item in data.get('runs') or []:
        _import_run(item, report, dry_run, force)
    verb = 'Would import' if dry_run else 'Imported'
    report.say(f"{verb} {report.imported} run{'s' if report.imported != 1 else ''}.")
    return report


def _import_run(item, report, dry_run, force):
    scenario = Scenario.objects.filter(pk=item.get('scenario_id')).first()
    label = f"Scenario {item.get('scenario_id')} \"{item.get('scenario_name', '')}\""
    if scenario is None:
        report.say(f'{label}: skipped, the scenario is not on this server.')
        return
    created_at = parse_datetime(item.get('created_at') or '')
    if ProposalGenerationRun.objects.filter(
        pk=item.get('id'), scenario=scenario, created_at=created_at,
    ).exists():
        report.say(f'{label}: skipped, this run is already on this server.')
        return
    current = ProposalGenerationRun.objects.filter(scenario=scenario, is_current=True).first()
    if current and created_at and current.created_at > created_at and not force:
        report.say(
            f'{label}: skipped, the server has newer proposals '
            f'({current.created_at:%Y-%m-%d %H:%M}). Use --force to replace them.'
        )
        return
    status = job_status(scenario.id)
    if status and status.get('running'):
        report.say(f'{label}: skipped, a proposal job is running for it on this server.')
        return

    activities = Activity.objects.filter(phase__scenario=scenario).in_bulk(
        [p.get('activity_id') for p in item.get('proposals') or []]
    )
    usable, skipped = [], []
    for proposal in item.get('proposals') or []:
        activity = activities.get(proposal.get('activity_id'))
        if activity is None or activity.name != proposal.get('activity_name'):
            skipped.append(proposal.get('activity_name') or f"activity {proposal.get('activity_id')}")
        else:
            usable.append((proposal, activity))
    if item.get('proposals') and not usable:
        report.say(
            f'{label}: skipped, none of its activities match this server any more '
            f"({', '.join(skipped)})."
        )
        return

    note = f"{len(usable)} proposal{'s' if len(usable) != 1 else ''}"
    if skipped:
        note += f"; left out (activity deleted or renamed since the dump): {', '.join(skipped)}"
    report.imported += 1
    if dry_run:
        report.say(f'{label}: would import {note}.')
        return
    with transaction.atomic():
        _write_run(item, scenario, created_at, usable)
    report.say(f'{label}: imported {note}; it is now the current run.')


def _write_run(item, scenario, created_at, usable):
    # The exported version if the server has it, else the current one.
    version = ScenarioVersion.objects.filter(
        pk=item.get('scenario_version_id'), scenario=scenario,
    ).first()
    run = ProposalGenerationRun.start_new(
        scenario,
        User.objects.filter(pk=item.get('created_by_id')).first(),
        scenario_version=version,
        evidence_scope=item.get('evidence_scope') or 'local',
        evidence_version_ids=item.get('evidence_version_ids') or [],
        evidence_summary=item.get('evidence_summary') or {},
        current=False,
    )
    if created_at:
        ProposalGenerationRun.objects.filter(pk=run.pk).update(created_at=created_at)

    new_ids = {}
    for data, activity in usable:
        proposal = ActivityProposal.objects.create(
            scenario=scenario,
            generation_run=run,
            phase=activity.phase,
            activity=activity,
            status='new',
            **{name: data.get(name) for name in _PROPOSAL_FIELDS if name in data},
        )
        new_ids[data.get('id')] = proposal
        if data.get('created_at'):
            ActivityProposal.objects.filter(pk=proposal.pk).update(
                created_at=parse_datetime(data['created_at']),
            )
        flags = []
        for flag_data in data.get('flags') or []:
            flag = ActivityFlag.objects.create(
                activity=activity, scenario=scenario, phase=activity.phase,
                **{name: flag_data.get(name) for name in _FLAG_FIELDS if name in flag_data},
            )
            if flag_data.get('flagged_on'):
                ActivityFlag.objects.filter(pk=flag.pk).update(
                    flagged_on=parse_datetime(flag_data['flagged_on']),
                )
            flags.append(flag)
        proposal.flag.set(flags)
        proposal.categories_in_risk.set([
            CategoryTag.objects.get_or_create(name=name)[0]
            for name in data.get('categories') or []
        ])

    for failure in item.get('failures') or []:
        created = ProposalStructuralFailure.objects.create(
            scenario=scenario,
            generation_run=run,
            proposal=new_ids.get(failure.get('proposal_id')),
            activity=Activity.objects.filter(
                pk=failure.get('activity_id'), phase__scenario=scenario,
            ).first(),
            selected_action=failure.get('selected_action') or '',
            stage=failure.get('stage') or 'generation',
            errors=failure.get('errors') or [],
            raw_output=failure.get('raw_output') or '',
        )
        if failure.get('created_at'):
            ProposalStructuralFailure.objects.filter(pk=created.pk).update(
                created_at=parse_datetime(failure['created_at']),
            )
    run.make_current()
