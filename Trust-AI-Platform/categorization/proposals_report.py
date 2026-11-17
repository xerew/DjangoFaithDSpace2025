"""How often teachers accept the AI activity proposals, and how that changes over time.

Read live from the database on each page load: there are few proposals, and
nothing here needs training. Every teacher's decision counts once, so a
proposal reviewed by five teachers counts five times. A pending review is
created as soon as a teacher opens a scenario's proposal list, so "opened,
not decided" means exactly that, not work that was assigned.
"""

from collections import Counter, defaultdict
from itertools import combinations

from django.utils import timezone
from django.utils.translation import gettext as _

from authoringtool.models import REJECTION_REASON_CHOICES, ActivityProposal, UserProposalReview

from .report import _period_label

DECIDED = ('accepted', 'rejected')


def _month(when):
    if timezone.is_aware(when):
        when = timezone.localtime(when)
    return f'{when.year:04d}-{when.month:02d}'


def _rate(accepted, rejected):
    decided = accepted + rejected
    return accepted / decided if decided else None


def _tally():
    return {'proposals': set(), 'accepted': 0, 'rejected': 0, 'undecided': 0}


def _count(tally, proposal_id, status):
    tally['proposals'].add(proposal_id)
    if status == 'accepted':
        tally['accepted'] += 1
    elif status == 'rejected':
        tally['rejected'] += 1
    else:
        tally['undecided'] += 1


def _row(label, tally, **extra):
    return {
        'label': label,
        'proposals': len(tally['proposals']),
        'accepted': tally['accepted'],
        'rejected': tally['rejected'],
        'undecided': tally['undecided'],
        'decisions': tally['accepted'] + tally['rejected'],
        'acceptance': _rate(tally['accepted'], tally['rejected']),
        **extra,
    }


def _source(proposal):
    return 'bandit' if proposal['selection_policy'] else 'before'


def build():
    proposals = {
        p['id']: p for p in ActivityProposal.objects.values(
            'id', 'scenario_id', 'scenario__name', 'proposal_type', 'selection_policy',
            'llm_model', 'created_at',
        )
    }
    reviews = list(UserProposalReview.objects.filter(proposal_id__in=proposals).values(
        'proposal_id', 'user_id', 'user__username', 'status', 'feedback_type',
        'rejection_reasons', 'reviewed_at', 'was_edited',
    ).order_by('proposal_id', 'user_id'))

    months, types, sources, models, scenarios = (defaultdict(_tally) for _unused in range(5))

    def groups(proposal):
        return (
            (months, _month(proposal['created_at'])),
            (types, proposal['proposal_type']),
            (sources, _source(proposal)),
            (models, proposal['llm_model']),
            (scenarios, proposal['scenario_id']),
        )

    # Every proposal is counted, reviewed or not.
    for proposal in proposals.values():
        for tallies, key in groups(proposal):
            tallies[key]['proposals'].add(proposal['id'])
    by_proposal = defaultdict(list)
    reasons = Counter()
    for review in reviews:
        proposal = proposals[review['proposal_id']]
        status = review['status']
        for tallies, key in groups(proposal):
            _count(tallies[key], proposal['id'], status)
        if status in DECIDED:
            by_proposal[proposal['id']].append(review)
        if status == 'rejected':
            reasons.update(review['rejection_reasons'] or [])

    accepted = sum(1 for r in reviews if r['status'] == 'accepted')
    rejected = sum(1 for r in reviews if r['status'] == 'rejected')
    summary = {
        'proposals': len(proposals),
        'proposals_decided': len(by_proposal),
        'decisions': accepted + rejected,
        'accepted': accepted,
        'rejected': rejected,
        'structural': sum(1 for r in reviews if r['status'] == 'rejected' and r['feedback_type'] == 'structural'),
        'acceptance': _rate(accepted, rejected),
        'undecided': len(reviews) - accepted - rejected,
        'undecided_share': (len(reviews) - accepted - rejected) / len(reviews) if reviews else None,
        'reviewers': len({r['user_id'] for r in reviews if r['status'] in DECIDED}),
        'scenarios': len({p['scenario_id'] for p in proposals.values()}),
    }

    month_rows = [_row(_period_label(key, key), months[key], month=key) for key in sorted(months)]
    type_labels = dict(ActivityProposal.PROPOSAL_TYPE_CHOICES)
    source_labels = {
        'bandit': _('Chosen by the learning bandit'),
        'before': _('Before the bandit (no selection policy recorded)'),
    }
    kinds = [
        {'title': _('By proposal type'), 'rows': [
            _row(type_labels.get(key, key), types[key]) for key in sorted(types)]},
        {'title': _('By how the action was chosen'), 'rows': [
            _row(source_labels[key], sources[key]) for key in ('bandit', 'before') if key in sources]},
        {'title': _('By language model'), 'rows': [
            _row(key or _('Not recorded'), models[key]) for key in sorted(models)]},
    ]
    names = {p['scenario_id']: p['scenario__name'] for p in proposals.values()}
    scenario_rows = sorted(
        (_row(names[key], tally) for key, tally in scenarios.items()),
        key=lambda row: (-row['decisions'], row['label']),
    )

    agreement = _agreement(by_proposal, proposals, type_labels)
    reason_labels = dict(REJECTION_REASON_CHOICES)
    reason_rows = [
        {'code': code, 'label': reason_labels.get(code, code), 'count': count}
        for code, count in reasons.most_common()
    ]
    decisions = [
        {
            'proposal_id': r['proposal_id'],
            'scenario': proposals[r['proposal_id']]['scenario__name'],
            'proposal_type': type_labels.get(proposals[r['proposal_id']]['proposal_type'], ''),
            'source': source_labels[_source(proposals[r['proposal_id']])],
            'llm_model': proposals[r['proposal_id']]['llm_model'],
            'proposed_at': proposals[r['proposal_id']]['created_at'],
            'reviewer': r['user__username'],
            'decision': r['status'],
            'feedback_type': r['feedback_type'],
            'reasons': [reason_labels.get(code, code) for code in r['rejection_reasons'] or []],
            'decided_at': r['reviewed_at'],
            'was_edited': r['was_edited'],
        }
        for r in reviews if r['status'] in DECIDED
    ]
    return {
        'summary': summary,
        'months': month_rows,
        'kinds': kinds,
        'scenarios': scenario_rows,
        'agreement': agreement,
        'reasons': reason_rows,
        'decisions': decisions,
        'chart': {
            'labels': [row['label'] for row in month_rows],
            'acceptance': [row['acceptance'] for row in month_rows],
            'decisions': [row['decisions'] for row in month_rows],
            'accepted': [row['accepted'] for row in month_rows],
            'rejected': [row['rejected'] for row in month_rows],
            'undecided': [row['undecided'] for row in month_rows],
        },
    }


def _agreement(by_proposal, proposals, type_labels):
    """Teachers against each other, on proposals that two or more of them decided."""
    shared = {pid: decided for pid, decided in by_proposal.items() if len(decided) >= 2}
    pairs = agreeing = unanimous = 0
    split = []
    for pid, decided in shared.items():
        statuses = [r['status'] for r in decided]
        for first, second in combinations(statuses, 2):
            pairs += 1
            agreeing += first == second
        if len(set(statuses)) == 1:
            unanimous += 1
        else:
            proposal = proposals[pid]
            split.append({
                'proposal_id': pid,
                'scenario': proposal['scenario__name'],
                'proposal_type': type_labels.get(proposal['proposal_type'], proposal['proposal_type']),
                'accepted': statuses.count('accepted'),
                'rejected': statuses.count('rejected'),
            })
    return {
        'proposals': len(shared),
        'unanimous': unanimous,
        'pairs': pairs,
        'pairwise': agreeing / pairs if pairs else None,
        'split': sorted(split, key=lambda row: row['proposal_id']),
    }
