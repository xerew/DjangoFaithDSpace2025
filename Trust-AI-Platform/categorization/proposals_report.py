"""How often teachers accept the AI activity proposals, and how that changes over time.

Read live from the database on each page load: there are few proposals, and
nothing here needs training. Every reviewer's decision counts once, so a
proposal decided by five teachers counts five times. A pending review is
created for every listed proposal as soon as someone opens a scenario's
proposal list, so an undecided review means the page was opened, not that the
proposal was read. It is "open" (still decidable) while the proposal is on
the review page, and "closed" (no longer decidable) once a newer generation
replaced it or its activity was removed. The overall rate on the chart adds
up every decision from the first month to each month, so one busy or quiet
month does not swing it.
"""

from collections import Counter, defaultdict
from itertools import combinations

from django.utils import timezone
from django.utils.translation import gettext as _

from authoringtool.models import (
    REJECTION_REASON_CHOICES, ActivityProposal, ProposalGenerationRun, UserProposalReview,
)

from .report import _period_label

DECIDED = ('accepted', 'rejected')
WHO_CHOICES = ('all', 'no_superusers', 'teachers')


def _month(when):
    if timezone.is_aware(when):
        when = timezone.localtime(when)
    return f'{when.year:04d}-{when.month:02d}'


def _months_between(first, last):
    """Every month from first to last ('YYYY-MM'), so gaps show as gaps."""
    year, month = map(int, first.split('-'))
    months = []
    while f'{year:04d}-{month:02d}' <= last:
        months.append(f'{year:04d}-{month:02d}')
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _rate(accepted, rejected):
    decided = accepted + rejected
    return accepted / decided if decided else None


def _tally():
    return {'proposals': set(), 'accepted': 0, 'rejected': 0, 'open': 0, 'closed': 0}


def _count(tally, proposal, status):
    tally['proposals'].add(proposal['id'])
    if status in DECIDED:
        tally[status] += 1
    elif proposal['open']:
        tally['open'] += 1
    else:
        tally['closed'] += 1


def _row(label, tally, **extra):
    return {
        'label': label,
        'proposals': len(tally['proposals']),
        'accepted': tally['accepted'],
        'rejected': tally['rejected'],
        'open': tally['open'],
        'closed': tally['closed'],
        'decisions': tally['accepted'] + tally['rejected'],
        'acceptance': _rate(tally['accepted'], tally['rejected']),
        **extra,
    }


def _source(proposal):
    return 'bandit' if proposal['selection_policy'] else 'before'


def _mark_open(proposals):
    """Whether teachers can still decide each proposal, as the review page shows them.

    The review page lists a scenario's current generation run, or, for a
    scenario without runs, the proposals saved before runs existed; a
    proposal whose activity was removed is never listed.
    """
    with_current_run = set(
        ProposalGenerationRun.objects.filter(is_current=True).values_list('scenario_id', flat=True)
    )
    for proposal in proposals.values():
        if proposal['activity_id'] is None:
            proposal['open'] = False
        elif proposal['generation_run_id'] is None:
            proposal['open'] = proposal['scenario_id'] not in with_current_run
        else:
            proposal['open'] = bool(proposal['generation_run__is_current'])


def _left_out(review, who):
    if who == 'teachers':
        return review['user__is_staff'] or review['user__is_superuser']
    if who == 'no_superusers':
        return review['user__is_superuser']
    return False


def _add_overall(month_rows):
    """Each month's overall rate: every decision from the first month up to it."""
    accepted = decided = 0
    for row in month_rows:
        accepted += row['accepted']
        decided += row['decisions']
        row['overall_accepted'] = accepted
        row['overall_decisions'] = decided
        row['overall_acceptance'] = _rate(accepted, decided - accepted)


def build(who='all'):
    """who: 'all' reviewers, 'no_superusers' (superuser accounts left out), or
    'teachers' only (staff and superuser accounts left out)."""
    proposals = {
        p['id']: p for p in ActivityProposal.objects.values(
            'id', 'scenario_id', 'scenario__name', 'proposal_type', 'selection_policy',
            'llm_model', 'created_at', 'activity_id', 'generation_run_id', 'generation_run__is_current',
        )
    }
    _mark_open(proposals)
    all_reviews = list(UserProposalReview.objects.values(
        'proposal_id', 'user_id', 'user__username', 'user__is_staff', 'user__is_superuser', 'status',
        'feedback_type', 'rejection_reasons', 'reviewed_at', 'was_edited',
    ).order_by('proposal_id', 'user_id'))
    staff_decisions = sum(
        1 for r in all_reviews
        if r['status'] in DECIDED and (r['user__is_staff'] or r['user__is_superuser'])
    )
    superuser_decisions = sum(
        1 for r in all_reviews if r['status'] in DECIDED and r['user__is_superuser']
    )
    reviews = [r for r in all_reviews if r['proposal_id'] in proposals and not _left_out(r, who)]

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
            _count(tallies[key], proposal, status)
        if status in DECIDED:
            by_proposal[proposal['id']].append(review)
        if status == 'rejected':
            reasons.update(review['rejection_reasons'] or [])

    accepted = sum(1 for r in reviews if r['status'] == 'accepted')
    rejected = sum(1 for r in reviews if r['status'] == 'rejected')
    undecided = [r for r in reviews if r['status'] not in DECIDED]
    still_open = sum(1 for r in undecided if proposals[r['proposal_id']]['open'])
    summary = {
        'who': who,
        'proposals': len(proposals),
        'proposals_decided': len(by_proposal),
        'decisions': accepted + rejected,
        'accepted': accepted,
        'rejected': rejected,
        'structural': sum(1 for r in reviews if r['status'] == 'rejected' and r['feedback_type'] == 'structural'),
        'acceptance': _rate(accepted, rejected),
        'open': still_open,
        'closed': len(undecided) - still_open,
        'reviewers': len({r['user_id'] for r in reviews if r['status'] in DECIDED}),
        'staff_decisions': staff_decisions,
        'superuser_decisions': superuser_decisions,
        'scenarios': len({p['scenario_id'] for p in proposals.values()}),
    }

    keys = _months_between(min(months), max(months)) if months else []
    month_rows = [_row(_period_label(key, key), months[key] if key in months else _tally(), month=key)
                  for key in keys]
    _add_overall(month_rows)
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

    reason_labels = dict(REJECTION_REASON_CHOICES)
    status_labels = {'accepted': _('Accepted'), 'rejected': _('Rejected')}
    feedback_labels = dict(UserProposalReview._meta.get_field('feedback_type').choices)
    decisions = []
    for r in reviews:
        if r['status'] not in DECIDED:
            continue
        proposal = proposals[r['proposal_id']]
        decisions.append({
            'proposal_id': r['proposal_id'],
            'scenario': proposal['scenario__name'],
            'proposal_type': type_labels.get(proposal['proposal_type'], proposal['proposal_type']),
            'source': source_labels[_source(proposal)],
            'llm_model': proposal['llm_model'],
            'proposed_at': proposal['created_at'],
            'reviewer': r['user__username'],
            'staff': bool(r['user__is_staff'] or r['user__is_superuser']),
            'decision': status_labels[r['status']],
            'feedback_type': feedback_labels.get(r['feedback_type'], r['feedback_type']),
            'reasons': [reason_labels.get(code, code) for code in r['rejection_reasons'] or []],
            'last_changed': r['reviewed_at'],
            'was_edited': r['was_edited'],
        })
    return {
        'summary': summary,
        'months': month_rows,
        'kinds': kinds,
        'scenarios': scenario_rows,
        'agreement': _agreement(by_proposal, proposals, type_labels),
        'reasons': [
            {'code': code, 'label': reason_labels.get(code, code), 'count': count}
            for code, count in reasons.most_common()
        ],
        'decisions': decisions,
        'chart': {
            'labels': [row['label'] for row in month_rows],
            'overall': [row['overall_acceptance'] for row in month_rows],
            'overall_accepted': [row['overall_accepted'] for row in month_rows],
            'overall_decisions': [row['overall_decisions'] for row in month_rows],
            'decisions': [row['decisions'] for row in month_rows],
            'accepted': [row['accepted'] for row in month_rows],
            'rejected': [row['rejected'] for row in month_rows],
            'open': [row['open'] for row in month_rows],
            'closed': [row['closed'] for row in month_rows],
        },
    }


def _agreement(by_proposal, proposals, type_labels):
    """Reviewers against each other, on proposals that two or more of them decided.

    Pairs are pooled: a proposal decided by 6 people adds 15 pairs, one decided
    by 2 adds 1. Chance agreement is what two reviewers deciding independently,
    at the overall acceptance rate of these decisions, would reach.
    """
    shared = {pid: decided for pid, decided in by_proposal.items() if len(decided) >= 2}
    pairs = agreeing = unanimous = 0
    split = []
    statuses_all = []
    for pid, decided in shared.items():
        statuses = [r['status'] for r in decided]
        statuses_all.extend(statuses)
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
    chance = shared_rate = None
    if statuses_all:
        shared_rate = statuses_all.count('accepted') / len(statuses_all)
        chance = shared_rate ** 2 + (1 - shared_rate) ** 2
    return {
        'proposals': len(shared),
        'unanimous': unanimous,
        'pairs': pairs,
        'pairwise': agreeing / pairs if pairs else None,
        'shared_rate': shared_rate,
        'chance': chance,
        # All accepted or all rejected: any two reviewers agree, whatever they think.
        'one_sided': len(set(statuses_all)) == 1,
        'split': sorted(split, key=lambda row: row['proposal_id']),
    }
