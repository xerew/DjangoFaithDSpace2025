from celery import shared_task
import numpy as np
import pandas as pd
from django.conf import settings
from .models import (
    Activity,
    ActivityFlag,
    ActivityProposal,
    ActivityType,
    Answer,
    BanditPolicyConfiguration,
    CategoryTag,
    EvQuestionBranching,
    NextQuestionLogic,
    Phase,
    ProposalGenerationRun,
    ProposalStructuralFailure,
    QValue,
    QuestionBunch,
    Scenario,
    UserAnswer,
    UserProposalReview,
)
from .evidence import (
    get_evidence_answers,
    get_evidence_context,
    get_evidence_versions,
    normalize_evidence_language,
    normalize_evidence_scope,
)
from .graph_validation import (
    ScenarioGraphValidationError,
    assert_scenario_graph_integrity,
)
from django.core.cache import cache
from datetime import timedelta
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.contrib.auth.models import User
from usergroups.models import UserGroupMembership
from django.utils.dateparse import parse_date
from django.db.models import Sum, Count, Q, Max, Min, Avg, F, FloatField, ExpressionWrapper, Prefetch
from collections import defaultdict
from .utils import (
    get_eligible_user_answers,
    get_first_answers,
    get_last_answers,
    get_scenario_evidence_cache_paths,
)
import os, io, json
import csv
from django.utils.timezone import now, make_aware
import datetime
from datetime import datetime
import statistics
import time
import requests
import re
from collections import deque
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import chromadb
import glob
import random
import math
from django.db import transaction
import shutil, errno, stat
from pathlib import Path
import traceback


@shared_task
def scan_scenario_family_candidates_task(
    scenario_id=None,
    scenario_ids=None,
    force_profiles=False,
):
    """Build profiles and discover reviewable scenario-family candidates."""
    from .scenario_matching import scan_scenario_family_candidates

    return scan_scenario_family_candidates(
        scenario_id=scenario_id,
        scenario_ids=scenario_ids,
        force_profiles=force_profiles,
    )


@shared_task
def review_scenario_family_candidate_with_llm_task(candidate_id):
    """Generate and persist one non-binding Ollama family review."""
    from .scenario_matching import review_candidate_with_llm

    try:
        candidate = review_candidate_with_llm(candidate_id)
    except Exception as exc:
        return {
            'candidate_id': candidate_id,
            'status': 'failed',
            'error': str(exc)[:500],
        }
    return {
        'candidate_id': candidate.id,
        'status': candidate.llm_status,
        'suggestion': candidate.llm_suggested_relationship,
        'confidence': (
            float(candidate.llm_confidence)
            if candidate.llm_confidence is not None
            else None
        ),
    }


@shared_task
def review_scenario_family_candidates_with_llm_task(candidate_ids):
    """Review a bounded administrator-selected candidate batch."""
    results = {
        'requested': len(candidate_ids),
        'completed': 0,
        'failed': 0,
    }
    for candidate_id in candidate_ids:
        result = review_scenario_family_candidate_with_llm_task(
            candidate_id
        )
        if result['status'] == 'completed':
            results['completed'] += 1
        else:
            results['failed'] += 1
    return results


def _date_to_aware(d):
    """Convert a date object to a timezone-aware datetime at midnight."""
    return make_aware(datetime.combine(d, datetime.min.time()))


def _on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree on Windows:
    * If the failure is due to a permission error, make the file writable and retry.
    * Otherwise, re-raise.
    """
    exc_type, exc_value, _ = exc_info
    if exc_type is PermissionError:
        # remove read-only flag and retry
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise exc_value
    
@shared_task
def compute_sankey_data(scenario_id, group_ids, start_date, end_date):
    version_id = (
        Scenario.objects.get(pk=scenario_id).ensure_current_version().id
    )
    cache_key = f"analytics:sankey:{scenario_id}:v{version_id}:{group_ids}:{start_date}:{end_date}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    scenario = get_object_or_404(Scenario, id=scenario_id)
    start_date = start_date
    end_date = end_date

    phases = Phase.objects.filter(scenario=scenario).order_by('id')
    
    nodes = []
    links = []
    phase_performance = {phase.id: {'High': 0, 'Moderate': 0, 'Low': 0} for phase in phases}

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)

    # 25 FEB
    if group_ids:
        group_ids = [int(g) for g in group_ids.split(',')]  # Convert string IDs to list of integers
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct()# .exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) & 
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct()
    
    users = users.exclude(groups__name='teachers')

    if start_date:
        start_date = parse_date(start_date)
        if start_date:
            last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    
    if end_date:
        end_date = parse_date(end_date)
        if end_date:
            last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))

    # Get the minimum activity ID in the scenario
    min_activity_id = scenario.get_start_activity()

    if not min_activity_id:
        return {"error": "No activities found for this scenario", "status": 400}

    valid_users = []  # List of users who started with the minimum activity

    for user in users:
        # Check if user has answered the minimum activity
        if last_answers.filter(user=user, activity=min_activity_id).exists():
            valid_users.append(user)

    user_performance = {user.id: [] for user in valid_users}

    # ── Bulk prefetch to avoid O(users × phases × activities) queries ────────
    all_scenario_activities = list(Activity.objects.filter(scenario=scenario))
    scenario_activity_ids = [a.id for a in all_scenario_activities]

    # 1. QuestionBunch dict keyed by activity_primary_id
    bunch_map = {
        qb.activity_primary_id: qb.activity_ids
        for qb in QuestionBunch.objects.filter(activity_primary_id__in=scenario_activity_ids)
    }

    # 2. UserAnswer dict keyed by (user_id, activity_id) — last answer per pair
    ua_qs = last_answers.filter(user_id__in=[u.id for u in valid_users])
    last_answer_map = {}
    for ua in ua_qs.select_related('answer'):
        key = (ua.user_id, ua.activity_id)
        if key not in last_answer_map or ua.id > last_answer_map[key].id:
            last_answer_map[key] = ua

    # 3. Max answer weight dict keyed by activity_id
    max_weight_map = {
        row['activity_id']: row['max_weight']
        for row in Answer.objects.filter(activity_id__in=scenario_activity_ids)
        .values('activity_id')
        .annotate(max_weight=Max('answer_weight'))
    }

    # Activities grouped by phase for quick lookup
    activities_by_phase = defaultdict(list)
    for act in all_scenario_activities:
        activities_by_phase[act.phase_id].append(act)

    for user in valid_users:
        for phase in phases:
            activities = activities_by_phase.get(phase.id, [])
            total_primary_score = 0
            total_primary_max_score = 0
            processed_activities = set()  # To avoid duplicates

            # Handle primary evaluatable activities (only)
            primary_evaluatable_activities = [a for a in activities if a.is_evaluatable and a.is_primary_ev]
            primary_count = len(primary_evaluatable_activities)

            if primary_count > 0:
                primary_weight_share = 100 / primary_count  # Each primary activity contributes equally

                for primary_activity in primary_evaluatable_activities:
                    bunch_act_ids = bunch_map.get(primary_activity.id, [primary_activity.id])

                    for act_id in bunch_act_ids:
                        if act_id not in processed_activities:
                            user_last_answer = last_answer_map.get((user.id, act_id))
                            if user_last_answer and user_last_answer.answer:
                                total_primary_score += (user_last_answer.answer.answer_weight * primary_weight_share) / 100

                            # Calculate max score for each bunch activity
                            max_weight = max_weight_map.get(act_id)
                            if max_weight:
                                total_primary_max_score += (max_weight * primary_weight_share) / 100

                            # Mark bunch activity as processed
                            processed_activities.add(act_id)

            if total_primary_max_score > 0:
                percentage_score = (total_primary_score / total_primary_max_score) * 100

                # Determine the performance category
                if percentage_score >= 83.3:
                    phase_performance[phase.id]['High'] += 1
                    user_performance[user.id].append('High')
                elif percentage_score >= 49.7:
                    phase_performance[phase.id]['Moderate'] += 1
                    user_performance[user.id].append('Moderate')
                else:
                    phase_performance[phase.id]['Low'] += 1
                    user_performance[user.id].append('Low')
            else:
                user_performance[user.id].append('None')

    # Prepare nodes
    for phase in phases:
        nodes.append({'name': f'High Performers {phase.name}', 'position': 1})
        nodes.append({'name': f'Moderate Performers {phase.name}', 'position': 2})
        nodes.append({'name': f'Low Performers {phase.name}', 'position': 3})

    # Prepare links
    transition_counts = {}
    for i in range(len(phases) - 1):
        current_phase = phases[i]
        next_phase = phases[i + 1]
        transition_counts[(current_phase.id, next_phase.id)] = {
            'High-High': 0, 'High-Moderate': 0, 'High-Low': 0,
            'Moderate-High': 0, 'Moderate-Moderate': 0, 'Moderate-Low': 0,
            'Low-High': 0, 'Low-Moderate': 0, 'Low-Low': 0
        }

        for user in valid_users:
            current_performance = user_performance[user.id][i]
            next_performance = user_performance[user.id][i + 1]

            if current_performance != 'None' and next_performance != 'None':
                transition_counts[(current_phase.id, next_phase.id)][f'{current_performance}-{next_performance}'] += 1

    for (current_phase_id, next_phase_id), counts in transition_counts.items():
        current_phase = Phase.objects.get(id=current_phase_id)
        next_phase = Phase.objects.get(id=next_phase_id)

        for transition, count in counts.items():
            if count > 0:
                source_performance, target_performance = transition.split('-')
                source_name = f'{source_performance} Performers {current_phase.name}'
                target_name = f'{target_performance} Performers {next_phase.name}'
                if source_name in [node['name'] for node in nodes] and target_name in [node['name'] for node in nodes]:
                    links.append({
                        'source': source_name,
                        'target': target_name,
                        'value': count
                    })

    # Filter nodes to only include those with links
    linked_nodes = set()
    for link in links:
        linked_nodes.add(link['source'])
        linked_nodes.add(link['target'])

    nodes = [node for node in nodes if node['name'] in linked_nodes]
    data = {'nodes': nodes, 'links': links}

    cache.set(cache_key, data, timeout=3600)
    return data

@shared_task
def compute_final_performance(scenario_id, group_ids, start_date, end_date):
    version_id = (
        Scenario.objects.get(pk=scenario_id).ensure_current_version().id
    )
    cache_key = f"analytics:final_perf:{scenario_id}:v{version_id}:{group_ids}:{start_date}:{end_date}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    scenario = get_object_or_404(Scenario, id=scenario_id)
    start_date = start_date
    end_date = end_date

    phases = Phase.objects.filter(scenario=scenario).order_by('id')
    
    # Define the weights for each phase in the order they come
    phase_weights = [0.2, 0.2, 0.45, 0.15]
    if len(phases) > 4:
        phase_weights = [0.2, 0.2, 0.3, 0.15, 0.15]

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)
    
    # FEB 28
    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct()# .exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) & 
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct()
    
    users = users.exclude(groups__name='teachers')
    
    # Apply start_date and end_date filters
    if start_date:
        start_date = parse_date(start_date)
        if start_date:
            last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    
    if end_date:
        end_date = parse_date(end_date)
        if end_date:
            last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))

    performance_counts = {'High': 0, 'Moderate': 0, 'Low': 0}
    user_performance = {user.id: {'weighted_score': 0, 'max_weighted_score': 0, 'phases_completed': 0} for user in users}

    # Get the minimum activity ID in the scenario
    min_activity_id = scenario.get_start_activity()

    if not min_activity_id:
        return {"error": "No activities found for this scenario", "status": 400}

    valid_users = []  # List of users who started with the minimum activity

    for user in users:
        # Check if user has answered the minimum activity
        if last_answers.filter(user=user, activity=min_activity_id).exists():
            valid_users.append(user)

    # ── Bulk prefetch to avoid O(users × phases × activities) queries ────────
    all_scenario_activities = list(Activity.objects.filter(scenario=scenario))
    scenario_activity_ids = [a.id for a in all_scenario_activities]

    # 1. QuestionBunch dict keyed by activity_primary_id
    bunch_map = {
        qb.activity_primary_id: qb.activity_ids
        for qb in QuestionBunch.objects.filter(activity_primary_id__in=scenario_activity_ids)
    }

    # 2. UserAnswer dict keyed by (user_id, activity_id) — last answer per pair
    ua_qs = last_answers.filter(user_id__in=[u.id for u in valid_users])
    last_answer_map = {}
    for ua in ua_qs.select_related('answer'):
        key = (ua.user_id, ua.activity_id)
        if key not in last_answer_map or ua.id > last_answer_map[key].id:
            last_answer_map[key] = ua

    # 3. Max answer weight dict keyed by activity_id
    max_weight_map = {
        row['activity_id']: row['max_weight']
        for row in Answer.objects.filter(activity_id__in=scenario_activity_ids)
        .values('activity_id')
        .annotate(max_weight=Max('answer_weight'))
    }

    # Activities grouped by phase for quick lookup
    activities_by_phase = defaultdict(list)
    for act in all_scenario_activities:
        activities_by_phase[act.phase_id].append(act)

    for user in valid_users:
        for index, phase in enumerate(phases):
            primary_total_score = 0
            primary_max_score = 0
            processed_activities = set()

            # Handle primary evaluatable activities
            primary_evaluatable_activities = [a for a in activities_by_phase.get(phase.id, []) if a.is_evaluatable and a.is_primary_ev]
            primary_count = len(primary_evaluatable_activities)

            if primary_count > 0:
                primary_weight_share = 100 / primary_count  # Each primary activity contributes equally

                for primary_activity in primary_evaluatable_activities:
                    bunch_act_ids = bunch_map.get(primary_activity.id, [primary_activity.id])

                    for act_id in bunch_act_ids:
                        if act_id not in processed_activities:
                            user_last_answer = last_answer_map.get((user.id, act_id))
                            if user_last_answer and user_last_answer.answer:
                                primary_total_score += (user_last_answer.answer.answer_weight * primary_weight_share) / 100

                            # Calculate max score for each bunch activity
                            max_weight = max_weight_map.get(act_id)
                            if max_weight:
                                primary_max_score += (max_weight * primary_weight_share) / 100

                            # Mark the bunch activity as processed
                            processed_activities.add(act_id)

            # Calculate weighted scores using the corresponding phase weight
            if primary_max_score > 0:
                weight = phase_weights[index]  # Get the weight based on phase order
                weighted_score = (primary_total_score / primary_max_score) * weight
                max_weighted_score = weight
                user_performance[user.id]['weighted_score'] += weighted_score
                user_performance[user.id]['max_weighted_score'] += max_weighted_score
                user_performance[user.id]['phases_completed'] += 1  # Track completed phases
    
    total_students = len(valid_users)
    waterfall_data = [{'name': 'Total', 'value': total_students}]
    
    for user in valid_users:
        user_data = user_performance[user.id]
        if user_data['max_weighted_score'] > 0 and user_data['phases_completed'] > 0:
            percentage_score = (user_data['weighted_score'] / user_data['max_weighted_score']) * 100
            if percentage_score >= 83.3:
                performance_category = 'High'
            elif percentage_score >= 49.7:
                performance_category = 'Moderate'
            else:
                performance_category = 'Low'

            performance_counts[performance_category] += 1
    
    waterfall_data.append({'name': 'High', 'value': performance_counts['High']})
    waterfall_data.append({'name': 'Moderate', 'value': performance_counts['Moderate']})
    waterfall_data.append({'name': 'Low', 'value': performance_counts['Low']})

    data = {'waterfall_data': waterfall_data}

    cache.set(cache_key, data, timeout=3600)
    return data

@shared_task
def compute_activity_answers_data(scenario_id, group_ids, start_date, end_date, activity_type):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    data_type = activity_type
    start_date = start_date
    end_date = end_date
    
    # Validate and parse dates
    if start_date:
        start_date = parse_date(str(start_date))
    if end_date:
        end_date = parse_date(str(end_date))

    data = {
        'categories': [],
        'correct': [],
        'incorrect': []
    }

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)
    
    # Filter users based on your criteria
    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct().exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) & 
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct().exclude(groups__name='teachers')

    # Get the IDs of the filtered users
    user_ids = users.values_list('id', flat=True)

    # Filter `last_answers` to include only entries from the specified users
    filtered_last_answers = last_answers.filter(user_id__in=user_ids)

    # Get the minimum activity ID in the scenario
    min_activity = scenario.get_start_activity()

    if not min_activity:
        return {"error": "No activities found for this scenario", "status": 400}

    # List of valid user IDs who started with the minimum activity
    valid_user_ids = []

    for user in users:
        # Check if the user has answered the minimum activity
        if filtered_last_answers.filter(user=user, activity=min_activity).exists():
            valid_user_ids.append(user.id)

    # Now filter user_ids to include only those in valid_user_ids
    user_ids = [user_id for user_id in user_ids if user_id in valid_user_ids]

    # Filter `filtered_last_answers` to include only answers from valid users
    last_answers = filtered_last_answers.filter(user_id__in=user_ids) # filtered_last_answers

    # Apply start_date and end_date filters to last answers
    if start_date:
        last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    if end_date:
        last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))

    if data_type == 'activities':
        # Handle activities as in the previous logic
        activity_type = get_object_or_404(ActivityType, name='Question')
        activities = Activity.objects.filter(scenario=scenario, activity_type=activity_type)

        for activity in activities:
            data['categories'].append(activity.name)

            # Filter last answers for this specific activity
            activity_last_answers = last_answers.filter(activity=activity)

            # Separate correct and incorrect answers based on the last answer
            correct_answers = activity_last_answers.filter(answer__is_correct=True)
            incorrect_answers = activity_last_answers.filter(answer__is_correct=False)

            # Get distinct user IDs for correct and incorrect answers
            correct_user_ids = correct_answers.values_list('user_id', flat=True).distinct()
            incorrect_user_ids = incorrect_answers.exclude(user_id__in=correct_user_ids).values_list('user_id', flat=True).distinct()

            data['correct'].append(len(correct_user_ids))
            data['incorrect'].append(len(incorrect_user_ids))
            
    else:
        # Now we're focusing on phases
        phases = Phase.objects.filter(scenario=scenario).order_by('id')

        for phase in phases:
            data['categories'].append(phase.name)

            # Initialize counters for the current phase
            total_correct_in_phase = 0
            total_incorrect_in_phase = 0

            # Fetch activities for the phase
            activities = Activity.objects.filter(phase=phase)

            for activity in activities:
                # Filter last answers for each activity in the phase
                activity_last_answers = last_answers.filter(activity=activity)

                # Separate correct and incorrect answers
                correct_answers = activity_last_answers.filter(answer__is_correct=True)
                incorrect_answers = activity_last_answers.filter(answer__is_correct=False)

                # Get distinct user IDs for correct and incorrect answers
                correct_user_ids = list(correct_answers.values_list('user_id', flat=True).distinct())
                incorrect_user_ids = list(incorrect_answers.values_list('user_id', flat=True).distinct())

                # Remove correct user IDs from incorrect user IDs
                incorrect_user_ids = [uid for uid in incorrect_user_ids if uid not in correct_user_ids]

                # Add the counts to the phase totals
                total_correct_in_phase += len(correct_user_ids)
                total_incorrect_in_phase += len(incorrect_user_ids)

            # Append the summed counts for the entire phase
            data['correct'].append(total_correct_in_phase)
            data['incorrect'].append(total_incorrect_in_phase)

    return data

@shared_task
def compute_performance_data(scenario_id, group_ids, start_date, end_date):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    start_date = start_date
    end_date = end_date
    
    phases = Phase.objects.filter(scenario=scenario)

    phase_performance = {phase.name: {'High': 0, 'Mid': 0, 'Low': 0} for phase in phases}
    phase_total_users = {phase.name: 0 for phase in phases}

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)

    # Apply start_date and end_date filters
    if start_date:
        start_date = parse_date(start_date)
        if start_date:
            last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    
    if end_date:
        end_date = parse_date(end_date)
        if end_date:
            last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))

    # Get the minimum activity ID in the scenario
    min_activity_id = scenario.get_start_activity()

    if not min_activity_id:
        return {"error": "No activities found for this scenario", "status": 400}

    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct()# .exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) &
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct()

    users = users.exclude(groups__name='teachers')

    valid_users = []  # List of users who started with the minimum activity

    for user in users:
        # Check if user has answered the minimum activity
        if last_answers.filter(user=user, activity=min_activity_id).exists():
            valid_users.append(user)

    # ── Bulk prefetch to avoid O(users × phases × activities) queries ────────
    all_scenario_activities = list(Activity.objects.filter(scenario=scenario))
    scenario_activity_ids = [a.id for a in all_scenario_activities]

    # 1. QuestionBunch dict keyed by activity_primary_id
    bunch_map = {
        qb.activity_primary_id: qb.activity_ids
        for qb in QuestionBunch.objects.filter(activity_primary_id__in=scenario_activity_ids)
    }

    # 2. UserAnswer dict keyed by (user_id, activity_id) — last answer per pair
    ua_qs = last_answers.filter(user_id__in=[u.id for u in valid_users])
    last_answer_map = {}
    for ua in ua_qs.select_related('answer'):
        key = (ua.user_id, ua.activity_id)
        if key not in last_answer_map or ua.id > last_answer_map[key].id:
            last_answer_map[key] = ua

    # 3. Max answer weight dict keyed by activity_id
    max_weight_map = {
        row['activity_id']: row['max_weight']
        for row in Answer.objects.filter(activity_id__in=scenario_activity_ids)
        .values('activity_id')
        .annotate(max_weight=Max('answer_weight'))
    }

    # Activities grouped by phase for quick lookup
    activities_by_phase = defaultdict(list)
    for act in all_scenario_activities:
        activities_by_phase[act.phase_id].append(act)

    for user in valid_users:
        for phase in phases:
            activities = activities_by_phase.get(phase.id, [])
            total_primary_score = 0
            total_primary_max_score = 0
            processed_activities = set()  # To avoid duplicates

            # Handle only primary evaluatable activities (and their bunches)
            primary_evaluatable_activities = [a for a in activities if a.is_evaluatable and a.is_primary_ev]
            primary_count = len(primary_evaluatable_activities)

            if primary_count > 0:
                primary_weight_share = 100 / primary_count  # Each primary activity contributes equally

                for primary_activity in primary_evaluatable_activities:
                    bunch_act_ids = bunch_map.get(primary_activity.id, [primary_activity.id])

                    for act_id in bunch_act_ids:
                        if act_id not in processed_activities:
                            user_last_answer = last_answer_map.get((user.id, act_id))
                            if user_last_answer and user_last_answer.answer:
                                total_primary_score += (user_last_answer.answer.answer_weight * primary_weight_share) / 100

                            # Calculate max score for each bunch activity
                            max_weight = max_weight_map.get(act_id)
                            if max_weight:
                                total_primary_max_score += (max_weight * primary_weight_share) / 100

                            # Mark bunch activity as processed
                            processed_activities.add(act_id)

            # Calculate performance based on total primary score
            if total_primary_max_score > 0:
                percentage_score = (total_primary_score / total_primary_max_score) * 100

                # Apply performance thresholds for primary evaluatable activities
                if percentage_score >= 83.3:
                    phase_performance[phase.name]['High'] += 1
                elif percentage_score >= 49.7:
                    phase_performance[phase.name]['Mid'] += 1
                else:
                    phase_performance[phase.name]['Low'] += 1

                phase_total_users[phase.name] += 1

    # Normalize the performance percentages by total users
    for phase in phases:
        total_users = phase_total_users[phase.name]
        if total_users > 0:
            phase_performance[phase.name]['High'] = (phase_performance[phase.name]['High'] / total_users) * 100
            phase_performance[phase.name]['Mid'] = (phase_performance[phase.name]['Mid'] / total_users) * 100
            phase_performance[phase.name]['Low'] = (phase_performance[phase.name]['Low'] / total_users) * 100

    return phase_performance

@shared_task
def compute_time_spent_data(scenario_id, group_ids, start_date, end_date, activity_type):
    version_id = (
        Scenario.objects.get(pk=scenario_id).ensure_current_version().id
    )
    cache_key = f"analytics:time_spent:{scenario_id}:v{version_id}:{group_ids}:{start_date}:{end_date}:{activity_type}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    scenario = get_object_or_404(Scenario, id=scenario_id)
    data_type = activity_type

    _sd = parse_date(start_date) if isinstance(start_date, str) and start_date else None
    _ed = parse_date(end_date) if isinstance(end_date, str) and end_date else None
    start_dt = _date_to_aware(_sd) if _sd else None
    end_dt = _date_to_aware(_ed) if _ed else None

    activity_type_q = get_object_or_404(ActivityType, name='Question')

    # ── 1. Build filtered student user ID set once ──────────────────────────
    if group_ids:
        user_ids = list(
            UserGroupMembership.objects
            .filter(group_id__in=group_ids)
            .values_list('user_id', flat=True)
            .distinct()
        )
    else:
        user_ids = list(
            User.objects
            .filter(
                Q(school_department__isnull=False) |
                Q(id__in=UserGroupMembership.objects.values('user_id'))
            )
            .values_list('id', flat=True)
            .distinct()
        )
    teacher_ids = list(
        User.objects.filter(groups__name='teachers').values_list('id', flat=True)
    )
    user_ids = [uid for uid in user_ids if uid not in set(teacher_ids)]

    # ── 2. Build base last_answers queryset with date + user filters applied ─
    last_answers = get_last_answers(scenario_id).filter(user_id__in=user_ids)
    if start_dt:
        last_answers = last_answers.filter(created_on__gte=start_dt)
    if end_dt:
        last_answers = last_answers.filter(created_on__lte=end_dt + timedelta(days=1))

    data = {'categories': [], 'time_spent': []}

    if data_type == 'activities_timing':
        activities = list(Activity.objects.filter(scenario=scenario, activity_type=activity_type_q))

        # ONE query: total timing + distinct student count per activity
        stats = {
            row['activity_id']: row
            for row in (
                last_answers
                .filter(activity__in=activities)
                .values('activity_id')
                .annotate(total_time=Sum('timing'), student_count=Count('user', distinct=True))
            )
        }

        for activity in activities:
            data['categories'].append(activity.name)
            s = stats.get(activity.id, {})
            total = s.get('total_time') or 0
            count = s.get('student_count') or 1
            data['time_spent'].append(total / count)

    else:
        phases = list(Phase.objects.filter(scenario=scenario))

        # ONE query: total timing + distinct student count per phase
        stats = {
            row['activity__phase_id']: row
            for row in (
                last_answers
                .values('activity__phase_id')
                .annotate(total_time=Sum('timing'), student_count=Count('user', distinct=True))
            )
        }

        for phase in phases:
            data['categories'].append(phase.name)
            s = stats.get(phase.id, {})
            total = s.get('total_time') or 0
            count = s.get('student_count') or 1
            data['time_spent'].append(total / count)

    cache.set(cache_key, data, timeout=3600)
    return data

@shared_task
def compute_detailed_phase_scores_data(scenario_id, group_ids, start_date, end_date):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    start_date = start_date
    end_date = end_date

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)
    
    # Apply start_date and end_date filters
    if start_date:
        start_date = parse_date(start_date)
        last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    
    if end_date:
        end_date = parse_date(end_date)
        last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))
    
    # Get the minimum activity ID in the scenario
    min_activity_id = scenario.get_start_activity()

    if not min_activity_id:
        return {"error": "No activities found for this scenario", "status": 400}

    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct()# .exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) &
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct()

    users = users.exclude(groups__name='teachers')

    valid_users = []  # List of users who started with the minimum activity

    for user in users:
        # Check if user has answered the minimum activity
        if last_answers.filter(user=user, activity=min_activity_id).exists():
            valid_users.append(user)

    phases = Phase.objects.filter(scenario=scenario)

    phase_scores = {phase.name: {'low': 0, 'mid': 0, 'high': 0, 'low_score': 0, 'mid_score': 0, 'high_score': 0, 'total_users': 0} for phase in phases}

    # ── Bulk prefetch to avoid O(users × phases × activities) queries ────────
    all_scenario_activities = list(Activity.objects.filter(scenario=scenario))
    scenario_activity_ids = [a.id for a in all_scenario_activities]

    # 1. QuestionBunch dict keyed by activity_primary_id
    bunch_map = {
        qb.activity_primary_id: qb.activity_ids
        for qb in QuestionBunch.objects.filter(activity_primary_id__in=scenario_activity_ids)
    }

    # 2. UserAnswer dict keyed by (user_id, activity_id) — last answer per pair
    ua_qs = last_answers.filter(user_id__in=[u.id for u in valid_users])
    last_answer_map = {}
    for ua in ua_qs.select_related('answer'):
        key = (ua.user_id, ua.activity_id)
        if key not in last_answer_map or ua.id > last_answer_map[key].id:
            last_answer_map[key] = ua

    # 3. Max answer weight dict keyed by activity_id
    max_weight_map = {
        row['activity_id']: row['max_weight']
        for row in Answer.objects.filter(activity_id__in=scenario_activity_ids)
        .values('activity_id')
        .annotate(max_weight=Max('answer_weight'))
    }

    # Activities grouped by phase for quick lookup
    activities_by_phase = defaultdict(list)
    for act in all_scenario_activities:
        activities_by_phase[act.phase_id].append(act)

    for user in valid_users:
        for phase in phases:
            total_primary_score = 0
            total_primary_max_score = 0
            processed_activities = set()

            activities_in_phase = activities_by_phase.get(phase.id, [])

            # Process primary evaluatable activities (and their bunches)
            primary_evaluatable_activities = [a for a in activities_in_phase if a.is_evaluatable and a.is_primary_ev]
            primary_count = len(primary_evaluatable_activities)

            if primary_count > 0:
                primary_weight_share = 100 / primary_count  # Each primary activity contributes equally

                for primary_activity in primary_evaluatable_activities:
                    bunch_act_ids = bunch_map.get(primary_activity.id, [primary_activity.id])

                    for act_id in bunch_act_ids:
                        if act_id not in processed_activities:
                            user_last_answer = last_answer_map.get((user.id, act_id))
                            if user_last_answer and user_last_answer.answer:
                                total_primary_score += (user_last_answer.answer.answer_weight * primary_weight_share) / 100

                            # Calculate max score for each bunch activity
                            max_weight = max_weight_map.get(act_id)
                            if max_weight:
                                total_primary_max_score += (max_weight * primary_weight_share) / 100

                            # Mark bunch activity as processed
                            processed_activities.add(act_id)

            # Calculate performance based on primary evaluatable activities only
            if total_primary_max_score > 0:
                percentage_score = (total_primary_score / total_primary_max_score) * 100
                if percentage_score >= 83.3:
                    phase_scores[phase.name]['high'] += 1
                    phase_scores[phase.name]['high_score'] += percentage_score
                elif percentage_score >= 49.7:
                    phase_scores[phase.name]['mid'] += 1
                    phase_scores[phase.name]['mid_score'] += percentage_score
                else:
                    phase_scores[phase.name]['low'] += 1
                    phase_scores[phase.name]['low_score'] += percentage_score

                phase_scores[phase.name]['total_users'] += 1

    # Prepare response data
    data = {
        'categories': [],
        'low': [],
        'mid': [],
        'high': [],
        'average': []
    }

    for phase in phases:
        data['categories'].append(phase.name)
        total_users = phase_scores[phase.name]['total_users']
        if total_users > 0:
            low_avg = phase_scores[phase.name]['low_score'] / phase_scores[phase.name]['low'] if phase_scores[phase.name]['low'] > 0 else 0
            mid_avg = phase_scores[phase.name]['mid_score'] / phase_scores[phase.name]['mid'] if phase_scores[phase.name]['mid'] > 0 else 0
            high_avg = phase_scores[phase.name]['high_score'] / phase_scores[phase.name]['high'] if phase_scores[phase.name]['high'] > 0 else 0
            overall_avg = (phase_scores[phase.name]['low_score'] + phase_scores[phase.name]['mid_score'] + phase_scores[phase.name]['high_score']) / total_users
        else:
            low_avg = mid_avg = high_avg = overall_avg = 0
        data['low'].append(low_avg)
        data['mid'].append(mid_avg)
        data['high'].append(high_avg)
        data['average'].append(overall_avg)

    return data

@shared_task
def compute_performers_data(scenario_id, group_ids, start_date, end_date):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    start_date = start_date
    end_date = end_date
    
    phases = Phase.objects.filter(scenario=scenario)

    phase_performance = {phase.name: {'low': 0, 'mid': 0, 'high': 0, 'total_users': 0, 'total_score': 0} for phase in phases}

    # Get the last answers (only the latest answer for each user/activity combination)
    last_answers = get_last_answers(scenario_id)

    # Apply start_date and end_date filters
    if start_date:
        start_date = parse_date(start_date)
        if start_date:
            last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    
    if end_date:
        end_date = parse_date(end_date)
        if end_date:
            last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))

    # Get the minimum activity ID in the scenario
    min_activity_id = scenario.get_start_activity()

    if not min_activity_id:
        return {"error": "No activities found for this scenario", "status": 400}

    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct()# .exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) &
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct()

    users = users.exclude(groups__name='teachers')

    valid_users = []  # List of users who started with the minimum activity

    for user in users:
        # Check if user has answered the minimum activity
        if last_answers.filter(user=user, activity=min_activity_id).exists():
            valid_users.append(user)

    # ── Bulk prefetch to avoid O(users × phases × activities) queries ────────
    all_scenario_activities = list(Activity.objects.filter(scenario=scenario))
    scenario_activity_ids = [a.id for a in all_scenario_activities]

    # 1. QuestionBunch dict keyed by activity_primary_id
    bunch_map = {
        qb.activity_primary_id: qb.activity_ids
        for qb in QuestionBunch.objects.filter(activity_primary_id__in=scenario_activity_ids)
    }

    # 2. UserAnswer dict keyed by (user_id, activity_id) — last answer per pair
    ua_qs = last_answers.filter(user_id__in=[u.id for u in valid_users])
    last_answer_map = {}
    for ua in ua_qs.select_related('answer'):
        key = (ua.user_id, ua.activity_id)
        if key not in last_answer_map or ua.id > last_answer_map[key].id:
            last_answer_map[key] = ua

    # 3. Max answer weight dict keyed by activity_id
    max_weight_map = {
        row['activity_id']: row['max_weight']
        for row in Answer.objects.filter(activity_id__in=scenario_activity_ids)
        .values('activity_id')
        .annotate(max_weight=Max('answer_weight'))
    }

    # Activities grouped by phase for quick lookup
    activities_by_phase = defaultdict(list)
    for act in all_scenario_activities:
        activities_by_phase[act.phase_id].append(act)

    for user in valid_users:
        for phase in phases:
            activities_in_phase = activities_by_phase.get(phase.id, [])
            total_primary_score = 0
            total_primary_max_score = 0

            # Track processed activities to avoid duplicates
            processed_activities = set()

            # Process primary evaluatable activities (and their bunches)
            primary_evaluatable_activities = [a for a in activities_in_phase if a.is_evaluatable and a.is_primary_ev]
            primary_count = len(primary_evaluatable_activities)

            if primary_count > 0:
                primary_weight_share = 100 / primary_count  # Each primary activity contributes equally

                for primary_activity in primary_evaluatable_activities:
                    bunch_act_ids = bunch_map.get(primary_activity.id, [primary_activity.id])

                    for act_id in bunch_act_ids:
                        if act_id not in processed_activities:
                            user_last_answer = last_answer_map.get((user.id, act_id))
                            if user_last_answer and user_last_answer.answer:
                                total_primary_score += (user_last_answer.answer.answer_weight * primary_weight_share) / 100

                            # Calculate max score for each bunch activity
                            max_weight = max_weight_map.get(act_id)
                            if max_weight:
                                total_primary_max_score += (max_weight * primary_weight_share) / 100

                            # Mark bunch activity as processed
                            processed_activities.add(act_id)

            # Calculate performance based on primary evaluatable activities
            if total_primary_max_score > 0:
                percentage_score = (total_primary_score / total_primary_max_score) * 100
                if percentage_score >= 83.3:
                    phase_performance[phase.name]['high'] += 1
                elif percentage_score >= 49.7:
                    phase_performance[phase.name]['mid'] += 1
                else:
                    phase_performance[phase.name]['low'] += 1

                # Update total score and user count
                phase_performance[phase.name]['total_score'] += percentage_score
                phase_performance[phase.name]['total_users'] += 1

    # Prepare data for response
    data = {
        'categories': [],
        'low': [],
        'mid': [],
        'high': [],
        'metric': []
    }

    for phase in phases:
        data['categories'].append(phase.name)
        data['low'].append(phase_performance[phase.name]['low'])
        data['mid'].append(phase_performance[phase.name]['mid'])
        data['high'].append(phase_performance[phase.name]['high'])
        if phase_performance[phase.name]['total_users'] > 0:
            average_score = phase_performance[phase.name]['total_score'] / phase_performance[phase.name]['total_users']
        else:
            average_score = 0
        data['metric'].append(average_score)

    return data

@shared_task
def compute_time_spent_by_performer_type(scenario_id, group_ids, start_date, end_date):
    version_id = (
        Scenario.objects.get(pk=scenario_id).ensure_current_version().id
    )
    cache_key = f"analytics:time_by_perf:{scenario_id}:v{version_id}:{group_ids}:{start_date}:{end_date}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    scenario = get_object_or_404(Scenario, id=scenario_id)

    _sd = parse_date(start_date) if start_date else None
    _ed = parse_date(end_date) if end_date else None
    start_dt = _date_to_aware(_sd) if _sd else None
    end_dt = _date_to_aware(_ed) if _ed else None

    phases = list(Phase.objects.filter(scenario=scenario).order_by('id'))

    # ── 1. Build student user ID set (same logic as before) ────────────────
    if group_ids:
        user_ids = set(
            UserGroupMembership.objects
            .filter(group_id__in=group_ids)
            .values_list('user_id', flat=True)
        )
    else:
        user_ids = set(
            User.objects
            .filter(
                Q(userscenarioscore__scenario=scenario) &
                (Q(school_department__isnull=False) |
                 Q(id__in=UserGroupMembership.objects.values('user_id')))
            )
            .values_list('id', flat=True)
        )
    teacher_ids = set(
        User.objects.filter(groups__name='teachers').values_list('id', flat=True)
    )
    user_ids -= teacher_ids

    # ── 2. ONE query: fetch all relevant UserAnswer rows into memory ─────────
    # We build the "last answer per (user, activity)" dict here rather than
    # using get_last_answers() so we can also apply the date filter upfront.
    answers_qs = (
        get_eligible_user_answers(scenario_id)
        .filter(user_id__in=user_ids)
        .select_related('answer')
    )
    if start_dt:
        answers_qs = answers_qs.filter(created_on__gte=start_dt)
    if end_dt:
        answers_qs = answers_qs.filter(created_on__lte=end_dt + timedelta(days=1))

    # Keep only the row with the highest id per (user_id, activity_id)
    last_answer_map = {}  # {(user_id, activity_id): UserAnswer}
    for ua in answers_qs:
        key = (ua.user_id, ua.activity_id)
        if key not in last_answer_map or ua.id > last_answer_map[key].id:
            last_answer_map[key] = ua

    # ── 3. Valid users = those who answered the first activity ──────────────
    min_activity = scenario.get_start_activity()
    if not min_activity:
        return {'categories': [], 'low': [], 'mid': [], 'high': []}

    valid_user_ids = {uid for (uid, act_id) in last_answer_map if act_id == min_activity.id}

    # ── 4. ONE query: all activities for the scenario, grouped by phase ─────
    all_activities = list(Activity.objects.filter(scenario=scenario))
    activities_by_phase = defaultdict(list)
    for act in all_activities:
        activities_by_phase[act.phase_id].append(act)

    # ── 5. ONE query: QuestionBunches {primary_activity_id: [act_ids]} ──────
    bunches = {
        qb.activity_primary_id: qb.activity_ids
        for qb in QuestionBunch.objects.filter(activity_primary__scenario=scenario)
    }

    # ── 6. ONE query: max answer weight per activity ─────────────────────────
    max_weights = {
        row['activity_id']: row['max_weight']
        for row in (
            Answer.objects
            .filter(activity__scenario=scenario)
            .values('activity_id')
            .annotate(max_weight=Max('answer_weight'))
        )
    }

    # ── 7. Compute entirely in Python — zero additional DB queries ───────────
    time_spent = {
        phase.name: {'low': 0, 'mid': 0, 'high': 0,
                     'low_count': 0, 'mid_count': 0, 'high_count': 0}
        for phase in phases
    }

    for user_id in valid_user_ids:
        for phase in phases:
            phase_activities = activities_by_phase.get(phase.id, [])

            # Skip phase if user has no answers here
            if not any((user_id, act.id) in last_answer_map for act in phase_activities):
                continue

            total_time = 0
            total_primary_score = 0
            total_primary_max_score = 0
            processed = set()

            primary_ev = [a for a in phase_activities if a.is_evaluatable and a.is_primary_ev]
            primary_count = len(primary_ev)

            if primary_count > 0:
                weight_share = 100 / primary_count

                for primary_act in primary_ev:
                    bunch_act_ids = bunches.get(primary_act.id, [primary_act.id])
                    for act_id in bunch_act_ids:
                        if act_id not in processed:
                            ua = last_answer_map.get((user_id, act_id))
                            if ua:
                                if ua.answer_id and ua.answer:
                                    total_primary_score += (ua.answer.answer_weight * weight_share) / 100
                                if ua.timing:
                                    total_time += ua.timing
                            total_primary_max_score += (max_weights.get(act_id, 0) * weight_share) / 100
                            processed.add(act_id)

            # Non-primary activities: count time only
            for act in phase_activities:
                if act.id not in processed:
                    ua = last_answer_map.get((user_id, act.id))
                    if ua and ua.timing:
                        total_time += ua.timing
                    processed.add(act.id)

            if total_primary_max_score > 0:
                pct = (total_primary_score / total_primary_max_score) * 100
                if pct >= 83.3:
                    time_spent[phase.name]['high'] += total_time
                    time_spent[phase.name]['high_count'] += 1
                elif pct >= 49.7:
                    time_spent[phase.name]['mid'] += total_time
                    time_spent[phase.name]['mid_count'] += 1
                else:
                    time_spent[phase.name]['low'] += total_time
                    time_spent[phase.name]['low_count'] += 1

    # ── 8. Format output ────────────────────────────────────────────────────
    data = {'categories': [], 'low': [], 'mid': [], 'high': []}
    for phase in phases:
        ts = time_spent[phase.name]
        data['categories'].append(phase.name)
        data['low'].append(ts['low'] / ts['low_count'] if ts['low_count'] > 0 else 0)
        data['mid'].append(ts['mid'] / ts['mid_count'] if ts['mid_count'] > 0 else 0)
        data['high'].append(ts['high'] / ts['high_count'] if ts['high_count'] > 0 else 0)

    cache.set(cache_key, data, timeout=3600)
    return data

@shared_task
def compute_scenario_paths(scenario_id, group_ids, start_date, end_date):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    
    # Parse start and end dates from request
    start_date = start_date
    end_date = end_date

    # Get the first answers (only the earliest answer for each user/activity combination)
    last_answers = get_first_answers(scenario_id)

    # Apply start_date and end_date filters
    if start_date:
        start_date = parse_date(start_date)
        if start_date:
            last_answers = last_answers.filter(created_on__gte=_date_to_aware(start_date))
    
    if end_date:
        end_date = parse_date(end_date)
        if end_date:
            last_answers = last_answers.filter(created_on__lte=_date_to_aware(end_date + timedelta(days=1)))

    # Get the minimum activity ID in the scenario
    min_activity = scenario.get_start_activity()

    if not min_activity:
        return {"error": "No activities found for this scenario", "status": 400}

    # users = User.objects.filter(
    #     Q(userscenarioscore__scenario=scenario) & 
    #     (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
    # ).distinct()
    if group_ids:
        users = User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct()# .exclude(groups__name='teachers')
    else:
        users = User.objects.filter(
            Q(userscenarioscore__scenario=scenario) & 
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
            ).distinct()
    
    users = users.exclude(groups__name='teachers')
    
    valid_users = []  # List of users who started with the minimum activity

    for user in users:
        # Check if user has answered the minimum activity
        if last_answers.filter(user=user, activity=min_activity).exists():
            valid_users.append(user)
    
    scenario_path_counts = defaultdict(lambda: {'count': 0, 'user_ids': []})
    phase_path_counts = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'user_ids': []}))
    all_activity_ids = set() # Mar
    all_phase_ids = set() # Mar

    for user in valid_users:
        user_answers = last_answers.filter(user=user).order_by('created_on')
        
        user_path = []
        phase_paths = defaultdict(list)

        for answer in user_answers:
            aid = answer.activity.id # Mar
            pid = answer.activity.phase.id # Mar
            all_activity_ids.add(aid) # Mar
            all_phase_ids.add(pid) # Mar

            activity_id = answer.activity.id
            phase = answer.activity.phase
            
            user_path.append(activity_id)
            phase_paths[phase.id].append(activity_id)

        path_tuple = tuple(user_path)
        if path_tuple:
            scenario_path_counts[path_tuple]['count'] += 1
            scenario_path_counts[path_tuple]['user_ids'].append(user.id)

        #for phase_id, phase_path in phase_paths.items():
        for pid, phase_path in phase_paths.items(): # Mar
            phase_path_tuple = tuple(phase_path)
            if phase_path_tuple:
                phase_path_counts[pid][phase_path_tuple]['count'] += 1 # phase_id
                phase_path_counts[pid][phase_path_tuple]['user_ids'].append(user.id) # phase_id
    
    # Fetch all activities and phases in bulk
    activities = Activity.objects.filter(id__in=all_activity_ids)
    activity_dict = {a.id: a for a in activities}
    phases = Phase.objects.filter(id__in=all_phase_ids)
    phase_dict = {p.id: p for p in phases}

    # Helper function to serialize activity
    def serialize_activity(aid):
        activity = activity_dict[aid]
        return {
            'id': activity.id,
            'name': activity.name,
            'short_name': (activity.name[:20] + '...') if len(activity.name) > 20 else activity.name,
            'tooltip': f"{activity.name} | {activity.phase.name if activity.phase else 'Unknown Phase'}",
            'url': f'/authoringtool/scenarios/{activity.scenario.id}/viewPhase/{activity.phase.id}/viewActivity/{activity.id}/'
        }

    max_scenario_count = max((data['count'] for data in scenario_path_counts.values()), default=0) # max_count
    # common_scenario_paths = [path for path, data in scenario_path_counts.items() if data['count'] == max_count]
    common_scenario_path = next((path for path, data in scenario_path_counts.items() if data['count'] == max_scenario_count), None)
    unique_scenario_path_count = len(scenario_path_counts)

    most_common_scenario = [serialize_activity(aid) for aid in common_scenario_path] if common_scenario_path else []

    # All scenario paths for modal
    all_scenario_paths = [
        {
            'path': [serialize_activity(aid) for aid in path],
            'count': data['count'],
            'user_ids': data['user_ids']
        }
        for path, data in scenario_path_counts.items()
    ]

    phase_path_data = {}

    for pid, paths in phase_path_counts.items():
        max_phase_count = max((data['count'] for data in paths.values()), default=0)
        common_phase_path = next((path for path, data in paths.items() if data['count'] == max_phase_count), None)
        unique_phase_path_count = len(paths)
        phase = phase_dict.get(pid)

        most_common_phase = [serialize_activity(aid) for aid in common_phase_path] if common_phase_path else []

        # All paths for modal
        all_phase_paths = [
            {
                'path': [serialize_activity(aid) for aid in path],
                'count': data['count'],
                'user_ids': data['user_ids']
            }
            for path, data in paths.items()
        ]

        # Final phase data
        phase_path_data[pid] = {
            'phase_name': phase.name if phase else "Unknown Phase",
            'most_common_path': {
                'path': most_common_phase,
                'count': max_phase_count
            },
            'paths': all_phase_paths,
            'unique_path_count': unique_phase_path_count
        }

    # Final structured return (Celery-friendly)
    return {
        'scenario_paths': {
            'most_common_path': {
                'path': most_common_scenario,
                'count': max_scenario_count
            },
            'paths': all_scenario_paths,
            'unique_path_count': unique_scenario_path_count
        },
        'phase_paths': phase_path_data
    }

from django.utils import timezone
from datetime import datetime, time


def _target_activity_routes(activity):
    next_low = next_mid = next_high = ''
    if activity.is_evaluatable:
        branching = EvQuestionBranching.objects.filter(
            activity=activity,
        ).first()
        if branching:
            next_low = (
                branching.next_question_on_low.name
                if branching.next_question_on_low
                else ''
            )
            next_mid = (
                branching.next_question_on_mid.name
                if branching.next_question_on_mid
                else ''
            )
            next_high = (
                branching.next_question_on_high.name
                if branching.next_question_on_high
                else ''
            )
    else:
        fallback = NextQuestionLogic.objects.filter(
            activity=activity,
            answer__isnull=True,
        ).select_related('next_activity').first()
        if fallback and fallback.next_activity:
            next_low = next_mid = next_high = fallback.next_activity.name
        else:
            linked_nexts = (
                NextQuestionLogic.objects
                .filter(activity=activity, answer__isnull=False)
                .exclude(next_activity__isnull=True)
                .values_list('next_activity__name', flat=True)
            )
            route_name = ', '.join(sorted(set(linked_nexts)))
            next_low = next_mid = next_high = route_name
    return next_low, next_mid, next_high


def _compute_compatible_category_metrics_data(
    scenario,
    group_ids=None,
    start_date=None,
    end_date=None,
    evidence_language='',
):
    """Aggregate structurally compatible evidence by activity lineage."""
    target_phases = list(
        Phase.objects.filter(scenario=scenario).order_by('created_on', 'id')
    )
    target_activities = list(
        Activity.objects
        .filter(scenario=scenario, phase__isnull=False)
        .select_related('phase', 'activity_type')
        .order_by('phase__created_on', 'phase_id', 'created_on', 'id')
    )
    target_by_lineage = {
        activity.lineage_key: activity
        for activity in target_activities
    }
    target_phase_by_lineage = {
        activity.lineage_key: activity.phase_id
        for activity in target_activities
    }

    source_versions = list(
        get_evidence_versions(
            scenario,
            'compatible',
            evidence_language,
        )
        .select_related('scenario')
    )
    source_version_by_id = {
        version.id: version for version in source_versions
    }
    source_scenario_ids = {
        version.scenario_id for version in source_versions
    }
    source_activities = list(
        Activity.objects
        .filter(
            scenario_id__in=source_scenario_ids,
            lineage_key__in=target_by_lineage,
        )
        .select_related('scenario', 'phase', 'activity_type')
    )
    source_activity_by_id = {
        activity.id: activity for activity in source_activities
    }

    answers = (
        get_evidence_answers(
            scenario,
            'compatible',
            evidence_language,
        )
        .filter(activity_id__in=source_activity_by_id)
        .select_related(
            'answer',
            'activity__scenario',
            'scenario_version',
        )
    )
    if group_ids:
        allowed_user_ids = UserGroupMembership.objects.filter(
            group_id__in=group_ids,
        ).values_list('user_id', flat=True)
        answers = answers.filter(user_id__in=allowed_user_ids)
    else:
        allowed_user_ids = (
            User.objects
            .filter(
                Q(school_department__isnull=False)
                | Q(id__in=UserGroupMembership.objects.values('user_id'))
            )
            .exclude(groups__name='teachers')
            .values_list('id', flat=True)
        )
        answers = answers.filter(user_id__in=allowed_user_ids)

    if start_date:
        parsed_start = parse_date(start_date)
        if parsed_start:
            answers = answers.filter(
                created_on__gte=_date_to_aware(parsed_start)
            )
    if end_date:
        parsed_end = parse_date(end_date)
        if parsed_end:
            answers = answers.filter(
                created_on__lt=_date_to_aware(
                    parsed_end + timedelta(days=1)
                )
            )

    latest_answers = {}
    for answer in answers:
        key = (
            answer.scenario_version_id,
            answer.implementation_id,
            answer.activity_id,
        )
        previous = latest_answers.get(key)
        if previous is None or answer.id > previous.id:
            latest_answers[key] = answer

    start_activity_by_scenario = {
        source_scenario_id: (
            Scenario.objects.get(pk=source_scenario_id).get_start_activity()
        )
        for source_scenario_id in source_scenario_ids
    }
    valid_implementations = set()
    for version_id, implementation_id, activity_id in latest_answers:
        version = source_version_by_id.get(version_id)
        start_activity = (
            start_activity_by_scenario.get(version.scenario_id)
            if version
            else None
        )
        if start_activity and activity_id == start_activity.id:
            valid_implementations.add((version_id, implementation_id))

    source_primary_activities = [
        activity
        for activity in source_activities
        if activity.is_evaluatable and activity.is_primary_ev
    ]
    primary_ids = [activity.id for activity in source_primary_activities]
    bunch_map = {
        bunch.activity_primary_id: bunch.activity_ids
        for bunch in QuestionBunch.objects.filter(
            activity_primary_id__in=primary_ids
        )
    }
    all_scoring_activity_ids = set(primary_ids)
    for activity_ids in bunch_map.values():
        all_scoring_activity_ids.update(activity_ids)
    max_weight_by_activity = {
        row['activity_id']: row['max_weight']
        for row in (
            Answer.objects
            .filter(activity_id__in=all_scoring_activity_ids)
            .values('activity_id')
            .annotate(max_weight=Max('answer_weight'))
        )
    }

    source_primaries_by_scenario_phase = defaultdict(list)
    for activity in source_primary_activities:
        target_phase_id = target_phase_by_lineage.get(
            activity.lineage_key
        )
        if target_phase_id:
            source_primaries_by_scenario_phase[
                (activity.scenario_id, target_phase_id)
            ].append(activity)

    implementation_categories = {}
    for version_id, implementation_id in valid_implementations:
        version = source_version_by_id[version_id]
        for target_phase in target_phases:
            total_score = 0
            max_score = 0
            processed = set()
            primaries = source_primaries_by_scenario_phase.get(
                (version.scenario_id, target_phase.id),
                [],
            )
            for primary in primaries:
                scoring_ids = bunch_map.get(primary.id, [primary.id])
                for activity_id in scoring_ids:
                    if activity_id in processed:
                        continue
                    user_answer = latest_answers.get(
                        (
                            version_id,
                            implementation_id,
                            activity_id,
                        )
                    )
                    if user_answer and user_answer.answer:
                        total_score += user_answer.answer.answer_weight
                    max_score += max_weight_by_activity.get(activity_id, 0)
                    processed.add(activity_id)
            if max_score:
                percentage = (total_score / max_score) * 100
                category = (
                    'High'
                    if percentage >= 83.3
                    else 'Moderate'
                    if percentage >= 49.7
                    else 'Low'
                )
                implementation_categories[
                    (
                        version_id,
                        implementation_id,
                        target_phase.id,
                    )
                ] = category

    aggregates = defaultdict(
        lambda: {
            'response_count': 0,
            'correct_flags': [],
            'same_language_timings': [],
        }
    )
    target_language = (scenario.language or '').strip().casefold()
    for (
        version_id,
        implementation_id,
        source_activity_id,
    ), user_answer in latest_answers.items():
        if (
            version_id,
            implementation_id,
        ) not in valid_implementations:
            continue
        source_activity = source_activity_by_id.get(source_activity_id)
        if not source_activity:
            continue
        target_activity = target_by_lineage.get(
            source_activity.lineage_key
        )
        if not target_activity:
            continue
        category = implementation_categories.get(
            (
                version_id,
                implementation_id,
                target_activity.phase_id,
            )
        )
        if not category or not user_answer.timing:
            continue

        aggregate = aggregates[(target_activity.id, category)]
        aggregate['response_count'] += 1
        if user_answer.answer:
            aggregate['correct_flags'].append(
                user_answer.answer.is_correct
            )
        source_language = (
            source_activity.scenario.language or ''
        ).strip().casefold()
        if source_language == target_language:
            aggregate['same_language_timings'].append(user_answer.timing)

    context = get_evidence_context(
        scenario,
        'compatible',
        evidence_language,
    )
    source_languages = ', '.join(context['languages'])
    source_version_ids = ','.join(
        str(version_id) for version_id in context['version_ids']
    )
    combined_data = []
    for target_phase in target_phases:
        phase_activities = [
            activity
            for activity in target_activities
            if activity.phase_id == target_phase.id
        ]
        for activity in phase_activities:
            type_name = (
                activity.activity_type.name
                if activity.activity_type
                else 'Unknown'
            )
            next_low, next_mid, next_high = _target_activity_routes(
                activity
            )
            for category in ['High', 'Moderate', 'Low']:
                aggregate = aggregates[(activity.id, category)]
                total = aggregate['response_count']
                correct_flags = aggregate['correct_flags']
                correct = sum(correct_flags) if correct_flags else 0
                wrong = total - correct if correct_flags else 0
                timings = aggregate['same_language_timings']
                combined_data.append({
                    'Phase': target_phase.name,
                    'Activity': activity.name,
                    'Type': type_name,
                    'Category': category,
                    'Total': total,
                    'Correct': correct if correct_flags else '',
                    'Wrong': wrong if correct_flags else '',
                    '% Correct': (
                        round((correct / total) * 100, 1)
                        if total
                        else ''
                    ),
                    '% Wrong': (
                        round((wrong / total) * 100, 1)
                        if total
                        else ''
                    ),
                    'Avg Time': (
                        round(np.mean(timings), 1) if timings else ''
                    ),
                    'Timing Total': len(timings),
                    'Next Low': next_low,
                    'Next Moderate': next_mid,
                    'Next High': next_high,
                    'Evidence Scope': 'compatible',
                    'Source Version IDs': source_version_ids,
                    'Source Languages': source_languages,
                    'Evidence Language Filter': (
                        evidence_language or 'All compatible languages'
                    ),
                })
    return combined_data

@shared_task
def compute_student_performance_metrics(scenario_id, group_ids, start_date, end_date, include_activity_detail=False):
    scenario = get_object_or_404(Scenario, id=scenario_id)
    phases = list(Phase.objects.filter(scenario=scenario).order_by('id'))

    phase_weights = [0.2, 0.2, 0.45, 0.15]
    if len(phases) > 4:
        phase_weights = [0.2, 0.2, 0.3, 0.15, 0.15]

    # Parse dates
    start_dt = end_dt = None
    if start_date:
        d = parse_date(start_date)
        if d:
            start_dt = _date_to_aware(d)
    if end_date:
        d = parse_date(end_date)
        if d:
            end_dt = _date_to_aware(d + timedelta(days=1))

    # Get users
    if group_ids:
        users = list(User.objects.filter(
            id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
        ).distinct().exclude(groups__name='teachers'))
    else:
        users = list(User.objects.filter(
            Q(userscenarioscore__scenario=scenario) &
            (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
        ).distinct().exclude(groups__name='teachers'))

    # Get min activity
    min_activity = scenario.get_start_activity()
    if not min_activity:
        return {"error": "No activities found for this scenario", "status": 400}

    # Build date-filtered answer base first, then derive valid users from it
    last_answers_base = get_last_answers(scenario_id)
    date_filtered_base = last_answers_base
    if start_dt:
        date_filtered_base = date_filtered_base.filter(created_on__gte=start_dt)
    if end_dt:
        date_filtered_base = date_filtered_base.filter(created_on__lte=end_dt)

    # Only include users who have at least one answer within the date range
    users_who_started = set(
        date_filtered_base.values_list('user_id', flat=True)
    )
    valid_users = [u for u in users if u.id in users_who_started]
    valid_user_ids = {u.id for u in valid_users}

    # Materialise date-filtered answers into a dict for O(1) lookup
    filtered_qs = date_filtered_base.filter(user_id__in=valid_user_ids)
    last_answers_dict = {
        (ua.user_id, ua.activity_id): ua
        for ua in filtered_qs.select_related('answer', 'activity')
    }

    # Pre-fetch all activities for this scenario grouped by phase
    all_acts = list(Activity.objects.filter(phase__scenario=scenario).select_related('phase', 'activity_type').order_by('id'))
    phase_activities_map = defaultdict(list)
    phase_primary_map = defaultdict(list)
    for act in all_acts:
        phase_activities_map[act.phase_id].append(act)
        if act.is_evaluatable and act.is_primary_ev:
            phase_primary_map[act.phase_id].append(act)

    # Pre-fetch QuestionBunches and resolve their activity lists
    primary_ids = [act.id for acts in phase_primary_map.values() for act in acts]
    bunches = {b.activity_primary_id: b.activity_ids
               for b in QuestionBunch.objects.filter(activity_primary_id__in=primary_ids)}
    all_bunch_act_ids = {aid for ids in bunches.values() for aid in ids}
    bunch_activity_objs = {a.id: a for a in Activity.objects.filter(id__in=all_bunch_act_ids)}

    # Pre-fetch max answer weight per activity (single query)
    max_weight_map = {
        row['activity_id']: row['max_w']
        for row in Answer.objects.filter(
            activity__phase__scenario=scenario
        ).values('activity_id').annotate(max_w=Max('answer_weight'))
    }

    # Pre-fetch phase start times in one aggregated query
    phase_start_qs = get_eligible_user_answers(scenario_id).filter(
        user_id__in=valid_user_ids,
        activity__phase__in=phases,
    )
    if start_dt:
        phase_start_qs = phase_start_qs.filter(created_on__gte=start_dt)
    if end_dt:
        phase_start_qs = phase_start_qs.filter(created_on__lte=end_dt)
    phase_start_map = {
        (row['user_id'], row['activity__phase_id']): row['first_ts']
        for row in phase_start_qs.values('user_id', 'activity__phase_id').annotate(first_ts=Min('created_on'))
    }

    # Main loop -- no DB queries inside
    csv_data = []
    for user in valid_users:
        row = [user.id, user.username, scenario.name]
        total_weighted_score = 0
        total_max_weighted_score = 0

        for index, phase in enumerate(phases):
            phase_total_time = 0
            phase_total_score = 0
            phase_max_score = 0
            processed_activities = set()

            phase_start_time = phase_start_map.get((user.id, phase.id))

            for activity in phase_activities_map[phase.id]:
                ua = last_answers_dict.get((user.id, activity.id))
                if ua and ua.timing:
                    phase_total_time += ua.timing

            for primary_activity in phase_primary_map[phase.id]:
                if primary_activity.id in bunches:
                    bunch_acts = [bunch_activity_objs[aid]
                                  for aid in bunches[primary_activity.id]
                                  if aid in bunch_activity_objs]
                else:
                    bunch_acts = [primary_activity]

                for bunch_activity in bunch_acts:
                    if bunch_activity.id not in processed_activities:
                        ua = last_answers_dict.get((user.id, bunch_activity.id))
                        if ua and ua.answer:
                            phase_total_score += ua.answer.answer_weight
                        phase_max_score += max_weight_map.get(bunch_activity.id, 0)
                        processed_activities.add(bunch_activity.id)

            if phase_max_score > 0:
                weight = phase_weights[index] if index < len(phase_weights) else 0
                total_weighted_score += (phase_total_score / phase_max_score) * weight
                total_max_weighted_score += weight

            phase_percentage = (phase_total_score / phase_max_score) * 100 if phase_max_score > 0 else 0
            if phase_percentage >= 83.3:
                phase_categorization = 'High'
            elif phase_percentage >= 49.7:
                phase_categorization = 'Moderate'
            else:
                phase_categorization = 'Low'

            row.extend([
                phase_categorization,
                phase_start_time.strftime('%Y-%m-%d %H:%M:%S') if phase_start_time else '',
                phase_total_time,
                phase_total_score,
            ])

        final_percentage = (total_weighted_score / total_max_weighted_score) * 100 if total_max_weighted_score > 0 else 0
        if final_percentage >= 83.3:
            final_categorization = 'High'
        elif final_percentage >= 49.7:
            final_categorization = 'Moderate'
        else:
            final_categorization = 'Low'

        row.append(final_categorization)

        if include_activity_detail:
            for phase in phases:
                for activity in sorted(phase_activities_map[phase.id], key=lambda a: a.id):
                    act_type = activity.activity_type.name if activity.activity_type else ''
                    ua = last_answers_dict.get((user.id, activity.id))
                    timing = ua.timing if ua and ua.timing is not None else ''
                    if activity.is_evaluatable and ua and ua.answer:
                        score = ua.answer.answer_weight
                    else:
                        score = ''
                    row.extend([act_type, timing, score])

        csv_data.append(row)

    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    header = ['User ID', 'Username', 'Scenario Name']
    for phase in phases:
        header.extend([phase.name + ' Categorization', phase.name + ' Start Time', phase.name + ' Time', phase.name + ' Score'])
    header.append('Final Categorization')

    if include_activity_detail:
        for phase in phases:
            for activity in sorted(phase_activities_map[phase.id], key=lambda a: a.id):
                prefix = f"{phase.name} > {activity.name}"
                header.extend([f"{prefix} Type", f"{prefix} Time (s)", f"{prefix} Score"])

    csv_writer.writerow(header)
    csv_writer.writerows(csv_data)
    csv_buffer.seek(0)

    return {
        "csv_content": csv_buffer.getvalue(),
        "message": "Student performance metrics computed successfully."
    }


@shared_task
def compute_category_metrics_per_phase_activity(
    scenario_id,
    group_ids=None,
    start_date=None,
    end_date=None,
    evidence_scope='local',
    evidence_language='',
):
    # base_dir = os.path.join(settings.BASE_DIR, 'ai_metrics_cache')
    base_dir = settings.AI_METRICS_CACHE_ROOT
    os.makedirs(base_dir, exist_ok=True)
    scenario = get_object_or_404(Scenario, id=scenario_id)
    evidence_scope = normalize_evidence_scope(evidence_scope)
    evidence_language = normalize_evidence_language(evidence_language)
    file_path = get_scenario_evidence_cache_paths(
        scenario,
        scope=evidence_scope,
        language=evidence_language,
    )['metrics']

    evidence_answers = get_evidence_answers(
        scenario,
        evidence_scope,
        evidence_language,
    )
    latest_answer_time = evidence_answers.aggregate(
        Max('created_on')
    )['created_on__max']

    if latest_answer_time and os.path.exists(file_path):
        file_timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
        file_modified = make_aware(file_timestamp)
        if latest_answer_time <= file_modified:
            return {
                "scenario_id": scenario_id,
                "evidence_scope": evidence_scope,
                "evidence_language": evidence_language,
            }

    if evidence_scope == 'compatible':
        combined_data = _compute_compatible_category_metrics_data(
            scenario,
            group_ids=group_ids,
            start_date=start_date,
            end_date=end_date,
            evidence_language=evidence_language,
        )
        fieldnames = [
            'Phase',
            'Activity',
            'Type',
            'Category',
            'Total',
            'Correct',
            'Wrong',
            '% Correct',
            '% Wrong',
            'Avg Time',
            'Timing Total',
            'Next Low',
            'Next Moderate',
            'Next High',
            'Evidence Scope',
            'Source Version IDs',
            'Source Languages',
            'Evidence Language Filter',
        ]
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_data)
        return {
            "scenario_id": scenario_id,
            "evidence_scope": evidence_scope,
            "evidence_language": evidence_language,
        }

    phases = list(Phase.objects.filter(scenario=scenario).order_by('id'))
    last_answers = get_last_answers(
        scenario_id,
        evidence_scope,
        evidence_language,
    )

    users = User.objects.filter(
        id__in=UserGroupMembership.objects.filter(group_id__in=group_ids).values_list('user_id', flat=True)
    ) if group_ids else User.objects.filter(
        Q(userscenarioscore__scenario=scenario) &
        (Q(school_department__isnull=False) | Q(id__in=UserGroupMembership.objects.values('user_id')))
    )

    users = users.exclude(groups__name='teachers').distinct()
    if start_date:
        last_answers = last_answers.filter(created_on__gte=parse_date(start_date))
    if end_date:
        last_answers = last_answers.filter(created_on__lte=parse_date(end_date) + timedelta(days=1))

    min_activity = scenario.get_start_activity()
    if not min_activity:
        return {"error": "No activities found for this scenario"}

    candidate_user_ids = set(users.values_list('id', flat=True))
    valid_implementation_ids = set(
        last_answers.filter(
            user_id__in=candidate_user_ids,
            activity=min_activity,
        ).values_list('implementation_id', flat=True)
    )
    latest_answers = {
        (
            user_answer.implementation_id,
            user_answer.activity_id,
        ): user_answer
        for user_answer in last_answers.filter(
            implementation_id__in=valid_implementation_ids,
        ).select_related('answer')
    }

    all_activities = list(
        Activity.objects
        .filter(phase__scenario=scenario)
        .select_related('activity_type')
        .order_by('created_on', 'id')
    )
    activities_by_phase = defaultdict(list)
    primaries_by_phase = defaultdict(list)
    for activity in all_activities:
        activities_by_phase[activity.phase_id].append(activity)
        if activity.is_evaluatable and activity.is_primary_ev:
            primaries_by_phase[activity.phase_id].append(activity)

    primary_ids = [
        activity.id
        for activities in primaries_by_phase.values()
        for activity in activities
    ]
    bunch_map = {
        bunch.activity_primary_id: bunch.activity_ids
        for bunch in QuestionBunch.objects.filter(
            activity_primary_id__in=primary_ids,
        )
    }
    max_weight_by_activity = {
        row['activity_id']: row['max_weight']
        for row in (
            Answer.objects
            .filter(activity__phase__scenario=scenario)
            .values('activity_id')
            .annotate(max_weight=Max('answer_weight'))
        )
    }

    implementation_phase_category = {}
    for implementation_id in valid_implementation_ids:
        implementation_phase_category[implementation_id] = {}
        for phase in phases:
            total_score, max_score = 0, 0
            processed = set()
            for primary in primaries_by_phase[phase.id]:
                scoring_ids = bunch_map.get(primary.id, [primary.id])
                for activity_id in scoring_ids:
                    if activity_id in processed:
                        continue
                    ua = latest_answers.get(
                        (implementation_id, activity_id)
                    )
                    if ua and ua.answer:
                        total_score += ua.answer.answer_weight
                    max_score += max_weight_by_activity.get(activity_id, 0)
                    processed.add(activity_id)
            if max_score > 0:
                pct = (total_score / max_score) * 100
                cat = 'High' if pct >= 83.3 else 'Moderate' if pct >= 49.7 else 'Low'
                implementation_phase_category[
                    implementation_id
                ][phase.id] = cat

    activity_type_map = dict(ActivityType.objects.values_list('id', 'name'))
    phase_activity_sequences = {
        phase.id: activities_by_phase[phase.id]
        for phase in phases
    }

    combined_data = []
    for phase in phases:
        activities = phase_activity_sequences[phase.id]
        for activity in activities:
            type_name = activity_type_map.get(activity.activity_type_id, 'Unknown')

            next_low = next_mid = next_high = ''

            if activity.is_evaluatable:
                try:
                    bunch = QuestionBunch.objects.get(activity_primary=activity)
                    branching = EvQuestionBranching.objects.filter(activity=activity).first()
                    if branching:
                        next_low = branching.next_question_on_low.name if branching.next_question_on_low else ''
                        next_mid = branching.next_question_on_mid.name if branching.next_question_on_mid else ''
                        next_high = branching.next_question_on_high.name if branching.next_question_on_high else ''
                except QuestionBunch.DoesNotExist:
                    pass
            else:
                fallback = NextQuestionLogic.objects.filter(activity=activity, answer__isnull=True).first()
                if fallback and fallback.next_activity:
                    next_low = next_mid = next_high = fallback.next_activity.name
                else:
                    answers = Answer.objects.filter(activity=activity)
                    linked_nexts = NextQuestionLogic.objects.filter(activity=activity, answer__in=answers).values_list('next_activity__name', flat=True)
                    unique_nexts = sorted(set(linked_nexts))
                    name = ", ".join(unique_nexts)
                    next_low = next_mid = next_high = name

            category_data = {'High': [], 'Moderate': [], 'Low': []}
            correctness_data = {'High': [], 'Moderate': [], 'Low': []}

            for implementation_id in valid_implementation_ids:
                cat = implementation_phase_category.get(
                    implementation_id,
                    {},
                ).get(phase.id)
                if not cat:
                    continue
                ua = latest_answers.get(
                    (implementation_id, activity.id)
                )
                if not ua or not ua.timing:
                    continue
                category_data[cat].append(ua.timing)
                if ua.answer:
                    correctness_data[cat].append(ua.answer.is_correct)

            for cat in ['High', 'Moderate', 'Low']:
                timings = category_data[cat]
                correct_flags = correctness_data[cat]
                total = len(timings)
                correct = sum(correct_flags) if correct_flags else 0
                wrong = total - correct if correct_flags else 0
                pct_c = round((correct / total) * 100, 1) if total else ''
                pct_w = round((wrong / total) * 100, 1) if total else ''
                avg_time = round(np.mean(timings), 1) if timings else ''

                combined_data.append({
                    'Phase': phase.name,
                    'Activity': activity.name,
                    'Type': type_name,
                    'Category': cat,
                    'Total': total,
                    'Correct': correct if correct_flags else '',
                    'Wrong': wrong if correct_flags else '',
                    '% Correct': pct_c,
                    '% Wrong': pct_w,
                    'Avg Time': avg_time,
                    'Timing Total': total,
                    'Next Low': next_low,
                    'Next Moderate': next_mid,
                    'Next High': next_high,
                    'Evidence Scope': evidence_scope,
                    'Source Version IDs': (
                        ''
                        if evidence_scope == 'historical'
                        else str(scenario.current_version_id or '')
                    ),
                    'Source Languages': (
                        (scenario.language or '').strip()
                        or 'Unspecified'
                    ),
                    'Evidence Language Filter': (
                        evidence_language or 'All compatible languages'
                    ),
                })

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'Phase', 'Activity', 'Type', 'Category', 'Total', 'Correct',
            'Wrong', '% Correct', '% Wrong', 'Avg Time', 'Timing Total',
            'Next Low', 'Next Moderate', 'Next High', 'Evidence Scope',
            'Source Version IDs', 'Source Languages',
            'Evidence Language Filter',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_data:
            writer.writerow(row)

    return {
        "scenario_id": scenario_id,
        "evidence_scope": evidence_scope,
        "evidence_language": evidence_language,
    }

@shared_task
def calculate_activities_in_risk(
    scenario_id,
    evidence_scope='local',
    evidence_language='',
):
    # ====== 1. Load the dataset ======
    # base_dir = os.path.join(settings.BASE_DIR, 'ai_metrics_cache')
    base_dir = settings.AI_METRICS_CACHE_ROOT
    os.makedirs(base_dir, exist_ok=True)
    scenario = get_object_or_404(Scenario, id=scenario_id)
    evidence_scope = normalize_evidence_scope(evidence_scope)
    evidence_language = normalize_evidence_language(evidence_language)
    evidence_context = get_evidence_context(
        scenario,
        evidence_scope,
        evidence_language,
    )
    cache_paths = get_scenario_evidence_cache_paths(
        scenario,
        scope=evidence_scope,
        language=evidence_language,
    )
    file_path = cache_paths['metrics']
    flags_file_path = cache_paths['flags']

    if not os.path.exists(file_path):
        # either build it or fail gracefully
        compute_category_metrics_per_phase_activity(
            scenario_id,
            evidence_scope=evidence_scope,
            evidence_language=evidence_language,
        )
        if not os.path.exists(file_path):
            return {"error": f"Metrics CSV not found: {file_path}"}
        
    df = pd.read_csv(file_path)

    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
    if 'Timing Total' not in df.columns:
        df['Timing Total'] = df['Total']
    df['Timing Total'] = pd.to_numeric(
        df['Timing Total'],
        errors='coerce',
    ).fillna(0)

    # Normalize category names
    df['Category'] = df['Category'].str.strip().str.capitalize()

    # Container for all flags
    flags = []

    # ====== 2. Function for Engagement (Timing Outlier) Analysis ======
    def flag_timing_outliers(activity_df, activity, activity_type):
        """
        Flags categories whose Avg Time is an outlier relative to other categories.
        Works with any 2+ categories present (does not require all three).
        Uses coefficient of variation (CV) instead of a fixed-second threshold so
        the check is scale-independent (short activities vs long ones).
        """
        results = []
        # Collect whichever categories have valid timing data
        ordered_cats = [
            c for c in ['High', 'Moderate', 'Low']
            if not activity_df[activity_df['Category'] == c].empty
            and activity_df[
                activity_df['Category'] == c
            ]['Timing Total'].values[0] >= 10
            and pd.notna(activity_df[activity_df['Category'] == c]['Avg Time'].values[0])
            and activity_df[activity_df['Category'] == c]['Avg Time'].values[0] != ''
        ]
        if len(ordered_cats) < 2:
            return results

        times = {
            c: float(activity_df[activity_df['Category'] == c]['Avg Time'].values[0])
            for c in ordered_cats
        }
        time_vals = list(times.values())
        mean_time = np.mean(time_vals)
        std_time  = np.std(time_vals, ddof=0)

        if mean_time == 0:
            return results

        # CV (coefficient of variation) makes the threshold scale-independent.
        # Require >25% relative spread AND >5s absolute spread to avoid noise
        # on very short activities where a few seconds look like a lot.
        # 5s floor is sufficient: genuine signals (e.g. 16s vs 31s, σ=6s) pass,
        # while true noise (e.g. 8s vs 9s, σ<1s) is suppressed.
        cv = std_time / mean_time
        if cv < 0.25 or std_time < 5:
            return results

        upper = mean_time + std_time
        lower = mean_time - std_time
        for cat, t in times.items():
            if t > upper:
                results.append({
                    "Activity": activity,
                    "Category": cat,
                    "Flag": f"Engagement risk (too slow, {activity_type})",
                    "Reason": (
                        f"Engagement risk: {cat} group's Avg Time ({t:.1f}s) is above mean+std "
                        f"({mean_time:.1f}s ± {std_time:.1f}s, CV={cv:.2f}). "
                        f"Significantly slower than the other groups — possible confusion or disengagement."
                    )
                })
            elif t < lower:
                results.append({
                    "Activity": activity,
                    "Category": cat,
                    "Flag": f"Engagement risk (too fast, {activity_type})",
                    "Reason": (
                        f"Engagement risk: {cat} group's Avg Time ({t:.1f}s) is below mean-std "
                        f"({mean_time:.1f}s ± {std_time:.1f}s, CV={cv:.2f}). "
                        f"Suspiciously fast relative to other groups — may indicate guessing or skipping."
                    )
                })
        return results

    # ====== 3. Part 1: Question Activities Risk Analysis ======
    question_df = df[
        (df['Type'].str.lower() == 'question')
        & (df['Total'] >= 10)
    ]

    for activity in question_df['Activity'].unique():
        act_df = question_df[question_df['Activity'] == activity]

        # Timing outlier check runs for any 2+ categories present (moved before the guard)
        flags.extend(flag_timing_outliers(act_df, activity, activity_type="question"))

        # Checks 1–3 compare H vs M vs L directly — require all three
        if set(act_df['Category']) != {'High', 'Moderate', 'Low'}:
            continue

        H = act_df[act_df['Category'] == 'High'].iloc[0]
        M = act_df[act_df['Category'] == 'Moderate'].iloc[0]
        L = act_df[act_df['Category'] == 'Low'].iloc[0]
        wrongs = {'High': H['% Wrong'], 'Moderate': M['% Wrong'], 'Low': L['% Wrong']}
        times  = {'High': H['Avg Time'],  'Moderate': M['Avg Time'],  'Low': L['Avg Time']}

        # --- 1. Non-monotonic performance distribution ---
        # Expected natural order: High wrong% ≤ Moderate wrong% ≤ Low wrong%
        # (High-category students should make the fewest mistakes.)
        # Any significant violation of this order is a calibration/design signal.
        MARGIN = 12  # percentage-point gap required before flagging (avoids noise)

        # Case A: Moderate is the worst of all three (unexpected V-shape)
        if (M['% Wrong'] > H['% Wrong'] + MARGIN and M['% Wrong'] > L['% Wrong'] + MARGIN):
            flags.append({
                "Activity": activity,
                "Category": "Moderate",
                "Flag": "Extreme correctness pattern",
                "Reason": (
                    f"Unexpected V-shape: Moderate group ({wrongs['Moderate']}% wrong) performs worse than "
                    f"both High ({wrongs['High']}%) and Low ({wrongs['Low']}%) groups. "
                    f"Activity may be poorly calibrated for mid-range learners."
                )
            })
        else:
            # Case B: Low outperforms Moderate (Low has fewer errors than Moderate)
            if L['% Wrong'] < M['% Wrong'] - MARGIN and M['% Wrong'] > 40:
                flags.append({
                    "Activity": activity,
                    "Category": "Moderate",
                    "Flag": "Extreme correctness pattern",
                    "Reason": (
                        f"Low group ({wrongs['Low']}% wrong) outperforms Moderate ({wrongs['Moderate']}% wrong) "
                        f"— unexpected reversal in performance order. "
                        f"Activity content or answer choices may not align with the expected difficulty level."
                    )
                })
            # Case C: High underperforms Moderate
            if H['% Wrong'] > M['% Wrong'] + MARGIN and H['% Wrong'] > 40:
                flags.append({
                    "Activity": activity,
                    "Category": "High",
                    "Flag": "Extreme correctness pattern",
                    "Reason": (
                        f"High group ({wrongs['High']}% wrong) performs worse than Moderate ({wrongs['Moderate']}% wrong). "
                        f"High-category students should outperform Moderate — this reversal may indicate "
                        f"activity miscalibration or a mismatch between the activity and the scoring rubric."
                    )
                })

        # --- 2. Timing Discrepancy Under Extreme Correctness ---
        # Flags categories that scored at a performance extreme (0% or 100% wrong)
        # yet spent disproportionately more time than every other category.
        # The two extremes carry different pedagogical meanings and produce
        # distinct flag types. A minimum absolute gap (TAU_ABS_GAP) prevents
        # spurious flags when all groups respond quickly and the 1.5x ratio
        # is met trivially (e.g. 9s vs 6s).
        TAU_RATIO   = 1.5   # flagged category must spend >= 1.5x each other group
        TAU_ABS_GAP = 10.0  # AND must spend >= 10s more than each other group

        for cat, wrong in wrongs.items():
            others = [c for c in ['High', 'Moderate', 'Low'] if c != cat]
            if wrong not in [0.0, 100.0]:
                continue
            # Both the ratio and the absolute gap must hold for every comparison group
            ratio_holds = all(times[cat] >= TAU_RATIO * times[other] for other in others)
            gap_holds   = all(times[cat] - times[other] >= TAU_ABS_GAP for other in others)
            if not (ratio_holds and gap_holds):
                continue

            if wrong == 100.0:
                flag_type = "Timing discrepancy under extreme correctness"
                interpretation = (
                    f"{cat} group answered everything incorrectly (100% wrong) yet spent "
                    f"{times[cat]}s — ≥{TAU_RATIO}x and ≥{TAU_ABS_GAP}s more than the other groups "
                    f"({times[others[0]]}s, {times[others[1]]}s). "
                    f"Prolonged time with total failure indicates active confusion or cognitive overload, "
                    f"not disengagement."
                )
            else:  # wrong == 0.0
                flag_type = "Timing discrepancy under extreme correctness"
                interpretation = (
                    f"{cat} group answered everything correctly (0% wrong) yet spent "
                    f"{times[cat]}s — ≥{TAU_RATIO}x and ≥{TAU_ABS_GAP}s more than the other groups "
                    f"({times[others[0]]}s, {times[others[1]]}s). "
                    f"Perfect outcome with excessive time may indicate effortful retrieval at the boundary "
                    f"of the group's ability range, rather than confident mastery."
                )

            flags.append({
                "Activity": activity,
                "Category": cat,
                "Flag": flag_type,
                "Reason": interpretation,
            })

        # --- 3. Dynamic Cutoff for Correctness (with systemic failure detection) ---
        wrong_arr  = np.array([H['% Wrong'], M['% Wrong'], L['% Wrong']])
        mean_wrong = wrong_arr.mean()
        std_wrong  = wrong_arr.std(ddof=0)
        spread     = wrong_arr.max() - wrong_arr.min()

        # Thresholds (all in percentage points)
        # TAU_BASE is grounded in the paper's PISA-aligned Low-performance boundary:
        # S_u,Pi < 0.497 correct ≡ > 50.3% wrong. If all three groups fall below
        # this threshold on a single activity, they are all performing at Low level
        # regardless of their overall scenario category — a content-level failure.
        TAU_BASE       = 50.3   # all-individual systemic failure gate (PISA Low boundary)
        TAU_SPREAD     = 15.0   # minimum spread for a per-category flag to be meaningful
        TAU_ABS_FLAG   = 45.0   # minimum absolute error rate for the flagged category
        TAU_LOW_EXEMPT = 40.0   # Low=100% exemption only when High is performing well

        # Systemic failure: all three categories individually exceed TAU_BASE.
        # Using individual checks (not mean) avoids the cliff-edge of a single cutoff:
        # an activity where all groups are above the baseline is universally too hard,
        # regardless of whether one group is slightly worse than the others.
        if H['% Wrong'] > TAU_BASE and M['% Wrong'] > TAU_BASE and L['% Wrong'] > TAU_BASE:
            flags.append({
                "Activity": activity,
                "Category": "All",
                "Flag": "Systemic failure",
                "Reason": (
                    f"All performance categories individually exceed the PISA Low-performance boundary "
                    f"({TAU_BASE}% wrong): "
                    f"High {wrongs['High']}%, Moderate {wrongs['Moderate']}%, Low {wrongs['Low']}% wrong. "
                    f"All groups score below the Low-performance threshold on this activity, indicating a "
                    f"content-level failure independent of learner category — "
                    f"review the question wording, answer options, or prerequisite content."
                )
            })
        else:
            # Per-category outlier: one group is notably worse than the others.
            # Guard 1 — minimum spread: if the max-min range is too small, the
            # mean+std threshold is unreliable (self-referential with n=3) and
            # any "outlier" is likely within statistical noise.
            if spread >= TAU_SPREAD:
                threshold = mean_wrong + std_wrong
                for cat, wrong in wrongs.items():
                    # Guard 2 — Low=100% exemption: only suppress when High is
                    # performing well, confirming this is an expected lower bound
                    # rather than part of a wider problem.
                    if cat == "Low" and wrong == 100.0 and H['% Wrong'] < TAU_LOW_EXEMPT:
                        continue
                    # Guard 3 — minimum absolute error: prevents flagging a category
                    # that technically exceeds mean+std but is still performing
                    # acceptably in absolute terms.
                    if wrong < TAU_ABS_FLAG:
                        continue
                    if wrong > threshold:
                        flags.append({
                            "Activity": activity,
                            "Category": cat,
                            "Flag": "Dynamic correctness threshold",
                            "Reason": (
                                f"Dynamic correctness: {cat} group's %Wrong ({wrong}%) exceeds "
                                f"mean+std threshold ({threshold:.2f}%) with a category spread of {spread:.1f}pp. "
                                f"This group is struggling significantly more than the others on this activity."
                            )
                        })

    # ====== 4. Part 2: Non-Question Activities Engagement Analysis ======
    nonq_df = df[df['Type'].str.lower() != 'question']
    for activity in nonq_df['Activity'].unique():
        act_df = nonq_df[nonq_df['Activity'] == activity]
        flags.extend(flag_timing_outliers(act_df, activity, activity_type="non-question"))

    # ====== 5. Output the results as a DataFrame ======
    flags_df = pd.DataFrame(flags)
    (
        ActivityFlag.objects
        .filter(
            activity__scenario_id=scenario_id,
            evidence_scope=evidence_scope,
            auto_flagged=True,
            proposals__isnull=True,
        )
        .delete()
    )
    for flag in flags:
        try:
            # activity_obj = Activity.objects.get(name=flag["Activity"], phase__scenario_id=scenario_id)
            activity_obj = Activity.objects.filter(
                name=flag["Activity"],
                phase__scenario_id=scenario_id
            ).order_by("id").first()

            if not activity_obj:
                continue
        except Activity.DoesNotExist:
            print(f"Activity not found in DB: {flag['Activity']}")
            continue

        ActivityFlag.objects.create(
            activity=activity_obj,
            category=flag["Category"],
            flag_type=flag["Flag"],
            flag_reason=flag["Reason"],
            is_at_risk=True,
            auto_flagged=True,
            evidence_scope=evidence_scope,
            evidence_signature=evidence_context['signature'],
            evidence_sources=evidence_context['sources'],
        )

    # Show the top results and save for further use
    flags_df.to_csv(flags_file_path, index=False)
    print("\nAll flagged activities and reasons saved to 'flagged_activities_with_reasons.csv'")
    return {
        "scenario_id": scenario_id,
        "evidence_scope": evidence_scope,
        "evidence_language": evidence_language,
    }

def get_data_insight(flag):
    """
    Returns a combined insight + resolution suggestion string
    based on the flag's type, reason, and category.
    """
    flag_type = flag.flag_type
    reason = flag.flag_reason
    category = flag.category
    activity = flag.activity.name

    # Try to parse numeric values
    percent_wrong = None
    if "group's %Wrong (" in reason:
        try:
            percent_wrong = float(reason.split("group's %Wrong (")[1].split('%')[0])
        except:
            pass

    avg_time = None
    if "Avg Time (" in reason:
        try:
            avg_time = float(reason.split("Avg Time (")[1].split('s)')[0])
        except:
            pass

    insights = []
    resolutions = []

    # --- Dynamic Correctness Threshold ---
    if flag_type == "Dynamic correctness threshold" and percent_wrong is not None:
        if percent_wrong >= 80:
            insights.append(
                f"{category} group error rate is very high ({percent_wrong}%), indicating a conceptual gap."
            )
            resolutions.append(
                "Suggest **creating** interactive explanation covering the missing concept before this activity with an informational video or data."
            )
        elif percent_wrong <= 25:
            insights.append(
                f"{category} group error rate is very low ({percent_wrong}%), suggesting this activity may be too easy."
            )
            resolutions.append(
                "Consider **skipping** this activity for this group or **revising** by increasing difficulty or adding extension questions."
            )
        else:
            insights.append(
                f"{category} group error rate deviates significantly ({percent_wrong}%)."
            )
            resolutions.append(
                "Recommend **revising** the activity wording or adding a short worked example to scaffold understanding."
            )

    # --- Engagement Risk: Too Slow ---
    if "Engagement risk (too slow" in flag_type and avg_time is not None:
        insights.append(
            f"{category} group spends {avg_time}s, well above average, suggesting confusion or cognitive overload."
        )
        resolutions.append(
            "Add a concise **video walkthrough** or **annotated image** explaining each step, to reduce load."
        )

    # --- Engagement Risk: Too Fast ---
    if "Engagement risk (too fast" in flag_type and avg_time is not None:
        insights.append(
            f"{category} group spends only {avg_time}s, suggesting skimming or that the activity is trivial."
        )
        resolutions.append(
            "Either **skip** this activity for this group or **revise** by adding reflective questions or richer multimedia (e.g., an image-based quiz)."
        )

    # --- Extreme Correctness Pattern ---
    if flag_type == "Extreme correctness pattern":
        insights.append(
            f"{category} group performance is misaligned with peers, signaling a difficulty mismatch."
        )
        resolutions.append(
            "Introduce a **bridging activity**—e.g., a conceptual animation or mini-experiment—targeted to this group."
        )

    # --- Timing Discrepancy Under Extreme Correctness ---
    if flag_type == "Timing discrepancy under extreme correctness":
        if "100% wrong" in reason:
            # Confusion under total failure sub-case
            insights.append(
                f"{category} group answered everything incorrectly yet spent disproportionately long on "
                f"this activity. Prolonged time with total failure indicates active confusion or cognitive "
                f"overload, not disengagement."
            )
            resolutions.append(
                "Add a **worked example** or **conceptual video** before this activity to reduce cognitive "
                "load, and consider breaking it into smaller steps."
            )
        elif "0% wrong" in reason:
            # Unexpected deliberation under perfect performance sub-case
            insights.append(
                f"{category} group answered everything correctly yet spent disproportionately long. "
                f"Perfect outcome with excessive time may indicate effortful retrieval at the boundary "
                f"of the group's ability range, rather than confident mastery."
            )
            resolutions.append(
                "Consider **revising** the activity to streamline decision points, or **creating** a "
                "faster-paced variant to build fluency for this group."
            )
        else:
            insights.append(
                f"{category} group's timing anomaly suggests flow or clarity issues despite correctness extremes."
            )
            resolutions.append(
                "Consider **revising** instructions to clarify decision points, or **creating** a quick formative check-in slide."
            )

    # --- Systemic Failure ---
    if flag_type == "Systemic failure":
        insights.append(
            "All performance groups exceed the PISA Low-performance boundary on this activity, "
            "indicating a content-level failure independent of learner category."
        )
        resolutions.append(
            "**Revise** the question wording, answer options, or prerequisite content — "
            "this activity is too difficult for all groups in the current instructional sequence."
        )

    # --- Disconnected Activity ---
    if getattr(flag, 'is_isolated', False):
        insights.append("Activity appears disconnected from the learning sequence.")
        resolutions.append(
            "Either **skip** this activity or **create** a transition activity that links it pedagogically."
        )

    # Combine and return
    combined_insights = " ".join(insights)
    combined_resolutions = " ".join(resolutions)
    return combined_insights, combined_resolutions

def get_activity_context_for_category(activity, category):
    """
    Get previous and next activities for a specific performance category ('High', 'Moderate', 'Low').
    """
    previous = set()
    next_ = set()
    
    # --- NEXT: from activity to others ---
    # Direct next questions
    for logic in NextQuestionLogic.objects.filter(activity=activity):
        if logic.next_activity:
            next_.add(logic.next_activity)

    # Branching: category-specific next
    if activity.is_evaluatable:
        try:
            branching = EvQuestionBranching.objects.get(activity=activity)
            if category == 'High' and branching.next_question_on_high:
                next_.add(branching.next_question_on_high)
            elif category == 'Moderate' and branching.next_question_on_mid:
                next_.add(branching.next_question_on_mid)
            elif category == 'Low' and branching.next_question_on_low:
                next_.add(branching.next_question_on_low)
        except EvQuestionBranching.DoesNotExist:
            pass

    # --- PREVIOUS: from others to this activity ---
    # Direct previous questions
    for logic in NextQuestionLogic.objects.filter(next_activity=activity):
        previous.add(logic.activity)

    # Branching: include only if this activity is the next for the category
    category_map = {
        'High': 'next_question_on_high',
        'Moderate': 'next_question_on_mid',
        'Low': 'next_question_on_low'
    }
    query = {f"{category_map[category]}": activity}
    prev_branchings = EvQuestionBranching.objects.filter(**query)
    for branching in prev_branchings:
        previous.add(branching.activity)

    return {
        "previous": list(previous),
        "next": list(next_),
    }

def get_all_previous_activities(activity):
    """
    Returns a list of all activities that can lead (directly or indirectly)
    into the given activity, without duplicates.
    """
    seen = set()
    result = []
    queue = deque([activity])

    while queue:
        current = queue.popleft()
        # 1) NextQuestionLogic predecessors
        for logic in NextQuestionLogic.objects.filter(next_activity=current):
            prev = logic.activity
            if prev and prev.id not in seen:
                seen.add(prev.id)
                result.append(prev)
                queue.append(prev)

        # 2) Branching predecessors
        branches = EvQuestionBranching.objects.filter(
            Q(next_question_on_high=current) |
            Q(next_question_on_mid=current) |
            Q(next_question_on_low=current)
        )
        for br in branches:
            prev = br.activity
            if prev and prev.id not in seen:
                seen.add(prev.id)
                result.append(prev)
                queue.append(prev)

    return result

def get_phase_based_prior_summary(activity):
    """
    Returns a bullet list of all activities that come
    before `activity` in the scenario, by phase order then activity creation order.
    """
    scenario = activity.phase.scenario
    # 1. Get all phases in order
    phases = Phase.objects.filter(scenario=scenario).order_by('id')  # or 'order' if you have one
    summary_lines = []

    for phase in phases:
        # If we haven't reached this activity's phase yet, dump whole phase
        if phase.id < activity.phase.id:
            acts = Activity.objects.filter(phase=phase).order_by('created_on', 'id')
        # If this is the same phase, only include those before the activity
        elif phase.id == activity.phase.id:
            acts = Activity.objects.filter(phase=phase, id__lt=activity.id).order_by('created_on', 'id')
        else:
            break

        if not acts:
            continue

        summary_lines.append(f"Phase: {phase.name}")
        for act in acts:
            # Use a short summary field—e.g. first sentence of llm_teacher_eval
            short = act.short_llm_summary or ""
            summary_lines.append(f"  • {act.name}: {short or 'No summary.'}")

    return "\n".join(summary_lines)

# BASE_RAG_DIR = os.path.join(os.path.dirname(__file__), '..', 'rag_pdfs')
# BASE_INDEX_DIR = os.path.join(settings.RAG_INDEX_ROOT, f"scenario_{id}")# os.path.join(settings.BASE_DIR, "rag_indexes")
BASE_RAG_DIR = settings.RAG_PDFS_ROOT          # e.g. /data/rag/rag_pdfs
BASE_INDEX_DIR = settings.RAG_INDEX_ROOT       # e.g. /data/rag/rag_indexes
def get_pdf_dir(scenario_id: int) -> str:
    return os.path.join(BASE_RAG_DIR, f"scenario_{scenario_id}")

def get_index_dir(scenario_id: int) -> str:
    return os.path.join(BASE_INDEX_DIR, f"scenario_{scenario_id}")

# 2026
# def get_pdf_dir(scenario_id):
#     return os.path.join(BASE_RAG_DIR, f"scenario_{scenario_id}")

# def get_index_dir(scenario_id):
#     return os.path.join(BASE_INDEX_DIR, f"scenario_{scenario_id}")

def ensure_rag_index(scenario_id):
    """
    Build or reuse a Chroma index for the scenario's PDFs.
    """
    index_dir = os.path.join(BASE_INDEX_DIR, f"scenario_{scenario_id}")
    os.makedirs(index_dir, exist_ok=True)
    collection_name = f"scenario_{scenario_id}"
    client = chromadb.PersistentClient(path=index_dir)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )

def chunk_and_index_pdfs(scenario_id):
    """
    Ingest all PDFs under rag_pdfs/scenario_<ID>/ and index into Chroma.
    """
    pdf_dir = get_pdf_dir(scenario_id)
    index_dir = get_index_dir(scenario_id)
    # only clear the *index* folder, not the entire scenario folder
    # if os.path.exists(index_dir):
    #     shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)  # ensure the PDF folder always exists

    rag_store = ensure_rag_index(scenario_id)
    # 2) If already indexed, skip re-ingest
    try:
        if rag_store._collection.count() > 0:
            print(f"[RAG] index for scenario {scenario_id} already populated, skipping.")
            return rag_store
    except Exception:
        pass  # fallback to indexing
    # pdf_dir = os.path.join(BASE_RAG_DIR, f"scenario_{scenario_id}")
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"[RAG] no PDFs found for scenario {scenario_id}, skipping indexing.")
        return rag_store

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []
    seen_texts = set()

    for path in pdf_files:
        loader = PyPDFLoader(path)
        raw_docs = loader.load_and_split(text_splitter=splitter)
        for d in raw_docs:
            text = d.page_content.strip()
            if text in seen_texts:
                continue
            seen_texts.add(text)
            d.metadata["source_file"] = os.path.basename(path)
            all_docs.append(d)

    if all_docs:
        rag_store.add_documents(all_docs)
        print(f"[RAG] indexed {len(all_docs)} new chunks for scenario {scenario_id}.")
    else:
        print(f"[RAG] no new chunks to index for scenario {scenario_id}.")
    return rag_store

ACTIONS = ["create", "revise", "skip"]

_EXPLORE_UNTIL = int(getattr(settings, "BANDIT_EXPLORE_UNTIL", 200))
_EXPLOIT_FROM = int(getattr(settings, "BANDIT_EXPLOIT_FROM", 500))
_EXPLOIT_EPSILON = min(
    1.0,
    max(0.0, float(getattr(settings, "BANDIT_EXPLOIT_EPSILON", 0.05))),
)
_EXPLORATION_ACTION_WEIGHTS = (
    ("create", max(0.0, float(getattr(settings, "BANDIT_CREATE_WEIGHT", 0.50)))),
    ("skip", max(0.0, float(getattr(settings, "BANDIT_SKIP_WEIGHT", 0.30)))),
    ("revise", max(0.0, float(getattr(settings, "BANDIT_REVISE_WEIGHT", 0.20)))),
)
_PROPOSAL_GENERATION_ATTEMPTS = max(
    1,
    int(getattr(settings, "LLM_PROPOSAL_GENERATION_ATTEMPTS", 3)),
)

SUPPORTED_PROPOSAL_ACTIVITY_TYPES = ("Question", "Explanation", "Experiment")


class ProposalValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def record_proposal_structural_failure(
    *,
    scenario,
    stage,
    errors,
    generation_run=None,
    proposal=None,
    activity=None,
    selected_action="",
    raw_output="",
):
    if isinstance(errors, (ProposalValidationError, ScenarioGraphValidationError)):
        errors = getattr(errors, "errors", None) or getattr(errors, "issues", [])
    elif not isinstance(errors, (list, tuple)):
        errors = [str(errors)]
    return ProposalStructuralFailure.objects.create(
        scenario=scenario,
        generation_run=generation_run,
        proposal=proposal,
        activity=activity,
        selected_action=selected_action or "",
        stage=stage,
        errors=list(errors),
        raw_output=raw_output or "",
    )


def get_total_bandit_rewards():
    """Return the number of reward observations applied to all Q-value arms."""
    return QValue.objects.aggregate(total=Sum("reward_count"))["total"] or 0


def _get_exploration_rate(total_rewards=None):
    """Return ε for weighted exploration with linear decay from 200 to 500."""
    total = get_total_bandit_rewards() if total_rewards is None else total_rewards
    if total <= _EXPLORE_UNTIL:
        return 1.0
    if total >= _EXPLOIT_FROM:
        return _EXPLOIT_EPSILON
    progress = (total - _EXPLORE_UNTIL) / (_EXPLOIT_FROM - _EXPLORE_UNTIL)
    return 1.0 - progress * (1.0 - _EXPLOIT_EPSILON)


def _get_bandit_policy_configuration():
    return BanditPolicyConfiguration.get_active()


def _configured_action_weights(configuration=None):
    if configuration and configuration.pk:
        return (
            ("create", max(0.0, configuration.create_weight)),
            ("skip", max(0.0, configuration.skip_weight)),
            ("revise", max(0.0, configuration.revise_weight)),
        )
    return _EXPLORATION_ACTION_WEIGHTS


def _select_weighted_exploration_action(configuration=None, allowed_actions=None):
    """Sample create/skip/revise using admin-configured cold-start weights."""
    weights = _configured_action_weights(configuration)
    if allowed_actions is not None:
        allowed = set(allowed_actions)
        weights = tuple(
            (action, weight)
            for action, weight in weights
            if action in allowed
        )
    total_weight = sum(weight for _, weight in weights)
    if total_weight <= 0:
        choices = [action for action, _ in weights] or list(ACTIONS)
        return random.choice(choices)

    roll = random.random() * total_weight
    cumulative = 0.0
    for action, weight in weights:
        cumulative += weight
        if roll < cumulative:
            return action
    return weights[-1][0]


def _context_rows(contexts):
    rows = QValue.objects.filter(
        flag_type__in=[flag_type for flag_type, _ in contexts],
        category__in=[category for _, category in contexts],
    )
    return {
        (row.flag_type, row.category, row.action): row
        for row in rows
        if (row.flag_type, row.category) in contexts
    }


def _select_thompson_action(contexts, rows, configuration):
    scores = {action: 0.0 for action in ACTIONS}
    prior_alpha = max(0.0001, configuration.thompson_prior_alpha)
    prior_beta = max(0.0001, configuration.thompson_prior_beta)
    for flag_type, category in contexts:
        for action in ACTIONS:
            row = rows.get((flag_type, category, action))
            successes = row.positive_reward_count if row else 0
            failures = row.negative_reward_count if row else 0
            scores[action] += random.betavariate(
                prior_alpha + successes,
                prior_beta + failures,
            )
    return max(scores, key=scores.get)


def _select_ucb_action(contexts, rows, configuration):
    total_observations = sum(row.reward_count for row in rows.values())
    untried_actions = set()
    scores = {action: 0.0 for action in ACTIONS}
    strength = max(0.0, configuration.ucb_exploration_strength)

    for flag_type, category in contexts:
        for action in ACTIONS:
            row = rows.get((flag_type, category, action))
            if not row or row.reward_count == 0:
                untried_actions.add(action)
                continue
            mean_reward = row.reward_sum / row.reward_count
            bonus = strength * math.sqrt(
                math.log(max(total_observations, 2)) / row.reward_count
            )
            scores[action] += mean_reward + bonus

    if untried_actions:
        return _select_weighted_exploration_action(
            configuration,
            allowed_actions=untried_actions,
        )
    return max(scores, key=scores.get)


def _select_contextual_bandit_action(contexts):
    configuration = _get_bandit_policy_configuration()
    contexts = set(contexts)
    if not contexts:
        return _select_weighted_exploration_action(configuration)

    rows = _context_rows(contexts)
    context_reward_counts = {
        context: sum(
            row.reward_count
            for (flag_type, category, _), row in rows.items()
            if (flag_type, category) == context
        )
        for context in contexts
    }
    if any(
        reward_count < configuration.minimum_context_rewards
        for reward_count in context_reward_counts.values()
    ):
        return _select_weighted_exploration_action(configuration)

    if configuration.policy == "ucb":
        return _select_ucb_action(contexts, rows, configuration)
    return _select_thompson_action(contexts, rows, configuration)

def select_action(flag_type, category):
    return _select_contextual_bandit_action([(flag_type, category)])

def get_best_q_action(flag_list):
    """
    Select one action across all unique flag type/category contexts.

    Cold contexts use weighted exploration. Mature contexts use the active
    Thompson Sampling or UCB policy selected in Django admin.
    """
    return _select_contextual_bandit_action(
        [
            (flag.flag_type, flag.category)
            for flag in flag_list
        ]
    )


def proposal_requires_insert_after(activity, action):
    """Return whether a create proposal targets the scenario's entry activity."""
    if (
        (action or "").strip().lower() != "create"
        or activity is None
        or not getattr(activity, "scenario_id", None)
    ):
        return False

    scenario = activity.scenario
    entry_activity_id = scenario.start_activity_id
    if not entry_activity_id:
        entry_activity_id = (
            Activity.objects.filter(scenario_id=activity.scenario_id)
            .order_by("id")
            .values_list("id", flat=True)
            .first()
        )
    return activity.id == entry_activity_id


def build_proposal_json_schema(
    required_action,
    expected_activity_type=None,
    expected_answer_count=None,
    require_insert_after=False,
):
    """Build the Ollama structured-output schema for one policy-selected action."""
    required_action = (required_action or "").strip().lower()
    if required_action not in ACTIONS:
        raise ValueError(f"Unsupported proposal action: {required_action}")

    allowed_activity_types = list(SUPPORTED_PROPOSAL_ACTIVITY_TYPES)
    if expected_activity_type in SUPPORTED_PROPOSAL_ACTIVITY_TYPES:
        allowed_activity_types = [expected_activity_type]

    answers_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "is_correct": {"type": "boolean"},
                "weight": {"type": "integer", "minimum": 1, "maximum": 3},
            },
            "required": ["text", "is_correct", "weight"],
        },
        "minItems": 0,
        "maxItems": 4,
    }
    if required_action == "skip" or (
        expected_activity_type and expected_activity_type != "Question"
    ):
        answers_schema["maxItems"] = 0
    elif expected_activity_type == "Question":
        if expected_answer_count and 2 <= expected_answer_count <= 4:
            answers_schema["minItems"] = expected_answer_count
            answers_schema["maxItems"] = expected_answer_count
        else:
            answers_schema["minItems"] = 2

    allowed_insert_locations = (
        ["after"]
        if required_action == "create" and require_insert_after
        else ["before", "after"]
    )

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": [required_action]},
            "activity_name": {"type": "string"},
            "activity_type": {"type": "string", "enum": allowed_activity_types},
            "content": {"type": "string"},
            "answers": answers_schema,
            "insert_location": {
                "type": "string",
                "enum": allowed_insert_locations,
            },
            "explanation": {"type": "string", "minLength": 1},
        },
        "required": [
            "action",
            "activity_name",
            "activity_type",
            "content",
            "answers",
            "insert_location",
            "explanation",
        ],
    }


_ANSWER_LABEL_RE = re.compile(r"^\s*[A-Za-z]\s*[\.\):\-]\s*")


def format_proposal_answer_text(text, index):
    """Return answer text with exactly one positional label such as `A. `."""
    if not isinstance(text, str):
        return text
    answer_body = _ANSWER_LABEL_RE.sub("", text.strip()).strip()
    if not answer_body:
        return ""
    return f"{chr(ord('A') + index)}. {answer_body}"


def normalize_proposal_data(data):
    """Normalize harmless formatting differences without inventing missing data."""
    if not isinstance(data, dict):
        raise ProposalValidationError(["The proposal must be a JSON object."])

    normalized = dict(data)
    normalized["action"] = str(normalized.get("action") or "").strip("* ").lower()

    activity_type = str(normalized.get("activity_type") or "").strip()
    type_by_lower = {name.lower(): name for name in SUPPORTED_PROPOSAL_ACTIVITY_TYPES}
    normalized["activity_type"] = type_by_lower.get(activity_type.lower(), activity_type)

    for field in ("activity_name", "content", "explanation"):
        value = normalized.get(field)
        normalized[field] = value.strip() if isinstance(value, str) else value

    location = str(normalized.get("insert_location") or "").strip().lower()
    if "before" in location:
        location = "before"
    elif "after" in location:
        location = "after"
    normalized["insert_location"] = location

    raw_answers = normalized.get("answers")
    if raw_answers is None:
        raw_answers = []
    normalized_answers = []
    if isinstance(raw_answers, list):
        for index, answer in enumerate(raw_answers):
            if not isinstance(answer, dict):
                normalized_answers.append(answer)
                continue
            normalized_answer = dict(answer)
            text = normalized_answer.get("text")
            normalized_answer["text"] = format_proposal_answer_text(text, index)
            weight = normalized_answer.get("weight")
            if isinstance(weight, str) and weight.strip().isdigit():
                normalized_answer["weight"] = int(weight.strip())
            normalized_answers.append(normalized_answer)
    else:
        normalized_answers = raw_answers
    normalized["answers"] = normalized_answers
    return normalized


def validate_proposal_data(
    data,
    expected_action=None,
    expected_activity_type=None,
    expected_answer_count=None,
    require_insert_after=False,
):
    """Validate action-specific invariants and return normalized proposal data."""
    normalized = normalize_proposal_data(data)
    errors = []

    action = normalized.get("action")
    if action not in ACTIONS:
        errors.append("action must be create, revise, or skip")
    if expected_action and action != expected_action:
        errors.append(f"action must remain {expected_action}")

    explanation = normalized.get("explanation")
    if not isinstance(explanation, str) or not explanation:
        errors.append("explanation must be a non-empty string")

    if action == "skip":
        normalized["answers"] = []
        if errors:
            raise ProposalValidationError(errors)
        return normalized

    for field in ("activity_name", "content"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{field} must be a non-empty string")

    activity_type = normalized.get("activity_type")
    if activity_type not in SUPPORTED_PROPOSAL_ACTIVITY_TYPES:
        errors.append(
            "activity_type must be Question, Explanation, or Experiment"
        )
    if expected_activity_type and activity_type != expected_activity_type:
        errors.append(f"activity_type must remain {expected_activity_type}")

    if normalized.get("insert_location") not in ("before", "after"):
        errors.append("insert_location must be before or after")
    elif (
        action == "create"
        and require_insert_after
        and normalized.get("insert_location") != "after"
    ):
        errors.append(
            "create proposals for the scenario's first activity must use "
            "insert_location after"
        )

    answers = normalized.get("answers")
    if not isinstance(answers, list):
        errors.append("answers must be an array")
        answers = []

    if activity_type == "Question":
        if not 2 <= len(answers) <= 4:
            errors.append("Question activities must contain 2 to 4 answers")
        if (
            expected_answer_count
            and 2 <= expected_answer_count <= 4
            and len(answers) != expected_answer_count
        ):
            errors.append(
                f"Question revisions must keep exactly {expected_answer_count} answers"
            )

        seen_texts = set()
        correct_count = 0
        for index, answer in enumerate(answers, start=1):
            if not isinstance(answer, dict):
                errors.append(f"answer {index} must be an object")
                continue
            text = answer.get("text")
            is_correct = answer.get("is_correct")
            weight = answer.get("weight")
            if not isinstance(text, str) or not text:
                errors.append(f"answer {index} text must be non-empty")
            elif _ANSWER_LABEL_RE.sub("", text).strip().casefold() in seen_texts:
                errors.append("Question answer text must be unique")
            else:
                seen_texts.add(
                    _ANSWER_LABEL_RE.sub("", text).strip().casefold()
                )
            if not isinstance(is_correct, bool):
                errors.append(f"answer {index} is_correct must be boolean")
            elif is_correct:
                correct_count += 1
            if type(weight) is not int or weight not in (1, 2, 3):
                errors.append(f"answer {index} weight must be 1, 2, or 3")
            elif isinstance(is_correct, bool) and is_correct != (weight == 3):
                errors.append(
                    f"answer {index} must be correct exactly when its weight is 3"
                )
        if correct_count != 1:
            errors.append("Question activities must contain exactly one correct answer")
    elif answers:
        errors.append("Explanation and Experiment activities must not contain answers")

    if errors:
        raise ProposalValidationError(errors)
    return normalized


def merge_proposal_edits(base, edited):
    """Merge teacher-editable fields while preserving policy and answer metadata."""
    if base is not None and not isinstance(base, dict):
        raise ProposalValidationError(["The stored proposal must be a JSON object."])
    if edited is not None and not isinstance(edited, dict):
        raise ProposalValidationError(["Teacher edits must be a JSON object."])
    base = dict(base or {})
    edited = dict(edited or {})
    merged = dict(base)

    for field in ("activity_name", "content", "explanation"):
        if field in edited:
            merged[field] = edited[field]

    if "answers" in edited:
        base_answers = base.get("answers") or []
        edited_answers = edited.get("answers") or []
        merged_answers = []
        for index, edited_answer in enumerate(edited_answers):
            base_answer = (
                base_answers[index]
                if index < len(base_answers) and isinstance(base_answers[index], dict)
                else {}
            )
            merged_answer = dict(base_answer)
            if isinstance(edited_answer, dict):
                merged_answer.update(edited_answer)
            else:
                merged_answer["text"] = edited_answer
            merged_answers.append(merged_answer)
        merged["answers"] = merged_answers

    # These fields are controlled by the policy/generated proposal, not the
    # teacher's content-only edit form.
    for field in ("action", "activity_type", "target_category", "insert_location"):
        merged[field] = base.get(field)
    return merged


def request_validated_proposal(
    prompt,
    required_action,
    expected_activity_type=None,
    expected_answer_count=None,
    require_insert_after=False,
):
    """Request schema-constrained JSON and retry when semantic validation fails."""
    schema = build_proposal_json_schema(
        required_action,
        expected_activity_type=expected_activity_type,
        expected_answer_count=expected_answer_count,
        require_insert_after=require_insert_after,
    )
    correction = ""
    last_error = None

    for attempt in range(1, _PROPOSAL_GENERATION_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "qwen3.6:35b",
                    "prompt": prompt + correction,
                    "format": schema,
                    # qwen3.6 otherwise places schema-constrained JSON in the
                    # Ollama envelope's `thinking` field and leaves `response`
                    # empty. Disabling thinking returns the JSON as response.
                    "think": False,
                    "options": {"temperature": 0.2},
                    "stream": False,
                },
            )
            response.raise_for_status()
            envelope = response.json()
            raw = (
                envelope.get("response")
                or envelope.get("thinking")
                or ""
            ).strip()
            if not raw:
                raise ProposalValidationError([
                    "Ollama returned neither response nor thinking content "
                    f"(done_reason={envelope.get('done_reason', 'unknown')})"
                ])
            parsed = json.loads(raw)
            structured = validate_proposal_data(
                parsed,
                expected_action=required_action,
                expected_activity_type=expected_activity_type,
                expected_answer_count=expected_answer_count,
                require_insert_after=require_insert_after,
            )
            return raw, structured
        except (requests.RequestException, ValueError, TypeError, ProposalValidationError) as exc:
            last_error = exc
            details = exc.errors if isinstance(exc, ProposalValidationError) else [str(exc)]
            correction = (
                "\n\nYour previous response was invalid. Return a corrected JSON object. "
                "Fix every validation error below and keep the required action unchanged:\n- "
                + "\n- ".join(details)
            )
            print(
                f"[proposal] Generation attempt {attempt}/"
                f"{_PROPOSAL_GENERATION_ATTEMPTS} failed: {exc}"
            )

    raise ProposalValidationError([
        f"Could not generate a valid {required_action} proposal: {last_error}"
    ])

# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")

@shared_task
def generate_llm_context_for_scenario(scenario_id, force_rebuild=False, triggered_by_id=None):
    scenario = Scenario.objects.get(id=scenario_id)
    triggered_by = User.objects.filter(id=triggered_by_id).first() if triggered_by_id else scenario.created_by
    scenario_version = scenario.ensure_current_version(
        created_by=triggered_by,
    )
    evidence_context = get_evidence_context(
        scenario,
        scope='compatible',
    )
    print(f"\nStarting LLM context generation for scenario '{scenario.name}' (ID: {scenario_id})")

    # 0️⃣ Build or refresh RAG index from scenario PDFs
    index_dir = Path(get_index_dir(scenario_id))
    if force_rebuild and index_dir.exists():
        print(f"Removing old RAG index at {index_dir}")
        shutil.rmtree(index_dir, onerror=_on_rm_error)

    if force_rebuild:
        print(f'Clearing cached LLM context from DB for scenario {scenario_id}')
        Activity.objects.filter(phase__scenario=scenario).update(
            llm_context='', short_llm_summary='', related_act_llm_context=''
        )
        Phase.objects.filter(scenario=scenario).update(llm_context='')
        scenario.llm_context = ''
        scenario.save(update_fields=['llm_context'])

    rag_store = chunk_and_index_pdfs(scenario_id)
    retriever = rag_store.as_retriever(search_kwargs={"k": 5})

    # 1️⃣ ACTIVITY-LEVEL EVALUATION
    phases     = Phase.objects.filter(scenario=scenario).prefetch_related('activities')
    activities = Activity.objects.filter(phase__scenario=scenario).select_related('phase').prefetch_related('answers')

    for activity in activities:
        # Skip ones we've already done
        if activity.llm_context and activity.short_llm_summary:
            continue

        print(f"\n[Activity] {activity.name} (Phase '{activity.phase.name}')")

        # — Image description (Gemma) —
        image_llm = ""
        m = re.search(r'data:image/png;base64,([^"]+)', activity.text or "")
        if m:
            try:
                payload = {
                    "model":  "gemma3:4b",
                    "prompt": "Describe this image as a teacher would, focusing on what students should notice.",
                    "images": [m.group(1)],
                    "stream": False
                }
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload) # localhost
                image_llm = resp.json().get("response","").strip()
            except Exception as e:
                image_llm = f"[Gemma error: {e}]"
                print("  Gemma error:", e)
        activity.llm_image_description = image_llm

        # — Full teacher review (Qwen) —
        prompt = (
            "You are a pedagogical reviewer. Assess clarity, educational value, and engagement.\n\n"
            f"Activity: {activity.name}\n"
            f"Content: {activity.plain_text or '—'}\n"
            f"Image Description: {image_llm or '—'}\n\n"
            "Please give:\n"
            "1) A one-sentence summary of the activity's goal.\n"
        )
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model":  "qwen3.6:35b",
                "prompt": prompt,
                "stream": False
            })
            activity.llm_context = resp.json().get("response", "").strip()
        except Exception as e:
            activity.llm_context = f"[LLM error: {e}]"
            print(f"  Ollama error (activity context): {e}")
        activity.save()

        # — Short summary for indexing & RAG queries —
        sum_prompt = (
            "Summarize this activity in ONE short sentence for a busy teacher:\n\n"
            f"{activity.llm_context}"
        )
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model":  "qwen3.6:35b",
                "prompt": sum_prompt,
                "stream": False
            })
            activity.short_llm_summary = resp.json().get("response", "").strip()
        except Exception as e:
            activity.short_llm_summary = activity.name
            print(f"  Ollama error (activity summary): {e}")
        activity.save()
        print("  Activity context + summary saved.")

    # 2️⃣ SCENARIO-LEVEL CONTEXT
    if not scenario.llm_context:
        lines = []
        for phase in phases:
            lines.append(f"\nPhase: {phase.name}\nDesc: {phase.description or '—'}\nActivities:")
            for act in phase.activities.all():
                lines.append(f"- {act.name}: {act.plain_text or '—'}")
        big_prompt = (
            "You are reviewing this full learning scenario to help a teacher understand its flow.\n\n"
            f"Scenario: {scenario.name}\n"
            f"{scenario.description or '—'}\n\n"
            "Structure:\n" + "\n".join(lines) + "\n\n"
            "Please give a 2-sentence summary of what this scenario teaches and how it progresses."
        )
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model":  "qwen3.6:35b",
                "prompt": big_prompt,
                "stream": False
            })
            scenario.llm_context = resp.json().get("response", "").strip()
        except Exception as e:
            scenario.llm_context = f"[LLM error: {e}]"
            print(f"  Ollama error (scenario context): {e}")
        scenario.save()
        print("  Scenario context saved.")

    # 3️⃣ PHASE-LEVEL CONTEXT
    for phase in phases:
        if phase.llm_context:
            continue
        act_contexts = [f"- {a.name}: {a.llm_context or '—'}" for a in phase.activities.all()]
        prompt = (
            "You are reviewing a single phase in a learning scenario.\n\n"
            f"Phase: {phase.name}\nDesc: {phase.description or '—'}\n\n"
            "Activities:\n" + "\n".join(act_contexts) + "\n\n"
            "Please give:\n"
            "1) A one-sentence summary of this phase's learning goal.\n"
            "2) A one-sentence note on the sequence's coherence."
        )
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model":  "qwen3.6:35b",
                "prompt": prompt,
                "stream": False
            })
            phase.llm_context = resp.json().get("response", "").strip()
        except Exception as e:
            phase.llm_context = f"[LLM error: {e}]"
            print(f"  Ollama error (phase context): {e}")
        phase.save()
        print(f"  Phase '{phase.name}' context saved.")

    # 4️⃣ FLOW ANALYSIS BY CATEGORY
    cats = ['High','Moderate','Low']
    for activity in activities:
        if activity.related_act_llm_context:
            continue
        flow = {}
        for cat in cats:
            ctx = get_activity_context_for_category(activity, cat)
            prev = ", ".join(a.name for a in ctx['previous']) or '—'
            nxt  = ", ".join(a.name for a in ctx['next'])     or '—'
            prompt = (
                f"As an educational path expert, review the flow for the {cat} performer.\n\n"
                f"Activity: {activity.name}\n"
                f"Previous: {prev}\nNext: {nxt}\n\n"
                "Summarize what students learn in this path."
            )
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                    "model":  "qwen3.6:35b",
                    "prompt": prompt,
                    "stream": False
                })
                flow[cat] = resp.json().get("response", "").strip()
            except Exception as e:
                flow[cat] = f"[LLM error: {e}]"
                print(f"  Ollama error (flow {cat}): {e}")
        activity.related_act_llm_context = flow
        activity.save()

    # 5️⃣ FLAG-DRIVEN PROPOSALS
    # — archive the previous run (if any) and start a fresh current one —
    generation_run = ProposalGenerationRun.start_new(
        scenario,
        triggered_by,
        scenario_version=scenario_version,
        evidence_scope='compatible',
        evidence_version_ids=evidence_context['version_ids'],
        evidence_summary=evidence_context,
    )
    print(f"Started new ProposalGenerationRun {generation_run.id} for scenario {scenario_id}")

    calculate_activities_in_risk(
        scenario_id,
        evidence_scope='compatible',
    )
    flags = ActivityFlag.objects.filter(
        activity__phase__scenario=scenario,
        evidence_scope='compatible',
        evidence_signature=evidence_context['signature'],
    )
    print(f"Found Flags")
    by_act = defaultdict(list)
    for f in flags:
        by_act[f.activity].append(f)

    for activity, flag_list in by_act.items():
        phase = activity.phase

        # — combine reasons, insights, resolutions —
        insights, reasons, resolutions = [], [], []
        for f in flag_list:
            ins, res = get_data_insight(f)
            if ins: insights.append(f"[{f.category}] {ins}")
            if res: resolutions.append(f"[{f.category}] {res}")
            reasons.append(f"[{f.category}] {f.flag_type}: {f.flag_reason}")

        combined_insight    = "\n".join(insights)
        combined_resolution = "\n".join(resolutions)
        combined_reason     = "\n".join(reasons)

        prior_summary = get_phase_based_prior_summary(activity)

        # — pull RAG snippets, de-duped —
        query = activity.short_llm_summary or activity.plain_text or activity.name
        docs = retriever.invoke(query)
        seen, chunks = set(), []
        for d in docs:
            txt = d.page_content.strip()
            if txt not in seen:
                seen.add(txt)
                chunks.append(txt)
        pdf_context = "\n\n".join(chunks)

        print(f"PDF CONTENT:\n\n {pdf_context} \n\n\n\n")

        # — build "Activity Content" & "Answers" blocks —
        content = f"Content:\n{activity.plain_text or '—'}"
        if activity.activity_type.name == "Experiment":
            if activity.simulation:
                content += f"\nSimulation: {activity.simulation.name}"
            elif activity.experiment_ll:
                content += f"\nRemote Lab URL: {activity.experiment_ll.launch_url}"

        answers = ""
        if activity.activity_type.name == "Question":
            lines = []
            for idx, ans in enumerate(activity.answers.all().order_by('answer_weight'), start=1):
                tag = "Correct" if ans.is_correct else "Incorrect"
                lines.append(f"{idx}. ({tag}) {ans.text} — weight {ans.answer_weight}")
            answers = "Answers:\n" + "\n".join(lines)

        selected_action = get_best_q_action(flag_list)
        expected_activity_type = (
            activity.activity_type.name if selected_action == "revise" else None
        )
        current_answer_count = activity.answers.count()
        expected_answer_count = (
            current_answer_count
            if (
                selected_action == "revise"
                and expected_activity_type == "Question"
                and 2 <= current_answer_count <= 4
            )
            else None
        )
        answer_count_instruction = (
            f"A Question revision must return exactly {expected_answer_count} answers "
            "so existing answer routing remains intact.\n"
            if expected_answer_count
            else ""
        )
        require_insert_after = proposal_requires_insert_after(
            activity,
            selected_action,
        )
        insert_location_instruction = (
            "The flagged activity is the scenario's first activity. A new "
            'activity cannot be inserted in front of it, so insert_location MUST be "after".\n'
            if require_insert_after
            else ""
        )

        # — final proposal prompt —
        prompt = (
            "You are an expert instructional designer. The learning policy has already "
            "selected the action below. You generate the educational content; you do not "
            "choose or override the action.\n\n"
            "=== REQUIRED ACTION ===\n"
            f"{selected_action.upper()}\n"
            f'Your JSON "action" value MUST be "{selected_action}".\n\n'

            "=== CONTEXT ===\n"
            "Evidence Pool:\n"
            f"- Compatible implementations: {evidence_context['implementation_count']}\n"
            f"- Source scenarios: {evidence_context['scenario_count']}\n"
            f"- Source languages: {', '.join(evidence_context['languages']) or 'Unspecified'}\n"
            "- Correctness evidence may be pooled across compatible languages; "
            "timing evidence is restricted to the target language.\n\n"
            f"Scenario Insight:\n{scenario.llm_context}\n\n"
            f"Phase Insight:\n{phase.llm_context}\n\n"
            "Flagged Activity:\n"
            f"- Name: {activity.name}\n"
            f"- Type: {activity.activity_type.name}\n"
            f"- Content: {content}\n"
            + (f"- {answers}\n" if answers else "")
            + "\nWhy students were flagged:\n"
            f"{combined_insight}\n\n"
            "Existing proposals:\n"
            f"{combined_resolution or 'None'}\n\n"
            "Prior activities summary:\n"
            f"{prior_summary}\n\n"
            "PDF snippets (in English, for inspiration only):\n"
            f"{pdf_context or 'None'}\n\n"

            "=== GUIDELINES ===\n"
            "CREATE can be:\n"
            "- A multiple-choice question (2–4 options, with weights)\n"
            "- A short explanation (clear and concise, no bullet points)\n"
            "- A hands-on experiment using the existing simulation or lab (never invented by the student)\n"
            "For Question activities, put every choice in the JSON answers array, never only in content. "
            "Include 2–4 unique choices with exactly one correct choice. Every answer must include "
            "text, is_correct, and weight. Prefix answer text by position exactly as A. Answer, "
            "B. Answer, C. Answer, and D. Answer when needed. Weight 3 means correct; weights "
            "1 or 2 mean incorrect. "
            "Do NOT reveal full solution steps or formulas.\n"
            f"{answer_count_instruction}"
            f"{insert_location_instruction}"
            "For explanations or experiments, keep student-facing text clear and age-appropriate.\n"
            "REVISE must keep the existing activity type and improve its clarity or correctness.\n"
            "For SKIP, copy the existing activity name, type, and content into the JSON, use an empty "
            "answers array, and explain why no change is useful.\n\n"
            "- Do NOT include open-ended or discussion prompts; only MCQs for Question type.\n"
            "- Students perform experiments—they do not invent them here.\n"
            "- Respond entirely in English.\n"
            "- Never use any percentage-based thresholds or \"if more than X%\" language.\n\n"

            "=== OUTPUT ===\n"
            "Return only the JSON object required by the supplied schema. Use before or after for "
            "insert_location. The explanation is teacher-only and should be two concise sentences."
        )
        try:
            raw, structured = request_validated_proposal(
                prompt,
                required_action=selected_action,
                expected_activity_type=expected_activity_type,
                expected_answer_count=expected_answer_count,
                require_insert_after=require_insert_after,
            )
        except ProposalValidationError as exc:
            record_proposal_structural_failure(
                scenario=scenario,
                generation_run=generation_run,
                activity=activity,
                selected_action=selected_action,
                stage="generation",
                errors=exc,
            )
            print(f"[proposal] Skipping invalid proposal for activity {activity.id}: {exc}")
            continue

        structured["target_category"] = sorted(
            {flag.category for flag in flag_list}
        ) or "all"
        try:
            translated_structured = translate_proposal_fields(
                structured,
                scenario.language,
            )
            translated_structured = validate_proposal_data(
                translated_structured,
                expected_action=selected_action,
                expected_activity_type=expected_activity_type,
                expected_answer_count=expected_answer_count,
                require_insert_after=require_insert_after,
            )
            translated_structured["target_category"] = structured["target_category"]
        except Exception as _te:
            record_proposal_structural_failure(
                scenario=scenario,
                generation_run=generation_run,
                activity=activity,
                selected_action=selected_action,
                stage="translation",
                errors=_te,
                raw_output=json.dumps(structured, ensure_ascii=False),
            )
            print(f"[proposal] Translation error: {_te}")
            translated_structured = structured
        translated_raw = json.dumps(translated_structured, ensure_ascii=False)
        translated_structured_json = translated_raw

        # — save the proposal —
        prop = ActivityProposal.objects.create(
            scenario=scenario,
            generation_run=generation_run,
            phase=phase,
            activity=activity,
            proposal_type=selected_action,
            suggested_action=raw,
            translated_action=translated_raw,
            json_action=json.dumps(structured, ensure_ascii=False),
            json_translated_action=translated_structured_json,
            status='new',
        )
        prop.flag.set(flag_list)
        for cat in {f.category for f in flag_list}:
            tag, _ = CategoryTag.objects.get_or_create(name=cat)
            prop.categories_in_risk.add(tag)

    latest_version = (
        Scenario.objects.get(pk=scenario_id).ensure_current_version()
    )
    latest_evidence_context = get_evidence_context(
        Scenario.objects.get(pk=scenario_id),
        scope='compatible',
    )
    if (
        latest_version.id != scenario_version.id
        or latest_evidence_context['source_signature']
        != evidence_context['source_signature']
    ):
        ProposalGenerationRun.objects.filter(pk=generation_run.pk).update(
            is_current=False
        )
        raise RuntimeError(
            'The scenario or its compatible evidence pool changed while '
            'proposals were being generated. This run was archived; '
            'generate proposals again for the current evidence sources.'
        )

    # 6️⃣ Write out a CSV snapshot of all proposals
    base = os.path.join(settings.BASE_DIR, 'ai_metrics_cache')
    fp   = os.path.join(base, f'scenario_{scenario_id}_activity_proposals.csv')
    qs   = ActivityProposal.objects.filter(scenario=scenario).prefetch_related('flag', 'categories_in_risk')
    rows = qs.values('id', 'activity__name', 'phase__name', 'proposal_type', 'status', 'created_at', 'reviewed_at', 'reviewer__username')
    df   = pd.DataFrame(rows)
    prop_map = {p.id: p for p in qs}
    df['flag_ids']   = df['id'].map(lambda i: list(prop_map[i].flag.values_list('id', flat=True)))
    df['categories'] = df['id'].map(lambda i: ", ".join(prop_map[i].categories_in_risk.values_list('name', flat=True)))
    df.to_csv(fp, index=False)

    return f"Completed LLM analysis + proposal generation for scenario {scenario_id}"

def parse_llm_proposal(response_suggested_action, proposal=None):
    try:
        parsed = json.loads(response_suggested_action)
        required_fields = {"action", "activity_name", "activity_type", "content"}
        if all(field in parsed for field in required_fields):
            if "answers" not in parsed:
                parsed["answers"] = []
            if proposal:
                parsed["target_category"] = [c.name for c in proposal.categories_in_risk.all()]
            return parsed
    except Exception:
        pass

    data = {
        "action": None,
        "activity_name": None,
        "activity_type": None,
        "content": "",
        "answers": [],
        "target_category": None,
        "insert_location": None,
        "explanation": "",
    }

    lines = response_suggested_action.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    in_content = False
    in_answers = False
    in_explanation = False
    content_lines = []
    explanation_lines = []

    for line in lines:
        if line.lower().startswith("action:"):
            data["action"] = line.split(":", 1)[1].strip().lower()
            in_content = in_answers = in_explanation = False

        elif line.startswith("- Activity Name:"):
            data["activity_name"] = line.split(":", 1)[1].strip()
            in_content = in_answers = in_explanation = False

        elif line.startswith("- Activity Type:"):
            data["activity_type"] = line.split(":", 1)[1].strip()
            in_content = in_answers = in_explanation = False

        elif line.startswith("- Content:"):
            in_content = True
            in_answers = in_explanation = False
            content_lines.append(line.split(":", 1)[1].strip())

        elif line.startswith("- Answers"):
            in_answers = True
            in_content = in_explanation = False

        elif "insert location" in line.lower():
            insert_value = line.split(":", 1)[1].strip().lower()
            if "before" in insert_value:
                data["insert_location"] = "before"
            elif "after" in insert_value:
                data["insert_location"] = "after"
            else:
                data["insert_location"] = "before"  # default fallback
            in_content = in_answers = in_explanation = False

        elif line.lower().startswith("explanation:"):
            in_explanation = True
            in_content = in_answers = False
            explanation_text = line.split(":", 1)[1].strip()
            if explanation_text:
                explanation_lines.append(explanation_text)

        elif in_content:
            content_lines.append(line)

        elif in_answers:
            match_1 = re.match(r"^\d+\.\s*\((Correct|Incorrect)\)\s*([A-Z])\.?\s+(.*?)[—-]\s*weight\s+(\d+)", line, re.IGNORECASE)
            match_2 = re.match(r"^([A-Z])\.?\s+(.*?)\s*\(weight\s*(\d+)\)", line, re.IGNORECASE)

            if match_1:
                correctness = match_1.group(1).strip().lower() == "correct"
                letter = match_1.group(2).strip()
                text = match_1.group(3).strip()
                weight = int(match_1.group(4))
                data["answers"].append({
                    "text": f"{letter}. {text}",
                    "is_correct": correctness,
                    "weight": weight
                })

            elif match_2:
                letter = match_2.group(1).strip()
                text = match_2.group(2).strip()
                weight = int(match_2.group(3))
                data["answers"].append({
                    "text": f"{letter}. {text}",
                    "is_correct": (weight == 3),
                    "weight": weight
                })

        elif in_explanation:
            explanation_lines.append(line)

    data["content"] = "\n".join(content_lines).strip()
    data["explanation"] = "\n".join(explanation_lines).strip()

    if proposal:
        data["target_category"] = [c.name for c in proposal.categories_in_risk.all()]

    return data

def translate_text(text, target_language):
    if text is None:
        return ""

    target_language = target_language.strip().capitalize()

    if target_language.lower() in ['english', 'en', '']:
        return text

    prompt = (
        f"You are a professional translator. Translate the following sentence from English to {target_language}."
        f"Respond ONLY with the translation. Do not add explanations or extra information.\n\n"
        f"English: {text}\n{target_language}:"
    )
    try:
        translation_response = requests.post(f"{OLLAMA_URL}/api/generate", json={ # localhost
                "model": "aya-expanse:32b",
                "prompt": prompt,
                "stream": False
            })
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return text
    except ValueError:
        print("Failed to parse JSON. Raw response:")
        print(translation_response.text)
        return text
    
    # print(f"TO RESPONSE EINAI: {translation_response.text}")
    translated_text = translation_response.json().get("response", "").strip()

    return translated_text if translated_text else text

def translate_proposal_fields(structured_data, scenario_language):
    translated = json.loads(json.dumps(structured_data, ensure_ascii=False))
    if not scenario_language or scenario_language.strip().lower() in ["english", "en"]:
        return translated

    for field in ("activity_name", "content", "explanation"):
        value = translated.get(field)
        if value:
            translated[field] = translate_text(value, scenario_language)

    for answer in translated.get("answers", []):
        if answer.get("text"):
            answer["text"] = translate_text(answer["text"], scenario_language)

    return translated

@shared_task
def apply_user_proposals_to_new_scenario(scenario_id, user_id):
    try:
        return _build_personal_scenario(scenario_id, user_id)
    except ScenarioGraphValidationError as exc:
        scenario = Scenario.objects.filter(pk=scenario_id).first()
        if scenario:
            record_proposal_structural_failure(
                scenario=scenario,
                stage="graph_integrity",
                errors=exc,
            )
        print("Scenario graph validation failed:")
        print(traceback.format_exc())
    except Exception as exc:
        scenario = Scenario.objects.filter(pk=scenario_id).first()
        if scenario:
            record_proposal_structural_failure(
                scenario=scenario,
                stage="application",
                errors=exc,
            )
        print("Exception in apply_user_proposals_to_new_scenario:")
        print(traceback.format_exc())
    return "scenario error"


@transaction.atomic
def _build_personal_scenario(scenario_id, user_id):
    print(f"SCENARIO IS : {scenario_id}")
    try:
        original_scenario = get_object_or_404(Scenario, pk=scenario_id)
        user = User.objects.get(pk=user_id)
        family = original_scenario.ensure_family()
        new_scenario_name_base = f"{original_scenario.name} Copy Made by {user.username}"
        new_scenario_name = new_scenario_name_base
        
        # Check if scenario with the new name already exists and modify the name if needed
        counter = 1
        while Scenario.objects.filter(name=new_scenario_name).exists():
            new_scenario_name = f"{new_scenario_name_base} {counter}"
            counter += 1
        
        new_scenario = Scenario.objects.create(
            name=new_scenario_name,
            learning_goals=original_scenario.learning_goals,
            description=original_scenario.description,
            age_of_students=original_scenario.age_of_students,
            subject_domains=original_scenario.subject_domains,
            language=original_scenario.language,
            suggested_learning_time=original_scenario.suggested_learning_time,
            image=original_scenario.image,
            created_by=user,
            updated_by=user,
            is_personal=True,
            origin_scenario=original_scenario,
            family=family,
            variant_type='adaptation',
            ai_metrics_min_implementations=(
                original_scenario.ai_metrics_min_implementations
            ),
        )
        new_scenario.subjects.set(original_scenario.subjects.all())

        # Map for original activities to their duplicates
        activity_mapping = {}
        answer_mapping = {}
        phase_mapping = {}

        # Duplicate phases first so phase-less activities are preserved too.
        for phase in original_scenario.phases.all():
            new_phase = Phase.objects.create(
                name=phase.name,
                description=phase.description,
                image=phase.image,
                scenario=new_scenario,
                created_by=user,
                updated_by=user
            )
            phase_mapping[phase.id] = new_phase

        for activity in original_scenario.activities.order_by('id'):
            new_activity = Activity.objects.create(
                name=activity.name,
                text=activity.text,
                plain_text=activity.plain_text,
                correct_count=activity.correct_count,
                incorrect_count=activity.incorrect_count,
                is_evaluatable=activity.is_evaluatable,
                is_primary_ev=activity.is_primary_ev,
                must_wait=activity.must_wait,
                score_limit=activity.score_limit,
                scenario=new_scenario,
                phase=phase_mapping.get(activity.phase_id),
                activity_type=activity.activity_type,
                helper=activity.helper,
                simulation=activity.simulation,
                experiment_ll=activity.experiment_ll,
                vr_ar_experiment=activity.vr_ar_experiment,
                lineage_key=activity.lineage_key,
                concept=activity.concept,
                created_by=user,
                updated_by=user
            )

            # Copy answers
            for answer in activity.answers.all():
                new_answer = Answer.objects.create(
                    activity=new_activity,
                    text=answer.text,
                    is_correct=answer.is_correct,
                    answer_weight=answer.answer_weight,
                    image=answer.image,
                    vid_url=answer.vid_url,
                    created_by=user,
                    updated_by=user
                )
                answer_mapping[answer.id] = new_answer

            activity_mapping[activity.id] = new_activity

        mapped_start = activity_mapping.get(original_scenario.start_activity_id)
        if mapped_start:
            new_scenario.start_activity = mapped_start
            new_scenario.save(update_fields=["start_activity"])

        # Duplicate Next Question Logic
        for original_activity_id, new_activity in activity_mapping.items():
            original_activity = Activity.objects.get(pk=original_activity_id)
            for logic in original_activity.next_logic.all():
                NextQuestionLogic.objects.create(
                    activity=new_activity,
                    answer=answer_mapping.get(logic.answer.id, None) if logic.answer else None,
                    next_activity=activity_mapping.get(logic.next_activity.id, None) if logic.next_activity else None
                )

        # Duplicate EvQuestionBranching and QuestionBunch for Evaluatable Activities
        for original_activity_id, new_activity in activity_mapping.items():
            original_activity = Activity.objects.get(pk=original_activity_id)
            if original_activity.is_evaluatable:
                if hasattr(original_activity, 'branching'):
                    branching = original_activity.branching
                    EvQuestionBranching.objects.create(
                        activity=new_activity,
                        next_question_on_high=activity_mapping.get(branching.next_question_on_high.id, None) if branching.next_question_on_high else None,
                        next_question_on_high_feedback=branching.next_question_on_high_feedback,
                        next_question_on_mid=activity_mapping.get(branching.next_question_on_mid.id, None) if branching.next_question_on_mid else None,
                        next_question_on_mid_feedback=branching.next_question_on_mid_feedback,
                        next_question_on_low=activity_mapping.get(branching.next_question_on_low.id, None) if branching.next_question_on_low else None,
                        next_question_on_low_feedback=branching.next_question_on_low_feedback,
                    )

                # Duplicate QuestionBunch
                question_bunch = QuestionBunch.objects.filter(activity_primary=original_activity).first()
                if question_bunch:
                    new_bunch = QuestionBunch.objects.create(
                        activity_primary=new_activity,
                        activity_ids=[activity_mapping[aid].id for aid in question_bunch.activity_ids]
                    )
        
        assert_scenario_graph_integrity(new_scenario)

        # After new_scenario is created and before apply_proposals_to_cloned_scenario
        existing_simulation = None
        existing_experiment_ll = None
        existing_vr_ar_experiment = None

        for activity in original_scenario.activities.all():
            if activity.simulation and not existing_simulation:
                existing_simulation = activity.simulation
            if activity.experiment_ll and not existing_experiment_ll:
                existing_experiment_ll = activity.experiment_ll
            if activity.vr_ar_experiment and not existing_vr_ar_experiment:
                existing_vr_ar_experiment = activity.vr_ar_experiment

        apply_proposals_to_cloned_scenario(
            original_scenario=original_scenario,
            new_scenario=new_scenario,
            activity_mapping=activity_mapping,
            user=user,
            default_simulation=existing_simulation,
            default_experiment_ll=existing_experiment_ll,
            default_vr_ar_experiment=existing_vr_ar_experiment,
        )

        assert_scenario_graph_integrity(new_scenario)

        new_scenario.ensure_current_version(
            created_by=user,
            change_summary='Personal adaptation created from accepted proposals',
        )
        # return redirect('updateScenario', id=new_scenario.id)
        return new_scenario.id

    except Exception:
        print("Exception in apply_user_proposals_to_new_scenario:")
        print(traceback.format_exc())
        raise
    
def get_accepted_reviews_for_personal_scenario(original_scenario, user):
    current_version = original_scenario.ensure_current_version()
    current_run = (
        ProposalGenerationRun.objects
        .filter(
            scenario=original_scenario,
            is_current=True,
            scenario_version=current_version,
        )
        .first()
    )
    if not current_run:
        return UserProposalReview.objects.none()
    if (
        current_run.evidence_scope == 'compatible'
        and current_run.evidence_summary.get('source_signature')
        != get_evidence_context(
            original_scenario,
            scope='compatible',
        )['source_signature']
    ):
        current_run.is_current = False
        current_run.save(update_fields=['is_current'])
        return UserProposalReview.objects.none()
    return UserProposalReview.objects.filter(
        user=user,
        proposal__scenario=original_scenario,
        proposal__generation_run=current_run,
        status='accepted'
    ).select_related('proposal', 'proposal__activity', 'proposal__phase')


def apply_proposals_to_cloned_scenario(original_scenario, new_scenario, activity_mapping, user,
                                       default_simulation=None, default_experiment_ll=None, default_vr_ar_experiment=None):

    refactor_all_proposals_to_json()

    accepted_reviews = get_accepted_reviews_for_personal_scenario(original_scenario, user)

    for review in accepted_reviews:
        proposal = review.proposal
        flagged_old = proposal.activity
        flagged_new = activity_mapping.get(flagged_old.id)
        if not flagged_new:
            print(f"Skipping proposal: flagged activity {flagged_old.id} not found in mapping.")
            continue


        # try:
        #     # data = json.loads(proposal.json_action)
        #     data = json.loads(review.teacher_edited_json or proposal.json_action)
        #     # raw_json = review.teacher_edited_json if isinstance(review.teacher_edited_json, str) else json.dumps(review.teacher_edited_json)
        #     # data = json.loads(raw_json or proposal.json_action)
        # except json.JSONDecodeError:
        #     continue
        #############################################################
        # try:
        #     raw = review.teacher_edited_json or proposal.json_action
        #     # print(f"EDITED:\n{review.teacher_edited_json}")
        #     # print(f"NOT EDITED:\n{proposal.json_action}")
        #     if isinstance(raw, dict):
        #         data = raw
        #     else:
        #         data = json.loads(raw)
        # except (json.JSONDecodeError, TypeError) as e:
        #     print(f"❌ Skipping proposal {proposal.id} due to invalid JSON:", e)
        #     continue
        try:
            # Load edited data (from teacher) and original proposal
            edited_raw = review.teacher_edited_json
            base_raw = proposal.json_translated_action or proposal.json_action # proposal.json_action

            # Ensure at least one is present
            if not edited_raw and not base_raw:
                print(f"Skipping proposal {proposal.id}: no JSON available in review or proposal.")
                continue

            try:
                edited = edited_raw if isinstance(edited_raw, dict) else json.loads(edited_raw) if edited_raw else {}
                base = base_raw if isinstance(base_raw, dict) else json.loads(base_raw) if base_raw else {}
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[proposal] Skipping proposal {proposal.id} due to invalid JSON:", e)
                continue

            data = merge_proposal_edits(base, edited)

            expected_type = (
                flagged_new.activity_type.name
                if proposal.proposal_type == "revise"
                else None
            )
            answer_count = flagged_new.answers.count()
            data = validate_proposal_data(
                data,
                expected_action=proposal.proposal_type,
                expected_activity_type=expected_type,
                expected_answer_count=(
                    answer_count
                    if expected_type == "Question" and 2 <= answer_count <= 4
                    else None
                ),
                require_insert_after=proposal_requires_insert_after(
                    flagged_old,
                    proposal.proposal_type,
                ),
            )

        except (json.JSONDecodeError, TypeError, ProposalValidationError) as e:
            record_proposal_structural_failure(
                scenario=original_scenario,
                generation_run=proposal.generation_run,
                proposal=proposal,
                activity=flagged_old,
                selected_action=proposal.proposal_type,
                stage="application",
                errors=e,
                raw_output=str(base_raw or ""),
            )
            print(f"[proposal] Skipping proposal {proposal.id} due to invalid data:", e)
            continue
        
        print(f"DATA:\n{data}")
        content = data.get("content")
        # if not content:
        #     data = parse_suggested_action_block(review.proposal.suggested_action)
        action = data.get("action")
        phase = flagged_new.phase

        if action == "revise":
            print(f"Found proposal to revise: {data}")
            print(f"Flagged activity: {flagged_old.name} (ID={flagged_old.id})")
            flagged_new.text = data.get("content", flagged_new.text)
            flagged_new.plain_text = data.get("content", flagged_new.plain_text)
            flagged_new.save()

            # Only update answers if it's a question AND new answers are provided
            if flagged_new.activity_type.name.lower() == "question" and "answers" in data:
                existing_answers = list(flagged_new.answers.all())
                new_answers = data.get("answers", [])

                # Match by index to preserve connection
                for i, ans_data in enumerate(new_answers):
                    if i < len(existing_answers):
                        existing = existing_answers[i]
                        existing.text = ans_data["text"]
                        existing.is_correct = ans_data["is_correct"]
                        existing.answer_weight = ans_data["weight"]
                        existing.updated_by = user
                        existing.save()
                    else:
                        # Optional: only add extra ones if needed
                        Answer.objects.create(
                            activity=flagged_new,
                            text=ans_data["text"],
                            is_correct=ans_data["is_correct"],
                            answer_weight=ans_data["weight"],
                            created_by=user,
                            updated_by=user
                        )

        # elif action == "create":
        #     new_act = create_activity_from_json_action(data, phase, new_scenario, user)
        #     connect_created_activity(
        #         flagged_new,
        #         new_act,
        #         data.get("insert_location", "after"),
        #         data.get("target_category", "All")
        #     )
        
        # elif action == "create":
        #     # 🔁 Split if proposal targets more than one category
        #     for single_cat_action in split_multi_category_action(data):
        #         new_act = create_activity_from_json_action(single_cat_action, phase, new_scenario, user)
        #         connect_created_activity(
        #             flagged_new,
        #             new_act,
        #             single_cat_action.get("insert_location", "after"),
        #             single_cat_action.get("target_category", "All")
        #         )
        elif action == "create":
            uses_branching = flagged_new.is_evaluatable and hasattr(flagged_new, "branching")

            if uses_branching:
                # 🔁 Split if proposal targets multiple categories
                for single_cat_action in split_multi_category_action(data):
                    new_act = create_activity_from_json_action(single_cat_action, phase, new_scenario, user,
                                                               default_simulation, default_experiment_ll, default_vr_ar_experiment)
                    connect_created_activity(
                        flagged_new,
                        new_act,
                        single_cat_action.get("insert_location", "after"),
                        single_cat_action.get("target_category", "All")
                    )
            else:
                # 👇 Don't split – treat as one general activity
                print(f"Skipping split: {flagged_new.name} does not use branching. Using 'All' for category.")
                data["target_category"] = "All"
                new_act = create_activity_from_json_action(data, phase, new_scenario, user,
                                                           default_simulation, default_experiment_ll, default_vr_ar_experiment)
                connect_created_activity(
                    flagged_new,
                    new_act,
                    data.get("insert_location", "after"),
                    "All"
                )

        elif action == "skip":
            # 1. Track where the skipped activity was pointing
            outgoing_targets = {
                logic.answer.id if logic.answer else None: logic.next_activity
                for logic in flagged_new.next_logic.all()
                if logic.next_activity
            }

            # 2. If evaluatable, get branching targets
            if flagged_new.is_evaluatable and hasattr(flagged_new, 'branching'):
                branching = flagged_new.branching
                branch_targets = {
                    'low': branching.next_question_on_low,
                    'mid': branching.next_question_on_mid,
                    'high': branching.next_question_on_high
                }
            else:
                branch_targets = {}

            if new_scenario.start_activity_id == flagged_new.id:
                successor_ids = {
                    target.id
                    for target in list(outgoing_targets.values())
                    + list(branch_targets.values())
                    if target
                }
                if len(successor_ids) != 1:
                    raise ScenarioGraphValidationError([
                        {
                            "code": "ambiguous_start_replacement",
                            "activity_id": flagged_new.id,
                            "message": (
                                f"Skipping start activity '{flagged_new.name}' "
                                "does not leave exactly one replacement entry "
                                "activity."
                            ),
                        }
                    ])
                new_scenario.start_activity_id = successor_ids.pop()
                new_scenario.save(update_fields=["start_activity"])

            # 3. Reconnect logic pointing to the flagged activity
            for in_logic in NextQuestionLogic.objects.filter(next_activity=flagged_new):
                replacement = outgoing_targets.get(
                    in_logic.answer.id if in_logic.answer else None
                )
                if replacement:
                    in_logic.next_activity = replacement
                    in_logic.save()

            # 4. Rewire branching from other activities
            for b in EvQuestionBranching.objects.filter(
                next_question_on_low=flagged_new
            ) | EvQuestionBranching.objects.filter(
                next_question_on_mid=flagged_new
            ) | EvQuestionBranching.objects.filter(
                next_question_on_high=flagged_new
            ):
                if b.next_question_on_low == flagged_new:
                    b.next_question_on_low = branch_targets.get('low')
                if b.next_question_on_mid == flagged_new:
                    b.next_question_on_mid = branch_targets.get('mid')
                if b.next_question_on_high == flagged_new:
                    b.next_question_on_high = branch_targets.get('high')
                b.save()

            # 5. Clean up outbound logic
            flagged_new.next_logic.all().delete()

            # 6. Remove activity
            flagged_new.delete()

@transaction.atomic
def create_activity_from_json_action(data, phase, scenario, user,
                                     default_simulation=None, default_experiment_ll=None, default_vr_ar_experiment=None):

    data = validate_proposal_data(data, expected_action="create")
    activity_type = ActivityType.objects.get(name__iexact=data["activity_type"])

    simulation = None
    experiment_ll = None
    vr_ar_experiment = None

    if data["activity_type"].lower() == "experiment":
        simulation = default_simulation
        experiment_ll = default_experiment_ll
        vr_ar_experiment = default_vr_ar_experiment

    new_activity = Activity.objects.create(
        name=data["activity_name"],
        text=data["content"],
        plain_text=data["content"],
        activity_type=activity_type,
        phase=phase,
        scenario=scenario,
        simulation=simulation,
        experiment_ll=experiment_ll,
        vr_ar_experiment=vr_ar_experiment,
        created_by=user,
        updated_by=user,
    )

    if activity_type.name.lower() == "question":
        for ans in data["answers"]:
            Answer.objects.create(
                activity=new_activity,
                text=ans["text"],
                is_correct=ans["is_correct"],
                answer_weight=ans["weight"],
                created_by=user,
                updated_by=user
            )

    return new_activity

def connect_created_activity(flagged_activity, new_activity, insert_location, target_category):
    is_eval = flagged_activity.is_evaluatable and hasattr(flagged_activity, "branching")

    if isinstance(target_category, str):
        categories = [target_category.lower()]
    elif isinstance(target_category, list):
        categories = [c.lower() for c in target_category]
    else:
        categories = []

    if not any(c in ["low", "moderate", "high"] for c in categories):
        categories = ["low", "mid", "high"]

    is_question = new_activity.activity_type.name.lower() == "question"
    if is_question:
        new_activity = Activity.objects.prefetch_related("answers").get(id=new_activity.id)

    # ========== INSERT BEFORE ==========
    if insert_location == "before":
        if flagged_activity.scenario.start_activity_id == flagged_activity.id:
            flagged_activity.scenario.start_activity = new_activity
            flagged_activity.scenario.save(update_fields=["start_activity"])

        # Redirect incoming links
        for link in NextQuestionLogic.objects.filter(next_activity=flagged_activity):
            link.next_activity = new_activity
            link.save()

        for branch in EvQuestionBranching.objects.filter(
            Q(next_question_on_low=flagged_activity) |
            Q(next_question_on_mid=flagged_activity) |
            Q(next_question_on_high=flagged_activity)
        ):
            if branch.next_question_on_low == flagged_activity:
                branch.next_question_on_low = new_activity
            if branch.next_question_on_mid == flagged_activity:
                branch.next_question_on_mid = new_activity
            if branch.next_question_on_high == flagged_activity:
                branch.next_question_on_high = new_activity
            branch.save()

        # Connect new → flagged
        if is_question:
            for answer in new_activity.answers.all():
                NextQuestionLogic.objects.get_or_create(
                    activity=new_activity,
                    answer=answer,
                    defaults={"next_activity": flagged_activity}
                )
        else:
            NextQuestionLogic.objects.create(activity=new_activity, next_activity=flagged_activity)

    # ========== INSERT AFTER ==========
    else:
        if is_eval:
            branching = flagged_activity.branching
            for branch_cat in categories:
                attr = {
                    "low": "next_question_on_low",
                    "moderate": "next_question_on_mid",
                    "high": "next_question_on_high",
                    "mid": "next_question_on_mid",
                }[branch_cat]

                original_target = getattr(branching, attr)

                # new → original_target
                if original_target:
                    if is_question:
                        for answer in new_activity.answers.all():
                            NextQuestionLogic.objects.get_or_create(
                                activity=new_activity,
                                answer=answer,
                                defaults={"next_activity": original_target}
                            )
                    else:
                        NextQuestionLogic.objects.create(activity=new_activity, next_activity=original_target)

                # flagged → new
                setattr(branching, attr, new_activity)
            branching.save()

        else:
            existing_links = list(flagged_activity.next_logic.all())
            flagged_activity.next_logic.all().delete()

            for logic in existing_links:
                orig_target = logic.next_activity
                answer = logic.answer

                if answer:
                    NextQuestionLogic.objects.create(activity=flagged_activity, answer=answer, next_activity=new_activity)
                else:
                    NextQuestionLogic.objects.create(activity=flagged_activity, next_activity=new_activity)

                if orig_target:
                    if is_question:
                        for new_answer in new_activity.answers.all():
                            NextQuestionLogic.objects.get_or_create(
                                activity=new_activity,
                                answer=new_answer,
                                defaults={"next_activity": orig_target}
                            )
                    else:
                        NextQuestionLogic.objects.create(activity=new_activity, next_activity=orig_target)

def parse_suggested_action_block(text):
    # Try JSON first
    try:
        parsed = json.loads(text)
        # Clean extra asterisks, e.g. "**revise**"
        parsed["action"] = parsed.get("action", "").strip("* ").lower()
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback to line-based format (e.g. LLM suggestion format)
    data = {
        "action": None,
        "activity_name": None,
        "activity_type": None,
        "content": None,
        "answers": [],
        "target_category": None,
        "insert_location": None,
        "explanation": None,
    }

    lines = text.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    in_answers = False
    in_explanation = False
    explanation_lines = []

    for line in lines:
        if line.lower().startswith("action:"):
            data["action"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("- Activity Name:"):
            data["activity_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Activity Type:"):
            data["activity_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Content:"):
            data["content"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Answers"):
            in_answers = True
        elif line.startswith("- Target Category:"):
            raw_cats = line.split(":", 1)[1].strip()
            data["target_category"] = [cat.strip() for cat in raw_cats.split(",") if cat.strip()] or "all"
            in_answers = False
        elif line.startswith("- Insert Location:"):
            data["insert_location"] = line.split(":", 1)[1].strip() or "before"
        elif line.startswith("Explanation:"):
            in_explanation = True
        elif in_answers:
            match = re.match(r"^\d+\.\s*(.*?)[—-]\s*(Correct|Incorrect)\s+weight\s+(\d+)", line)
            if match:
                text_part = match.group(1).strip()
                correctness = match.group(2).strip().lower() == "correct"
                weight = int(match.group(3).strip())
                data["answers"].append({
                    "text": text_part,
                    "is_correct": correctness,
                    "weight": weight
                })
        elif in_explanation:
            explanation_lines.append(line)

    if explanation_lines:
        data["explanation"] = " ".join(explanation_lines)

    return data

def parse_answers_from_content_block(content: str):
    """
    Extracts answers from the markdown-style text inside `content`
    when the structured `answers` key is empty or missing.
    """
    answers = []
    pattern = r"\d+\.\s*\((Correct|Incorrect)\)\s*([A-Z])\.\s*(.*?)\s*—\s*weight\s*(\d+)"

    for match in re.finditer(pattern, content, re.DOTALL):
        correctness, label, text, weight = match.groups()
        answers.append({
            "text": f"{label}. {text.strip()}",
            "is_correct": correctness.lower() == "correct",
            "weight": int(weight),
        })

    return answers

def refactor_all_proposals_to_json():
    updated = 0
    failed = 0
    skipped = 0
    no_need = 0

    for proposal in ActivityProposal.objects.all():
        try:
            suggested = proposal.suggested_action or ""
            if not suggested.strip():
                print(f"Skipping empty proposal ID {proposal.id}")
                skipped += 1
                continue
            
            if not proposal.json_action:
                parsed = parse_llm_proposal(suggested)
                if not proposal.json_translated_action:
                    translated_parsed = parse_llm_proposal_translated(suggested, proposal.scenario.language, proposal)
                else:
                    translated_parsed = json.loads(proposal.json_translated_action)

                # Dynamically inject categories_in_risk from M2M relationship
                # parsed["target_category"] = ", ".join(proposal.categories_in_risk.values_list("name", flat=True))
                parsed["target_category"] = list(proposal.categories_in_risk.values_list("name", flat=True)) or "all"
                parsed["insert_location"] = parsed.get("insert_location") or "before"

                if not parsed.get("action") or not parsed.get("activity_name"):
                    raise ValueError("Parsed data missing required fields")

                with transaction.atomic():
                    proposal.json_action = json.dumps(parsed, ensure_ascii=False)
                    proposal.json_translated_action = json.dumps(translated_parsed, ensure_ascii=False)
                    proposal.save()
                    print(f"Updated proposal ID {proposal.id}")
                    updated += 1
            else:
                no_need += 1

        except Exception as inner:
            print(f"Failed to fix proposal ID {proposal.id}: {inner}")
            failed += 1

    print("\nSummary:")
    print(f"  Updated: {updated}")
    print(f"  Skipped (empty): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (no need to update): {no_need}")

# def split_multi_category_action(parsed_action):
#     """
#     If a proposed action targets multiple categories (e.g., ["Low", "High"]),
#     split it into multiple single-category actions.
#     """
#     target_categories = parsed_action.get("target_category")
#     if not isinstance(target_categories, list) or len(target_categories) <= 1:
#         return [parsed_action]

#     base_name = parsed_action.get("activity_name", "Unnamed Activity")
#     split_actions = []

#     for cat in target_categories:
#         cloned = parsed_action.copy()
#         cloned["activity_name"] = f"{base_name} ({cat})"
#         cloned["target_category"] = [cat]
#         split_actions.append(cloned)

#     return split_actions

def split_multi_category_action(parsed):
    """
    If the parsed action targets multiple categories, split it into individual
    actions per category with suffixed activity names.
    """
    target = parsed.get("target_category") or []

    if isinstance(target, str):
        target = [target]

    if len(target) <= 1:
        return [parsed]  # No need to split

    out = []
    for cat in target:
        copy = parsed.copy()
        suffix = f" ({cat.capitalize()})"
        copy["activity_name"] = parsed["activity_name"] + suffix
        copy["target_category"] = [cat]
        out.append(copy)

    return out

def parse_llm_proposal_translated(response_suggested_action, scenario_language, proposal=None):
    try:
        parsed = json.loads(response_suggested_action)
        required_fields = {"action", "activity_name", "activity_type", "content"}
        if all(field in parsed for field in required_fields):
            if "answers" not in parsed:
                parsed["answers"] = []
            if proposal:
                parsed["target_category"] = [c.name for c in proposal.categories_in_risk.all()]
            return parsed
    except Exception:
        pass

    data = {
        "action": None,
        "activity_name": None,
        "activity_type": None,
        "content": "",
        "answers": [],
        "target_category": None,
        "insert_location": None,
        "explanation": "",
    }

    lines = response_suggested_action.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    in_content = False
    in_answers = False
    in_explanation = False
    content_lines = []
    explanation_lines = []

    for line in lines:
        if line.lower().startswith("action:"):
            data["action"] = line.split(":", 1)[1].strip().lower()
            in_content = in_answers = in_explanation = False

        elif line.startswith("- Activity Name:"):
            data["activity_name"] = line.split(":", 1)[1].strip()
            if data["activity_name"]:
                data["activity_name"] = translate_text(data["activity_name"], scenario_language)
            in_content = in_answers = in_explanation = False

        elif line.startswith("- Activity Type:"):
            data["activity_type"] = line.split(":", 1)[1].strip()
            in_content = in_answers = in_explanation = False

        elif line.startswith("- Content:"):
            in_content = True
            in_answers = in_explanation = False
            content_lines.append(line.split(":", 1)[1].strip())

        elif line.startswith("- Answers"):
            in_answers = True
            in_content = in_explanation = False

        elif "insert location" in line.lower():
            insert_value = line.split(":", 1)[1].strip().lower()
            if "before" in insert_value:
                data["insert_location"] = "before"
            elif "after" in insert_value:
                data["insert_location"] = "after"
            else:
                data["insert_location"] = "before"
            in_content = in_answers = in_explanation = False

        elif line.lower().startswith("explanation") and ":" in line:
            in_explanation = True
            in_content = in_answers = False
            explanation_text = line.split(":", 1)[1].strip()
            if explanation_text:
                explanation_lines.append(explanation_text)

        elif in_content:
            content_lines.append(line)

        elif in_answers:
            match_1 = re.match(r"^\d+\.\s*\((Correct|Incorrect)\)\s*([A-Z])\.?\s+(.*?)[—-]\s*weight\s+(\d+)", line, re.IGNORECASE)
            match_2 = re.match(r"^([A-Z])\.?\s+(.*?)\s*\(weight\s*(\d+)\)", line, re.IGNORECASE)

            if match_1:
                correctness = match_1.group(1).strip().lower() == "correct"
                letter = match_1.group(2).strip()
                text = match_1.group(3).strip()
                weight = int(match_1.group(4))
                data["answers"].append({
                    "text": f"{letter}. {text}",
                    "is_correct": correctness,
                    "weight": weight
                })

            elif match_2:
                letter = match_2.group(1).strip()
                text = match_2.group(2).strip()
                weight = int(match_2.group(3))
                data["answers"].append({
                    "text": f"{letter}. {text}",
                    "is_correct": (weight == 3),
                    "weight": weight
                })

        elif in_explanation:
            explanation_lines.append(line)

    # Translate content
    data["content"] = "\n".join(content_lines).strip()
    if data["content"]:
        data["content"] = translate_text(data["content"], scenario_language)

    # Translate explanation only if meaningful
    data["explanation"] = "\n".join(explanation_lines).strip()
    if data["explanation"] and "english" not in data["explanation"].lower():
        data["explanation"] = translate_text(data["explanation"], scenario_language)

    # Translate answers
    # for ans in data["answers"]:
    #     if ans.get("text"):
    #         ans["text"] = translate_text(ans["text"], scenario_language)
    for ans in data["answers"]:
        if ans.get("text"):
            # Extract prefix like "A." using regex
            match = re.match(r"^([A-Z]\.)(\s*)(.*)", ans["text"])
            if match:
                prefix, space, body = match.groups()
                translated_body = translate_text(body.strip(), scenario_language)
                ans["text"] = f"{prefix}{space}{translated_body}"
            else:
                # fallback to full translation if format is unexpected
                ans["text"] = translate_text(ans["text"], scenario_language)

    if proposal:
        data["target_category"] = [c.name for c in proposal.categories_in_risk.all()]

    return data
