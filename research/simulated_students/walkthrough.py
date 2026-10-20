"""Simulated students go through whole scenarios, as real students do in Plato.

Each simulated student (model x persona x replicate) starts at the scenario's
first activity, reads the pages, answers the questions and is routed exactly
as Plato (Rasa) routes students: by the chosen answer, or at evaluation
points by the average weight of the answers in the bunch. The whole lesson is
one conversation, so the student remembers what it read and answered.

Input: scenario_bank.jsonl from `manage.py export_scenario_bank`.
Output: one JSONL line per simulated student; running the same command again
skips finished students, so long runs can be resumed.
"""
import argparse
import base64
import json
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import prompts
from .simulate import default_think

# Change whenever the wording or the reply format changes.
PROMPT_VERSION = 'sw-2026-10-20'
PAGE_LIMIT = 2000  # characters of one page kept in the conversation


class RouteError(Exception):
    """The scenario is configured so that Plato cannot continue."""


def load_scenarios(path, scenario_ids=None):
    with open(path, encoding='utf-8') as handle:
        scenarios = [json.loads(line) for line in handle if line.strip()]
    if scenario_ids:
        scenarios = [s for s in scenarios if s['scenario_id'] in set(scenario_ids)]
    return scenarios


def nodes_of(scenario):
    return {int(key): node for key, node in scenario['activities'].items()}


def route(nodes, node, option_id, answered):
    """Next activity id (None = end of scenario) and evaluation band, as Rasa decides.

    answered: (activity id, option id) of every answer so far, this one included.
    Must match authoringtool/management/commands/export_scenario_bank.py.
    """
    if node['kind'] == 'page':
        return node['next'], None
    evaluation = node['evaluation']
    if evaluation is None:
        option = next((o for o in node['options'] if o['id'] == option_id), None)
        return (option['next'] if option else None), None
    if not evaluation['high'] or evaluation['high']['to'] is None or evaluation['mid']['to'] is None:
        raise RouteError(f'activity {node["id"]}: evaluation branching incomplete')
    weights = {o['id']: o['weight'] for n in nodes.values() for o in n['options']}
    bunch = set(evaluation['bunch'])
    scored = [(activity, option) for activity, option in answered if activity in bunch]
    average = sum(weights.get(option, 0) for _, option in scored) / (len({a for a, _ in scored}) or 1)
    for band in ('high', 'mid'):
        if average >= evaluation[band]['limit']:
            return evaluation[band]['to'], band
    return evaluation['low']['to'], 'low'


def run_key(model, scenario_id, persona, replicate, temperature):
    return f'{model}|{scenario_id}|{persona}|{replicate}|{PROMPT_VERSION}|{temperature}'


def seed_for(*parts):
    """Same seed for every model, so all models see identical option orders."""
    return zlib.crc32('|'.join(str(p) for p in parts).encode())


def finished_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path, encoding='utf-8') as handle:
        return {json.loads(line)['key'] for line in handle if line.strip()}


# ── prompts ───────────────────────────────────────────────────────────────
def system_prompt(scenario, persona):
    low, high = scenario.get('age_min'), scenario.get('age_max')
    age = f'a {low}-{high} year old' if low and high else 'a secondary-school'
    language = scenario.get('language') or 'English'
    return '\n\n'.join([
        f'You are role-playing {age} school student whose first language is {language}.',
        prompts.PERSONAS[persona],
        f'You are going through an online science lesson ("{scenario.get("scenario", "")}") in a chat. '
        'The platform shows you pages to read and asks you questions. Answer each question exactly '
        'as this student would, not as an expert; students like you do sometimes choose a wrong answer. '
        'Use what you read and answered earlier in the lesson, as far as this student would remember it.',
    ])


def question_message(scenario, node, order, pages, vision):
    """The user message for one question: pages read since the last question, then the question."""
    language = scenario.get('language') or 'English'
    parts, image_notes, letters = [], [], {}
    if pages:
        read = '\n\n'.join(page['text'][:PAGE_LIMIT] for page in pages)
        parts.append(f'You read:\n"""\n{read}\n"""')
        if any(page['interactive'] for page in pages):
            parts.append('(One of these pages had an interactive simulation or experiment, which you explored.)')
        for page in pages:
            image_notes += [f'image {n} of the page you read' for n in range(1, len(page['images']) + 1)]
    parts.append(f'Question (in {language}):\n{node["text"]}')
    image_notes += [f'image {n} of the question' for n in range(1, len(node['images']) + 1)]
    lines = []
    for position, index in enumerate(order):
        option = node['options'][index]
        letter = prompts.LETTERS[position]
        letters[letter] = option['id']
        lines.append(f'{letter}. {option["text"]}')
        if option.get('image'):
            image_notes.append(f'option {letter}')
    parts.append('Options:\n' + '\n'.join(lines))
    if image_notes:
        if vision:
            parts.append('The attached images are, in order: ' + '; '.join(image_notes) + '.')
        else:
            parts.append('(You cannot see the images; rely on the text.)')
    parts.append(
        'Reply in JSON with the letter you choose, how confident you are (0-100) '
        "and one short sentence in the student's own words explaining why."
    )
    return '\n\n'.join(parts), letters


def images_for(node, pages, base_dir):
    paths = [p for page in pages for p in page['images']] + prompts.image_paths(node)
    encoded = []
    for relative in paths:
        with open(os.path.join(base_dir, relative), 'rb') as handle:
            encoded.append(base64.b64encode(handle.read()).decode())
    return encoded


# ── one simulated student ─────────────────────────────────────────────────
def walk(client, scenario, model, persona, replicate, temperature, base_dir,
         vision=False, think=None, num_ctx=16384, max_steps=200):
    nodes = nodes_of(scenario)
    scenario_id = scenario['scenario_id']
    messages = [{'role': 'system', 'content': system_prompt(scenario, persona)}]
    steps, answered, pages, visits = [], [], [], {}
    row = {
        'key': run_key(model, scenario_id, persona, replicate, temperature),
        'scenario_id': scenario_id, 'model': model, 'persona': persona, 'replicate': replicate,
        'temperature': temperature, 'prompt_version': PROMPT_VERSION, 'used_images': False,
        'steps': steps, 'end_reason': None, 'error': None,
    }
    current = scenario['start']
    try:
        while True:
            if current is None:
                row['end_reason'] = 'end'
                break
            if len(steps) >= max_steps:
                row['end_reason'] = 'max_steps'
                break
            node = nodes.get(current)
            if node is None:
                raise RouteError(f'activity {current} is not in this scenario')
            if node['kind'] == 'page':
                steps.append({'activity_id': current, 'kind': 'page'})
                pages.append(node)
                current, _ = route(nodes, node, None, answered)
                continue
            if not node['options']:
                raise RouteError(f'question {current} has no options')

            visits[current] = visits.get(current, 0) + 1
            seed = seed_for(scenario_id, current, persona, replicate, visits[current])
            order = prompts.option_order(node, seed)
            content, letters = question_message(scenario, node, order, pages, vision)
            message = {'role': 'user', 'content': content}
            images = images_for(node, pages, base_dir) if vision else []
            if images:
                message['images'] = images
                row['used_images'] = True
            # One retry for an unusable reply or a network error, so a single
            # hiccup does not end a whole lesson.
            parsed, latency = None, 0.0
            for attempt in range(2):
                try:
                    text, seconds = client.chat(
                        model, messages + [message], prompts.reply_schema(len(order)),
                        temperature, seed + attempt, think, num_ctx,
                    )
                except Exception:
                    if attempt:
                        raise
                    continue
                latency += seconds
                parsed = prompts.parse_reply(text, letters)
                if parsed:
                    break
            if parsed is None:
                row['end_reason'] = 'invalid_reply'
                row['error'] = f'unusable reply: {text[:1000]}'
                break
            option_id, letter, confidence, reason = parsed
            option = next(o for o in node['options'] if o['id'] == option_id)
            graded = any(o['is_correct'] for o in node['options'])
            steps.append({
                'activity_id': current, 'kind': 'question',
                'offered_order': [node['options'][i]['id'] for i in order],
                'choice_option_id': option_id, 'choice_letter': letter,
                'correct': option['is_correct'] if graded else None, 'weight': option['weight'],
                'confidence': confidence, 'reason': reason, 'latency_s': round(latency, 3),
            })
            # Images are seen when the question is asked; afterwards the student remembers the text.
            messages += [{'role': 'user', 'content': content}, {'role': 'assistant', 'content': text}]
            pages = []
            answered.append((current, option_id))
            current, _ = route(nodes, node, option_id, answered)
    except RouteError as exc:
        row['end_reason'], row['error'] = 'route_error', str(exc)
    except Exception as exc:  # network or model errors are recorded, the run goes on
        row['end_reason'], row['error'] = 'error', f'{type(exc).__name__}: {exc}'[:300]

    questions = [s for s in steps if s['kind'] == 'question']
    row.update(
        total_score=sum(s['weight'] for s in questions),
        questions=len(questions),
        correct=sum(1 for s in questions if s['correct']),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    return row


# ── many simulated students ───────────────────────────────────────────────
def run(client, scenarios_path, out_path, models, personas, replicates, temperature, workers,
        scenario_ids=None, num_ctx=16384, max_steps=200):
    scenarios = load_scenarios(scenarios_path, scenario_ids)
    base_dir = os.path.dirname(os.path.abspath(scenarios_path))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = finished_keys(out_path)
    vision = {model: client.has_vision(model) for model in models}
    jobs = [
        (model, scenario, persona, replicate)
        for model in models for scenario in scenarios
        for persona in personas for replicate in range(replicates)
        if run_key(model, scenario['scenario_id'], persona, replicate, temperature) not in done
    ]
    lock = threading.Lock()

    def work(job):
        model, scenario, persona, replicate = job
        row = walk(client, scenario, model, persona, replicate, temperature, base_dir,
                   vision=vision[model], think=default_think(model), num_ctx=num_ctx, max_steps=max_steps)
        with lock:
            with open(out_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, jobs))
    return {'students_run': len(jobs), 'scenarios': len(scenarios), 'vision': vision}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenarios', required=True, help='scenario_bank.jsonl from export_scenario_bank')
    parser.add_argument('--out', required=True, help='results JSONL (appended to, resumable)')
    parser.add_argument('--model', action='append', required=True, help='Ollama model, repeatable')
    parser.add_argument('--persona', action='append', choices=sorted(prompts.PERSONAS))
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--scenario', type=int, action='append', help='only these scenario ids')
    parser.add_argument('--num-ctx', type=int, default=16384, help='Ollama context length in tokens')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--ollama-url', default=os.environ.get('OLLAMA_URL', 'http://localhost:11434'))
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args(argv)

    from .ollama import OllamaClient

    summary = run(
        client=OllamaClient(args.ollama_url, timeout=args.timeout),
        scenarios_path=args.scenarios, out_path=args.out, models=args.model,
        personas=args.persona or ['low', 'mid', 'high'], replicates=args.replicates,
        temperature=args.temperature, workers=args.workers, scenario_ids=args.scenario,
        num_ctx=args.num_ctx, max_steps=args.max_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
