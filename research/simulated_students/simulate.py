"""Run simulated students on the exported item bank.

Each job is one simulated student (model x persona x context x replicate)
answering one activity. Results are appended to a JSONL file; running the
same command again skips finished jobs, so long runs can be resumed.
"""
import argparse
import base64
import json
import os
import random
import threading
from collections import defaultdict
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import prompts


def load_items(path, limit=None):
    with open(path, encoding='utf-8') as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    return items[:limit] if limit else items


def sample_items(items, n, seed=0):
    """A reproducible random sample that keeps every language represented:
    one activity per language first, the rest in proportion to their size."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    by_language = defaultdict(list)
    for item in items:
        by_language[item.get('language') or ''].append(item)
    languages = sorted(by_language)
    quota = {language: 1 for language in languages} if n >= len(languages) else {}
    spare = n - sum(quota.values())
    for language in languages:
        share = int(spare * len(by_language[language]) / len(items))
        quota[language] = min(len(by_language[language]), quota.get(language, 0) + share)
    picked = []
    for language in languages:
        picked += rng.sample(by_language[language], quota.get(language, 0))
    chosen = {item['item_id'] for item in picked}
    rest = [item for item in items if item['item_id'] not in chosen]
    rng.shuffle(rest)
    picked += rest[:n - len(picked)]
    return sorted(picked, key=lambda item: item['item_id'])


def job_seed(item_id, persona, context, replicate):
    """Same seed for every model, so all models see identical prompts and shuffles."""
    return zlib.crc32(f'{item_id}|{persona}|{context}|{replicate}'.encode())


def job_key(model, item_id, persona, context, replicate, temperature):
    return f'{model}|{item_id}|{persona}|{context}|{replicate}|{prompts.PROMPT_VERSION}|{temperature}'


def finished_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path, encoding='utf-8') as handle:
        return {json.loads(line)['key'] for line in handle if line.strip()}


def encode_images(item, base_dir):
    encoded = []
    for relative in prompts.image_paths(item):
        with open(os.path.join(base_dir, relative), 'rb') as handle:
            encoded.append(base64.b64encode(handle.read()).decode())
    return encoded


def default_think(model):
    # Qwen 3 otherwise spends its tokens "thinking" and may leave the JSON empty.
    return False if model.lower().startswith('qwen3') else None


def run(client, items_path, out_path, models, personas, contexts, replicates,
        temperature, workers, limit=None, sample=None, sample_seed=0):
    items = load_items(items_path, limit)
    if sample:
        items = sample_items(items, sample, sample_seed)
    base_dir = os.path.dirname(os.path.abspath(items_path))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = finished_keys(out_path)
    vision = {model: client.has_vision(model) for model in models}
    skipped = {model: 0 for model in models}

    jobs = []
    for model in models:
        for item in items:
            has_images = bool(prompts.image_paths(item))
            if has_images and not vision[model]:
                skipped[model] += 1
                continue
            for persona in personas:
                for context in contexts:
                    for replicate in range(replicates):
                        key = job_key(model, item['item_id'], persona, context, replicate, temperature)
                        if key not in done:
                            jobs.append((key, model, item, persona, context, replicate))

    lock = threading.Lock()

    def work(job):
        key, model, item, persona, context, replicate = job
        seed = job_seed(item['item_id'], persona, context, replicate)
        order = prompts.option_order(item, seed)
        prompt, letters = prompts.build_prompt(item, persona, context, order)
        images = encode_images(item, base_dir) if vision[model] else []
        row = {
            'key': key, 'item_id': item['item_id'], 'model': model, 'persona': persona,
            'context': context, 'replicate': replicate, 'seed': seed,
            'temperature': temperature, 'prompt_version': prompts.PROMPT_VERSION,
            'offered_order': [item['options'][i]['id'] for i in order],
            'used_images': bool(images),
            'choice_option_id': None, 'choice_letter': None, 'correct': None,
            'confidence': None, 'reason': None, 'latency_s': None, 'error': None,
        }
        try:
            text, latency = client.generate(
                model, prompt, prompts.reply_schema(len(order)), images,
                temperature, seed, default_think(model),
            )
            row['latency_s'] = round(latency, 3)
            parsed = prompts.parse_reply(text, letters)
            if parsed is None:
                row['error'] = f'unusable reply: {text[:200]}'
            else:
                option_id, letter, confidence, reason = parsed
                correct_id = next(o['id'] for o in item['options'] if o['is_correct'])
                row.update(choice_option_id=option_id, choice_letter=letter,
                           correct=option_id == correct_id, confidence=confidence, reason=reason)
        except Exception as exc:  # network or model errors are recorded, the run goes on
            row['error'] = f'{type(exc).__name__}: {exc}'[:300]
        row['finished_at'] = datetime.now(timezone.utc).isoformat()
        with lock:
            with open(out_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, jobs))
    return {'jobs_run': len(jobs), 'skipped_image_items': skipped, 'vision': vision}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--items', required=True, help='item_bank.jsonl from export_item_bank')
    parser.add_argument('--out', required=True, help='results JSONL (appended to, resumable)')
    parser.add_argument('--model', action='append', required=True, help='Ollama model, repeatable')
    parser.add_argument('--persona', action='append', choices=sorted(prompts.PERSONAS))
    parser.add_argument('--context', action='append', choices=prompts.CONTEXTS)
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, help='only the first N items')
    parser.add_argument('--sample', type=int,
                        help='a random sample of N items, spread over languages (pilots)')
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--ollama-url', default=os.environ.get('OLLAMA_URL', 'http://localhost:11434'))
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args(argv)

    from .ollama import OllamaClient

    summary = run(
        client=OllamaClient(args.ollama_url, timeout=args.timeout),
        items_path=args.items, out_path=args.out, models=args.model,
        personas=args.persona or ['low', 'mid', 'high'], contexts=args.context or ['none'],
        replicates=args.replicates, temperature=args.temperature,
        workers=args.workers, limit=args.limit,
        sample=args.sample, sample_seed=args.sample_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
