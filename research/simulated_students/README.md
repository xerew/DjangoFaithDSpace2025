# Simulated students

Can LLM "students" predict how a real class will do? Simulated students of
low, middle and high ability answer the platform's multiple-choice activities;
their results are compared with real students' first attempts.

**Questions**

1. Do simulated students find the same activities hard as real students
   (rank correlation of the share of correct answers)?
2. Do they choose the same wrong options (most popular distractor, and the
   divergence between answer distributions)?
3. Do the low/mid/high personas behave like real low/mid/high students?
4. Does it hold across languages (English, Greek, Portuguese, Spanish,
   Estonian), with and without images, and with or without the material
   students read just before the question?

**Data and privacy.** `export_item_bank` writes one line per activity with
aggregate results only (counts and shares, overall and per ability group).
Nothing about individual students leaves the database. By default only
activities from **public** scenarios are exported. Exports and results go to
`data/research/` on the server, which is not tracked by git. Do not commit
them: the repository is public.

## Files

| File | Purpose |
|---|---|
| `Trust-AI-Platform/authoringtool/management/commands/export_item_bank.py` | Export activities + real aggregate results |
| `prompts.py` | Personas, prompt, reply format (`PROMPT_VERSION`) |
| `simulate.py` | Runs the simulated students on Ollama (resumable) |
| `analyze.py` | Compares with real students, writes `report.md`, `summary.json`, `items.csv` |
| `metrics.py` | Spearman, Pearson, bootstrap CI, Jensen-Shannon divergence |

Design notes:

- Options are shuffled per simulated student with a seed that depends on the
  activity, persona, context and replicate, but not on the model, so every
  model sees identical prompts.
- Personas are described qualitatively; giving a target accuracy would make
  the model reproduce the number instead of student reasoning.
- Ability groups of real students are based on their score on the scenario's
  *other* activities, so an activity never rates itself.
- Activities with images are sent only to models that accept images (asked
  from Ollama); text-only models skip them, and the report shows "text only"
  and "with images" separately. `missing_images` counts images hosted on
  other websites that could not be copied.
- Change `PROMPT_VERSION` whenever the prompt changes; results are keyed by it.

## Running on the server

From the project folder, after deploying the code that contains
`export_item_bank` (`git pull` and `up -d --build`):

```sh
alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml'
R='-v ./research:/research -w /research'

# 1. Export the item bank (public scenarios, at least 30 students per activity)
dc exec web python manage.py export_item_bank --output /data/research/item_bank.jsonl

# 2. Which models does the NTUA Ollama have?
dc exec web python -c "import os, requests; print([m['name'] for m in requests.get(os.environ['OLLAMA_URL'] + '/api/tags').json()['models']])"

# 3. Pilot: 30 random activities across languages, 3 personas, 3 replicates
dc run --rm --no-deps $R web python -m simulated_students.simulate \
  --items /data/research/item_bank.jsonl --out /data/research/runs/pilot.jsonl \
  --model gemma3:4b --sample 30 --replicates 3

# 4. Full run (hours; resumable, so it can be stopped and started again)
nohup docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm --no-deps $R web \
  python -m simulated_students.simulate \
  --items /data/research/item_bank.jsonl --out /data/research/runs/full.jsonl \
  --model gemma3:4b --model qwen3.6:35b --model aya-expanse:32b \
  --context none --context material --replicates 10 --workers 2 \
  > /data/research/runs/full.log 2>&1 &

# 5. Report
dc run --rm --no-deps $R web python -m simulated_students.analyze \
  --items /data/research/item_bank.jsonl --results /data/research/runs/full.jsonl \
  --out /data/research/report
```

The runner reads `OLLAMA_URL` from the container (the NTUA tunnel). Results,
logs and the report are under `data/research/` on the host.

## Tests

```sh
cd research && python -m unittest discover -s simulated_students/tests -t .
```
