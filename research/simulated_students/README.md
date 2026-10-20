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

## Whole scenarios (walk-through)

Single activities show whether simulated students find the same questions
hard. The walk-through asks whether they also go through a **scenario** like
real students: each simulated student starts at the first activity, reads the
pages, answers the questions and is routed exactly as Plato routes students
(by the chosen answer, or at evaluation points by the average answer weight
of the bunch against the destination's score limit). The lesson is one
conversation, so the student remembers what it read and answered.

**Questions**

1. At choice points, do simulated students go the same ways as real ones
   (Jensen-Shannon divergence of the next activity)?
2. Do they land in the same evaluation bands (high/mid/low)?
3. Are their total scores distributed like the real ones (Wasserstein distance)?
4. Do they find the same activities hard, and do the same activities separate
   strong from weak students (difficulty and point-biserial rank correlation)?
5. Do the personas order themselves low < mid < high over a whole scenario?

| File | Purpose |
|---|---|
| `Trust-AI-Platform/authoringtool/management/commands/export_scenario_bank.py` | Export scenario graphs + real aggregate runs |
| `walkthrough.py` | Simulated students go through scenarios (resumable) |
| `analyze_walkthrough.py` | Compares with real runs, writes `report.md`, `summary.json`, `scenarios.csv` |

Design notes:

- The export repeats Plato's (Rasa's) routing: page → its route; question →
  the chosen answer's route; evaluation point → high/mid/low destination by
  the bunch's average weight. The branching feedback texts are not exported
  because Plato does not show them to students.
- A real student's run is their first implementation; consecutive repeated
  answers to the same activity (double clicks) count once.
- Teachers edit scenarios while classes use them, so some real runs no longer
  match the current routes. The export counts them (`follows_routes`) but
  summarises only the runs that match the current routes, the scenario the
  simulated students walk. `--min-students` applies to those runs.
- Whole paths rarely repeat (most scenarios have no path taken by 5
  students), so the comparison is mostly per step: routes at choice points,
  bands, item statistics, scores. Paths taken by fewer than
  `--min-path-students` (default 5) real students are not exported.
- Real students drop out (lesson time ends); simulated students do not.
  Completion is reported for context, and score and path comparisons use
  completed runs.
- Pages with an interactive simulation are marked as explored; the model
  cannot use the simulation itself.
- Ollama's default context is too short for a whole scenario; `--num-ctx`
  (default 16384) sets it. Images are attached when a question is asked;
  later in the conversation the student remembers only the text.

```sh
# 1. Export (public scenarios, at least 30 students on the current routes)
dc exec web python manage.py export_scenario_bank --output /data/research/scenario_bank.jsonl

# 2. Pilot: two scenarios, 3 personas, 2 replicates
dc run --rm --no-deps $R web python -m simulated_students.walkthrough \
  --scenarios /data/research/scenario_bank.jsonl --out /data/research/runs/walk_pilot.jsonl \
  --model gemma3:4b --scenario 97 --scenario 100 --replicates 2

# 3. Full run (resumable)
nohup docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm --no-deps $R web \
  python -m simulated_students.walkthrough \
  --scenarios /data/research/scenario_bank.jsonl --out /data/research/runs/walk.jsonl \
  --model gemma3:4b --model qwen3.6:35b --model aya-expanse:32b --replicates 10 --workers 2 \
  > /data/research/runs/walk.log 2>&1 &

# 4. Report
dc run --rm --no-deps $R web python -m simulated_students.analyze_walkthrough \
  --scenarios /data/research/scenario_bank.jsonl --results /data/research/runs/walk.jsonl \
  --out /data/research/report_walk
```

## Tests

```sh
cd research && python -m unittest discover -s simulated_students/tests -t .
```
