# Generate FAITH AI proposals on this PC, then copy them to the server

Instructions for Claude Code running on the user's home PC (Windows, Docker
Desktop, RTX 4080 Super 16 GB, 32 GB RAM). Work through the parts in order.
Run the commands yourself where you can, and stop to ask the user where a
step says so.

## What this is for

The FAITH platform's server normally generates AI proposals with an Ollama
server at NTUA, which is often unreachable. Instead, this PC runs its own
copy of the platform in Docker, with:

- a fresh dump of the server's database
- the scenario PDFs
- this PC's own Ollama

The user generates the proposals here. Then only the proposals go back to
the server as one JSON file, which the server imports.

**Nothing connects to this PC.** The user copies two files from the server
(a database dump and the PDFs) and one file back (the proposals). Nothing
else changes on the network.

## Rules (do not break these)

- **The site stays local.** It must be reachable only at
  `http://127.0.0.1:8000`. Never publish ports on `0.0.0.0`, never forward
  ports on the router, and never set `OLLAMA_HOST=0.0.0.0`.
- **No emails.** The dump contains every student's and teacher's email
  address. `EMAIL_HOST_PASSWORD` must stay empty in the local `.env` (the
  platform then only prints emails to its log). **Never start
  `celerybeat`**, nor `rasa`, `rasa_actions`, `rasa-eval`, `nginx`,
  `certbot`, `db-backup` or `ollama-tunnel`. Start only `db`, `redis`,
  `web` and `celery`, exactly as shown below.
- **Personal data.** The dump holds personal data of students and teachers.
  - Keep it only inside the project folder on this PC.
  - Never upload it anywhere, and never paste its contents into the chat.
  - Delete it when the work is done (Part 6).
- **The local copy is for generating proposals only.** Do not edit
  scenarios, users or anything else in it, and do not accept or reject
  proposals here. Review happens on the server after the import.
- **Never commit `.env`, the dump, the PDFs or the exported JSON to git.**

## Part 1 — Ollama and the models

1. Check that Ollama is installed (`ollama --version`). If it is missing,
   install it with `winget install --id Ollama.Ollama -e`, or ask the user
   to install it from ollama.com.
2. It must listen on 127.0.0.1 only:
   `Get-NetTCPConnection -LocalPort 11434 -State Listen | Select LocalAddress`
   should show `127.0.0.1`.
3. Set these user environment variables, then restart Ollama (quit it from
   the tray and start it again):
   `[Environment]::SetEnvironmentVariable('NAME','value','User')`
   - `OLLAMA_MAX_LOADED_MODELS=1`
   - `OLLAMA_KEEP_ALIVE=30m`
   - `OLLAMA_FLASH_ATTENTION=1`
4. About 75 GB of free disk space is needed for the models. Pull exactly
   these names, one after the other, and report progress:
   ```
   ollama pull qwen3.6:35b
   ollama pull aya-expanse:32b
   ollama pull gemma4:31b
   ollama pull gemma3:4b
   ```
   If a name is not found, stop and tell the user. Do not substitute
   another model: the server's proposals are made with these.
5. Test: `ollama run gemma3:4b "Say hello in Greek."`

## Part 2 — The platform's code

1. Check that Docker Desktop is installed and running (`docker version`).
   If not, ask the user to install it (it needs WSL 2) and start it.
2. Ask the user where to put the project (suggest `C:\faith`). Clone it
   (the user may need to sign in to GitHub):
   ```
   git clone https://github.com/xerew/DjangoFaithDSpace2025.git C:\faith
   cd C:\faith
   git checkout main
   git pull
   ```
   The repository is large; let the clone finish.
3. Check that `docker-compose.home.yml` exists in the folder, and that
   `Trust-AI-Platform/authoringtool/management/commands/export_proposals.py`
   exists. If not, the code is older than this guide: tell the user.

## Part 3 — The local `.env` (never copy the server's)

Create `C:\faith\.env` with the contents below. Fill the two random values
with PowerShell:

```
-join ((48..57)+(65..90)+(97..122) | Get-Random -Count 50 | % {[char]$_})
```

```
DJANGO_SECRET_KEY=<random 50 characters>
DJANGO_DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CSRF_TRUSTED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
SITE_URL=http://127.0.0.1:8000

STUDENT_CREDENTIALS_KEY=
EMAIL_HOST=localhost
EMAIL_HOST_USER=
EMAIL_HOST_PASSWORD=
DEFAULT_FROM_EMAIL=local@localhost

POSTGRES_DB=faith
POSTGRES_USER=faith
POSTGRES_PASSWORD=<another random 50 characters>
POSTGRES_HOST=db
POSTGRES_PORT=5432

CELERY_BROKER_URL=redis://redis:6379/0
RAG_PDFS_DIR=/data/rag/rag_pdfs
RAG_INDEXES_DIR=/data/rag/rag_indexes
AI_METRICS_CACHE_DIR=/data/ai_metrics_cache
OLLAMA_URL=http://host.docker.internal:11434
SCENARIO_SIMILARITY_EMBEDDINGS_ENABLED=0
```

Leave `EMAIL_HOST_PASSWORD` empty (see the rules).

## Part 4 — The data from the server (needs the user)

Ask the user to run these **on the server**, in the platform's folder (the
one with `docker-compose.yml`). Show them the commands; do not try to
connect to the server yourself.

```
mkdir -p ~/faith-export
docker compose exec -T db sh -c 'pg_dump --format=custom --compress=6 --no-owner --no-privileges -U "$POSTGRES_USER" "$POSTGRES_DB"' > ~/faith-export/faith.dump
grep RAG_PDFS_DIR .env        # normally /data/rag/rag_pdfs, i.e. ./data/rag/rag_pdfs on the server
tar czf ~/faith-export/rag_pdfs.tgz -C data/rag rag_pdfs
ls -lh ~/faith-export
```

The user then fetches the two files to this PC, for example from
PowerShell on this PC:

```
scp <user>@<server>:~/faith-export/faith.dump C:\faith\data\
scp <user>@<server>:~/faith-export/rag_pdfs.tgz C:\faith\data\
```

Then ask the user to delete the copies on the server:
`rm -r ~/faith-export`.

On this PC, unpack the PDFs:
```
New-Item -ItemType Directory -Force C:\faith\data\rag | Out-Null
tar xzf C:\faith\data\rag_pdfs.tgz -C C:\faith\data\rag
```
Check that `C:\faith\data\rag\rag_pdfs` now contains folders or PDF files.

## Part 5 — Start, restore, generate, export

Every command runs in `C:\faith`. Define this once per PowerShell session.
It is a short name for Docker Compose with the local override file, so
nothing else can be started by mistake:

```
function fdc { docker compose -f docker-compose.yml -f docker-compose.home.yml @args }
```

(If you run commands through Git Bash instead, use
`docker compose -f docker-compose.yml -f docker-compose.home.yml` in place of
`fdc`. The commands below avoid nested double quotes, which Windows
PowerShell 5.1 passes on incorrectly.)

1. **Database only, then restore the dump.** It must be a fresh, empty
   database. If an old local copy exists, ask the user before removing it
   with `fdc down -v`.
   ```
   fdc up -d --no-deps db
   # wait until it is ready:
   fdc exec db sh -c 'until pg_isready -U $POSTGRES_USER; do sleep 2; done'
   docker cp C:\faith\data\faith.dump "$(fdc ps -q db):/tmp/faith.dump"
   fdc exec db sh -c 'pg_restore --no-owner --no-privileges -U $POSTGRES_USER -d $POSTGRES_DB /tmp/faith.dump; rm /tmp/faith.dump'
   ```
   `pg_restore` may print a few warnings. Report any line that says
   `error`, but keep going unless the table count below is zero:
   `fdc exec db sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB -tc ''select count(*) from authoringtool_scenario'''`
2. **Start the rest, and only the rest.** The first build takes a while.
   ```
   fdc up -d --build --no-deps redis web celery
   fdc ps
   ```
   Running services must be exactly `db`, `redis`, `web` and `celery`. If
   anything else is running, stop it: `fdc stop <name>`.
   The web container applies any newer database migrations on start; wait
   until `fdc logs --tail 20 web` shows the development
   server running.
3. **Check that Celery reaches this PC's Ollama.**
   ```
   fdc exec celery python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=10).read()[:200])"
   ```
   It must print model names. If it cannot connect, stop and tell the
   user. Do not change `OLLAMA_HOST`.
4. **Generate (the user does this).** Tell the user:
   - Open `http://127.0.0.1:8000` and log in with your usual platform
     account (it is in the dump).
   - For each scenario, open **Metrics & AI** and press **Generate**.
     Doing it on one translation of a scenario is enough: its confirmed
     translations get the same proposals.
   - A run can take a long time on this PC. The page shows its progress.
     Do not accept or reject proposals here.

   While a run is going, you can watch it with
   `fdc logs -f --tail 50 celery`.
5. **Export (when the user says the runs are finished).**
   ```
   fdc exec celery python manage.py export_proposals --output /data/proposals.json
   ```
   This writes `C:\faith\data\proposals.json` and lists each scenario with
   its number of proposals. Every scenario's current proposals are
   included; the server skips the ones it already has. To send only
   today's runs, add `--since 2026-11-28T00:00` (use the date the runs
   started).

## Part 6 — Copy the proposals to the server (needs the user), then clean up

1. Show the user how to copy the file and import it. From this PC:
   ```
   scp C:\faith\data\proposals.json <user>@<server>:<platform folder>/data/proposals.json
   ```
   Then **on the server**, in the platform folder:
   ```
   docker compose exec web python manage.py import_proposals /data/proposals.json --dry-run
   ```
   This changes nothing; it lists, per scenario, what would be imported
   and what is skipped and why. If it looks right:
   ```
   docker compose exec web python manage.py import_proposals /data/proposals.json
   rm data/proposals.json
   ```
   Each imported scenario's proposals become the current ones on the
   server, and the previous ones move to Proposal History. A scenario that
   got newer proposals on the server since the dump is skipped; add
   `--force` to replace them anyway. Proposals for activities that were
   deleted or renamed since the dump are left out and listed.
2. **Clean up this PC** (ask the user first). Stop the copy and delete its
   database and the files:
   ```
   fdc down -v
   Remove-Item C:\faith\data\faith.dump, C:\faith\data\rag_pdfs.tgz, C:\faith\data\proposals.json
   Remove-Item -Recurse C:\faith\data\rag, C:\faith\data\ai_metrics_cache
   ```
   The code, the Docker images and the Ollama models can stay for next
   time. Next time, repeat from Part 4 with a fresh dump.

## Report for the user

Finish with a short summary:

- which models were pulled
- whether the restore and the Ollama check worked
- which scenarios were exported, with their proposal counts
- the exact `scp` and `import_proposals` commands, filled in with the paths
  you used
- anything not finished, and why
