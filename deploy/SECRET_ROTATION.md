# Rotating secrets and deploying the October 2026 security fixes

The repository was public and contained the production `.env` (secret key,
database and SMTP passwords) and student account exports. Treat every value
that was ever in `.env` as known to others. Do the steps in this order, in one
maintenance window. Commands run on the server from the project directory.

`dc` below means `docker compose -f docker-compose.yml -f docker-compose.prod.yml`.

## 0. Before touching the server

1. GitHub → repository Settings → Danger Zone → **Change visibility → Private**.
2. Papaki → mailbox `info@trust-ai-lab.eu` → **change the password**. Check the
   sent folder and Papaki logs for mail you did not send.

## 1. Back up

```sh
dc run --rm db-backup backup-now
cp .env ~/env.backup-$(date +%F)
```

## 2. Generate new secrets (on the server, never paste them anywhere else)

```sh
python3 -c "import secrets; print(secrets.token_urlsafe(50))"   # DJANGO_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(24))"   # POSTGRES_PASSWORD
python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"   # STUDENT_CREDENTIALS_KEY
```

## 3. Change the database password (while the old stack still runs)

```sh
dc exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\password $POSTGRES_USER"'
```

Type the new POSTGRES_PASSWORD twice at the prompt.

## 4. Pull the code

Pulling the commit that stops tracking `.env` deletes it from the working
tree, so restore it right after:

```sh
git fetch origin
git checkout improvements/performance-and-responsive   # or the branch you deploy
git pull
cp ~/env.backup-$(date +%F) .env
```

## 5. Edit `.env`

Compare with `.env.example`. Set:

- `DJANGO_SECRET_KEY` = new value (logs everyone out once)
- `POSTGRES_PASSWORD` = the password from step 3
- `EMAIL_HOST_PASSWORD` = the new Papaki password
- `STUDENT_CREDENTIALS_KEY` = new value (add this line)

## 6. Deploy

```sh
dc up -d --build
dc logs -f web     # migrations 0071–0073, feedback 0002, usergroups 0003–0004 apply
```

`dc exec web python manage.py check` must not show `usergroups.W001`.

## 7. New teacher registration code

The old code is in the public source. Choose a new one and hash it:

```sh
dc exec web python manage.py shell -c "from django.contrib.auth.hashers import make_password; print(make_password(input('New teacher code: ')))"
```

Add it to `.env` **in single quotes** (the hash contains `$`):

```
TEACHER_ACCESS_CODE_HASHED='pbkdf2_sha256$...'
```

Then `dc up -d` to restart with it, and give the new code to teachers.

## 8. Lock guessable student passwords

Accounts made by the old `create_students` command use
`<username>password123123`, which anyone can work out from the source code.

```sh
dc exec web python manage.py lock_guessable_student_passwords          # report only
dc exec web python manage.py lock_guessable_student_passwords --apply  # lock them
```

The accounts and their work are kept; set a new password in the admin if a
student needs to log in again.

## 9. Check

- Log in as a teacher and as a student; the student chat answers.
- Download a group's credentials sheet; the passwords work.
- `https://trust-ai-lab.eu/rasa/` and `/rag_pdfs/` return 404.

## Afterwards

- Decide whether to remove the leaked files from git history (rewrites
  history; every clone must be re-cloned).
- Assess the personal-data exposure with your DPO (GDPR art. 33: 72 hours
  from becoming aware, if it is likely to put students at risk).
