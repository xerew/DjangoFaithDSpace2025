"""
Management command to normalize Organization.language values to match
the canonical spellings stored in the authoringtool.Language model.

- Single-language values are matched case-insensitively or via the alias map.
- Multi-language values (e.g. 'Greek, English', 'GREEK ENGLISH', 'English/Greek')
  are split and the non-English language is preferred.

Usage (dry run — shows what would change):
    python manage.py normalize_org_languages

Usage (apply changes):
    python manage.py normalize_org_languages --apply
"""
import re
from django.core.management.base import BaseCommand
from authoringtool.models import Language
from organization.models import Organization

# Maps lowercase alias -> canonical Language.name
ALIASES = {
    # Basque
    'euskara': 'Basque',
    'euskera': 'Basque',
    'vasco': 'Basque',

    # Catalan
    'català': 'Catalan',
    'catala': 'Catalan',
    'valencian': 'Catalan',

    # English
    'eng': 'English',
    'en': 'English',
    'inglés': 'English',
    'ingles': 'English',
    'anglais': 'English',

    # Estonian
    'eesti': 'Estonian',
    'et': 'Estonian',

    # French
    'français': 'French',
    'francais': 'French',
    'fr': 'French',

    # German
    'deutsch': 'German',
    'de': 'German',
    'allemand': 'German',
    'tedesco': 'German',

    # Greek
    'ελληνικά': 'Greek',
    'ελληνικα': 'Greek',
    'ellinika': 'Greek',
    'gr': 'Greek',
    'el': 'Greek',
    'grec': 'Greek',

    # Italian
    'italiano': 'Italian',
    'it': 'Italian',
    'italien': 'Italian',

    # Persian
    'farsi': 'Persian',
    'فارسی': 'Persian',
    'fa': 'Persian',
    'dari': 'Persian',

    # Portuguese
    'português': 'Portuguese',
    'portugues': 'Portuguese',
    'pt': 'Portuguese',
    'pt-br': 'Portuguese',
    'portuguese (brazil)': 'Portuguese',
    'portuguese (portugal)': 'Portuguese',

    # Romanian
    'română': 'Romanian',
    'romana': 'Romanian',
    'ro': 'Romanian',
    'roumain': 'Romanian',

    # Spanish
    'español': 'Spanish',
    'espanol': 'Spanish',
    'castellano': 'Spanish',
    'es': 'Spanish',
    'spa': 'Spanish',

    # Turkish
    'türkçe': 'Turkish',
    'turkce': 'Turkish',
    'turco': 'Turkish',
    'tr': 'Turkish',

    # Ukrainian
    'українська': 'Ukrainian',
    'ukrainska': 'Ukrainian',
    'ukraynaca': 'Ukrainian',
    'uk': 'Ukrainian',
}

# Delimiters that separate multiple languages in one string
SPLIT_PATTERN = re.compile(r'[,/]+|\s+and\s+|\s{2,}')


def resolve_token(token, canonical, canonical_values):
    """Return canonical language name for a single token, or None."""
    t = token.strip()
    if not t:
        return None
    if t in canonical_values:
        return t
    lower = t.lower()
    if lower in canonical:
        return canonical[lower]
    if lower in ALIASES:
        target = ALIASES[lower]
        if target in canonical_values:
            return target
    return None


def pick_language(resolved):
    """
    Given a list of resolved canonical names, prefer the non-English one.
    If multiple non-English languages exist, return the first.
    Falls back to English if that is the only recognized language.
    """
    non_english = [lang for lang in resolved if lang != 'English']
    if non_english:
        return non_english[0]
    if 'English' in resolved:
        return 'English'
    return None


class Command(BaseCommand):
    help = 'Normalize Organization language values to canonical Language model spellings.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--apply',
            action='store_true',
            help='Apply the changes (default is dry run).',
        )

    def handle(self, *args, **options):
        apply = options['apply']

        canonical = {lang.name.lower(): lang.name for lang in Language.objects.all()}
        canonical_values = set(canonical.values())

        self.stdout.write('\n=== Canonical languages in Language model ===')
        for name in sorted(canonical_values):
            self.stdout.write(f'  {name}')

        self.stdout.write('\n=== Scanning organizations ===')
        changed = []
        unmatched = []

        for org in Organization.objects.exclude(language=None).exclude(language=''):
            current = org.language.strip()

            if current in canonical_values:
                continue  # already canonical

            # --- Step 1: try the whole string as-is ---
            new_val = resolve_token(current, canonical, canonical_values)

            # --- Step 2: split into tokens and pick best ---
            if new_val is None:
                # Try splitting by delimiters first, then fall back to single spaces
                tokens = SPLIT_PATTERN.split(current)
                if len(tokens) == 1:
                    # No delimiter found — try splitting on single spaces
                    tokens = current.split(' ')

                resolved = [resolve_token(t, canonical, canonical_values) for t in tokens]
                resolved = [r for r in resolved if r is not None]
                new_val = pick_language(resolved)

            if new_val is None:
                unmatched.append((org, current))
                continue

            if new_val != current:
                changed.append((org, current, new_val))

        if changed:
            self.stdout.write(f'\n  {len(changed)} organization(s) to update:')
            for org, old, new in changed:
                self.stdout.write(f'    [{org.id}] {org.name!r}: {old!r}  ->  {new!r}')
            if apply:
                for org, old, new in changed:
                    org.language = new
                    org.save(update_fields=['language'])
                self.stdout.write(self.style.SUCCESS(f'\n  Applied {len(changed)} change(s).'))
            else:
                self.stdout.write(self.style.WARNING(
                    '\n  Dry run — pass --apply to save changes.'
                ))
        else:
            self.stdout.write(self.style.SUCCESS(
                '\n  All organization languages are already canonical. Nothing to do.'
            ))

        if unmatched:
            self.stdout.write(self.style.WARNING(
                f'\n  {len(unmatched)} organization(s) have an unrecognized language value:'
            ))
            for org, val in unmatched:
                self.stdout.write(f'    [{org.id}] {org.name!r}: {val!r}')
            self.stdout.write(
                '  Add the value to ALIASES in this command or to the Language model.'
            )
        else:
            self.stdout.write(self.style.SUCCESS('\n  No unrecognized language values.'))
