"""Export multiple-choice activities and how real students did on them.

Used by the simulated-students study (research/simulated_students). Each line
of the output is one activity: its text, options and preceding material, and
aggregate results of students' first attempts — overall, per option and per
ability group. No data about individual students is written.
"""
import base64
import html
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count, Min, Q
from django.utils.html import strip_tags

from authoringtool.models import Activity, Answer, UserAnswer

IMG_TAG = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
MATERIAL_LIMIT = 3000
# Teachers often type the option letter themselves ("A. ...", "b) ..."). The
# study shuffles options and adds its own letters, so remove theirs.
OPTION_LABEL = re.compile(r'^\(?[A-Za-zΑ-Ωα-ω][.)]\s+')
ABILITY_GROUPS = ('low', 'mid', 'high')


def plain(html_text):
    return ' '.join(html.unescape(strip_tags(html_text or '')).split())


class Command(BaseCommand):
    help = 'Export activities and aggregate student results for the simulated-students study.'

    def add_arguments(self, parser):
        parser.add_argument('--output', required=True, help='Path of the JSONL file to write.')
        parser.add_argument('--min-answers', type=int, default=30,
                            help='Minimum number of students who answered an activity.')
        parser.add_argument('--visibility', action='append', choices=['public', 'org', 'private'],
                            help='Scenario visibility to include (default: public only).')

    def handle(self, *args, output, min_answers, visibility, **options):
        visibility = visibility or ['public']
        out_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        candidates = self.single_answer_questions(visibility)
        first = self.first_attempts(candidates)
        per_item = defaultdict(list)
        for row in first:
            per_item[row['activity_id']].append(row)
        items = [item_id for item_id, rows in per_item.items() if len(rows) >= min_answers]
        ability = self.ability_groups(first)

        activities = {
            a.id: a for a in Activity.objects.filter(id__in=items)
            .select_related('scenario', 'phase').order_by()
        }
        written, skipped_images = 0, 0
        with open(output, 'w', encoding='utf-8') as handle:
            for item_id in sorted(items):
                record, skipped = self.record(activities[item_id], per_item[item_id], ability, out_dir)
                skipped_images += skipped
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                written += 1

        manifest = {
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'items': written,
            'min_answers': min_answers,
            'visibility': visibility,
            'external_images_skipped': skipped_images,
        }
        with open(output + '.manifest.json', 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
        self.stdout.write(self.style.SUCCESS(f'Exported {written} activities to {output}'))

    # ── selection ──────────────────────────────────────────────────────────
    def single_answer_questions(self, visibility):
        """Activities whose scenario is visible as chosen and that have exactly one correct option."""
        return list(
            Activity.objects.filter(
                scenario__visibility_status__in=visibility,
                scenario__archived_at__isnull=True,
            )
            .annotate(
                options=Count('answers', distinct=True),
                correct=Count('answers', filter=Q(answers__is_correct=True), distinct=True),
            )
            .filter(options__gte=2, correct=1)
            .order_by()
            .values_list('id', flat=True)
        )

    def first_attempts(self, activity_ids):
        """Each student's first recorded choice on each activity (teachers excluded)."""
        attempts = (
            UserAnswer.objects.filter(activity_id__in=activity_ids, answer__isnull=False)
            .exclude(user__groups__name='teachers')
        )
        first_ids = attempts.values('user_id', 'activity_id').annotate(first=Min('id')).values('first')
        return list(
            UserAnswer.objects.filter(id__in=first_ids).order_by()
            .values('user_id', 'activity_id', 'activity__scenario_id', 'answer_id',
                    'answer__is_correct', 'timing')
        )

    def ability_groups(self, first):
        """Map (activity, student) to low/mid/high by the student's score on the
        scenario's other activities, so an activity never rates itself."""
        by_scenario = defaultdict(lambda: defaultdict(dict))
        for row in first:
            by_scenario[row['activity__scenario_id']][row['user_id']][row['activity_id']] = bool(
                row['answer__is_correct']
            )
        groups = {}
        for answers_by_student in by_scenario.values():
            items = {item for answers in answers_by_student.values() for item in answers}
            for item in items:
                rest = []
                for student, answers in answers_by_student.items():
                    others = [ok for other, ok in answers.items() if other != item]
                    if item in answers and others:
                        rest.append((sum(others) / len(others), student))
                rest.sort()
                for position, (_, student) in enumerate(rest):
                    groups[(item, student)] = ABILITY_GROUPS[position * 3 // len(rest)]
        return groups

    # ── one activity ──────────────────────────────────────────────────────
    def record(self, activity, rows, ability, out_dir):
        options = list(Answer.objects.filter(activity=activity).order_by('id'))
        images, skipped = [], 0
        text = activity.text or ''
        for index, match in enumerate(IMG_TAG.finditer(text), start=1):
            path = self.copy_image(match.group(1), f'{activity.id}_{index}', out_dir)
            if path:
                images.append(path)
            else:
                skipped += 1
        numbers = iter(range(1, 1000))
        text = IMG_TAG.sub(lambda _match: f' [Image {next(numbers)}] ', text)

        option_records = []
        for option in options:
            image = None
            if option.image:
                image = self.copy_file(option.image.path, f'{activity.id}_option{option.id}', out_dir)
            option_records.append({
                'id': option.id, 'text': OPTION_LABEL.sub('', plain(option.text)),
                'is_correct': option.is_correct, 'image': image,
            })

        n = len(rows)
        correct = sum(1 for row in rows if row['answer__is_correct'])
        counts = defaultdict(int)
        for row in rows:
            counts[str(row['answer_id'])] += 1
        times = [row['timing'] for row in rows if row['timing'] and row['timing'] > 0]
        by_ability = {}
        for group in ABILITY_GROUPS:
            members = [row for row in rows if ability.get((activity.id, row['user_id'])) == group]
            if members:
                by_ability[group] = {
                    'n': len(members),
                    'p_correct': sum(1 for row in members if row['answer__is_correct']) / len(members),
                }

        ages = activity.scenario.age_of_students
        return {
            'item_id': activity.id,
            'scenario_id': activity.scenario_id,
            'scenario': activity.scenario.name,
            'phase': activity.phase.name if activity.phase_id else '',
            'language': activity.scenario.language or '',
            'age_min': getattr(ages, 'lower', None),
            'age_max': getattr(ages, 'upper', None),
            'learning_goals': plain(activity.scenario.learning_goals)[:MATERIAL_LIMIT],
            'text': plain(text),
            'options': option_records,
            'images': images,
            'missing_images': skipped,
            'preceding_material': self.preceding_material(activity),
            'real': {
                'n': n,
                'p_correct': correct / n,
                'option_shares': {key: value / n for key, value in counts.items()},
                'median_seconds': median(times) if times else None,
                'by_ability': by_ability,
            },
        }, skipped

    def preceding_material(self, activity):
        """Text of the non-question activities shown before this one in its phase."""
        if not activity.phase_id:
            return ''
        before = (
            Activity.objects.filter(phase_id=activity.phase_id)
            .filter(Q(order__lt=activity.order) | Q(order=activity.order, id__lt=activity.id))
            .exclude(activity_type__name='Question')
            .order_by('order', 'id')
        )
        material = '\n\n'.join(filter(None, (a.plain_text or plain(a.text) for a in before)))
        return material[-MATERIAL_LIMIT:]

    # ── images ────────────────────────────────────────────────────────────
    def copy_image(self, src, stem, out_dir):
        if src.startswith('data:image/'):
            header, _, data = src.partition(',')
            extension = header.split('/')[1].split(';')[0] or 'png'
            relative = os.path.join('images', f'{stem}.{extension}')
            with open(os.path.join(out_dir, relative), 'wb') as handle:
                handle.write(base64.b64decode(data))
            return relative.replace(os.sep, '/')
        media_url = settings.MEDIA_URL
        path = re.sub(r'^https?://[^/]+', '', src)
        if path.startswith(media_url):
            return self.copy_file(os.path.join(settings.MEDIA_ROOT, path[len(media_url):]), stem, out_dir)
        return None  # external image: not copied, counted in the manifest

    def copy_file(self, source, stem, out_dir):
        if not os.path.exists(source):
            return None
        extension = os.path.splitext(source)[1] or '.png'
        relative = os.path.join('images', f'{stem}{extension}')
        shutil.copyfile(source, os.path.join(out_dir, relative))
        return relative.replace(os.sep, '/')
