from django.db import migrations

SUBJECTS = [
    # (name, icon, color, category, order)
    ('Physics',                    'bi-lightning-charge',  '#1a56db', 'STEM', 1),
    ('Mathematics',                'bi-calculator',        '#6366f1', 'STEM', 2),
    ('Chemistry',                  'bi-eyedropper',        '#7c3aed', 'STEM', 3),
    ('Biology',                    'bi-flower1',           '#059669', 'STEM', 4),
    ('Computer Science',           'bi-cpu',               '#0891b2', 'STEM', 5),
    ('Environmental Science',      'bi-tree',              '#16a34a', 'STEM', 6),
    ('Earth & Space Science',      'bi-globe',             '#0d9488', 'STEM', 7),
    ('Engineering & Technology',   'bi-gear',              '#ea580c', 'STEM', 8),
    ('Statistics & Data Science',  'bi-bar-chart-line',   '#dc2626', 'STEM', 9),
]


def populate_subjects(apps, schema_editor):
    Subject = apps.get_model('authoringtool', 'Subject')
    for name, icon, color, category, order in SUBJECTS:
        Subject.objects.get_or_create(
            name=name,
            defaults=dict(icon=icon, color=color, category=category, order=order),
        )


def remove_subjects(apps, schema_editor):
    Subject = apps.get_model('authoringtool', 'Subject')
    Subject.objects.filter(name__in=[s[0] for s in SUBJECTS]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0050_subject_scenario_subjects'),
    ]

    operations = [
        migrations.RunPython(populate_subjects, reverse_code=remove_subjects),
    ]