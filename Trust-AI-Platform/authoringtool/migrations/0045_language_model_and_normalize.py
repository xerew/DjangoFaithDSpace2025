from django.db import migrations, models

INITIAL_LANGUAGES = [
    'Basque',
    'Catalan',
    'English',
    'Estonian',
    'French',
    'German',
    'Greek',
    'Italian',
    'Persian',
    'Portuguese',
    'Romanian',
    'Spanish',
    'Ukrainian',
]

# Maps lowercase variant -> canonical language name
LANGUAGE_NORM = {
    # English variants
    'english':    'English',
    'anglais':    'English',
    'inglês':     'English',
    'inglés':     'English',
    'englisch':   'English',
    # Greek variants
    'ελληνικά':   'Greek',
    'ε΄λληνικά':  'Greek',
    'ελληνικα':   'Greek',
    'gr':         'Greek',
    # Portuguese variants
    'português':  'Portuguese',
    'portugais':  'Portuguese',
    'portuguese': 'Portuguese',
    # German variants
    'deutsch':    'German',
    'german':     'German',
    # Spanish variants
    'español':    'Spanish',
    'spanish':    'Spanish',
    # Italian
    'italiano':   'Italian',
    'italian':    'Italian',
    # French
    'français':   'French',
    'french':     'French',
    # Romanian
    'romanian':   'Romanian',
    # Estonian
    'estonian':   'Estonian',
    # Basque
    'euskara':    'Basque',
    'basque':     'Basque',
    # Catalan
    'català':     'Catalan',
    'catalan':    'Catalan',
    # Ukrainian
    'українська': 'Ukrainian',
    'ukrainian':  'Ukrainian',
    # Persian
    'persian':    'Persian',
}


def seed_languages_and_normalize(apps, schema_editor):
    Language = apps.get_model('authoringtool', 'Language')
    Scenario = apps.get_model('authoringtool', 'Scenario')

    # Create canonical languages
    for name in INITIAL_LANGUAGES:
        Language.objects.get_or_create(name=name)

    # Normalize existing scenario language values
    for scenario in Scenario.objects.all():
        lang = (scenario.language or '').strip()
        canonical = LANGUAGE_NORM.get(lang.lower())
        if canonical and canonical != lang:
            scenario.language = canonical
            scenario.save(update_fields=['language'])


def reverse_migration(apps, schema_editor):
    Language = apps.get_model('authoringtool', 'Language')
    Language.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0044_auth_user_school_department'),
    ]

    operations = [
        migrations.CreateModel(
            name='Language',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
            ],
            options={
                'ordering': ['name'],
            },
        ),
        migrations.RunPython(seed_languages_and_normalize, reverse_migration),
    ]
