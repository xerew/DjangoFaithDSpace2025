from django.db import migrations


def add_turkish(apps, schema_editor):
    Language = apps.get_model('authoringtool', 'Language')
    Language.objects.get_or_create(name='Turkish')


def remove_turkish(apps, schema_editor):
    Language = apps.get_model('authoringtool', 'Language')
    Language.objects.filter(name='Turkish').delete()


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0047_scenariohealthproxy'),
    ]

    operations = [
        migrations.RunPython(add_turkish, remove_turkish),
    ]
