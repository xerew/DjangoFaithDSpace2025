from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0052_simulation_language'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='scenario',
            name='video_url',
        ),
        migrations.RemoveField(
            model_name='phase',
            name='video_url',
        ),
    ]
