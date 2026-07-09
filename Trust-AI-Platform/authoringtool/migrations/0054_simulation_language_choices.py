from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0053_remove_video_url'),
    ]

    operations = [
        migrations.AlterField(
            model_name='simulation',
            name='language',
            field=models.CharField(
                blank=True,
                default='',
                max_length=100,
                choices=[
                    ('', '— Select Language —'),
                    ('English', 'English'),
                    ('Greek', 'Greek'),
                    ('Spanish', 'Spanish'),
                    ('French', 'French'),
                    ('German', 'German'),
                    ('Italian', 'Italian'),
                    ('Portuguese', 'Portuguese'),
                    ('Dutch', 'Dutch'),
                    ('Polish', 'Polish'),
                    ('Romanian', 'Romanian'),
                    ('Turkish', 'Turkish'),
                    ('Arabic', 'Arabic'),
                    ('Other', 'Other'),
                ],
            ),
        ),
    ]
