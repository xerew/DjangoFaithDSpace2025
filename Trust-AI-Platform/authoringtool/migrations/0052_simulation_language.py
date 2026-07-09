from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0051_populate_subjects'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation',
            name='language',
            field=models.CharField(blank=True, default='', max_length=100),
        ),
    ]
