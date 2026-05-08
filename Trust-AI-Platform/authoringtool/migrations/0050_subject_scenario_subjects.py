from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0049_subject_model'),
    ]

    operations = [
        migrations.CreateModel(
            name='Subject',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('icon', models.CharField(default='bi-book', max_length=60)),
                ('color', models.CharField(default='#1a56db', max_length=20)),
                ('category', models.CharField(
                    choices=[
                        ('STEM', 'STEM'),
                        ('Humanities', 'Humanities'),
                        ('Social Sciences', 'Social Sciences'),
                        ('Arts', 'Arts'),
                        ('Other', 'Other'),
                    ],
                    default='STEM',
                    max_length=30,
                )),
                ('order', models.PositiveIntegerField(default=0)),
            ],
            options={
                'verbose_name': 'Subject',
                'verbose_name_plural': 'Subjects',
                'ordering': ['order', 'name'],
            },
        ),
        migrations.AddField(
            model_name='scenario',
            name='subjects',
            field=models.ManyToManyField(blank=True, related_name='scenarios', to='authoringtool.subject'),
        ),
    ]