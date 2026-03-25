from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0041_activityproposal_json_translated_action'),
    ]

    operations = [
        migrations.AddIndex(
            model_name='useranswer',
            index=models.Index(fields=['user', 'activity'], name='useranswer_user_activity_idx'),
        ),
    ]
