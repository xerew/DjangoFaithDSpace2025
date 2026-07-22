from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_userprofile_gender_userprofile_picture_and_more'),
        ('organization', '0004_orgchatmessage'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='BulkEmailCampaign',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('target_type', models.CharField(choices=[('all_teachers', 'All teachers'), ('selected_teachers', 'Selected teachers'), ('organizations', 'Selected organizations')], max_length=30)),
                ('subject', models.CharField(max_length=200)),
                ('body_html', models.TextField()),
                ('site_url', models.URLField(blank=True)),
                ('status', models.CharField(choices=[('queued', 'Queued'), ('sending', 'Sending'), ('completed', 'Completed'), ('partial', 'Completed with errors'), ('failed', 'Failed')], default='queued', max_length=20)),
                ('recipient_count', models.PositiveIntegerField(default=0)),
                ('sent_count', models.PositiveIntegerField(default=0)),
                ('failed_count', models.PositiveIntegerField(default=0)),
                ('error_summary', models.TextField(blank=True)),
                ('celery_task_id', models.CharField(blank=True, max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('created_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='created_bulk_email_campaigns', to=settings.AUTH_USER_MODEL)),
                ('organizations', models.ManyToManyField(blank=True, related_name='bulk_email_campaigns', to='organization.organization')),
                ('recipients', models.ManyToManyField(blank=True, related_name='bulk_email_campaigns', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at', '-id'],
            },
        ),
        migrations.AddIndex(
            model_name='bulkemailcampaign',
            index=models.Index(fields=['status', 'created_at'], name='bulk_email_status_created_idx'),
        ),
    ]
