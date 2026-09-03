from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0004_normalize_ai_metrics_threshold'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='MaintenanceNotice',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('reason', models.TextField(help_text='Explain why the platform is undergoing maintenance.')),
                ('starts_at', models.DateTimeField(db_index=True, verbose_name='start date and time')),
                ('ends_at', models.DateTimeField(db_index=True, verbose_name='end date and time')),
                ('is_enabled', models.BooleanField(db_index=True, default=True, help_text='Disable this notice without deleting its schedule.')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='maintenance_notices_created', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-starts_at', '-id'],
                'indexes': [models.Index(fields=['is_enabled', 'starts_at', 'ends_at'], name='maintenance_active_window_idx')],
            },
        ),
    ]
