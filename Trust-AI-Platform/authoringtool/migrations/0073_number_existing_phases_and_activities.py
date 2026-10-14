from django.db import migrations


def number_in_current_order(apps, schema_editor):
    """Give existing phases and activities the order they are shown in today."""
    Phase = apps.get_model('authoringtool', 'Phase')
    Activity = apps.get_model('authoringtool', 'Activity')

    def number(queryset, key):
        changed, group, position = [], object(), 0
        for item in queryset.order_by(key, 'created_on', 'id'):
            if getattr(item, key) != group:
                group, position = getattr(item, key), 0
            position += 1
            item.order = position
            changed.append(item)
        queryset.model.objects.bulk_update(changed, ['order'], batch_size=1000)

    number(Phase.objects.only('id', 'scenario_id', 'created_on'), 'scenario_id')
    number(Activity.objects.filter(phase__isnull=False).only('id', 'phase_id', 'created_on'), 'phase_id')
    number(Activity.objects.filter(phase__isnull=True).only('id', 'scenario_id', 'created_on'), 'scenario_id')


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0072_order_names_provenance_research_data'),
    ]

    operations = [
        migrations.RunPython(number_in_current_order, migrations.RunPython.noop),
    ]
