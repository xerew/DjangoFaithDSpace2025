from django.db import migrations

from authoringtool.risk_flag_text import parse_reason


def backfill(apps, schema_editor):
    """Give existing flags a reason code and numbers parsed from the stored
    English sentence. Flags that do not match any sentence are left alone."""
    ActivityFlag = apps.get_model('authoringtool', 'ActivityFlag')
    rows = ActivityFlag.objects.filter(reason_code='').only(
        'id', 'flag_type', 'flag_reason')
    for flag in rows.iterator():
        parsed = parse_reason(flag.flag_type, flag.flag_reason)
        if parsed:
            ActivityFlag.objects.filter(pk=flag.pk).update(
                reason_code=parsed[0], reason_params=parsed[1])


class Migration(migrations.Migration):

    dependencies = [
        ('authoringtool', '0076_activityflag_reason_code'),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
