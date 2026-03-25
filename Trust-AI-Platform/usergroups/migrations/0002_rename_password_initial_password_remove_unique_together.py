from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('usergroups', '0001_initial'),
    ]

    operations = [
        # Remove the redundant unique_together (OneToOneField already enforces global uniqueness on user)
        migrations.AlterUniqueTogether(
            name='usergroupmembership',
            unique_together=set(),
        ),
        # Rename password -> initial_password to make the intent clear (SEC-22)
        migrations.RenameField(
            model_name='usergroupmembership',
            old_name='password',
            new_name='initial_password',
        ),
        # Update field definition: add blank=True and help_text
        migrations.AlterField(
            model_name='usergroupmembership',
            name='initial_password',
            field=models.CharField(
                blank=True,
                max_length=128,
                help_text='Initial assigned password for teacher export. Not updated when student changes password.',
            ),
        ),
    ]
