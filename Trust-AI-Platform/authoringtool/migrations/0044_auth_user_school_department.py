import django.db.models.deletion
from django.db import migrations, models


class AddAuthUserSchoolDepartmentState(migrations.operations.base.Operation):
    reduces_to_sql = False
    reversible = True

    def state_forwards(self, app_label, state):
        state.add_field(
            "auth",
            "user",
            "school_department",
            models.ForeignKey(
                to="authoringtool.schooldepartment",
                on_delete=django.db.models.deletion.SET_NULL,
                null=True,
                blank=True,
            ),
            preserve_default=True,
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def describe(self):
        return "Adds school_department to auth.User in the migration state"


class Migration(migrations.Migration):

    dependencies = [
        ("authoringtool", "0043_indexes_and_activityproposal_fixes"),
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(
                    sql="""
                        ALTER TABLE auth_user
                        ADD COLUMN IF NOT EXISTS school_department_id bigint NULL;

                        CREATE INDEX IF NOT EXISTS auth_user_school_department_id_idx
                        ON auth_user (school_department_id);

                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1
                                FROM pg_constraint
                                WHERE conname = 'auth_user_school_department_id_fk'
                            ) THEN
                                ALTER TABLE auth_user
                                ADD CONSTRAINT auth_user_school_department_id_fk
                                FOREIGN KEY (school_department_id)
                                REFERENCES authoringtool_schooldepartment (id)
                                ON DELETE SET NULL
                                DEFERRABLE INITIALLY DEFERRED;
                            END IF;
                        END
                        $$;
                    """,
                    reverse_sql="""
                        ALTER TABLE auth_user
                        DROP CONSTRAINT IF EXISTS auth_user_school_department_id_fk;

                        DROP INDEX IF EXISTS auth_user_school_department_id_idx;

                        ALTER TABLE auth_user
                        DROP COLUMN IF EXISTS school_department_id;
                    """,
                ),
            ],
            state_operations=[
                AddAuthUserSchoolDepartmentState(),
            ],
        ),
    ]
