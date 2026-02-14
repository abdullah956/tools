from django.contrib.postgres.operations import TrigramExtension
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('workflow', '0002_workflow_user_query'),  # Replace with your last migration name
    ]

    operations = [
        TrigramExtension(),
    ]