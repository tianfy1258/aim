# Generated by Django 3.2 on 2022-11-07 19:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aim', '0002_rename_user_name_user_username'),
    ]

    operations = [
        migrations.RenameField(
            model_name='dataset',
            old_name='dataset_num',
            new_name='dataset_instances',
        ),
        migrations.AddField(
            model_name='dataset',
            name='labels_num',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]
