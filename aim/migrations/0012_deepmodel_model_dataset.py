# Generated by Django 3.2.16 on 2022-11-17 12:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aim', '0011_deepmodel_model_processor'),
    ]

    operations = [
        migrations.AddField(
            model_name='deepmodel',
            name='model_dataset',
            field=models.CharField(max_length=255, null=True),
        ),
    ]