# Generated by Django 3.2 on 2022-11-08 20:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aim', '0010_deepmodel_model_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='deepmodel',
            name='model_processor',
            field=models.CharField(default='', max_length=255),
            preserve_default=False,
        ),
    ]