# Generated by Django 3.0.4 on 2020-04-10 23:29

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('casestudy', '0035_pollutant'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pollutant',
            name='country',
        ),
    ]