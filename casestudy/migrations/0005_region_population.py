# Generated by Django 3.0.4 on 2020-03-29 13:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('casestudy', '0004_auto_20200328_2306'),
    ]

    operations = [
        migrations.AddField(
            model_name='region',
            name='population',
            field=models.PositiveIntegerField(null=True, verbose_name='Population'),
        ),
    ]