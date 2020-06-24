# Generated by Django 3.0.4 on 2020-04-10 17:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('casestudy', '0030_country_numeric'),
    ]

    operations = [
        migrations.AddField(
            model_name='country',
            name='alt1',
            field=models.CharField(max_length=200, null=True, verbose_name='Alternative Name 1'),
        ),
        migrations.AddField(
            model_name='country',
            name='alt2',
            field=models.CharField(max_length=200, null=True, verbose_name='Alternative Name 2'),
        ),
        migrations.CreateModel(
            name='Pollutants',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(verbose_name='Date')),
                ('longitude', models.FloatField(verbose_name='Longitude')),
                ('latitude', models.FloatField(verbose_name='Latitude')),
                ('temp', models.FloatField(null=True, verbose_name='Temperature')),
                ('dewpoint', models.FloatField(null=True, verbose_name='Dewpoint Temperature')),
                ('uvb', models.FloatField(null=True, verbose_name='UV Radiation')),
                ('evap', models.FloatField(null=True, verbose_name='Evaporation')),
                ('country', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='casestudy.Country')),
            ],
        ),
    ]