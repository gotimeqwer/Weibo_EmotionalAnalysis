# Generated by Django 4.0.4 on 2022-05-14 14:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ana_text', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UsersHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('userid', models.CharField(max_length=16)),
                ('history', models.CharField(max_length=64)),
            ],
        ),
    ]
