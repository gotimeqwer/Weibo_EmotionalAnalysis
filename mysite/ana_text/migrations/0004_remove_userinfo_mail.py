# Generated by Django 4.0.4 on 2022-05-31 15:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ana_text', '0003_userinfo_mail'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userinfo',
            name='mail',
        ),
    ]