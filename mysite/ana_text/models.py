from django.db import models

# Create your models here.


class UserInfo(models.Model):
    userid = models.CharField(max_length=16, primary_key=True)
    password = models.CharField(max_length=16)
    mail = models.CharField(max_length=16,null=True)



class UsersHistory(models.Model):
    userid = models.CharField(max_length=16)
    history = models.CharField(max_length=64)
    


class UserHistory(models.Model):
    userid = models.CharField(max_length=16)
    history = models.CharField(max_length=64)


