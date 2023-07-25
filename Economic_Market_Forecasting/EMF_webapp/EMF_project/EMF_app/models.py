from django.db import models

from django.db.models.fields import CharField, FloatField

class EMF(models.Model):
    target = models.CharField(max_length=150)

class EMF_data(models.Model):
    date = models.CharField(max_length=150)
    data = models.FloatField()

