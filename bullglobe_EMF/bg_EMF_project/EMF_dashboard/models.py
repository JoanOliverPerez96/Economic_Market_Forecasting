from django.db import models
from django.utils.text import slugify
from django.urls import reverse

class EMF(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(blank=True)

    def get_absolute_url(self):
        return reverse('EMF_dashboard:dashboard', kwargs={'slug': self.slug})

    @property
    def data(self):
        try:
            return self.dataitem.all()
        except:
            pass

    def __str__(self):
        return str(self.name)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)



class DataItem(models.Model):
    emf = models.ForeignKey(EMF, on_delete=models.CASCADE)
    value = models.PositiveSmallIntegerField()
    owner = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.owner} - {self.value}"
    
class MLData(models.Model):
    Date = models.DateField()
    SP500 = models.FloatField()
    SP500_prediction = models.FloatField()

    def __str__(self):
        return f"{self.owner} - {self.value}"