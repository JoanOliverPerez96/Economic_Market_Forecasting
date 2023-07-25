import csv
from datetime import date
from django.conf import settings
from django.core.management.base import BaseCommand
from EMF_app.models import *
import pandas as pd

class Command(BaseCommand):
    help = 'Load data from csv file'

    def add_arguments(self, parser):
        parser.add_argument('file', nargs='+', type=str)

    def handle(self, *args, **kwargs):
        file = settings.BASE_DIR / '/EMF_webapp/EMF_project/EMF_app/data/' / 'SP500_data.csv'

        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                dt = pd.to_datetime(row['Date'])

                EMF.objects.get_or_create(
                    date=dt,
                    target=row['SP500'],
                )
