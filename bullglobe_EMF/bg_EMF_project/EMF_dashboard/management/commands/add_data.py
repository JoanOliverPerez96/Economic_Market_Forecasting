from django.core.management.base import BaseCommand
import pandas as pd
from EMF_dashboard.models import MLData
from sqlalchemy import create_engine
from django.conf import settings

class Command(BaseCommand):
    help = 'Add data from Excel to DB'

    def handle(self, *args, **options):
        excel_file = "model_data_SP500.csv"
        df = pd.read_csv(excel_file)
        # engine = create_engine('sqlite:///db.sqlite3')
        
        user = settings.DATABASES['default']['USER']
        password = settings.DATABASES['default']['PASSWORD']
        host = settings.DATABASES['default']['HOST']
        port = settings.DATABASES['default']['PORT']
        database_name = settings.DATABASES['default']['NAME']

        database_url = 'postgresql://{user}:{password}@{host}:{port}/{database_name}'.format(
            user=user,
            password=password,
            host=host,
            port=port,
            database_name=database_name,
        )
        print(database_url)
        engine = create_engine(database_url, echo=False)

        df.to_sql(MLData._meta.db_table, con=engine, if_exists='replace', index=False)