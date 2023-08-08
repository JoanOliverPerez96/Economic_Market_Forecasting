import requests
import pandas as pd

headers = {'User-Agent': 'joanoliverperez96@gmail.com'}

companyTickers = requests.get(
    'https://www.sec.gov/files/company_tickers.json', headers=headers
)