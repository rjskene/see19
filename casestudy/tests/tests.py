import pandas as pd

from django.test import TestCase

from casestudy.models import Deaths, Cases, Tests, Region
from casestudy.update import BRAZREGIONS

class BrazilTestCase(TestCase):
    fixtures = ['casestudy']

    def test_animals_can_speak(self):
        """Brazil csv saves properly to database"""
        create = True
        brazil_url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
        df = pd.read_csv(brazil_url, parse_dates=['date'])

        cases = []
        deaths = []
        tests = []
        for i, row in df.iterrows():
            region = Region.objects.get(name=BRAZREGIONS[row.state])
            cases.append(Cases(date=row.date, cases=row.totalCases, region=region))
            deaths.append(Deaths(date=row.date, deaths=row.deaths, region=region))
            tests.append(Tests(date=row.date, tests=row.tests, region=region))
        
        if create:
            with transaction.atomic():
                Deaths.objects.filter(region__country_key__alpha3='BRA').delete()
                Cases.objects.filter(region__country_key__alpha3='BRA').delete()
                max_bulk_create(cases)
                max_bulk_create(deaths)