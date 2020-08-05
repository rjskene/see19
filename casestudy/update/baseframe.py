import math
from datetime import datetime as dt
import numpy as np
import pandas as pd
import collections

import gc
from decouple import config

from django.db.models import F

from casestudy.see19.see19.constants import ALL_RANGES, BASE_COLS, COUNT_TYPES, STRINDEX_SUBCATS, STRINDEX_CATS
from casestudy.models import Region, Country, Cases, Deaths, Tests, Hospitalizations, Measurements, Pollutant, Strindex, \
    Mobility, Cause, Travel, GDP, AppleMobility

def calc_rhum(temp, dewpoint):
    """
    Returns relative humidity derived from temperature and dewpoint temperature
    Formula found here: https://bmcnoldy.rsmas.miami.edu/Humidity.html
    """
    a = 17.625
    b = 243.04

    top = math.exp(a * dewpoint / (b + dewpoint)) 
    bottom = math.exp(a * temp / (b + temp))

    return 100 * (top / bottom)

def merge_or_return(func):
    def wrapper(self, **kwargs):
        df, merge_kwargs = func(self)

        kwargs['merge'] = True if 'merge' not in kwargs else kwargs['merge']
        if kwargs['merge']:
            if 'init_deaths' in str(func) or 'add_causes' in str(func):
                self.bf = df
            else:
                country = 'country' in merge_kwargs and merge_kwargs['country']
                regcol = 'country_id' if country  else 'region_id'
                on = ['date', regcol] if 'date' in df.columns else [regcol]
                self.bf = pd.merge(self.bf, df, how=merge_kwargs['how'], on=on).sort_values(by=['region_id', 'date'])

            return self
        else:
            return df

    return wrapper

def allmethods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            callable = isinstance(getattr(cls, attr), collections.Callable)
            if callable and attr in cls.mergefuncs:
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

@allmethods(merge_or_return)
class BaseMaker:
    basefuncs = [
        'init_deaths', 'add_cases', 'add_tests',
        'refresh_n_set_dates', 'add_polluts', 
        'backfill', 'add_msmts', 'add_strindex', 
        'add_gmobi', 'add_amobi', 'add_causes', 
        'add_travel', 'add_gdp'
    ]
    mergefuncs = [func for func in basefuncs if func not in ['refresh_n_set_dates', 'backfill']]

    def __init__(self, test=False, save=False):
        self.test = test
        self.save = save
        self.regions = Region.objects.filter(population__gt=0)    

        if config('HEROKU', cast=bool):
            self.SEE19PATH = config('ROOTPATH') + 'see19repo/'
        else:
            self.SEE19PATH = config('ROOTPATH') + 'casestudy/see19/'

    def init_deaths(self):
        deaths = Deaths.objects.filter(region__in=self.regions)

        age_annots = {age_col: F('region__' + age_col) for age_col in ALL_RANGES}
        deaths = deaths.annotate(
                region_name=F('region__name'), region_code=F('region__code'), 
                country=F('region__country_key__name'), 
                country_id=F('region__country_key__id'), country_code=F('region__country_key__alpha3'),
                population=F('region__population'), land_dens=F('region__land_density'),
                city_dens=F('region__city_dens'), **age_annots
            ) \
            .values(*[col for col in BASE_COLS if col not in ['cases', 'tests', 'land_KM2', 'city_KM2']])

        df = pd.DataFrame(deaths)
        df.country_id = df.country_id.astype('int64')
        df['date'] = df['date'].dt.normalize()
        df['land_KM2'] = df['population'] / df['land_dens']
        df['city_KM2'] = df['population'] / df['city_dens']

        return df, {}

    def add_cases(self):
        print ('adding cases')
        # Make Cases DF and Merge
        cases = Cases.objects.filter(region__in=self.regions).values('region_id', 'date', 'cases')
        df = pd.DataFrame(cases)
        df['date'] = df['date'].dt.normalize()

        return df, {'how': 'outer'}
    
    def add_tests(self):
        # Make Tests DF and Merge
        print ('adding tests')
        tests = Tests.objects.filter(region__in=self.regions).values('region_id', 'date', 'tests')
        df = pd.DataFrame(tests)
        df.date = df.date.dt.normalize()

        return df, {'how': 'inner'}

    def refresh_n_set_dates(self):
        # RE-ORDER COLUMNS
        self.bf = self.bf[BASE_COLS]
        
        # Update regions for those in the baseframe
        self.regions = Region.objects.filter(id__in=self.bf.region_id.unique())
        self.countries = Country.objects.filter(id__in=self.bf.country_id.unique())
    
        # Used to limit dates added to baseframe; this fixes issues with backfill crossing-up regions
        self.maxdate = self.bf.date.max()
        self.reg_maxdates = self.bf.groupby('region_id').date.max().to_dict()
        self.coun_maxdates = self.bf.groupby('country_id').date.max().to_dict()
    
    def add_polluts(self):
        # Make Polluts DF
        print ('adding pollutants')
        polluts = Pollutant.objects.filter(
            city__region__in=self.regions,
            date__lte=self.maxdate) \
            .values(
                'date', 'city__name', 
                'city__region__id', 'city__region__name', 
                'pollutant', 'median'
        )
        df = pd.DataFrame(polluts)
        df = df \
            .set_index(['city__region__id', 'pollutant', 'date']) \
            .sort_values(by=['city__region__id', 'pollutant', 'date'])
        
        dfs_polluts = []
        for region_id, df_group in df.groupby('city__region__id'):
            df_group = df_group.groupby(['pollutant', 'date'])['median'].mean().reset_index(level=['pollutant']).copy()
            df_group = df_group.reset_index().pivot('date', 'pollutant', 'median').reset_index()
            df_group['region_id'] = region_id
            dfs_polluts.append(df_group)

        df = pd.concat(dfs_polluts)
        dfs_polluts = [df_group[df_group.date <= self.reg_maxdates[region_id]] for region_id, df_group in df.groupby('region_id')]
        df = pd.concat(dfs_polluts)

        return df, {'how': 'outer'}

    def backfill(self):
        # Backfill time-static info
        print ('backfill time-static data')
        fill_cols = [col for col in BASE_COLS if col not in COUNT_TYPES]
        self.bf[fill_cols] = self.bf[fill_cols].bfill(axis='rows')

    def add_msmts(self):
        print ('adding measurements')
        msmts = Measurements.objects \
            .filter(region__in=self.regions, date__lte=self.maxdate) \
            .values('region_id', 'date', 'temp', 'dewpoint', 'uvb')
        
        df = pd.DataFrame(msmts)
        df['date'] = df['date'].dt.normalize()

        # Relative Humidity is calculated from temperature and dewpoint temperature 
        df['rhum'] = np.vectorize(calc_rhum)(df['temp'], df['dewpoint'])

        dfs_msmts = [df_group[df_group.date <= self.reg_maxdates[region_id]] for region_id, df_group in df.groupby('region_id')]
        df = pd.concat(dfs_msmts)

        return df, {'how': 'outer'}

    def add_strindex(self):
        # Merge in Oxford stringency index
        print ('adding strindex')
        strindex_fields = [f.name for f in Strindex._meta.get_fields() if f.name != 'id']
        strin_objs = Strindex.objects \
            .filter(country__in=self.countries, date__lte=self.maxdate) \
            .values(*strindex_fields).annotate(country_id=F('country')
        )
        df = pd.DataFrame(strin_objs).drop(columns='country')

        # Sum of the strindex columns have strange very large numbers within them
        # correct for these
        cols_w_big_nums = [col for col, val in (df[STRINDEX_SUBCATS] >= 10).any().iteritems() if val]
        for col in cols_w_big_nums:
            df[col] = np.where(df[col] < 10, df[col], 0)

        dfs_strindex = [df_group[df_group.date <= self.coun_maxdates[country_id]] for country_id, df_group in df.groupby('country_id')]
        df = pd.concat(dfs_strindex)

        return df, {'how': 'inner', 'country': True}
    
    def add_gmobi(self):
        # Merge in Google Mobility Index
        print ('adding Google Mobility')
        gmobi_fields = [f.name for f in Mobility._meta.get_fields() if f.name != 'id']
        gmobi_objs = Mobility.objects \
            .filter(region__in=self.regions, date__lte=self.maxdate) \
            .values(*gmobi_fields) \
            .annotate(region_id=F('region')
        )
        df = pd.DataFrame(gmobi_objs).drop(columns='region')
        dfs_gmobi = [df_group[df_group.date <= self.reg_maxdates[region_id]] for region_id, df_group in df.groupby('region_id')]
        df = pd.concat(dfs_gmobi)

        return df, {'how': 'left'}

    def add_amobi(self):
        # Merge in in Apple Mobility Index
        print ('adding Apple Mobility')
        amobi_fields = [f.name for f in AppleMobility._meta.get_fields() if f.name != 'id']
        amobis = AppleMobility.objects \
            .filter(region__in=self.regions, date__lte=self.maxdate) \
            .values(*amobi_fields) \
            .annotate(
                region_id=F('region'), transit_apple=F('transit'), 
                walking_apple=F('walking'), driving_apple=F('driving')
        )
        df = pd.DataFrame(amobis).drop(columns=['region', 'transit', 'walking', 'driving'])

        dfs_amobi = [df_group[df_group.date <= self.reg_maxdates[region_id]] for region_id, df_group in df.groupby('region_id')]
        df = pd.concat(dfs_amobi)

        return df, {'how': 'left'}

    def add_causes(self):
        # Merge in Cause of Death factors
        print ('adding Causes of Death')
        cause_fields = [f.name for f in Cause._meta.get_fields() if f.name != 'id']
        cause_objs = Cause.objects \
            .values(*cause_fields) \
            .annotate(region_id=F('region'), country_id=F('country')
        )
        df_cause = pd.DataFrame(cause_objs).drop(columns=['region', 'country'])

        # Split into regional level causes (USA and ITA only) or country level then concat
        df_cause_reg = df_cause[df_cause.region_id.notnull()].drop(columns='country_id').copy(deep=True)
        df_cause_reg.region_id = df_cause_reg.region_id.astype('int64')
        df_cause_reg = df_cause_reg.fillna(0)
        df_cause_reg = df_cause_reg.set_index('region_id')

        df_base_reg = self.bf[self.bf.region_id.isin(df_cause_reg.index)].copy(deep=True).sort_values(by=['region_id', 'country_id'])
        df_base_reg = pd.merge(df_base_reg, df_cause_reg, how='inner', on=['region_id'])

        df_cause_coun = df_cause[df_cause.country_id.notnull()].drop(columns='region_id').copy(deep=True)
        df_cause_coun.country_id = df_cause_coun.country_id.astype('int64')
        df_cause_coun = df_cause_coun.fillna(0)
        df_cause_coun = df_cause_coun.set_index('country_id')
        df_base_coun = self.bf[self.bf.country_id.isin(df_cause_coun.index)].copy(deep=True).sort_values(by=['region_id', 'country_id'])
        df_base_coun = pd.merge(df_base_coun, df_cause_coun, how='left', on=['country_id'])

        return pd.concat([df_base_reg, df_base_coun]).sort_values(by=['region_id', 'date']), {}
 
    def add_travel(self):
        # Merge in travel popularity
        print ('adding Travel popularity')
        travel_objs = Travel.objects \
            .filter(region__in=self.regions) \
            .annotate(travel_year=F('year')) \
            .values('region_id', 'travel_year', 'visitors')
        
        return pd.DataFrame(travel_objs), {'how': 'left'}

    def add_gdp(self):
        # Merge in GDP data
        print ('adding GDP')
        gdp_objs = GDP.objects \
            .filter(region__in=self.regions) \
            .annotate(gdp_year=F('year')) \
            .values('region_id', 'gdp_year', 'gdp')
        
        return pd.DataFrame(gdp_objs), {'how': 'left'}
          
    def add_hospis(self):
        # Make Hospitalizations DF and Merge
        print ('adding hospitalizations')
        hospi = Hospitalizations.objects. \
            filter(region__in=self.regions) \
            .values('region_id', 'date', 'hospitalizations')
        d_hospi = pd.DataFrame(hospi)
        df.date = df.date.dt.normalize()

        return df, {'how': 'left'}
        
    def save_frame(self, inside_test=False):
        if inside_test:
            print ('saving dataset...')
        else:
            print ('saving testset')

        file_date = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

        datafile = self.SEE19PATH + 'dataset/see19-{}.csv'.format(file_date)
        testfile = self.SEE19PATH + 'testset/see19-TEST-{}.csv'.format(file_date)
        filename = testfile if inside_test else datafile
        
        self.bf.to_csv(filename, index=False)

        txtfile = self.SEE19PATH + 'latest_testset.txt' if inside_test else self.SEE19PATH + 'latest_dataset.txt' 
        
        with open(txtfile, 'w') as filetowrite:
            filetowrite.write(file_date)

    def make(self, save=None, test=None):
        save = save if save is not None else self.save
        test = test if test is not None else self.test

        # Call each function dynamically
        # This allows for testing at each stage
        for func in self.basefuncs:
            getattr(self, func)()

        self.bf.country_id = self.bf.country_id.astype('int64')
        self.bf.date = pd.to_datetime(self.bf.date).dt.tz_localize(None)
    
        if save:
            self.save_frame()

        if test:
            print ('adding test items')
            self.tf = self.add_hospis()

            if save:
                self.save_frame(inside_test=True)
        print ('COMPLETE')

class Tester(BaseMaker):

    def make(self, testfunc, basefuncs=[]):
        basefuncs = basefuncs if basefuncs else self.basefuncs
        for func in basefuncs:
            getattr(self, func)()
            testfunc(self.bf)