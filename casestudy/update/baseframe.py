import math
from datetime import datetime as dt
import numpy as np
import pandas as pd

import gc
from decouple import config

from django.db.models import F

from casestudy.see19.see19.constants import ALL_RANGES, BASE_COLS, COUNT_TYPES, STRINDEX_SUBCATS, STRINDEX_CATS
from casestudy.models import Region, Country, Cases, Deaths, Tests, Measurements, Pollutant, Strindex, \
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

def make(save=False, test=False):
    if config('HEROKU', cast=bool):
        SEE19PATH = config('ROOTPATH') + 'see19repo/'
    else:
        SEE19PATH = config('ROOTPATH') + 'casestudy/see19/'

    print ('initializing')
    print ('adding fatalities')
    # Make Deaths DF
    regions = Region.objects.filter(population__gt=0)
    deaths = Deaths.objects.filter(region__in=regions)
    age_annots = {age_col: F('region__' + age_col) for age_col in ALL_RANGES}

    deaths = deaths.annotate(
            region_name=F('region__name'), region_code=F('region__code'), 
            country=F('region__country_key__name'), 
            country_id=F('region__country_key__id'), country_code=F('region__country_key__alpha3'),
            population=F('region__population'), land_dens=F('region__land_density'),
            city_dens=F('region__city_dens'), **age_annots
        ) \
        .values(*[col for col in BASE_COLS if col not in ['cases', 'tests', 'land_KM2', 'city_KM2']])

    df_deaths = pd.DataFrame(deaths)
    df_deaths.country_id = df_deaths.country_id.astype('int64')
    df_deaths['date'] = df_deaths['date'].dt.normalize()
    df_deaths['land_KM2'] = df_deaths['population'] / df_deaths['land_dens']
    df_deaths['city_KM2'] = df_deaths['population'] / df_deaths['city_dens']

    # Make Cases DF and Merge
    print ('adding cases')
    cases = Cases.objects.filter(region__in=regions).values('region_id', 'date', 'cases')
    df_cases = pd.DataFrame(cases)
    df_cases['date'] = df_cases['date'].dt.normalize()
    df_base = pd.merge(df_deaths, df_cases, how='outer', on=['date', 'region_id']).sort_values(by=['region_id', 'date'])

    # Make Tests DF and Merge
    print ('adding tests')
    tests = Tests.objects.filter(region__in=regions).values('region_id', 'date', 'tests')
    df_tests = pd.DataFrame(tests)
    # df_tests.date = pd.to_datetime(df_tests.date).dt.tz_localize(None)
    df_tests.date = df_tests.date.dt.normalize()
    df_base = pd.merge(df_base, df_tests, how='left', on=['date', 'region_id']).sort_values(by=['region_id', 'date'])

    # RE-ORDER COLUMNS
    df_base = df_base[BASE_COLS]

    # Update regions for those in the baseframe
    regions = Region.objects.filter(id__in=df_base.region_id.unique())
    countries = Country.objects.filter(id__in=df_base.country_id.unique())

    # Used to limit dates added to baseframe; this fixes issues with backfill crossing-up regions
    maxdate = df_base.date.max()
    reg_maxdates = {region_id: df_group.date.max() for region_id, df_group in df_base.groupby('region_id')}
    coun_maxdates = {country_id: df_group.date.max() for country_id, df_group in df_base.groupby('country_id')}

    # Pause of for Garbage Collection
    del df_deaths
    del df_cases
    del df_tests
    gc.collect()
    df_deaths=df_cases=df_tests=pd.DataFrame()

    # Make Polluts DF
    print ('adding pollutants')
    polluts = Pollutant.objects.filter(city__region__in=regions, date__lte=maxdate).values('date', 'city__name', 'city__region__id', 'city__region__name', 'pollutant', 'median')
    df_polluts = pd.DataFrame(polluts)
    df_polluts = df_polluts.set_index(['city__region__id', 'pollutant', 'date']) \
        .sort_values(by=['city__region__id', 'pollutant', 'date'])

    dfs_polluts = []
    for region_id, df_group in df_polluts.groupby('city__region__id'):
        df_group = df_group.groupby(['pollutant', 'date'])['median'].mean().reset_index(level=['pollutant']).copy()
        df_group = df_group.reset_index().pivot('date', 'pollutant', 'median').reset_index()
        df_group['region_id'] = region_id
        dfs_polluts.append(df_group)
    df_polluts = pd.concat(dfs_polluts)
    
    dfs_polluts = [df_group[df_group.date <= reg_maxdates[region_id]] for region_id, df_group in df_polluts.groupby('region_id')]
    df_polluts = pd.concat(dfs_polluts)

    # Merge Pollutants
    df_base = pd.merge(df_base, df_polluts, how='outer', on=['date', 'region_id']).sort_values(by=['region_id', 'date'])

    del df_polluts
    del dfs_polluts
    gc.collect()
    df_polluts=pd.DataFrame()
    dfs_polluts = []

    # GET MSMTS
    print ('adding measurements')
    msmts = Measurements.objects.filter(region__in=regions, date__lte=maxdate).values('region_id', 'date', 'temp', 'dewpoint', 'uvb')
    df_msmts = pd.DataFrame(msmts)
    df_msmts['date'] = df_msmts['date'].dt.normalize()

    # Relative Humidity is calculated from temperature and dewpoint temperature 
    df_msmts['rhum'] = np.vectorize(calc_rhum)(df_msmts['temp'], df_msmts['dewpoint'])

    dfs_msmts = [df_group[df_group.date <= reg_maxdates[region_id]] for region_id, df_group in df_msmts.groupby('region_id')]
    df_msmts = pd.concat(dfs_msmts)

    # Merge Deaths and Msmts
    df_base = pd.merge(df_base, df_msmts, how='outer', on=['date', 'region_id']).sort_values(by=['region_id', 'date'])
    
    del df_msmts
    gc.collect()
    df_msmts=pd.DataFrame()
    
    # Backfill time-static info
    print ('backfill time-static data')
    fill_cols = [col for col in BASE_COLS if col not in COUNT_TYPES]
    df_base[fill_cols] = df_base[fill_cols].bfill(axis='rows')

    # Merge in Oxford stringency index
    print ('adding strindex')
    strindex_fields = [f.name for f in Strindex._meta.get_fields() if f.name != 'id']
    df_strindex = pd.DataFrame(Strindex.objects.filter(country__in=countries, date__lte=maxdate).values(*strindex_fields).annotate(country_id=F('country'))).drop(columns='country')

    # Sum of the strindex columns have strange very large numbers within them
    # correct for these
    cols_w_big_nums = [col for col, val in (df_strindex[STRINDEX_SUBCATS] >= 10).any().iteritems() if val]
    for col in cols_w_big_nums:
        df_strindex[col] = np.where(df_strindex[col] < 10, df_strindex[col], 0)

    dfs_strindex = [df_group[df_group.date <= coun_maxdates[country_id]] for country_id, df_group in df_strindex.groupby('country_id')]
    df_strindex = pd.concat(dfs_strindex)

    df_base = pd.merge(df_base, df_strindex, how='inner', on=['date', 'country_id']).sort_values(by=['region_id', 'date'])

    del df_strindex
    gc.collect()
    df_strindex=pd.DataFrame()

    # Merge in Google Mobility Index
    print ('adding Google Mobility')
    gmobi_fields = [f.name for f in Mobility._meta.get_fields() if f.name != 'id']
    df_gmobi = pd.DataFrame(Mobility.objects.filter(region__in=regions, date__lte=maxdate).values(*gmobi_fields).annotate(region_id=F('region'))).drop(columns='region')
    dfs_gmobi = [df_group[df_group.date <= reg_maxdates[region_id]] for region_id, df_group in df_gmobi.groupby('region_id')]
    df_gmobi = pd.concat(dfs_gmobi)

    df_base = pd.merge(df_base, df_gmobi, how='left', on=['date', 'region_id']).sort_values(by=['region_id', 'date'])

    del df_gmobi
    gc.collect()
    df_gmobi=pd.DataFrame()

    # Merge in in Apple Mobility Index
    print ('adding Apple Mobility')
    amobi_fields = [f.name for f in AppleMobility._meta.get_fields() if f.name != 'id']
    amobis = AppleMobility.objects.filter(region__in=regions, date__lte=maxdate) \
        .values(*amobi_fields) \
        .annotate(region_id=F('region'), transit_apple=F('transit'), walking_apple=F('walking'), driving_apple=F('driving'))
    df_amobi = pd.DataFrame(amobis).drop(columns=['region', 'transit', 'walking', 'driving'])

    dfs_amobi = [df_group[df_group.date <= reg_maxdates[region_id]] for region_id, df_group in df_amobi.groupby('region_id')]
    df_amobi = pd.concat(dfs_amobi)

    df_base = pd.merge(df_base, df_amobi, how='left', on=['date', 'region_id']).sort_values(by=['region_id', 'date'])

    del df_amobi
    gc.collect()
    df_amobi=pd.DataFrame()

    # Merge in Cause of Death factors
    print ('adding Causes of Death')
    cause_fields = [f.name for f in Cause._meta.get_fields() if f.name != 'id']
    df_cause = pd.DataFrame(Cause.objects.values(*cause_fields).annotate(region_id=F('region'), country_id=F('country'))).drop(columns=['region', 'country'])

    # Split into regional level causes (USA and ITA only) or country level then concat
    df_cause_reg = df_cause[df_cause.region_id.notnull()].drop(columns='country_id').copy(deep=True)
    df_cause_reg.region_id = df_cause_reg.region_id.astype('int64')
    df_cause_reg = df_cause_reg.fillna(0)
    df_cause_reg = df_cause_reg.set_index('region_id')

    df_base_reg = df_base[df_base.region_id.isin(df_cause_reg.index)].copy(deep=True).sort_values(by=['region_id', 'country_id'])
    df_base_reg = pd.merge(df_base_reg, df_cause_reg, how='inner', on=['region_id'])

    df_cause_coun = df_cause[df_cause.country_id.notnull()].drop(columns='region_id').copy(deep=True)
    df_cause_coun.country_id = df_cause_coun.country_id.astype('int64')
    df_cause_coun = df_cause_coun.fillna(0)
    df_cause_coun = df_cause_coun.set_index('country_id')
    df_base_coun = df_base[df_base.country_id.isin(df_cause_coun.index)].copy(deep=True).sort_values(by=['region_id', 'country_id'])
    df_base_coun = pd.merge(df_base_coun, df_cause_coun, how='left', on=['country_id'])

    df_base = pd.concat([df_base_reg, df_base_coun]).sort_values(by=['region_id', 'date'])
    
    del [[df_base_reg, df_base_coun]]
    gc.collect()
    df_base_reg=pd.DataFrame()
    df_base_coun=pd.DataFrame()

    # Merge in travel popularity
    print ('adding Travel popularity')
    df_travel = pd.DataFrame(Travel.objects.filter(region__in=regions).annotate(travel_year=F('year')).values('region_id', 'travel_year', 'visitors'))
    df_base = pd.merge(df_base, df_travel, how='left', on=['region_id']).sort_values(by=['region_id', 'date'])

    del df_travel
    gc.collect()
    df_travel=pd.DataFrame()

    # Merge in GDP data
    print ('adding GDP')
    df_gdp = pd.DataFrame(GDP.objects.filter(region__in=regions).annotate(gdp_year=F('year')).values('region_id', 'gdp_year', 'gdp'))
    df_base = pd.merge(df_base, df_gdp, how='left', on=['region_id']).sort_values(by=['region_name', 'date'])
    df_base.country_id = df_base.country_id.astype('int64')

    df_base.date = pd.to_datetime(df_base.date).dt.tz_localize(None)
    
    del df_gdp
    gc.collect()
    df_gdp=pd.DataFrame()

    if save:
        print ('saving...')
        file_date = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = SEE19PATH + 'dataset/see19-{}.csv'.format(file_date)
        df_base.to_csv(filename, index=False)

        filename = SEE19PATH + 'latest_dataset.txt' 
        with open(filename, 'w') as filetowrite:
            filetowrite.write(file_date)

    if test and False:
        print ('adding test items')
        # There Are Currently No Test Items; This section automatically set to falses
        if save:
            print ('saving testset')
            file_date = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = SEE19PATH + 'testset/see19-TEST-{}.csv'.format(file_date)
            df_base.to_csv(filename, index=False)

            filename = SEE19PATH + 'latest_testset.txt' 
            with open(filename, 'w') as filetowrite:
                filetowrite.write(file_date)

    print ('COMPLETE')
    return df_base
