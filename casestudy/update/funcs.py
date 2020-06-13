import time
import os
import math
from datetime import datetime as dt, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import requests
from io import BytesIO
import urllib.request
import shutil
from zipfile import ZipFile
import unidecode

from decouple import config
import cdsapi

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from django.db import transaction
from django.db.models import Q

from zooscraper.globals import max_bulk_create, ChromeInstantiator
from casestudy.models import Cases, Deaths, Tests, Region, Measurements, Pollutant, City, \
    Country, Strindex, Cause, Mobility, Travel, AppleMobility

ITALY_URL = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
BRAZIL_URL = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
US_URL = 'https://nssac.bii.virginia.edu/covid-19/dashboard/data/nssac-ncov-data-country-state.zip'
REST_URL = 'https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases'
AUSTESTS_URL = 'https://services1.arcgis.com/vHnIGBHHqDR6y0CR/arcgis/rest/services/COVID19_Time_Series/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json'
CADTESTS_URL = 'https://health-infobase.canada.ca/src/data/covidLive/covid19.csv'
USTESTS_URL = 'https://covidtracking.com/api/v1/states/daily.csv'
RESTTESTS_URL = 'https://covid.ourworldindata.org/data/owid-covid-data.xlsx'
OXCOV_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
GMOBI_URL = 'https://www.google.com/covid19/mobility/'
AMOBI_URL = 'https://www.apple.com/covid19/mobility/'
AQ_URL = 'https://aqicn.org/data-platform/covid19/report/10248-8ad6289d/2020/waqi-covid19-airqualitydata-2020.csv'

BRAZREGIONS = {
    'RO': 'Rondonia', 'AC': 'Acre', 'AM': 'Amazonas', 'RR': 'Roraima', 'TO': 'Tocantins', 'PA': 'Para', 'AP': 'Amapa',
    'MA': 'Maranhao', 'PI': 'Piaui', 'CE': 'Ceara', 'RN': 'Rio Grande Do Norte', 
    'PB': 'Paraiba', 'PE': 'Pernambuco', 'AL': 'Alagoas', 'SE': 'Sergipe', 'BA': 'Bahia',
    'ES': 'Espirito Santo', 'MG': 'Minas Gerais', 'RJ': 'Rio De Janeiro', 'SP': 'Sao Paulo',
    'PR': 'Parana', 'SC': 'Santa Catarina', 'RS': 'Rio Grande Do Sul',
    'GO': 'Goias', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso Do Sul', 'DF': 'Distrito Federal',
}

def get_italy(create=False):
    df = pd.read_csv(ITALY_URL)
    df_casi = df[df.casi_testati.notnull()]
    tamponi_per_testati = df_casi.tamponi.sum() / df_casi.casi_testati.sum()
    df['adj_testati'] = np.where(df.casi_testati.isnull(), df.tamponi / tamponi_per_testati, df.casi_testati)

    cases = []
    deaths = []
    tests = []
    for i, row in df.iterrows():
        date = pd.to_datetime(row['data'])
        region = Region.objects.get(name=row['denominazione_regione'])
        cases.append(Cases(date=date, cases=row['totale_casi'], region=region))
        deaths.append(Deaths(date=date, deaths=row['deceduti'], region=region))
        tests.append(Tests(date=date, tests=row['adj_testati'], region=region))
    
    if create:
        with transaction.atomic():
            Deaths.objects.filter(region__country_key__alpha3='ITA').delete()
            Cases.objects.filter(region__country_key__alpha3='ITA').delete()
            Tests.objects.filter(region__country_key__alpha3='ITA').delete()
            max_bulk_create(cases)
            max_bulk_create(deaths)
            max_bulk_create(tests)

def get_braz(create=False):
    df = pd.read_csv(BRAZIL_URL, parse_dates=['date'])
    df = df[df.state != 'TOTAL']

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
            Tests.objects.filter(region__country_key__alpha3='BRA').delete()
            max_bulk_create(cases)
            max_bulk_create(deaths)
            max_bulk_create(tests)

def get_US(create=False):
    EXCLUDED = ['USA (Aggregate Recovered)', 'Grand Princess', 'Wuhan Evacuee', 'Others (repatriated from Wuhan)',
       'Navajo Nation', 'Unknown location US', 'Federal Bureau of Prisons', 'Veteran Affair',
    ]
    cases = []
    deaths = []

    url = urllib.request.urlopen(US_URL)
    with ZipFile(BytesIO(url.read())) as zf:
        for contained_file in zf.namelist():
            if 'csv' in contained_file:
                df = pd.read_csv(zf.open(contained_file), error_bad_lines=False)
                df = df[df['Region'] == 'USA']
                df = df[~df['name'].isin(EXCLUDED)]
                for i, row in df.iterrows():
                    region_name = row['name'].strip()
                    region, created = Region.objects.get_or_create(name=region_name, country_key=Country.objects.get(alpha3='USA'))

                    if created:
                        print ('new region created: ', region.name)
                    date = pd.to_datetime('-'.join(contained_file.split('.')[0].split('-')[-3:]))
                    cases.append(Cases(region=region, date=date, cases=row['Confirmed']))
                    deaths.append(Deaths(region=region, date=date, deaths=row['Deaths']))
    
    if create:
        with transaction.atomic():
            Deaths.objects.filter(region__country_key__alpha3='USA').delete()
            Cases.objects.filter(region__country_key__alpha3='USA').delete()
            max_bulk_create(cases)
            max_bulk_create(deaths)

def create_rest(create=False):
    covreq = requests.get(REST_URL)
    strs = ['title="time_series_covid19_confirmed_global.csv"', 'title="time_series_covid19_deaths_global.csv"']

    csvs = {}
    for string in strs:
        case_start = covreq.text.find(string)
        case_text = covreq.text[case_start:]
        href_start = case_text.find('href')
        ref_text = case_text[href_start:]
        url_start = ref_text.find('"') + 1
        url_end = ref_text[url_start: ].find('"') + url_start
        csvs[string] = 'https://data.humdata.org/' + ref_text[url_start: url_end]
        
    with transaction.atomic():
        for filestr, csv in csvs.items():
            model = Cases if 'confirmed' in filestr else Deaths

            excluded = ['Diamond Princess', 'Grand Princess', 'Recovered', 'US', 'Italy', 'Brazil', 'MS Zaandam']
            df = pd.read_csv(csv)

            df = df[(~df['Province/State'].isin(excluded)) & (~df['Country/Region'].isin(excluded))]
            counts = []
            for i, row in df.iterrows():
                
                name = row['Country/Region'] if str(row['Province/State']) == 'nan' else row['Province/State']
                country = row['Country/Region']

                if name == 'Taiwan*':
                    name = 'Taiwan'
                    country = 'Taiwan'

                if name == 'Hong Kong':
                    country = 'Hong Kong'
                
                region, _ = Region.objects.get_or_create(name=name, country=country, defaults={
                     'latitude': row['Lat'], 'longitude': row['Long']
                })

                for col in df.columns[4:]:
                    if 'confirmed' in filestr:
                        counts.append(model(region=region, date=pd.to_datetime(col), cases=row[col]))
                    else:
                        counts.append(model(region=region, date=pd.to_datetime(col), deaths=row[col]))

            if create:
                with transaction.atomic():
                    model.objects.exclude(region__country_key__alpha3__in=['ITA', 'BRA', 'USA']).delete()
                    max_bulk_create(counts)

def update_austests(create=False):
    d = requests.get(AUSTESTS_URL).json()

    df = pd.json_normalize(d['features'])
    df.columns = [col.replace('attributes.', '') for col in df.columns]
    df.Date = pd.to_datetime(df.Date, unit='ms')
    cols = ['Date'] + [col for col in df.columns.tolist() if '_Tests' in col and '_Negative' not in col and 'Total' not in col]
    df = df[cols].fillna(0)
    df.columns = ['Date'] + [col.replace('_Tests', '') for col in df.columns[1:].tolist()]

    auskey = {
        'ACT': 'Australian Capital Territory', 
        'NSW': 'New South Wales', 
        'NT': 'Northern Territory', 
        'QLD': 'Queensland', 
        'SA': 'South Australia', 
        'TAS': 'Tasmania', 
        'VIC': 'Victoria', 
        'WA': 'Western Australia'
    }

    test_objs = []
    for i, row in df.iterrows():
        for j, item in row.iteritems():
            if j != 'Date':
                region = Region.objects.get(name=auskey[j])
                test_obj = Tests(**{'date': row.Date, 'region': region, 'tests': item})
                test_objs.append(test_obj)

    if create:
        with transaction.atomic():
            Tests.objects.filter(region__country_key__alpha3='AUS').delete()
            max_bulk_create(test_objs)

def update_cadtests(create=False):
    df = pd.read_csv(CADTESTS_URL)
    df.date = pd.to_datetime(df.date, format='%d-%m-%Y')
    df = df[~df.prname.isin(['Repatriated travellers', 'Canada'])]
    df = df.fillna(0)

    test_objs = []
    for i, row in df.iterrows():
        region = Region.objects.get(name=row.prname)
        test_obj = Tests(**{'date': row.date, 'region': region, 'tests': row.numtested})
        test_objs.append(test_obj)

    if create:
        with transaction.atomic():
            Tests.objects.filter(region__country_key__alpha3='CAN').delete()
            max_bulk_create(test_objs)

def update_ustests(create=False):
    df = pd.read_csv(USTESTS_URL, parse_dates=['date'])
    df = df.fillna(0).iloc[::-1]

    test_objs = []
    for i, row in df.iterrows():
        if row.state != 'AS':
            region = Region.objects.get(code=row.state, country_key__alpha3='USA')
            test_obj = Tests(**{'date': row.date, 'region': region, 'tests': row.totalTestResults})
            test_objs.append(test_obj)

    if create:
        with transaction.atomic():
            Tests.objects.filter(region__country_key__alpha3='USA').delete()
            max_bulk_create(test_objs)
        
def update_resttests(create=False):
    # Web Page location
    # 'https://data.humdata.org/dataset/total-covid-19-tests-performed-by-country'
    
    regions_w_subs_w_tests = ['Australia', 'Canada', 'Italy', 'United States', 'Brazil']
    df = pd.read_excel(RESTTESTS_URL, parse_dates=['date'])
    df = df[~(df.location.isin(['International', 'World'] + regions_w_subs_w_tests))]
    no_tests = [code for code, df_g in df.groupby('iso_code') if df_g.total_tests.isnull().all()]
    df = df[~(df.iso_code.isin(no_tests))]
    replace = {
        'South Korea': 'Korea, South',
        'Myanmar': 'Burma'
    }
    df.location = df.location.replace(replace)

    test_objs = []
    for i, row in df[['iso_code', 'location', 'date', 'total_tests']].iterrows():
        region = Region.objects.get(Q(name=row.location) | Q(name_alt=row.location))
        test_obj = Tests(**{'date': row.date, 'region': region, 'tests': row.total_tests})
        test_objs.append(test_obj)

    if create:
        with transaction.atomic():
            Tests.objects.exclude(region__country_key__alpha3__in=['AUS','CAD','USA', 'BRA', 'ITA']).delete()
            max_bulk_create(test_objs)

def update_strindex(create=False):
    df = pd.read_csv(OXCOV_URL, parse_dates=['Date'], infer_datetime_format=True)

    df = df[['CountryName', 'CountryCode', 'Date', 'C1_School closing',
       'C2_Workplace closing', 'C3_Cancel public events',
       'C4_Restrictions on gatherings', 'C5_Close public transport',
        'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
        'C8_International travel controls', 'E1_Income support',
        'E2_Debt/contract relief', 'E3_Fiscal measures',
        'E4_International support', 'H1_Public information campaigns',
        'H2_Testing policy', 'H3_Contact tracing',
        'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
        'StringencyIndex',
    ]]

    strindex_objs = []
    no_country = []
    with transaction.atomic():
        for i, row in df.iterrows():
            # Trying to remove very large numbers in sum of stringency categories
            factor_kwargs = {i.split('_')[0].lower() : float(factor) if factor < 10 else 0 for i, factor in row.iteritems() if '_' in i}
    
            # Set nans to None for saving database
            factor_kwargs = {k:None if math.isnan(v) else v for k, v in factor_kwargs.items()}
            try:
                country = Country.objects.get(alpha3=row['CountryCode'])
                kwargs = {
                    'country': country,
                    'date': row['Date'],
                    'strindex': float(row['StringencyIndex']),
                    **factor_kwargs,
                }
                strindex_objs.append(Strindex(**kwargs))
            except Country.DoesNotExist:
                no_country.append(row['CountryName'])
    if create:
        with transaction.atomic():
            Strindex.objects.all().delete()
            max_bulk_create(strindex_objs)

def update_gmobi(create=False):
    gmobi_req = requests.get(GMOBI_URL)
    string = 'Terms of Service'
    case_start = gmobi_req.text.find(string)
    case_text = gmobi_req.text[case_start:]
    href_start = case_text.find('href')
    ref_text = case_text[href_start:]
    url_start = ref_text.find('"') + 1
    url_end = ref_text[url_start: ].find('"') + url_start
    gmobi_url = ref_text[url_start: url_end]
    df = pd.read_csv(gmobi_url, parse_dates=['date'], infer_datetime_format=True)

    mobi_objs = []
    for alpha2, df_group in df.groupby('country_region_code'):
        if alpha2 == 'BR':
            regions = [unidecode.unidecode(''.join(reg.split('State of '))) for reg in df_group.sub_region_1.unique() if str(reg) != 'nan']
        else:
            regions = df_group.sub_region_1.unique()

        regions_objs = Region.objects.filter((Q(name__in=regions) | Q(name_alt__in=regions)) & Q(country_key__alpha2=alpha2))                       

        if regions_objs:
            df_group = df_group[~df_group.sub_region_1.isnull()]
            regions = df_group.sub_region_1.unique()

            for region, df_reg in df_group.groupby('sub_region_1'):
                df_reg = df_reg.copy(deep=True)
                df_reg = df_reg[df_reg.sub_region_2.isnull()]
                df_reg = df_reg.fillna(0)
                if alpha2 == 'BR':
                    region = unidecode.unidecode(''.join(region.split('State of ')))

                if region == 'Trentino-South Tyrol':
                    region = 'P.A. Trento'
                try:
                    filt = ((Q(name=region) | Q(name_alt=region) | Q(name_alt__contains=region[:6])) & Q(country_key__alpha2=alpha2))
                    region_obj = Region.objects.get(filt)

                    for i, row in df_reg.iterrows():
                        kwargs = {
                            'region': region_obj,
                            'date': row['date'],
                            'retail_n_rec': row['retail_and_recreation_percent_change_from_baseline'],
                            'groc_n_pharm': row['grocery_and_pharmacy_percent_change_from_baseline'],
                            'parks': row['parks_percent_change_from_baseline'],
                            'transit': row['transit_stations_percent_change_from_baseline'],
                            'workplaces': row['workplaces_percent_change_from_baseline'],
                            'residential': row['residential_percent_change_from_baseline']
                        }
                        mobi_objs.append(Mobility(**kwargs))
                except Region.MultipleObjectsReturned:
                    try: 
                        filt = ((Q(name=region)) & Q(country_key__alpha2=alpha2))
                        region_obj = Region.objects.get(filt)
                    except:
                        filt = ((Q(name=region) | Q(name_alt=region)) & Q(country_key__alpha2=alpha2))
                        region_obj = Region.objects.get(filt)

                    for i, row in df_reg.iterrows():
                        kwargs = {
                            'region': region_obj,
                            'date': row['date'],
                            'retail_n_rec': row['retail_and_recreation_percent_change_from_baseline'],
                            'groc_n_pharm': row['grocery_and_pharmacy_percent_change_from_baseline'],
                            'parks': row['parks_percent_change_from_baseline'],
                            'transit': row['transit_stations_percent_change_from_baseline'],
                            'workplaces': row['workplaces_percent_change_from_baseline'],
                            'residential': row['residential_percent_change_from_baseline']
                        }
                        mobi_objs.append(Mobility(**kwargs))
                except Region.DoesNotExist:
                    print (region)
        else:
            df_group = df_group[df_group.sub_region_1.isnull()]
            df_group = df_group.fillna(0)
            country_name = df_group['country_region'].iloc[0]

            if country_name == 'South Korea':
                country_name = 'Korea, South'
            elif country_name == 'Puerto Rico':
                alpha2 = 'US'
            elif country_name == "Côte d'Ivoire":
                country_name = "Cote d'Ivoire"
                alpha2 = ''

            try:
                region_obj = Region.objects.get((Q(name=country_name) | Q(name_alt=country_name)) & Q(country_key__alpha2=alpha2))
                for i, row in df_group.iterrows():
                    kwargs = {
                        'region': region_obj,
                        'date': row['date'],
                        'retail_n_rec': row['retail_and_recreation_percent_change_from_baseline'],
                        'groc_n_pharm': row['grocery_and_pharmacy_percent_change_from_baseline'],
                        'parks': row['parks_percent_change_from_baseline'],
                        'transit': row['transit_stations_percent_change_from_baseline'],
                        'workplaces': row['workplaces_percent_change_from_baseline'],
                        'residential': row['residential_percent_change_from_baseline']
                    }
                    mobi_objs.append(Mobility(**kwargs))
            except Region.DoesNotExist:
                print (country_name, alpha2)
                pass

    if create:
        with transaction.atomic():
            Mobility.objects.all().delete()
            max_bulk_create(mobi_objs)

def update_amobi(create=False, headless=True):
    with ChromeInstantiator(headless=headless) as chrome:
        chrome.get(AMOBI_URL)
        time.sleep(3)
        chrome.implicitly_wait(10)
        parent = chrome.find_element_by_class_name('download-button-container')
        btn_html = parent.get_attribute('innerHTML')
        ref_html = btn_html[btn_html.find('"') + 1:]
        csv = ref_html[:ref_html.find('"')]    

    df = pd.read_csv(csv)

    df.region = df.region.apply(unidecode.unidecode)
    df.at[df[df.region == 'Sao Paulo'].index, 'geo_type'] = 'sub-region'
    df.at[df[df.region == 'Washington DC'].index, 'geo_type'] = 'sub-region'
    df = df[~(df.geo_type == 'city')]
    df.alternative_name = df.alternative_name.fillna('')
    df['region'] = df['region'].str.replace(' Region', '')
    df['region'] = df['region'].str.replace(' Province', '')
    df['region'] = df['region'].str.replace(' do', ' Do')

    replace = {
        'Autonomous Trentino-Alto Adige/Sudtirol': 'P.A. Trento',
        'Sicily': 'Sicilia',
        'Lombardy': 'Lombardia',
        'Autonomous Friuli-Venezia Giulia': 'Friuli Venezia Giulia',
        'Piedmont': 'Piemonte',
        'Autonomous Aosta Valley': "Valle d'Aosta",
        'Autonomous Sardinia': 'Sardegna',
        'Apulia': 'Puglia',
        'Tuscany': 'Toscana',
        'Yukon Territory': 'Yukon',
        'Rio de Janeiro (state)': 'Rio De Janeiro',
        'Distrito Federal (Brazil)': 'Distrito Federal',
        'Washington DC': 'District of Columbia',
        'Tyrol': 'P.A. Bolzano',
        'Republic of Korea': 'Korea, South',
        'UK': 'United Kingdom',
    }
    df['region'] = df['region'].replace(replace)
    df['alternative_name'] = df['alternative_name'].replace(replace)

    df = df.dropna(subset=df.columns[6:], how='all')
    region_names = list(Region.objects.values_list('name', flat=True))
    df = df[df.region.isin(region_names)]

    df.iloc[:, 6:] = df.iloc[:, 6:].interpolate(method='linear', axis=1).round(1)

    mobi_objs = []
    for region_name, df_g in df.groupby('region'):
        region = Region.objects.get(name=region_name) if not region_name == 'Georgia' else Region.objects.get(name=region_name, country_key__alpha3='USA')
        for col in df.columns[6:]:
            date = dt.strptime(col, '%Y-%m-%d')
            mobi_kwargs = {'region': region, 'date': date}
            for i, row in df_g.loc[:,['transportation_type', col]].iterrows():
                mobi_kwargs[row.transportation_type] = row[col]

            mobi_obj = AppleMobility(**mobi_kwargs)
            mobi_objs.append(mobi_obj)

    if create:
        with transaction.atomic():
            AppleMobility.objects.all().delete()
            max_bulk_create(mobi_objs)

def update_msmts(create=False):

    MSMT_PATH = config('MSMTPATH')
    last_date = Measurements.objects.latest('date').date
    next_date = last_date + timedelta(1) 
    year = next_date.strftime('%Y')
    month = next_date.strftime('%m')
    days = [next_date.strftime('%d')]
    fileappend = month + days[0]+ '-' + month + days[-1] + '.nc'
    uvb_file = 'uvb-' + fileappend
    temp_file = 'temp-' + fileappend
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_dewpoint_temperature', '2m_temperature',
            ],
            'year': year,
            'month': month,
            'day': days,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'format': 'netcdf',
        },
        MSMT_PATH + temp_file
    )
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'downward_uv_radiation_at_the_surface',
            'year': '2020',
            'month': month,
            'day': days,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'format': 'netcdf',
        },
        MSMT_PATH + uvb_file
    )

    try:
        msmt_files = [file for file in os.listdir(MSMT_PATH) if '.nc' in file]
        temp_sets = []
        uvb_sets = []
        for msmt_file in msmt_files:
            dataset = xr.open_dataset(MSMT_PATH + msmt_file)
            if 'expver' in dataset.variables:
                dataset = dataset.sel(expver=5)
                del dataset['expver']
            if 't2m' in dataset.variables:
                temp_sets.append(dataset)
            elif 'uvb' in dataset.variables:
                uvb_sets.append(dataset)

        ds_temp = xr.combine_by_coords(temp_sets)
        ds_new = xr.Dataset({'d2m': ds_temp.d2m.resample(time='D', skipna=True).mean()})
        ds_new['t2m'] = ds_temp.t2m.resample(time='D').mean()

        ds_uvb = xr.combine_by_coords(uvb_sets)
        ds_uvb = ds_uvb.resample(time='D').sum()
        ds_new['uvb'] = ds_uvb.uvb

        msmt_objs = []
        for region in Region.objects.filter(longitude__isnull=False):
            ds_longlat = ds_new.sel(latitude=region.latitude, longitude=region.longitude, method='nearest')

            times = ds_longlat.t2m.time.values
            uvbs = ds_longlat.uvb.values
            t2ms = ds_longlat.t2m.values
            d2ms = ds_longlat.d2m.values

            for values in zip(list(times), uvbs, t2ms, d2ms):
                date_shift = values[0] + np.timedelta64(12, 'h')
                ts = (date_shift - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                date = dt.utcfromtimestamp(ts)
                msmt_objs.append(
                    Measurements(
                        region=region, longitude=region.longitude, 
                        latitude=region.latitude, date=date,
                        uvb=values[1], 
                        temp= values[2], dewpoint= values[3]
                ))

        if create:
            with transaction.atomic():
                max_bulk_create(msmt_objs)

        for msmt_file in msmt_files:
            shutil.move(MSMT_PATH + msmt_file, MSMT_PATH + 'obsolete/' + msmt_file)
    except Exception as e:
        shutil.move(MSMT_PATH + msmt_file, MSMT_PATH + 'failed/' + msmt_file)
        raise e

def update_pollutants(create=False):
    # Download takes some time as it is a big file
    update_date = dt.now() - timedelta(10)

    df = pd.read_csv(AQ_URL, delimiter=',', engine='python', comment='#', error_bad_lines=False)

    df = df[df['City'].isin(City.objects.values_list('name', flat=True))]
    df['date_as_ts'] = pd.to_datetime(df['Date'])
    df = df[df['date_as_ts'] >= update_date]

    pollu_objs = []
    no_city = []
    for i, row in df.iterrows():
        try:
            city = City.objects.get(name=row['City'], region__country_key__alpha2=row['Country'])
            pollu_objs.append(
                Pollutant(
                    date=row['Date'], city=city, pollutant=row['Specie'], count=row['count'],
                    minimum=row['min'], maximum=row['max'], median=row['median'], variance=row['variance'],
                )
            )
        except:
            no_city.append(row['City'])
    
    if create:
        with transaction.atomic():
            Pollutant.objects.filter(date__gte=update_date).delete()
            max_bulk_create(pollu_objs)

update_funcs = [
    get_italy, get_braz, get_US, create_rest, 
    update_austests, update_cadtests, update_ustests, update_resttests,
    update_strindex, update_gmobi, update_amobi, 
    update_msmts, update_pollutants
]