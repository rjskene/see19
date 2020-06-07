COUNTRIES_W_REGIONS = ['AUS', 'BRA', 'CAN', 'CHN', 'ITA', 'USA']
COUNTRIES_W_REGIONS_LONG = ['Australia', 'Brazil', 'Canada', 'China', 'Italy', 'United States of America (the)']
COVID_DRAGONS = ['Hong Kong', 'Taiwan', 'Korea, South', 'Malaysia']

AUSREGIONS = {
    'ACT': 'Australian Capital Territory', 
    'NSW': 'New South Wales', 
    'NT': 'Northern Territory', 
    'QLD': 'Queensland', 
    'SA': 'South Australia', 
    'TAS': 'Tasmania', 
    'VIC': 'Victoria', 
    'WA': 'Western Australia'
}
AUSABBRS = {AUSREGIONS[key]: key for key in AUSREGIONS.keys()}

BRAREGIONS = {
    'RO': 'Rondonia', 'AC': 'Acre', 'AM': 'Amazonas', 'RR': 'Roraima', 'TO': 'Tocantins', 'PA': 'Para', 'AP': 'Amapa',
    'MA': 'Maranhao', 'PI': 'Piaui', 'CE': 'Ceara', 'RN': 'Rio Grande Do Norte', 
    'PB': 'Paraiba', 'PE': 'Pernambuco', 'AL': 'Alagoas', 'SE': 'Sergipe', 'BA': 'Bahia',
    'ES': 'Espirito Santo', 'MG': 'Minas Gerais', 'RJ': 'Rio De Janeiro', 'SP': 'Sao Paulo',
    'PR': 'Parana', 'SC': 'Santa Catarina', 'RS': 'Rio Grande Do Sul',
    'GO': 'Goias', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso Do Sul', 'DF': 'Distrito Federal',
}
BRAABBRS = {BRAREGIONS[key]: key for key in BRAREGIONS.keys()}

CANREGIONS = {
    'AB': 'Alberta', 
    'BC': 'British Columbia', 
    'MB': 'Manitoba', 
    'NB': 'New Brunswick',
    'NFLD': 'Newfoundland and Labrador', 
    'NWT': 'Northwest Territories',
    'NS': 'Nova Scotia', 
    'ON': 'Ontario', 
    'PEI': 'Prince Edward Island', 
    'QC': 'Quebec',
    'SASK': 'Saskatchewan', 
    'YT': 'Yukon',
    'NU': 'Nunavut',
}
CANABBRS = {CANREGIONS[key]: key for key in CANREGIONS.keys()}

CHNREGIONS = {
    'AH': 'Anhui', 
    'BJ': 'Beijing', 
    'CQ': 'Chongqing', 
    'FJ': 'Fujian', 
    'GS': 'Gansu', 
    'GD': 'Guangdong',
    'GX': 'Guangxi', 
    'GZ': 'Guizhou', 
    'HI': 'Hainan', 
    'HE': 'Hebei', 
    'HL': 'Heilongjiang', 
    'HA': 'Henan',
    'HB': 'Hubei', 
    'HN': 'Hunan', 
    'NM': 'Inner Mongolia', 
    'JS': 'Jiangsu',
    'JX': 'Jiangxi',
    'JL': 'Jilin',
    'LN': 'Liaoning', 
    'MO': 'Macau', 
    'NX': 'Ningxia', 
    'QH': 'Qinghai', 
    'SN': 'Shaanxi', 
    'SD': 'Shandong',
    'SH': 'Shanghai', 
    'SX': 'Shanxi', 
    'SC': 'Sichuan', 
    'TJ': 'Tianjin', 
    'TX': 'Tibet', 
    'XJ': 'Xinjiang',
    'YN': 'Yunnan', 
    'ZJ': 'Zhejiang'
}
CHNABBRS = {CHNREGIONS[key]: key for key in CHNREGIONS.keys()}

ITAREGIONS = {
    'ABR': 'Abruzzo',
    'BAS': 'Basilicata', 
    'CAL': 'Calabria',
    'CAM': 'Campania',
    'EMI': 'Emilia-Romagna',
    'FRI': 'Friuli Venezia Giulia',
    'LAZ': 'Lazio',
    'LIG': 'Liguria',
    'LOM': 'Lombardia',
    'MAR': 'Marche',
    'MOL': 'Molise',
    'PIE': 'Piemonte',
    'PUG': 'Puglia',
    'SAR': 'Sardegna',
    'SIC': 'Sicilia',
    'TOS': 'Toscana',
    'TRE': 'P.A. Trento',
    'BZ': 'P.A. Bolzano',
    'UMB': 'Umbria',
    'VAL': "Valle d'Aosta",
    'VEN': 'Veneto',
}
ITAABBRS = {ITAREGIONS[key]: key for key in ITAREGIONS.keys()}
SUBABBRS = {
    'AUS': AUSABBRS, 'BRA': BRAABBRS, 'CAN': CANABBRS, 'CHN': CHNABBRS, 'ITA': ITAABBRS,
}

AGE_RANGES = [(i * 5, i * 5 + 4) for i in range(int(85 / 5))]
AGE_COLS = ['A' + str(age_range[0]).zfill(2) + '_' + str(age_range[1]).zfill(2) + 'B' for age_range in AGE_RANGES]

RANGES = {    
    'UNDERS': {
        'ranges': ['A' + str(age).zfill(2) + 'UNDERB' for age in [9, 14, 19, 24, 29, 34]],
        'range_slice': (1, 3),
        'fix_position': 0,
        'fix_direction': 'beg',
    },
    'OVERS': {
        'ranges': ['A' + str(age).zfill(2) + 'PLUSB' for age in [65, 70, 75, 80, 85]],
        'range_slice': (1, 3),
        'fix_position': 0,
        'fix_direction': 'beg',
    },
    'SCHOOL_GOERS': {
        'ranges': ['A05_' + str(age) + 'B' for age in [19, 24, 29, 34]],
        'range_slice': (-3, -1),
        'fix_position': 1,
        'fix_direction': 'beg',
    },
    'Y_MILLS': {
        'ranges': ['A15_' + str(age) + 'B' for age in [24, 29, 34]],
        'range_slice': (-3, -1),
        'fix_position': 3,
        'fix_direction': 'beg',
    },
    'MILLS': {
        'ranges': ['A20_' + str(age) + 'B' for age in [29, 34]],
        'range_slice': (-3, -1),
        'fix_position': 4,
        'fix_direction': 'beg',
    },
    'MID': {
        'ranges': ['A' + str(age) + '_54B' for age in [35, 40, 45]],
        'range_slice': (1, 3),
        'fix_position': -7,
        'fix_direction': 'end',
    },
    'MID_PLUS': {
        'ranges': ['A' + str(age) + '_64B' for age in [35, 40, 45]],
        'range_slice': (1, 3),
        'fix_position': -5,
        'fix_direction': 'end',
    }
}

CUSTOM_RANGES = [custom_range for ranges in RANGES.values() for custom_range in ranges['ranges']]

ALL_RANGES = AGE_COLS + CUSTOM_RANGES

GMOBIS = ['retail_n_rec', 'groc_n_pharm', 'parks', 'transit', 'workplaces', 'residential']
AMOBIS = ['transit_apple', 'driving_apple', 'walking_apple']
CAUSES = [
    'neoplasms', 'blood', 'endo', 'mental', 'nervous', 'circul', 'infectious', 'respir', 
    'digest', 'skin', 'musculo', 'genito', 'childbirth', 'perinatal', 'congenital',
    'other', 'external'
]
MAJOR_CAUSES = ['circul', 'infectious', 'respir', 'endo']
STRINDEX_SUBCATS = [
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
    'e1', 'e2', 'e3', 'e4', 'h1', 'h2', 'h3', 'h4', 'h5'
]
STRINDEX_CATS = STRINDEX_SUBCATS + ['strindex']
CONTAIN_CATS = [cat for cat in STRINDEX_CATS if 'c' in cat]

ECON_CATS = [cat for cat in STRINDEX_CATS if 'e' in cat and cat != 'strindex']
HEALTH_CATS = [cat for cat in STRINDEX_CATS if 'h' in cat]
KEY3_CATS = ['h1', 'h2', 'h3']
POLLUTS = [
    'co', 'pm25', 'o3', 'no2', 'so2',
    'dew', 'humidity', 'pm10', 'pressure', 'temperature',
    'wind gust', 'wind speed', 'wind-gust', 'wind-speed', 'wd',
    'precipitation', 'uvi', 'aqi', 'pol', 'mepaqi', 'pm1'
]
TEMP_MSMTS = ['temp', 'dewpoint']
MSMTS = ['uvb', 'rhum'] + TEMP_MSMTS

ALL_FACTORS = GMOBIS + AMOBIS + CAUSES + STRINDEX_CATS + POLLUTS + MSMTS

COUNT_TYPES = ['cases', 'deaths', 'tests']
COUNT_APPENDS = ['_dma', '_new', '_new_dma']
BASECOUNT_CATS = [count_type + count_append for count_type in COUNT_TYPES for count_append in COUNT_APPENDS] + COUNT_TYPES

PER_APPENDS = ['_per_1M', '_per_person_per_land_KM2', '_per_person_per_city_KM2']
PER_CATS = [count_cat + count_append for count_cat in BASECOUNT_CATS for count_append in PER_APPENDS]
BASE_PLUS_PER_CATS = BASECOUNT_CATS + PER_CATS
LOGNAT_CATS = [cat + '_lognat' for cat in BASE_PLUS_PER_CATS]
ALL_CATS =  BASECOUNT_CATS + PER_CATS + LOGNAT_CATS

META_COLS = ['region_id', 'country_id', 'region_code', 'region_name', 'country_code', 'country', 
    'date', 'cases', 'deaths', 'tests',
    'population', 'land_KM2', 'land_dens', 'city_KM2', 'city_dens',
]
BASE_COLS = META_COLS + ALL_RANGES

MANUAL_CHART_LABELS = {
    '': 'January 2020',
    'population': 'Population',
    'land_dens': 'Density of Land Area',
    'city_dens': 'Population Density of Largest City',
    'uvb': 'UV-B Radiation in J / M\u00b2',
    'rhum': 'Relative Humidity',
    'strindex': 'Oxford Stringency Index',
    'visitors': 'Annual Visitors',
    'visitors_%': 'Annual Visitors as % of Population',
    'gdp': 'Gross Domestic Product',
    'gdp_%': 'Gross Domestic Product per Capita',
    'retail_n_rec': 'Change in Retail n Recreation Mobility',
    'transit': 'Change in Transit Mobility',
    'workplaces': 'Change in WorkPlace Mobility',
    'residential': 'Change in Residential Mobility',
    'parks': 'Change in Parks Mobility',
    'groc_n_pharm': 'Change in Grocery & Pharmacy Mobility',
    'transit_apple': 'Change in Transit Mobility - Apple',
    'driving_apple': 'Change in Driving Mobility - Apple',
    'walking_apple': 'Change in Walking Mobility - Apple',
    'c1': 'School Closing', 'c2': 'Workplace Closing', 'c3': 'Cancel Public Events',
    'c4': 'Restrictions on Gatherings', 'c5': 'Close Public Transport', 'c6': 'Stay-at-Home Requirements',
    'c7': 'Restrictions on Internal Movement', 'c8': 'International Travel Controls',
    'e1': 'Income Support', 'e2': 'Debt / Contract Relief', 'e3': 'Fiscal Measures',
    'e4': 'International Support', 'h1': 'Public Information Campaigns',
    'h2': 'Testing Policy', 'h3': 'Contact Tracing',
    'h4': 'Emergency Investment in Health Care', 'h5': 'Investment in Vaccines',
    'key3_sum': 'Sum of the Key 3 Indicators',
    'key3_sum_earlier': 'Sum of Key 3 Oxford Stingency Factor Weighted to Earlier Dates',
    'make_sum': 'Custom Stringency Aggregate',
    'neoplasms': 'NeoPlasms Fatalities',
    'blood': 'Blood-based Fatalities',
    'endo': 'Endocrine Fatalities',
    'mental': 'Mental Fatalities',
    'nervous': 'Nervous System Fatalities',
    'circul': 'Circulatory Fatalities',
    'infectious': 'Infectious Fatalities',
    'respir': 'Respiratory Fatalities',
    'digest': 'Digestive Fatalities',
    'skin': 'Skin-related Fatalities',
    'musculo': 'Musculo-skeletal Fatalities',
    'genito': 'Genitourinary Fatalities',
    'childbirth': 'Maternal and Childbirth Fatalities',
    'perinatal': 'Perinatal Fatalities',
    'congenital': 'Congenital Fatalities',
    'other': 'Other Fatalities',
    'external': 'External Fatalities'
}