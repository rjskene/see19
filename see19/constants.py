COUNTRIES_W_REGIONS = ['AUS', 'BRA', 'CAN', 'CHN', 'ITA', 'USA']
COVID_DRAGONS = ['Hong Kong', 'Taiwan', 'Korea, South', 'Malaysia']

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

MOBIS = ['retail_n_rec', 'groc_n_pharm', 'parks', 'transit', 'workplaces', 'residential']
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

COUNT_TYPES = ['cases', 'deaths']
COUNT_APPENDS = ['_dma', '_new', '_new_dma']
BASECOUNT_CATS = [count_type + count_append for count_type in COUNT_TYPES for count_append in COUNT_APPENDS] + COUNT_TYPES

PER_APPENDS = ['_per_1M', '_per_person_per_land_KM2', '_per_person_per_city_KM2']
PER_CATS = [count_cat + count_append for count_cat in BASECOUNT_CATS for count_append in PER_APPENDS]
BASE_PLUS_PER_CATS = BASECOUNT_CATS + PER_CATS
LOGNAT_CATS = [cat + '_lognat' for cat in BASE_PLUS_PER_CATS]
ALL_CATS =  BASECOUNT_CATS + PER_CATS + LOGNAT_CATS

META_COLS = ['region_id', 'country_id', 'region_name', 'country_code', 'country', 
    'date', 'cases', 'deaths',
    'population', 'land_KM2', 'land_dens', 'city_KM2', 'city_dens',
]
BASE_COLS = META_COLS + ALL_RANGES
