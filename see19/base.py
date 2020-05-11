import requests
import numpy as np
import pandas as pd

from .helpers import accept_string_or_list
from .charts import CompChart2D, CompChart4D, HeatMap, BarCharts, ScatterFlow
from .constants import ALL_RANGES, RANGES, MOBIS, CAUSES, MAJOR_CAUSES, \
        STRINDEX_SUBCATS, STRINDEX_CATS, CONTAIN_CATS, ECON_CATS, HEALTH_CATS, POLLUTS, TEMP_MSMTS, MSMTS, \
        COUNTRIES_W_REGIONS, COUNT_TYPES, BASE_COLS, PER_APPENDS, \
        BASECOUNT_CATS, PER_CATS, BASE_PLUS_PER_CATS, LOGNAT_CATS, ALL_CATS, KEY3_CATS

CASE_COLS = [col for col in BASE_COLS if col not in ALL_RANGES]

def get_baseframe():
    url = 'https://raw.githubusercontent.com/ryanskene/see19/master/latest_dataset.txt'
    page = requests.get(url)
    df_url = 'https://raw.githubusercontent.com/ryanskene/see19/master/dataset/see19-{}.csv'.format(page.text)
    
    return pd.read_csv(df_url, parse_dates=['date'])

class CaseStudy:
    """
    Class for filtering the baseframe dataset, analysing, and generating graphs
    
    #### TO DO #####
        1: Make it so factors can take keywords msmts, age_ranges, etc to get multiple factors easily
    """
    COUNT_TYPES = COUNT_TYPES    
    BASECOUNT_CATS = BASECOUNT_CATS
    BASE_PLUS_PER_CATS = BASE_PLUS_PER_CATS
    LOGNAT_CATS = LOGNAT_CATS
    ALL_COUNT_CATS = ALL_CATS
    PER_COUNT_CATS = [cat for cat in ALL_CATS if 'per' in cat]
    DMA_COUNT_CATS = [cat for cat in ALL_CATS if 'dma' in cat]

    ALL_RANGES = ALL_RANGES
    TEMP_MSMTS = TEMP_MSMTS
    MSMTS = MSMTS    
    POLLUTS = POLLUTS
    CAUSES = CAUSES
    MAJOR_CAUSES = MAJOR_CAUSES
    STRINDEX_CATS = STRINDEX_CATS
    STRINDEX_SUBCATS = STRINDEX_SUBCATS
    CONTAIN_CATS = CONTAIN_CATS
    ECON_CATS = ECON_CATS
    HEALTH_CATS = HEALTH_CATS
    KEY3_CATS = KEY3_CATS

    MOBIS = MOBIS
    DMA_CATS = MSMTS + POLLUTS + STRINDEX_CATS

    PER_APPENDS = PER_APPENDS
    ALLOWED_FACTORS_TO_FAVOR_EARLIER = ['key3_sum', 'h_sum', 'e_sum', 'c_sum'] + STRINDEX_CATS
    
    def __init__(
        self, baseframe, count_dma=3, count_categories=[], factors=[], 
        regions=[], countries=[], excluded_regions=[], excluded_countries=[], 
        factor_dmas={}, mobi_dmas={},
        start_factor='deaths', start_hurdle=1, tail_factor='', tail_hurdle=1.2, 
        min_deaths=0, min_days_from_start=0, country_level=False, world_averages=False,
        temp_scale='C', lognat=False, favor_earlier=False, factors_to_favor_earlier=[]
    ):
        # Base DataFrame
        self.baseframe = baseframe
        
        # Period used for Daily moving average of count categories
        self.count_dma = count_dma
        
        # Limit the casestudy df to certain count categories
        self.count_categories = accept_string_or_list(count_categories)
        self.count_categories = [cat for cat in self.count_categories if cat not in CaseStudy.COUNT_TYPES]
        # Factors in focus for analysis
        self.factors = [factors] if isinstance(factors, str) else factors
        self.factor_dmas = factor_dmas
        
        # To remove specific regions or countries from the dataset
        self.regions = accept_string_or_list(regions)
        self.countries = accept_string_or_list(countries)
        self.excluded_regions = accept_string_or_list(excluded_regions)
        self.excluded_countries = accept_string_or_list(excluded_regions)
                
        # Hurdles to filter data and isolate regions/timeframes most pertinent to analysis
        self.min_deaths = min_deaths
        self.start_hurdle = start_hurdle
        self.start_factor = start_factor
        self.tail_hurdle = tail_hurdle
        self.tail_factor = tail_factor
        self.min_days_from_start = min_days_from_start
        
        self.country_level = country_level
        self.world_averages = world_averages
        self.lognat = lognat
        self.favor_earlier = favor_earlier
        self.factors_to_favor_earlier = accept_string_or_list(factors_to_favor_earlier)
        self.temp_scale = temp_scale
        
        # FACTORS
        self.temp_msmts = [msmt for msmt in self.TEMP_MSMTS if msmt in self.factors]
        self.polluts = [factor for factor in self.factors if factor in self.POLLUTS]        
        self.age_ranges = [factor for factor in self.factors if factor in ALL_RANGES]
        self.causes = [factor for factor in self.factors if factor in self.CAUSES]
        self.strindex_factors = [factor for factor in self.factors if factor in self.STRINDEX_CATS]
        
        self.mobis = [factor for factor in self.factors if factor in self.MOBIS]
        self.mobi_dmas = mobi_dmas
        
        self.pop_cats = self.age_ranges + self.causes
        if 'visitors' in factors:
            self.pop_cats.append('visitors')
        if 'gdp' in factors:
            self.pop_cats.append('gdp')
        
        if not all(factor in self.ALLOWED_FACTORS_TO_FAVOR_EARLIER for factor in self.factors_to_favor_earlier):
            raise AttributeError("""
                Only the following categories can included in factors_to_favor_earlier: {}
        """.format(' '.join(self.ALLOWED_FACTORS_TO_FAVOR_EARLIER)))

        if isinstance(self.factor_dmas, dict):
            dma_not_available = [key for key in self.factor_dmas.keys() if key not in self.DMA_CATS]
            if dma_not_available:
                raise AttributeError("""
                    DMA or growth not available for {}. Only available for {}.
                """.format(' ,'.join(dma_not_available, ' ,'.join(self.DMA_CATS))))
        
        self.abbreviate = 'initials'

        self.df = self._filter_baseframe()
        
        # Chart inner classes; pass the casestudy instance to make 
        # attributes accesible at the chart level
        self.comp_chart = CompChart2D(self)
        self.comp_chart4d = CompChart4D(self)
        self.heatmap = HeatMap(self)
        self.barcharts = BarCharts(self)
        self.scatterflow = ScatterFlow(self)

    def total_cases(self, date, regions=None):        
        """
        Parameter
        _________
        
        date:   type str, from '%Y-%m-%df' e.g. '2020-05-05'

        Returns 
        ________

        type float, total global cases as of specified date
        """
        baseframe = self.baseframe.copy(deep=True)
        if regions:
            baseframe = baseframe[baseframe.region_name.isin(regions)]

        return baseframe[baseframe.date == date].cases.sum()

    def total_deaths(self, date, regions=None):
        """
        Parameter
        _________
        
        date:   type str, from '%Y-%m-%df' e.g. '2020-05-05'

        Returns 
        ________

        type float, total global fatalities as of specified date
        """
        baseframe = self.baseframe.copy(deep=True)
        if regions:
            baseframe = baseframe[baseframe.region_name.isin(regions)]
        
        return baseframe[baseframe.date == date].deaths.sum()

    def _abbreviator(self, region, abbreviate=None):
        """
        Used to abbreviate region names

        available abbreviate option:
            'initials': finds initials 
        """
        abbreviate = abbreviate if abbreviate else self.abbreviate
        region = region.split(' ')
        if len(region) > 1:
            if self.abbreviate == 'first':
                first = region[0][0] + '.'
                return ' '.join([first] + region[1:])
            elif self.abbreviate == 'initials':
                initials = '.'.join(reg[0] for reg in region) + '.'
                return initials
        else:
            return region[0][:6]

    def _earlier_is_better(self, series, scale_factor=1):
        """
        Takes a series or array of values and scales it so that
        earlier values are worth more than later ones

        When scale_factor == 1, y will be an array between 1 and ~0
        If scale_factor is > 1, then y.min() will increase
        So for scale_factor == 20, y.min() ~= 0.4 

        """
        x = np.array([i for i in range(1, series.shape[0] + 1)])
        y = np.log(x) / np.log(1/series.shape[0]/scale_factor) + 1
        return np.multiply(series, y)

    def _agg_to_country_level(self, baseframe):
        # Different aggregation approaches for columns
        FIRST_ROW = ['travel_year', 'gdp_year', 'year', 'country', 'country_id', 'country_code']
        SUMS = ALL_RANGES + CAUSES + ['visitors', 'gdp', 'deaths', 'cases', 'land_KM2', 'city_KM2', 'population']
        AVERAGES = STRINDEX_CATS + MSMTS
        EXCLUDES = POLLUTS + MOBIS
        
        # Filter baseframe
        df_subs = baseframe[baseframe.country_code.isin(COUNTRIES_W_REGIONS)]

        # Loop through each country 
        dfs_country = []
        for code, df_group in df_subs.groupby('country_code'):
            region_id = 'reg_for_' + code
            region_name = df_group.iloc[0]['country']
            country_dict = df_group.iloc[0][FIRST_ROW]

            # Group each country frame by date, then aggregate column values on the date
            country_dicts = []
            for date, df_date in df_group.groupby('date'):
                country_dict = {'region_id': region_id, 'region_name': region_name, **country_dict}
                country_dict['date'] = date

                for sum_ in SUMS:
                    country_dict[sum_] = df_date[sum_].sum()
                for avg in AVERAGES:
                    country_dict[avg] = df_date[avg].mean()   

                country_dict['land_dens'] = country_dict['population'] / country_dict['land_KM2']
                country_dict['city_dens'] = country_dict['population'] / country_dict['city_KM2']
                country_dicts.append(country_dict)
            
            df_country = pd.DataFrame(country_dicts)
            dfs_country.append(df_country)

        df_countries = pd.concat(dfs_country)

        df_nosubs = baseframe[~baseframe.country_code.isin(COUNTRIES_W_REGIONS)]

        # Exclude values that don't aggregate across regions easily
        df_nosubs = df_nosubs.drop(EXCLUDES, axis=1)

        return pd.concat([df_nosubs, df_countries])

    def _filter_baseframe(self, baseframe=None, country_level=False, world_averages=False):
        """
        Filters and processes the base dataframe to isolate specific data for analysis

        A baseframe can be provided directly to augment outside of the standard usage
        """

        df = self.baseframe.copy(deep=True) if not baseframe else baseframe
        df.date = pd.to_datetime(df.date).dt.tz_localize(None)

        if self.country_level or country_level:
            df = self._agg_to_country_level(df)
            self.regions = df.region_name.unique().tolist()
        
        if self.regions:
            df = df[df['region_name'].isin(self.regions)]
        
        if self.countries:    
            df = df[df['country'].isin(self.countries)]            
        
        if self.excluded_regions:
            df = df[~df['region_name'].isin(self.excluded_regions)]
        
        if self.excluded_countries:    
            df = df[~df['country'].isin(self.excluded_countries)]            
        
        # Shrink DF to include only columns for factors in focus
        df = df[CASE_COLS + self.factors]

        """
        # Find dma for factors with timeshift
        # This must be completed before the main loop, because the shift may look to 
        # dates that occur BEFORE the start_hurdle. These will be cutoff in the main the loop
        """
        # Remove regions that don't meet a minimum death threshold if start_hurdle provided
        if self.start_factor == 'deaths' or self.min_deaths:
            min_deaths = self.min_deaths if self.min_deaths else self.start_hurdle
            max_deaths = df.groupby('region_id')['deaths'].max()
            regions_above_threshold = max_deaths.where(max_deaths > min_deaths).dropna().index.values
            df = df[df['region_id'].isin(regions_above_threshold)]
        
        # Loop through each region and append additional data
        new_dfs = []
        for region_id, df_group in df.groupby('region_id'):
            df_group = df_group.copy(deep=True)

            for count_type in self.COUNT_TYPES:
                df_group[count_type + '_new'] = df_group[count_type].diff()
                df_group[count_type + '_dma'] = df_group[count_type].rolling(window=self.count_dma).mean()
                df_group[count_type + '_new_dma'] = df_group[count_type + '_new'].rolling(window=self.count_dma).mean()
            
            for count_cat in self.BASECOUNT_CATS:
                df_group[count_cat + '_per_1M'] = df_group[count_cat] / df_group['population'] * 1000000
                df_group[count_cat + '_per_person_per_land_KM2'] = df_group[count_cat] / (df_group['land_dens'])
                df_group[count_cat + '_per_person_per_city_KM2'] = df_group[count_cat] / (df_group['city_dens'])
            
            if self.lognat:
                with np.errstate(divide='ignore', invalid='ignore'):
                    for cat in self.BASE_PLUS_PER_CATS:
                        df_group[cat + '_lognat'] = np.log(df_group[cat].fillna(0))
            
            for col in df_group.columns:
                if col not in CASE_COLS:
                    df_group['growth_' + col] = df_group[col] / df_group[col].shift(1)

            # Filter dataframe for count categories
            if self.count_categories:
                df_group = df_group[CASE_COLS + self.count_categories + self.factors]

            # Forward fill any nans in the strindex categories. For some, the entire columns is nans, so fill those with 0s.
            if self.strindex_factors:
                df_group[self.strindex_factors] = df_group[self.strindex_factors].fillna(method='ffill')
                df_group[self.strindex_factors] = df_group[self.strindex_factors].fillna(0)

            # SUM certain strindex categories
            if all(cat in self.factors for cat in self.CONTAIN_CATS):
                df_group['c_sum'] = df_group[self.CONTAIN_CATS].sum(axis=1)
            
            if all(cat in self.factors for cat in self.ECON_CATS):
                df_group['e_sum'] = df_group[self.ECON_CATS].sum(axis=1)

            if all(cat in self.factors for cat in self.HEALTH_CATS):
                df_group['h_sum'] = df_group[self.HEALTH_CATS].sum(axis=1)

            if all(cat in self.factors for cat in self.KEY3_CATS):
                df_group['key3_sum'] = df_group[self.KEY3_CATS].sum(axis=1)

            if self.favor_earlier:
                for factor in self.factors_to_favor_earlier:
                    df_group[factor + '_earlier'] = self._earlier_is_better(df_group[factor])
            
            # Adjust factors for percentage of population
            if self.pop_cats:
                for pop_cat in self.pop_cats:
                    df_group[pop_cat + '_%'] = df_group[pop_cat] / df_group['population']
            
            # Unit conversions for chosen temperature scale
            if self.temp_msmts:
                for temp_msmt in self.temp_msmts:
                    if self.temp_scale == 'C':
                        df_group[temp_msmt] = df_group[temp_msmt] - 273.15                            
                    elif self.temp_scale == 'F':
                        df_group[temp_msmt] = df_group[temp_msmt] * 9 / 5 - 459.67

            if self.factor_dmas:
                # Return empty dataframe if any factor has only NaNs for the time period and the region
                for factor, dma in self.factor_dmas.items():
                    df_group[factor + '_dma'] = df_group[factor].rolling(window=dma).mean()
                    df_group[factor + '_growth'] = df_group[factor] / df_group[factor].shift(1)
                    df_group[factor + '_growth_dma'] = df_group[factor + '_growth'].rolling(window=dma).mean()
            
            if self.mobis:
                for mobi in self.mobis:
                    df_group[mobi] /= 100
                for mobi, dma in self.mobi_dmas.items():
                    df_group[mobi + '_growth'] = df_group[mobi] + 1
                    df_group[mobi + '_growth_dma'] = df_group[mobi + '_growth'].rolling(window=dma).mean()
                    df_group[mobi + '_dma'] = df_group[mobi + '_growth_dma'] - 1

            # Filter observations on the front end
            # If there are no observations that satisfy the hurdle
            # set df_group to empty 
            if self.start_factor:
                hurdle_index = df_group[df_group[self.start_factor] >= self.start_hurdle].index
                
                if hurdle_index.size >= 1:
                    first_row = hurdle_index[0]
                    df_group = df_group.loc[first_row:]
                else:
                    df_group = pd.DataFrame()

            # Filter observations on the tailend
            if self.tail_factor:
                if (df_group[self.tail_factor] >= self.tail_hurdle).any():
                    last_row = df_group[df_group[self.tail_factor] >= self.tail_hurdle].index[-1]
                    df_group = df_group.loc[:last_row]
                else:
                    df_group = pd.DataFrame()

            if not df_group.empty:
                # Indicates the number of days since the hurdle was met
                df_group['days'] = df_group['date'] - df_group['date'].iloc[0]

            # If region data does not cover enough days, return empty Dataframe
            if len(df_group) < self.min_days_from_start:
                df_group = pd.DataFrame()
            
            new_dfs.append(df_group)
              
        df = pd.concat(new_dfs)

        # Handle -inf values arising from taking natural log
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.sort_values(by=['region_id', 'date'])

        # Find world averages for each column on each date
        if self.world_averages or world_averages:
            globe_rows = []
            for date, df_date in df.groupby('date'):
                globe_row = ['REG_FOR_WORLD_AVGS', 'COUNTRY_FOR_WORLD_AVGS', 'WorldAvg', 'WLDAVG', 'WorldAvg'] 
                globe_row += [date]

                df_date = df_date[[col for col in df_date.columns if col not in ['region_id', 'country_id', 'region_name', 'country_code', 'country', 'date']]]
                globe_row += df_date.mean().tolist()
                
                globe_row = [round(i, 2) if isinstance(i, float) else i for i in globe_row]
                globe_rows.append(globe_row)
            
            df_globe = pd.DataFrame(globe_rows, columns=df.columns)
            df = df.append(df_globe).sort_values(by=['region_name', 'date'])

        return df
