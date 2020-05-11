import math
import operator
import numpy as np
import pandas as pd
import copy

from bokeh.plotting import figure, output_file, ColumnDataSource
from bokeh.palettes import Category20, Viridis256
from bokeh.transform import dodge, linear_cmap
from bokeh.models import HoverTool, Label, LinearAxis, Range1d, Arrow, NormalHead, OpenHead, VeeHead
from bokeh.io import export_png

import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .helpers import accept_string_or_list
from .constants import META_COLS, COVID_DRAGONS

class BaseChart:
    """
    Base Class for common chart attributes and methods

    2 main roles:
        1) converts casestudy instance attributes to more useable form
        2) centralizes label formatting
    """
    def __init__(self, casestudy):
        self._casestudy = casestudy
        self.df = self._casestudy.df.copy(deep=True)
        self.start_hurdle = self._casestudy.start_hurdle
        self.start_factor = self._casestudy.start_factor
        self.count_dma = self._casestudy.count_dma
        self.temp_scale = self._casestudy.temp_scale

        # Computations
        self.regions = self._casestudy.regions
        
        self.degrees = '\xb0' if self.temp_scale in ['C', 'F'] else ''
            
        self.labels = {
            '': 'January 2020',
            'population': 'Population',
            'land_dens': 'Density of Land Area',
            'city_dens': 'Population Density of Largest City',
            'deaths': 'Cumulative Deaths',
            'deaths_new': 'Daily Fatalities',
            'deaths_new_dma_per_1M': 'Daily Fatalities per 1M ({}DMA)'.format(self.count_dma),
            'deaths_per_1M': 'Cumulative Fatalities per 1M',
            'deaths_new_per_1M': 'Daily Fatalities per 1M',
            'deaths_new_dma_per_1M_lognat':  'Daily Fatalities per 1M ({}DMA)\n(Natural Log)'.format(self.count_dma),
            'deaths_new_dma_per_person_per_land_KM2': 'Daily Fatalities / Person / Land KM\u00b2 ({}DMA)'.format(self.count_dma),
            'deaths_new_dma_per_person_per_city_KM2': 'Daily Fatalities / Person / City KM\u00b2 ({}DMA)'.format(self.count_dma),
            'deaths_new_dma_per_person_per_land_KM2_lognat': 'Daily Fatalities / Person / Land KM\u00b2 ({}DMA)\n(Natural Log)'.format(self.count_dma),
            'deaths_new_dma_per_person_per_city_KM2_lognat': 'Daily Fatalities / Person / City KM\u00b2 ({}DMA)\n(Natural Log)'.format(self.count_dma),
            'deaths_per_person_per_land_KM2': 'Total Fatalities / Person / Land KM\u00b2 ({}DMA)'.format(self.count_dma),
            'deaths_per_person_per_city_KM2': 'Total Fatalities / Person / City KM\u00b2 ({}DMA)'.format(self.count_dma),
            'cases': 'Cumulative Cases',
            'cases_per_1M': 'Cumulative Cases per 1M',
            'cases_new_per_1M': 'Daily Cases per 1M',
            'cases_per_person_per_city_KM2_lognat': 'Total Cases / Person / City KM\u00b2\n(Natural Log)',
            'cases_new_dma_per_1M': 'Daily Cases per 1M ({}DMA)'.format(self.count_dma),
            'cases_new_dma_per_1M_lognat': 'Daily Cases per 1M ({}DMA)\n(Natural Log)'.format(self.count_dma),
            'cases_new_dma_per_person_per_city_KM2': 'Daily Cases / Person / City KM\u00b2 ({}DMA)'.format(self.count_dma),
            'cases_new_dma_per_person_per_land_KM2': 'Daily Cases / Person / Land KM\u00b2 ({}DMA)'.format(self.count_dma),
            'cases_new_dma_per_person_per_city_KM2_lognat': 'Total Cases / Person / City KM\u00b2 ({}DMA)\n(Natural Log)'.format(self.count_dma),
            'temp': 'Temperature ({}{})'.format(self.degrees, self.temp_scale),
            'dewpoint': 'Dewpoint ({}{})'.format(self.degrees, self.temp_scale),
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
            'c1': 'School Closing', 'c2': 'Workplace Closing', 'c3': 'Cancel Public Events',
            'c4': 'Restrictions on Gatherings', 'c5': 'Close Public Transport', 'c6': 'Stay-at-Home Requirements',
            'c7': 'Restrictions on Internal Movement', 'c8': 'International Travel Controls',
            'e1': 'Income Support', 'e2': 'Debt / Contract Relief', 'e3': 'Fiscal Measures',
            'e4': 'International Support', 'h1': 'Public Information Campaigns',
            'h2': 'Testing Policy', 'h3': 'Contact Tracing',
            'h4': 'Emergency Investment in Health Care', 'h5': 'Investment in Vaccines',
            'key3_sum_earlier': 'Sum of Key 3 Oxford Stingency Factor Weighted to Earlier Dates',
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
        if self.start_factor == 'date':
            self.labels = {
                **self.labels,
                'days_' + self.start_factor: 'Days Since {}'.format(self.start_hurdle.strftime('%B %d, %Y')),
            }
        else:
            self.labels = {
                **self.labels,
                'days_' + self.start_factor: 'Days Since {} {}'.format(self.start_hurdle, self.labels[self.start_factor]),
                'days_to_max_' + self.start_factor : 'Days Since {} {} Until Max Fatality Rate'.format(self.start_hurdle, self.labels[self.start_factor]),
            }

        if self._casestudy.age_ranges:
            self.labels ={**self.labels, **{age_range + '_%': self._age_range_label_maker(age_range) for age_range in self._casestudy.age_ranges}}
        if self._casestudy.factor_dmas:
            for factor, dma in self._casestudy.factor_dmas.items():
                self.labels[factor + '_dma'] = self.labels[factor] + ' {}DMA'.format(dma)
        if self._casestudy.mobi_dmas:
            for factor, dma in self._casestudy.mobi_dmas.items():
                self.labels[factor + '_dma'] = self.labels[factor] + ' {}DMA'.format(dma)
        if self._casestudy.causes:
            self.labels = {**self.labels, **{cause + '_%': self._cause_of_death_label_maker(cause) for cause in self._casestudy.causes}}
        
    def _age_range_label_maker(self, age_range):
        if '_' in age_range:
            return '% of Population Between ' + age_range[1:3] + ' & ' + age_range[-3:-1]
        elif 'PLUS' in age_range:
            return '% of Population Over ' + age_range[1:3]

    def _cause_of_death_label_maker(self, cause):
        return self.labels[cause] + ' as % of Population'     

class CompChart2D(BaseChart):

    def _bar_incrementer(self, labels):
        """
        Used to space multiple bars around y-axis point
        """
        num_labels = len(labels)
        midpoint = num_labels / 2 if num_labels % 2 == 0 else num_labels / 2 - 0.5
        offset = self.base_inc * midpoint
        even_offset = self.base_inc / 2 if num_labels % 2 == 0 else 0

        return [i * self.base_inc - offset + even_offset for i in range(num_labels)]
    
    def _multiline_source(self, palette_shift=0):
        """
        Function to prepare dataframe for use in chart
        Loops through each region group
        """
        days = []
        values_by_region = []
        region_order = []
        for region_id, df_group in self.df_comp.groupby('region_id'):
            values = list(df_group[self.comp_category].values)
            days.append([i for i in range(len(values))])
            values_by_region.append(values)
            region_order.append(df_group.region_name.iloc[0])
        
        return {'x': days, 'y': values_by_region, 'regions': region_order}
    
    def _vbar_source(self):
        values_by_region = {}
        dates_by_region = {}
        for i, df_group in self.df_comp.groupby('region_name'):
            values = list(df_group[self.comp_category].values)
            dates = list(df_group['date'].dt.strftime('%b %d'))

            # If the counts for region are less than the max, add 0s until it matches the size of the longest region
            if len(values) < self.max_length:
                values += [0 for i in range(self.max_length - len(values))]
                dates += ['n/a' for i in range(self.max_length - len(dates))]

            values_by_region[i] = values
            dates_by_region[i + '_date'] = dates
        
        return {'x': self.days, **values_by_region, **dates_by_region}
    
    def make(
            self, comp_category='deaths_new_dma_per_1M', regions=None,
            comp_type='vbar', overlay=None, title='', 
            palette_base=Viridis256, palette_flip=False, palette_shift=0, 
            multiline_labels=True, label_offsets={}, fs_labels=8, 
            legend=False, legend_location='top_right',
            x_fontsize=10, y_fontsize=10,
            fs_xticks=16, fs_yticks=16,
            width=750, height=500, base_inc=.25,
            save_file=False, filename=None,
        ):
        self.df_comp = self.df[self.df[comp_category].notna()].copy(deep=True)
        
        # If regions are passed, filter the dataframe and reset attributes
        # via `_chart_setup`
        if regions:
            regions = [regions] if isinstance(regions, str) else regions
            self.df_comp = self.df_comp[self.df_comp['region_name'].isin(regions)]
        else:
            regions = list(self.regions)
        
        # Setup additional class attributes
        self.comp_category = comp_category
        self.comp_type = comp_type
        self.palette_base = palette_base
        self.base_inc = base_inc
        self.max_length = self.df.groupby('region_id')['days'].max().max().days + 1
        self.days = [i for i in range(self.max_length)]

        # Set chart attributes
        fs_labels = str(fs_labels) + 'pt'
        last_date = self.df_comp.iloc[-1]['date'].strftime('%B %d')
        min_y = self.df_comp[comp_category].min()
        min_y += min_y * .01
        max_y = self.df_comp[comp_category].max()
        if max_y < 0:
            max_y -= max_y * .1
        else:
            max_y += max_y * .1

        p = figure(
            y_range=Range1d(start=min_y, end=max_y),
            plot_height=height, plot_width=width,
            min_border=0,
            toolbar_location=None,
            title=title,
        )

        # Create the Color Palette
        # An extra color is added and changes in the coloridx are limited
        # to all but the last item, to allow for shifting of palette
        # via palette_shift
        palette_base = np.array(palette_base)
        coloridx = np.round(np.linspace(0, len(palette_base) - 1, len(regions) + 1)).astype(int)
        if palette_flip:
            coloridx[:-1] = coloridx[:-1][::-1]
        if palette_shift:
            coloridx[:-1] += palette_shift
        palette = palette_base[coloridx]

        if comp_type == 'multiline':
            ml_data = self._multiline_source()
            ml_data['color'] = palette[:-1]
            source_ml = ColumnDataSource(ml_data)
            
            p.multi_line(xs='x', ys='y', line_color='color', legend_group='regions', line_width=5, source=source_ml)

            # Setup labels for each line
            if multiline_labels:
                for i in range(len(ml_data['x'])):
                    x_label = int(ml_data['x'][i][-1])
                    y_label = ml_data['y'][i][-1]
                    x_offset = -20
                    y_offset = 5
                    label_region = ml_data['regions'][i]

                    if label_region in label_offsets.keys():
                        x_offset += label_offsets[label_region]['x_offset']
                        y_offset += label_offsets[label_region]['y_offset']
    
                    label = Label(
                        x=x_label, y=y_label, 
                        x_offset=x_offset, y_offset=y_offset,
                        text_font_size=fs_labels, text_color=palette[i], text_alpha=.8,
                        text=label_region, text_font_style='bold',
                        render_mode='canvas'
                    )
                    p.add_layout(label)

        if comp_type == 'vbar':
            vbar_data = self._vbar_source()
            vbar_source = ColumnDataSource(vbar_data)
            increments = self._bar_incrementer(regions)
            
            legend_items = []
            for i, region in enumerate(regions):
                region_start = vbar_data['{}_date'.format(region)][0]
                legend_label = region + ': {}'.format(region_start)

                v = p.vbar(
                    x=dodge('x', increments[i], range=p.x_range), top=region, width=.3, 
                   source=vbar_source, color=palette[i], legend_label=legend_label,
                )
        
        p.legend.visible = legend
        p.legend.title = 'Region: Start Date'
        p.legend.location = legend_location
        p.legend.border_line_color = 'black'

        p.xaxis.axis_label = self.labels['days_' + self.start_factor]
        p.xaxis.axis_label_text_font_size = str(x_fontsize) + 'pt'
        p.xaxis.major_label_text_font_size = str(fs_xticks) + 'pt'

        p.yaxis.axis_label = self.labels[comp_category]
        p.yaxis.axis_label_text_font_size = str(y_fontsize) + 'pt'
        p.yaxis.major_label_text_font_size = str(fs_yticks) + 'pt'

        p.xaxis.major_tick_line_color = None
        p.xgrid.grid_line_color = None

        p.min_border = 20

        if overlay:
            overlay_days = []
            overlay_by_region = []
            for region_id, df_group in self.df_comp.groupby('region_id'):
                overlays = list(df_group[overlay].dropna().values)
                overlay_days.append([i for i in range(len(overlays))])
                overlay_by_region.append(overlays)
                
            data2 = {'x': overlay_days, 'y': overlay_by_region, 'color': palette[:-1]}
            source2 = ColumnDataSource(data=data2)
            start = min(olay for region in overlay_by_region for olay in region) * 0.5
            end = max(olay for region in overlay_by_region for olay in region) * 1.5
            p.extra_y_ranges = {overlay: Range1d(start=start, end=end)}
            p.multi_line(xs='x', ys='y', line_color='color', line_width=4, source=source2,
                      y_range_name=overlay, alpha=.4,
            )

            right_axis_label = self.labels[overlay]
            p.add_layout(LinearAxis(y_range_name='{}'.format(overlay), axis_label=right_axis_label), 'right')

        if save_file:
            export_png(p, filename=filename)

        return p

class CompChart4D(BaseChart):
    """
    Class for 4D bar charts
    
    Utilizes matplotlib
    
    Inherits `__init__`, attributes, and methods from BaseChart
    
    ***NOTE***
    
    matplotlib 3d bar charts can have issues with clipping and depth. these can be overcome by
    building the chart manually, one bar at a time, as per this answer from astroMonkey on stackoverflow:
    
    https://stackoverflow.com/questions/18602660/matplotlib-bar3d-clipping-problems/37374864#comment108302737_37374864
    """
        
    def _sph2cart(self, r, theta, phi):
        """
        For manual bar creation:
        transforms spherical to cartesian 
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def _sphview(self, ax):
        """
        For manual bar creation:
        returns the camera position for 3D axes in spherical coordinates
        """
        r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
        theta, phi = np.radians((90-ax.elev, ax.azim))
        return r, theta, phi

    def _ravzip(self, *itr):
        '''flatten and zip arrays'''
        return zip(*map(np.ravel, itr))
  
    def _data_morph_for_bar4d(self, comp_category=None, comp_size=None, rank_category=None):
        comp_category = comp_category if comp_category else self.comp_category
        rank_category = rank_category if rank_category else comp_category
        comp_size = comp_size if comp_size else self.comp_size        
        
        df = self.df_4d.copy(deep=True)
        cols_to_keep = list(set([comp_category, rank_category]))
        if self.color_factor:
            cols_to_keep += [self.color_factor] 
        
        df = df[['region_id', 'region_name', 'country', 'date', 'days'] + cols_to_keep]
        df = df.dropna()
        
        dfs = []
        for region_id, df_group in df.groupby('region_id'):
            if (df_group[comp_category].isnull()).any():
                df_group = df_group.loc[df_group.index[df_group[comp_category].notna()]]
                df_group = df_group.reset_index().drop('index', axis=1)
            
            dfs.append(df_group)
        
        df = pd.concat(dfs) if dfs else df

        self.region_names = list(df.sort_values(rank_category, ascending=False)['region_name'].unique())[:comp_size]
        reg_cats = {region: i + 1 for i, region in enumerate(self.region_names)}

        df = df[df['region_name'].isin(self.region_names)].copy(deep=True)
        df['region_name'] = df['region_name'].astype('category')
        df['region_name'].cat.set_categories(self.region_names, inplace=True)
        df = df.sort_values(by=['region_name', 'date'])
        df['region_code'] = [reg_cats[region_name] for region_name in df['region_name'].values]
    
        return df
    
    def _grey_maker(self, rgb_value):
        return [rgb_value / 255 for i in range(3)] + [1.0]

    def make(self,
            comp_category='deaths_per_1M', rank_category=None, comp_size=0,
            regions=[], color_factor='',
            x_ticks=True,
            fs_xticks=12, fs_yticks=12, fs_zticks=12,
            fs_xlabel=12, fs_ylabel=12, fs_zlabel=12,
            title='', subtitle='', datetitle='',
            x_title=0, y_title=0, fs_title=19, rot_title=-10.6,
            x_subtitle=0, y_subtitle=0, fs_subtitle=14, rot_subtitle=-10.6,
            x_datetitle=0, y_datetitle=0, fs_datetitle=14, rot_datetitle=-10.6,
            palette_base='coolwarm', color_interval=(), bar_color='orange',
            x_colorbar=0.2, y_colorbar=-.05, h_colorbar=.3, w_colorbar=.01, a_colorbar= 'vertical',
            grid_grey= 87, pane_grey=200, tick_grey=30,
            cb_labelpad=-50,
            zaxis_left=False, gridlines=True, tight=True,
            width=16, height=8, save_file=False, filename=''
        ):
        # Create `make` specific dataframe
        self.df_4d = self.df[self.df[comp_category].notna()].copy(deep=True)

        # Update Object instance for new attributes if provided
        if regions:
            regions = [regions] if isinstance(regions, str) else regions
            self.df_4d = self.df_4d[self.df_4d['region_name'].isin(regions)]
        else:
            regions = self.regions

        # Setup class features through comp function
        self.comp_category = comp_category
        self.rank_category = rank_category
        self.color_factor = color_factor
        self.comp_size = comp_size if comp_size else len(regions)
        self.palette_base = palette_base
        
        # Prep data for 4d
        self.df_4d = self._data_morph_for_bar4d()

        last_date = self.df.iloc[-1]['date'].strftime('%B %d')
        title = 'Comparison of Daily Fatalities as of {}'.format(last_date) if not title else title

        """
        ### Create Base Grid and variables for building the plot manually ###
        """        
        x_length = len(self.region_names)    
        y_length = self.df_4d['days'].max().days + 1
    
        Y, X = np.mgrid[:y_length, :x_length]
        grid = np.array([X, Y])
        (dx,), (dy,) = 0.8*np.diff(X[0,:2]), 0.8*np.diff(Y[:2,0])
        
        # Zarray for zaxis values and farrray for color selection
        # both zarray and farray must be padded to much the size of the 2d x & y arrays
        zarrays = []
        farrays = []
        for i, group in self.df_4d.groupby('region_code'):
            len_array = len(group[comp_category].values)
            end_pad = y_length - len_array
            padded_zarray = np.pad(group[comp_category].values, (0, end_pad), 'constant', )
            zarrays.append(padded_zarray)   
            
            if self.color_factor:
                padded_farray = np.pad(group[self.color_factor].values, (0, end_pad), 'constant', constant_values=(0, -1))
            else:
                # If color is not variable, farray padded with -2, to indicate alpha of 0 later on
                padded_farray = np.pad(np.full_like(group[comp_category].values, -2), (0, end_pad), 'constant', constant_values=(0, -1))
            
            farrays.append(padded_farray)

        Z = np.stack((zarrays)).T
        M = np.stack((farrays)).T if farrays else np.full_like(Z, -2)
        
        """
        ### Instantiate the plot ###
        """

        # Main Plot
        fig = plt.figure(figsize=(width, height))
        ax1 = plt.subplot(projection='3d')    

        # Plot colors
        grid_rgba = self._grey_maker(grid_grey)
        pane_rgba = self._grey_maker(pane_grey)
        tick_rgba = self._grey_maker(tick_grey)
        
        # Color Bar must be defined before the 3d plot is built
        if self.color_factor:
            color_min, color_max = color_interval if color_interval else (self.df_4d[self.color_factor].min(), self.df_4d[self.color_factor].max())
            norm = mpc.Normalize(vmin=color_min, vmax=color_max)
            cmap = cm.get_cmap(palette_base)

            cax = inset_axes(ax1,
               width="60%",  # width = 5% of parent_bbox width
               height="1%",  # height : 50%
               loc='lower left',
               bbox_to_anchor=(x_colorbar, y_colorbar, w_colorbar, h_colorbar),
               bbox_transform=ax1.transAxes,
               borderpad=0,
            )
            cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), extend='both', cax=cax, label=self.labels[self.color_factor], orientation=a_colorbar)
            font = {
                'weight': 'normal',
                'size': 14,
            }

            cb.set_label(label=self.labels[self.color_factor], labelpad=cb_labelpad, color=tick_rgba, fontdict=font)

        # Establish plot visual characteristics
        xyz = np.array(self._sph2cart(*self._sphview(ax1)), ndmin=3).T       #camera position in xyz
        zo = np.multiply([X, Y, np.zeros_like(Z)], xyz).sum(0)  #"distance" of each bar from camera
        bars = np.empty(X.shape, dtype=object)

        #Build plot ... each loop is a bar
        for i, (x, y, dz, m, o) in enumerate(self._ravzip(X, Y, Z, M, zo)):
            if m == -1:
                color = 'white'
                alpha = 0
            elif m == -2:
                color = bar_color
                alpha = 1
            else:
                color = cmap(norm(m))
                alpha = 1
            j, k = divmod(i, x_length)
            bars[j, k] = pl = ax1.bar3d(x, y, 0, dx, dy, dz, color=color, alpha=alpha)
            pl._sort_zpos = o
        
        """
        ### AXIS FORMATS ###
        """
        # XAXIS
        xlab_font = {
            'weight': 'normal',
            'size': fs_xticks,
        }
        
        # Set ticklabels
        # If labels close to end of axis, truncate their names to fit in image
        if x_ticks:
            xticklabels = [self._casestudy._abbreviator(region) for region in self.region_names]
            for i in range(1, 7):
                if len(xticklabels[-i]) > 7:
                    xticklabels[-i] = xticklabels[-i][:6] + '.'
            ax1.set_xticks(np.arange(len(self.region_names)))
            ax1.set_xticklabels(xticklabels, rotation=50, fontdict=xlab_font, color=tick_rgba)
        else:
            xlab_font = {
                'weight': 'normal',
                'size': fs_xlabel,
            }
            xticklabels = ['' for region in self.region_names]
            ax1.set_xticklabels(xticklabels, rotation=50, fontdict=xlab_font, color=tick_rgba)
            ax1.set_xlabel('Regions', 
                labelpad=10, fontdict=xlab_font, color=tick_rgba
            )
        # Remove every nth label for multiples over 40 regions
        # Over 100 regions, forget the labels
        if len(regions) > 50 and len(regions) <= 100:
            for label in ax1.xaxis.get_ticklabels()[:: math.ceil(len(regions) / 40)]:
                label.set_visible(False)
        elif len(regions) > 100:
            for label in ax1.xaxis.get_ticklabels():
                label.set_visible(False)

        for tick in ax1.xaxis.get_majorticklabels():
            tick.set_horizontalalignment('right')

        ax1.tick_params(axis='x', labelsize=fs_xticks, grid_alpha=1, which='major', pad=0, color=pane_rgba)
        
        # YAXIS
        ylab_font = {
            'weight': 'normal',
            'size': fs_ylabel,
        }
        ax1.set_ylabel(self.labels[self.start_factor], 
            labelpad=10, fontdict=ylab_font, color=tick_rgba
        )
        ax1.tick_params(axis='y', labelsize=fs_yticks, pad=3, color=pane_rgba)
        plt.setp(ax1.get_yticklabels(), color=tick_rgba)

        # ZAXIS
        ax1.set_zlabel(self.labels[comp_category], labelpad=10, fontdict=ylab_font, color=tick_rgba)
        plt.setp(ax1.get_zticklabels(), color=tick_rgba)
        ax1.tick_params(axis='z', labelsize=fs_zticks, pad=5, color=tick_rgba)

        # move zaxis to left side if desired
        if zaxis_left:
            tmp_planes = ax1.zaxis._PLANES 
            ax1.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                tmp_planes[0], tmp_planes[1], 
                tmp_planes[4], tmp_planes[5]
            )

            view_2 = (25, -65)
            init_view = view_2
            ax1.view_init(*init_view)
        
        """
        ### GRID AND PANE FORMATS ###
        """
        # Remove gridlines if preferred
        ax1.grid(gridlines)
        
        plt.rcParams['grid.color'] = grid_rgba
        ax1.w_xaxis.set_pane_color(pane_rgba)
        ax1.w_yaxis.set_pane_color(pane_rgba)
        ax1.w_zaxis.set_pane_color(pane_rgba)

        """
        ### TEXT BOXES ### 
        """
        # Title
        bbox_features = {'boxstyle': 'round4', 'facecolor': pane_rgba, 'alpha': 1, 'edgecolor': grid_rgba}
        font = { 'weight': 'normal', 'size': fs_title }
        ax1.text2D(x_title, y_title, title, transform=ax1.transAxes, color=tick_rgba, rotation=rot_title, bbox=bbox_features, fontdict=font)

        # SubTitle
        if subtitle:
            bbox_features = {'boxstyle': 'square', 'facecolor': pane_rgba, 'alpha': 1, 'edgecolor': pane_rgba}
            font = { 'weight': 'normal', 'size': fs_subtitle }
            ax1.text2D(x_subtitle, y_subtitle, subtitle, transform=ax1.transAxes, color=tick_rgba, rotation=rot_subtitle, bbox=bbox_features, fontdict=font)
        
        if datetitle:
            font = { 'weight': 'normal', 'size': fs_datetitle }    
            bbox_features = {'boxstyle': 'round4', 'facecolor': pane_rgba, 'alpha': 1, 'edgecolor': pane_rgba}
            ax1.text2D(x_datetitle, y_datetitle, datetitle, transform=ax1.transAxes, color=tick_rgba, rotation=-10.5, bbox=bbox_features, fontdict=font)
        
        # For tight layout
        plt.subplots_adjust(left=0, bottom=-.3, right=1, top=1, wspace=0, hspace=0)
        
        if save_file:
            plt.savefig(filename, bbox_inches = "tight")
        
        return plt

class HeatMap(BaseChart):
    """
    Class for Heat Maps comparing a infection rates and a data factor
    
    Utilizes matplotlib    
    """
    def _data_morph_for_heatmap(self, comp_category=None, comp_size=None, rank_category=None):
        comp_category = comp_category if comp_category else self.comp_category
        rank_category = rank_category if rank_category else comp_category
        comp_size = comp_size if comp_size else self.comp_size        
    
        df = self.df.copy(deep=True)
        # Filter for variables in the compsize
        self.comp_region_ids = list(df.sort_values(rank_category, ascending=False)['region_id'].unique())[:comp_size]
        df = df[df['region_id'].isin(self.comp_region_ids)].copy(deep=True)

        map_list = []
        for region_id, df_group in df.groupby('region_id'):
            map_dict = {}
            map_dict['region_id'] = region_id
            map_dict['region_id'] = region_id
            map_dict['region_name'] = df_group['region_name'].iloc[0]
            map_dict[comp_category] = df_group[comp_category].max()

            if self.comp_factor == 'days_to_max':
                max_idx = df_group[comp_category].argmax()
                map_dict[self.comp_factor] = df_group.iloc[max_idx].days.days
            elif 'earlier' in self.comp_factor:
                map_dict[self.comp_factor] = df_group[self.comp_factor].mean()
            else:
                comp_idx = df_group[self.comp_factor].argmax() if self.comp_factor_start == 'max' else 0
                map_dict[self.comp_factor] = df_group[self.comp_factor].iloc[comp_idx]
            
            if self.color_factor:
                color_idx = df_group[self.color_factor].argmax() if self.color_factor_start == 'max' else 0
                map_dict[self.color_factor] = df_group[self.color_factor].iloc[color_idx]

            map_list.append(map_dict)

        df_hm = pd.DataFrame(map_list)
        df_hm = df_hm.sort_values(by=self.comp_factor, ascending=False)

        return df_hm

    def box_stats(self, date, dirxn, xbox, ybox, factor_in_the_box='', inverse=False):
        if dirxn == 'greater':
            comp = operator.lt if inverse else operator.gt
            box_params = (comp(self.df_hm[self.comp_factor], xbox) & \
                (self.df_hm[self.comp_category] > ybox)
            )
        elif dirxn == 'lesser':
            comp = operator.ge if inverse else operator.le
            box_params = (comp(self.df_hm[self.comp_factor], xbox) & \
                (self.df_hm[self.comp_category] <= ybox)
            )
            
        df_box = self.df_hm[box_params]
 
        total_deaths = self._casestudy.total_deaths(date)
        box_deaths = self._casestudy.total_deaths(date, df_box.region_name.tolist())
        kwargs = {
            'df': df_box,
            'num_regs': df_box.shape[0],
            'all_deaths': box_deaths / total_deaths,
            self.labels[self.comp_factor]: df_box[self.comp_factor].mean(),            
        }
        if factor_in_the_box == 'comp_category':
            kwargs[self.labels[self.comp_category]] = df_box[self.comp_category].mean()
        if factor_in_the_box == 'color_factor':
            kwargs[self.labels[self.color_factor]] = df_box[self.color_factor].mean()
        return kwargs

    def make(self,
        comp_category='deaths', rank_category=None, comp_size=None,
        regions=[], comp_factor='population', color_factor='', 
        comp_factor_start='start_hurdle', color_factor_start='start_hurdle',
        fs_xticks=12, fs_yticks=12,
        fs_xlabel=24, fs_ylabel=24, fs_clabel=16,
        pad_xlabel=20, pad_ylabel=12,
        annotations=[], hlines=[], hline_alpha=.1,
        palette_base='coolwarm', hexsize=27, bins=None,
        rects=[],
        width=16, height=12, save_file=False, filename=''
    ):
        factor_starts = ['start_hurdle', 'max']
        if comp_factor_start not in factor_starts or color_factor_start not in factor_starts:
            raise AttributeError("""
                comp_factor_start or color_factor_start must be one of {}
        """.format(factor_starts))

        # Update Object instance for new attributes if provided
        if regions:
            regions = [regions] if isinstance(regions, str) else regions
            self.df = self.df[self.df['region_name'].isin(regions)]
        
        # Setup class features through comp function
        self.comp_category = comp_category
        self.rank_category = rank_category
        self.comp_factor = comp_factor
        self.comp_factor_start = comp_factor_start
        self.color_factor = color_factor
        self.color_factor_start = color_factor_start
        
        self.comp_size = comp_size
        self.palette_base = palette_base
        
        # Prep data for 4d
        self.df_hm = self._data_morph_for_heatmap()
        
        
        x = self.df_hm[self.comp_factor]
        y = self.df_hm[self.comp_category]
        c = x

        if self.color_factor:
            c = self.df_hm[self.color_factor]

        fig, ax = plt.subplots(figsize=(width, height))
        h = ax.hexbin(x, y, C=c, gridsize=hexsize, cmap=palette_base, bins=bins)

        if hlines:
            for y_hline in hlines:
                hl = ax.axhline(y=y_hline, color='gray', linestyle='dashed', alpha=hline_alpha)

        ax.tick_params(axis='x', labelsize=fs_xticks, which='major', pad=4)
        xlabel = 'days_to_max_' + self.start_factor if self.comp_factor == 'days_to_max' else self.comp_factor    
        ax.set_xlabel(self.labels[xlabel], fontsize=fs_xlabel, labelpad=pad_xlabel)

        ax.tick_params(axis='y', labelsize=fs_yticks, which='major', pad=4)
        ylabel = self.labels[comp_category]
        ax.set_ylabel(ylabel, fontsize=fs_ylabel, labelpad=pad_ylabel)

        if self.color_factor:
            norm = mpc.Normalize(vmin=c.min(), vmax=c.max())
            cmap = cm.get_cmap(palette_base)
            
            cax = inset_axes(ax,
                width='3%',
                height='33%',
                loc='lower left',
                bbox_to_anchor=(1.03, 0., .3, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cb = fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap), extend='both', 
                cax=cax, orientation='vertical',
            )
            cb.set_label(self.labels[color_factor], labelpad=10, fontsize=fs_clabel)
        
        for annot in annotations:
            plt.text(*annot)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if rects:
            self.box_dfs = {}
            for i, rect in enumerate(rects):
                rect_patch = plt.Rectangle(*rect['args'], **rect['kwargs'])
                ax.add_patch(rect_patch)
                box_stats = self.box_stats(rect['date'], rect['dirxn'], *rect['args'][0], rect['factor_in_the_box'], rect['inverse'])
                self.box_dfs[i] = box_stats.pop('df')

                if rect['text']:
                    box_length = 35
                    text = 'Box {}{}'.format(i+1, ' ' * box_length)
                    text += '\n{}:{}{}'.format('# of Regions', ' ' * (box_length - 10), box_stats.pop('num_regs'))
                    text += '\n{}:{}{:.0%}'.format('% of All Deaths', ' ' * (box_length - 14), box_stats.pop('all_deaths'))
                    for statname, val in box_stats.items():
                        text += '\n{}:{}{:.{}}'.format(statname,' ' * (box_length - len(statname) - 5), val, '0%' if '%' in statname else '0f')
                    plt.text(
                        rect['x_text'], rect['y_text'], text, 
                        {'alpha': 1, 'color': 'black', 'style': 'italic', 'fontsize': 12, 'ha': rect['ha'], 'va': 'center', 'bbox':dict(facecolor=rect['kwargs']['color'], alpha=rect['alpha'], edgecolor='white')}
                    )

        if save_file:
            plt.savefig(filename, bbox_inches='tight')
        
        return plt

class BarCharts(BaseChart):

    def _data_morph_for_barcharts(self, df, regions=[]):
        first_counts = ['region_name', 'population', 'city_dens']
        first_counts += [a + b for a in [*self._casestudy.ALL_RANGES, *self._casestudy.MAJOR_CAUSES, 'gdp', 'visitors_%'] for b in ['', '_%']]
        last_counts = self._casestudy.COUNT_TYPES + [a + b for a in self._casestudy.COUNT_TYPES for b in self._casestudy.PER_APPENDS]
        max_counts = [a + '_new_dma' + b for a in self._casestudy.COUNT_TYPES for b in self._casestudy.PER_APPENDS]
        max_counts += ['uvb_dma', 'temp_dma']
        max_counts += [a + b for a in self._casestudy.STRINDEX_CATS for b in ['', '_dma']]
        mobi_counts = [a + b for a in self._casestudy.mobi_dmas.keys() for b in ['', '_dma']]
        
        groups = []
        for region_id, df_group in df.groupby('region_id'):
            groupdict = {}
            groupdict['region_id'] = region_id

            for col in df_group.columns:  
                if col in first_counts:
                    groupdict[col] = df_group[col].iloc[0]
                if col in last_counts:
                    groupdict[col] = df_group[col].iloc[-1]
                if col in max_counts:
                    groupdict[col] = df_group[col].max()
                if col in mobi_counts:
                    groupdict[col] = -df_group[col].min()
            groups.append(groupdict)

        df_comp = pd.DataFrame(groups)
        df_comp = df_comp.sort_values(by='deaths', ascending=False)
        
        df_comp = df_comp.T
        df_comp.columns = df_comp.iloc[1]
        df_comp = df_comp.drop(['region_name'])
        
        if regions:
            df_comp = df_comp[regions]
        
        return df_comp

    def make(self, factors=[], regions=[],
        title='', y_title=0.9,
        fs_xticks=12, fs_yticks=12,
        fs_xlabel=24, fs_title=16, fs_subtitle=12,
        pad_xlabel=20, pad_ylabel=12,
        annotations=[], hlines=[], hline_alpha=.1,
        colors=['red', 'magenta', 'blue'],
        width=16, height=12, save_file=False, filename=''    
    ):
        df_bcs = self.df.copy(deep=True)
        if regions:
            regions = accept_string_or_list(regions)
            df_bcs = df_bcs[df_bcs['region_name'].isin(regions)]

        regions = regions if regions else self.regions
        factors = accept_string_or_list(factors if factors else self._casestudy.factors)
        
        df_bcs = df_bcs[[*META_COLS, *[col for col in factors if col not in META_COLS]]]
        self.df_bcs = self._data_morph_for_barcharts(df_bcs)
        self.df_bcs = self.df_bcs[regions]
        
        if len(factors) > 1:
            if len(factors) % 2 != 0:
                factors.append('')
            factors = np.array(factors).reshape(-1, 2)
        else:
            factors = np.array(factors)

        fig, axs = plt.subplots(*factors.shape, figsize=(width, height))
        
        color_key = {dragon: colors[0] for dragon in COVID_DRAGONS}
        color_key['WorldAvg'] = colors[1]
        c = [color_key[region] if region in color_key.keys() else colors[2] for region in self.df_bcs.columns]
        
        x = [self._casestudy._abbreviator(region) for region in self.df_bcs.columns]
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, factor in np.ndenumerate(factors):
            if factor:
                if axs.shape == (2,):
                    i = i[1]
                y = self.df_bcs.loc[factor].tolist()
                axs[i].bar(x, y, color=c, width=.5)

                axs[i].tick_params(axis='x', labelsize=fs_xticks)
                axs[i].tick_params(axis='y', labelsize=fs_yticks)
                axs[i].set_title(self.labels[factor], fontsize=fs_subtitle)
        
        fig.suptitle(title, fontsize=fs_title, y=y_title)

        plt.subplots_adjust(hspace=0.4)

        if save_file:
            plt.savefig(filename, bbox_inches='tight')

        return plt

class ScatterFlow(BaseChart):
    def _inputs_for_scatter(self, region):
        df = self.df.copy(deep=True)
        df_reg = df[df.region_name == region][['date', 'days', *reversed(self._casestudy.STRINDEX_SUBCATS)]]
        xs = np.repeat(df_reg.days.dt.days, np.array([*reversed(self._casestudy.STRINDEX_SUBCATS)]).shape[0])
        ys = np.tile(self.subcats_keys, len(df_reg.days))
        
        zs = []
        for x in df_reg.days.values:
            counts = []
            for y in self.subcats_keys:
                count = df_reg[df_reg.days == x][self.subcats_key[y]].iloc[0]
                counts.append(count)
            zs.append(counts)
        zs = np.array(zs).flatten()

        return xs, ys, zs

    def make(self, regions=[],
        y_title=1.1, fs_title=20,
        fs_subtitle=10, fs_xlabel=16, fs_ylabel=16,
        fs_clabel=10, fs_legend=12, pad_clabel=10,
        xy_legend=(0, 0), xy_cbar=(0, 0),
        width=14, height=26,
        save_file=False, filename='',
    ):
        self.subcats_keys = np.arange(1, len(self._casestudy.STRINDEX_SUBCATS) + 1)
        self.subcats_key = {tup[0]: tup[1] for tup in zip(self.subcats_keys, reversed(self._casestudy.STRINDEX_SUBCATS))}

        if regions:
            regions = accept_string_or_list(regions)
        else:        
            regions = self.regions
        regions = copy.deepcopy(regions)

        input_key = {reg: self._inputs_for_scatter(reg) for reg in regions}
        
        # Shape grid into appropriate size for number of regions
        # Create empty plot for first box and last box
        if len(regions) > 1 and len(regions) <= 3:
            regions.insert(0, '')
            if len(regions) % 2 != 0:
                regions += ['']
            regions = np.array(regions).reshape(-1, 2)
        elif len(regions) > 3 and len(regions) <= 8: 
            regions.insert(0, '')
            while True:
                if len(regions) % 3 != 0:
                    regions += ['']
                else:
                    break
            regions = np.array(regions).reshape(-1, 3)
        elif len(regions) > 8: 
            regions.insert(0, '')
            while True:
                if len(regions) % 4 != 0:
                    regions += ['']
                else:
                    break
            regions = np.array(regions).reshape(-1, 4)
        else:
            regions = np.array(regions)
        
        fig, axs = plt.subplots(*regions.shape, figsize=(width, height))
        cmap = plt.cm.get_cmap('RdPu', 4)

        strindex_labels = [self.labels[val] for val in self.subcats_key.values()]
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, region in np.ndenumerate(regions):    
            if region:
                i = i[1] if axs.shape == (2,) else i
                xs, ys, zs = input_key[region]
                sc = axs[i].scatter(xs, ys, c=zs, vmin=zs.min(), vmax=zs.max(), cmap=cmap)
                axs[i].set_yticks(self.subcats_keys)
                axs[i].set_yticklabels(self.subcats_key.values() if regions.size > 1 else strindex_labels, fontsize=fs_ylabel)
                axs[i].set_title(region, fontsize=fs_subtitle,  fontweight='demibold')
            else:
                # Add legend in the first box
                if i == (0,0):
                    axs[i].set_title('LEGEND', 
                        {
                            'alpha': 1, 'color': 'black', 'va': 'center', 
                            'fontweight': 'demibold', 'fontsize': fs_subtitle
                        }
                    ) 
                    legend_text = ''
                    for val in reversed(list(self.subcats_key.values())):
                        legend_text += '\n{}: {}'.format(val, self.labels[val])

                    axs[i].text(*xy_legend, legend_text, 
                        {
                            'alpha': 1, 'color': 'black', 'fontsize': fs_legend, 'ha': 'left', 'va': 'center', 
                            'bbox':dict(facecolor='white', alpha=1, edgecolor='white')
                        }
                    )
                # Remove spines and other features for first and last box
                axs[i].tick_params(axis='both', which='both', length=0)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['bottom'].set_visible(False)

                plt.setp(axs[i].get_xticklabels(), visible=False)
                plt.setp(axs[i].get_yticklabels(), visible=False)

        # COLOR BAR SETUP
        if regions.size == 1:
            axis_for_color_bar = axs[0] 
            cb_ticks_pos = 'right'
            cb_label_pos = 'right'
        else:
            axis_for_color_bar = axs[0, 0]
            cb_ticks_pos = 'left'
            cb_label_pos = 'left'
        
        cax = inset_axes(axis_for_color_bar,
            width='6%',
            height='70%',
            loc='lower left',
            bbox_to_anchor=(*xy_cbar, .3, 1),
            bbox_transform=axis_for_color_bar.transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(
            sc, extend='both', 
            cax=cax, orientation='vertical',
        )
        cb.set_label('Score', labelpad=pad_clabel, fontsize=fs_clabel)
        cax.yaxis.set_ticks_position(cb_ticks_pos)
        cax.yaxis.set_label_position(cb_label_pos)

        ax = fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off', length=0)
        ax.set_title('Quarantine Evolution: Change in Oxford Stringency Policy Indicators', fontsize=fs_title, y=y_title)
        ax.set_xlabel('Days Since January 1', labelpad=20, fontsize=fs_xlabel)

        plt.subplots_adjust(left=.1, bottom=.1, right=1, top=1, wspace=0.2, hspace=0.22)
        
        if save_file:
            plt.savefig(filename, bbox_inches='tight')

        return plt

    def make_race(self, 
        indicator=None, make_sum=[], regions=[],
        title='', y_title=1.02, ylabel='',
        fs_title=16, fs_ylabel=12, fs_xlabel=14, fs_clabel=10, 
        pad_xlabel=10, pad_ylabel=10, pad_clabel=10, 
        width=8, height=6, xy_cbar=(0,0),
        annotations=[],
        save_file=False, filename='',
    ):
        regions = regions if regions else self.regions
        df = self.df.copy(deep=True)
        
        if make_sum:
            if indicator:
                raise AttributeError("You cannot provide indicator with make_sum")
            else:
                df['make_sum'] = df[make_sum].sum(axis=1)
                indicator = 'make_sum'

        df = df[['date', 'region_name', 'days', indicator]]
        df = df[df.region_name.isin(regions)]
        min_days = np.array([df_group.days.max() for i, df_group in df.groupby('region_name')]).min()
        df = df[df.days <= min_days]

        xs = np.repeat(df.days.dt.days, np.array(regions).shape[0])

        region_keys = np.arange(1, len(regions) + 1)
        ys = np.tile(region_keys, len(df.days))
        regions_key = {tup[0]: tup[1] for tup in zip(region_keys, regions)}
        
        zs = []
        for x in df.days:
            counts = []
            for y in region_keys:
                count = df[(df.days == x) & (df.region_name == regions_key[y])][indicator].iloc[0]
                counts.append(count)
            zs.append(counts)
        zs = np.array(zs).flatten()

        fig, ax = plt.subplots(figsize=(width, height))
        cmap = plt.cm.get_cmap('RdPu')

        sc = ax.scatter(xs, ys, c=zs, vmin=zs.min(), vmax=zs.max(), cmap=cmap)

        cax = inset_axes(ax,
            width='6%',
            height='50%',
            loc='lower right',
            bbox_to_anchor=(*xy_cbar, .3, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(
            sc, extend='both', 
            cax=cax, orientation='vertical',
        )
        cb.set_label('Score', labelpad=pad_clabel, fontsize=fs_clabel)
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')

        ax.set_title(title, fontsize=fs_title, y=y_title)
        ax.set_yticks(region_keys)
        ax.set_yticklabels(regions_key.values())
        ax.set_ylabel(ylabel, labelpad=pad_ylabel, fontsize=fs_ylabel)

        ax.set_xlabel('Days Since January 1', labelpad=pad_xlabel, fontsize=fs_xlabel)

        for annot in annotations:
            plt.text(*annot)

        if save_file:
            plt.savefig(filename, bbox_inches='tight')

        return plt
    