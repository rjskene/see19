from datetime import datetime as dt, timedelta
import pandas as pd

import time
import shutil

from decouple import config

import tweepy

from bokeh.io import show
from bokeh.palettes import Category20b

class TweetBot:
    CONSUMER_KEY = config('CONSUMER_KEY')
    CONSUMER_SECRET = config('CONSUMER_SECRET')

    ACCESS_TOKEN = config('ACCESS_TOKEN')
    ACCESS_SECRET = config('ACCESS_SECRET')

    CHARTPATH = config('CHARTPATH')

    def __init__(self, test=False, wait=True, save_file=True, twitter_height_for_bokeh=325, twitter_height_for_matplotlib=3):
        self.auth = tweepy.OAuthHandler(self.CONSUMER_KEY, self.CONSUMER_SECRET)
        self.auth.set_access_token(self.ACCESS_TOKEN, self.ACCESS_SECRET)
        
        self.test=test
        self.tweet = not self.test
        self.save_file = save_file
        self.wait = self.tweet
        
        # Store the filenames in the tweet bot and wipe them after every net tweet
        self.filenames = []

        self.now = dt.now()
        self.cs_date = (dt(self.now.year, self.now.month, self.now.day) - timedelta(1))
        self.day = (self.cs_date).strftime('%b %d')
        
        self.twitter_height_for_bokeh = twitter_height_for_bokeh
        self.twitter_height_for_mpl = twitter_height_for_matplotlib
        
    def new_tweet(self, status, tweetid=None, wait_time=15):
        auto_populate_reply_metadata = True if tweetid else False

        api = tweepy.API(self.auth)
        media_ids = []
        
        for filename in self.filenames:
             res = api.media_upload(filename)
             media_ids.append(res.media_id)

        hashtags = ' #COVID19 #COVID #coronavirus #C19'
        status += hashtags

        if self.tweet:
            api.update_status(status=status, media_ids=media_ids, in_reply_to_status_id = tweetid , auto_populate_reply_metadata=True)
        
        for filename in self.filenames:
            shutil.move(filename, self.CHARTPATH + 'old/' + filename.split(self.CHARTPATH)[1])
        
        # Wipe filenames to prepare for next tweet
        self.filenames = []
        
        if self.wait:
            time.sleep(60*wait_time)
            print ('Waiting {} mins'.format(str(wait_time)))
        
    def count_comparison(self, casestudy, count_type, regions):
        title = 'Comparing New {} as of {}'.format(count_type.title() if count_type == 'cases' else 'Fatalities', self.day)
        filename = self.CHARTPATH + title + '.png'
        kwargs ={    
            'title': title,
            'regions': regions,
            'comp_category': '{}_new_dma_per_1M'.format(count_type),
            'x_fontsize': 11, 
            'y_fontsize': 11,
            'fs_xticks': 8,
            'fs_yticks': 8,
            'fs_labels': 16,
            'fs_legend': 10,
            'legend': True,
            'legend_location': 'top_left',
            'legend_title': '',
            'height': self.twitter_height_for_bokeh,
            'width': 525,
            'bg_color': '#6883BA', 'bg_alpha': .02,
            'border_color': '#6883BA', 'border_alpha': .05,
            'multiline_labels': None,
            'save_file': self.save_file,
            'filename': filename,
        }

        if 'Japan' in regions and count_type != 'deaths':
            kwargs['legend_location'] = 'top_center'
        
        p = casestudy.comp_chart.make(comp_type='multiline', **kwargs)
        show(p)

        self.filenames.append(filename)
    
    def amobi_comparison(self, casestudy, regions):
        title = 'Apple Driving Queries as of {}'.format(self.day)
        filename = '/Users/spindicate/Documents/docs/covid19/charts/{} as of {}.png'.format(title, self.day)

        kwargs ={    
            'title': title,
            'regions': regions,
            'comp_category': 'driving_apple',
            'x_fontsize': 11, 
            'y_fontsize': 11,
            'fs_xticks': 8,
            'fs_yticks': 8,
            'fs_labels': 16,
            'fs_legend': 10,
            'bg_color': '#6883BA', 'bg_alpha': .02,
            'border_color': '#6883BA', 'border_alpha': .05,
            'multiline_labels': False,
            'legend': True,
            'legend_title': False,
            'legend_location': 'top_center',
            'height': self.twitter_height_for_bokeh,
            'width': 500,
            'save_file': self.save_file,
            'filename': filename
        }
        p = casestudy.comp_chart.make(comp_type='multiline', **kwargs)
        show(p)
        
        self.filenames.append(filename)
        
    def multiline_comparison(self, casestudy, comp_category, regions, title):
        filename = self.CHARTPATH + title + '.png'      
        legend = True if regions else False
        kwargs ={    
            'title': title,
            'regions': regions,
            'comp_category': comp_category,
            'legend_title': '',
            'palette_base': Category20b[20],
            'x_fontsize': 14, 
            'y_fontsize': 14,
            'fs_xticks': 8,
            'fs_yticks': 10,
            'fs_labels': 12,
            'height': self.twitter_height_for_bokeh,
            'width': 525,
            'h_legend': 10, 'w_legend': 10,
            'bg_color': '#6883BA', 'bg_alpha': .02,
            'border_color': '#6883BA', 'border_alpha': .05,
            'comp_type': 'multiline',
            'legend': legend,
            'legend_location': 'top_left',
            'multiline_labels': False,
            'save_file': self.save_file,
            'filename': filename,
        }

        p = casestudy.comp_chart.make(**kwargs)
        show(p)
        self.filenames.append(filename)
    
    def scatterflow(self, casestudy, comp_category, regions, title):
        filename = self.CHARTPATH + title + '.png'
        kwargs = {
            'regions': regions,
            'title': title,
            'y_title': 1.012,
            'fs_title': 12,
            'marker': 's',
            'ms': 225,
            'comp_category': comp_category,
            'width': 5, 
            'height': 4,
            'fs_ylabel': 7, 
            'fs_yticks': 8,
            'pad_clabel': 8, 
            'fs_clabel': 10, 
            'xy_cbar': (.62, .24),
            'w_cbar': .45,
            'h_cbar': 1.5,
            'save_file': self.save_file,
            'filename': filename,
        }
        plt = casestudy.scatterflow.make_race(**kwargs)

        plt.show()
        self.filenames.append(filename)
        
    def wavefinder(self, df, comp_category, key):
        """
        Separates regions into 3 categories based on count curve: Peak, Sustained, Weak
        """
        end_date = self.cs_date

        regions_str = 'Brazil' if key == 'BRA' else 'USA, CAD, AUS, Europe and Asia'
        start_date = end_date - timedelta(days=7)
        stretch_date = end_date - timedelta(days=21)
        close_date = end_date - timedelta(days=3)

        wave_types = {}
        wave_types['peak_wave'] = []
        wave_types['sustained_wave'] = []
        wave_types['weak_wave'] = []
        for region_name, df_g in df.groupby('region_name'):
            end_comp = df_g[df_g.date <= end_date][comp_category].values[-1]
            start_comp = df_g[df_g.date == start_date][comp_category].values[0]
            stretch_comp = df_g[df_g.date == stretch_date][comp_category].values[0]
            close_comp = df_g[df_g.date == close_date][comp_category].values[0]
            test_med = end_comp / start_comp
            test_far = end_comp / stretch_comp
            test_close = end_comp / close_comp

            if test_med > 1:
                wave_types['weak_wave'].append(region_name)

            if test_med > 1 and test_far > 1 and test_close > 1:
                wave_types['sustained_wave'].append(region_name)

            if end_comp >= df_g[comp_category].max():
                wave_types['peak_wave'].append(region_name)

        return wave_types, regions_str, end_date
    
    def positivity_bar(self, casestudy, factors=['cases_new_dma_per_test_new_dma']):
        if casestudy.df.date.max() != self.cs_date:
            raise AssertionError('The casestudy object does not have data for as of the required date {}'.format(self.cs_date.strftime('%b %d, %Y')))

        title = 'Daily New Cases Per Test ({}DMA) in All US States as of {}'.format(casestudy.count_dma, self.day)
        filename = self.CHARTPATH + title + '.png'
        kwargs = {
            'factors': factors, 'height': self.twitter_height_for_mpl, 'width': 16, 
            'colors': ['#D4AFB9', '#3D7068', '#529FD7'],
            'title': title, 'y_title': .86, 'fs_title': 12, 'fs_yticks': 10, 'fs_xticks': 10,
            'subtitles': '',
            'sort_cols': factors,
            'as_of': self.cs_date,
            'hlines': [{'y_hline':.1, 'color': 'red', 'alpha': .8}]
        }
        kwargs['save_file'] = self.save_file
        kwargs['filename'] = filename

        plt = casestudy.barcharts.make(**kwargs)
        self.filenames.append(filename)
        
    def positivity_race(self, casestudy):
        us_regions = casestudy.df[casestudy.df.country_code == 'USA'].sort_values(by='cases', ascending=False).region_name.unique().tolist()[:20]
        us_regions = [reg for reg in us_regions if reg != 'Georgia']
        filename = self.CHARTPATH + 'Positivity Ratio in US States.png'
        kwargs = {
            'regions': us_regions,
            'title': 'Positivity Ratio in US States',
            'y_title': 1.012,
            'fs_title': 12,
            'marker': 's',
            'ms': 100,
            'comp_category': 'cases_new_dma_per_test_new_dma',
            'width': 3, 
            'height': self.twitter_height_for_mpl,
            'fs_ylabel': 7, 
            'fs_yticks': 7,
            'pad_clabel': 8, 
            'fs_clabel': 8, 
            'xy_cbar': (.685, .24),
            'w_cbar': .4,
            'h_cbar': 1.5,
            'save_file': self.save_file,
            'filename': filename,
        }
        plt = casestudy.scatterflow.make_race(**kwargs)
        plt.show()
        self.filenames.append(filename)
        
    def strindex_race(self, casestudy):
        filename = self.CHARTPATH + 'Strindex update.png'
        kwargs = {
            'title': 'Oxford Stringency Index',
            'y_title': 1.012,
            'fs_title': 12,
            'marker': 's',
            'ms': 225,
            'comp_category': 'strindex',
            'width': 5, 
            'height': 4,
            'fs_ylabel': 7, 
            'fs_yticks': 10,
            'pad_clabel': 8, 
            'fs_clabel': 10, 
            'xy_cbar': (.50, .24),
            'w_cbar': .6,
            'h_cbar': 1.5,
            'save_file': self.save_file,
            'filename': filename,
        }
        plt = casestudy.scatterflow.make_race(**kwargs)
        plt.show()
        self.filenames.append(filename)