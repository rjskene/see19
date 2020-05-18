# see19

**An aggregation dataset and interface for visualizing and analyzing the epidemiology of Coronavirus Disease 2019 aka SARS-CoV-2 aka COVID19 aka C19**

***
***NOTE ON THE TEST DATASET***

As of May 15, a `testset` folder has been added to the master. Going forward, the testset will include new data  (either additional factors or new regions) that has not yet been incorporated into the `see19` interface. The goal is to integrate the new data into the interface over time. The `testset` will be update concurrently with the main dataset.

The existing `see19` package will ***NOT*** be compatiable with the testset, **HOWEVER** you can download the `testset` via `get_baseframe` by setting `test=True`.

Data currently available only in the testset:

* [Apple Mobility index](https://www.apple.com/covid19/mobility)

* Test counts
    * [Country level](https://data.humdata.org/dataset/total-covid-19-tests-performed-by-country)
    * [Italy](https://github.com/pcm-dpc/COVID-19)
        * **NOTE:** Italian testing has two categories that complicate the data somewhat
            * `tamponi` refers to swabs. Swabs have been recorded since very early on. There are generally multiple swabs per individual whereas most test counts are one test per individual.
            * `casi_testati` refers to the more standard one test per person. This metric was not reliably tract before mid-April
            * for metrics prior to mid-April, `see19` adjusts the `tamponi` counts by finding the average `tamponi` per `case_testati` across the all data then dividing the tampons by the average to estimate casi_testati
    * [Australia](https://services1.arcgis.com/vHnIGBHHqDR6y0CR/arcgis/rest/services/COVID19_Time_Series/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json)
    * [Canada](https://www.canada.ca/en/public-health/services/diseases/2019-novel-coronavirus-infection.html)
    * [United States](https://covidtracking.com/)


**NOTE:** Brazil is not included in the `tests` count data currently. Brazil test counts are only available on the country level whereas case and fatality data is available on a regional level. I am exploring methodsto allocate aggregate tests among the regions (perhaps simply as percentage of population or cases counts).

***
# Latest Analysis
[How Effective Is Social Distancing?](https://ryanskene.github.io/see19/analysis/How%20Effective%20Is%20Social%20Distancing%3F.html)

[What Factors Are Correlated With COVID19 Fatality Rates?](https://ryanskene.github.io/see19/analysis/What%20Factors%20Are%20Correlated%20With%20COVID19%20Fatality%20Rates%3F.html)

[The COVID Dragons](https://ryanskene.github.io/see19/analysis/The%20COVID%20Dragons.html)

***
# The Dataset
The dataset is in `csv` format and can be found [here](https://github.com/ryanskene/see19/tree/master/dataset)

You can find relevant statistics and detailed sourcing in the **[Guide](https://ryanskene.github.io/see19/)**

# The Package

the `see19` package is available on [pypi](https://pypi.org/project/see19/) and can be installed as follows:

`pip install see19`

The package provides a helpful `pandas`-based interface for working with the dataset. It also provides several visualization tools 

# The Guide
The **[Guide](https://ryanskene.github.io/see19/)** details data sources, structure, functionality, and visualization tools.

***
# Purpose

##### _"It is better to be vaguely right than exactly wrong."_   

_- Carveth Read, Logic, Chapter 22_

<br/>

**see19** is an early stage attempt to aggregate various data sources and analyze their impact (together and in isolation) on the virulence of SARS-CoV2.

* Ease-of-use is paramount, thus, all data from all sources have been compiled into a single structure, readily consumed and manipulated in the ubiquitous `csv` format.

**see19** aggregates the following data:

* COVID19 Data Characteristics:
    * Cumulative Cases for each region on each date
    * Cumulative Fatalities for each region on each date
    * State / Provincial-level data available for
* Factor Data Characteristics available for most regions include:
    * Longitude / Latitude, Population, Demographic Segmentation, Density
    * Climate Characteristics including temperatue and uvb radiation
    * Historical Health Outcomes
    * Travel Popularity
    * Social Distancing Implementation
    * And more and counting ...

There is no single all-encompassing data from an undoubted source that will serve the needs of every user for every use case. Thus, the dataset as it stands is an ad-hoc aggregation from multiple sources with *eyeball*-style approximations used in some instances. But while the dataset's imperfections are numerous, they cannot blunt the power of the insights that can be gleaned from an early exploratory analysis.

In addition to the dataset, `see19` is a python package that provides:
* Helpful `pandas`-based interface for manipulating the data
* Visualization tools in `bokeh` and `matplotlib` to compare factors across multiple dimensions ..
* Statistical analysis is also a goal of the project and I expect to add such analysis tools as time progresses. Until then, the data is available for all.

<br/>
<div align="center"><b> THIS IS A SOLO PROJECT. <br/>I FIND ERRORS WITHIN THE DATA REGULARLY. PLEASE FLAG ANY ISSUES YOU SEE!</b></div>

***
# Suggestions For Additional Data

I am always on the hunt for new additions to the dataset. If you have any suggestions, please contact me. Specifically, if you are aware of any datasets that might integrate nicely with `see19` in the following realms:

1. German daily, state-level case and fatality data
2. Russian daily, state-level case and fatality data
3. State or city level travel data
4. Global Commercial Airline route data (there seems to be plenty available, except only for a whopping price)

***
# Quick Demo

You can very quickly use see19 to develop visuals for COVID19 analysis and presentation.

The `see19` package can be installed via `pip`.

`pip install see19`

Then simply:


```python
# Required to use Bokeh with Jupyter notebooks
from bokeh.io import output_notebook, show
output_notebook()
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="1098">Loading BokehJS ...</span>
</div>





```python
from see19 import get_baseframe, CaseStudy
baseframe = get_baseframe()
```


```python
regions = ['Germany', 'Spain']
casestudy = CaseStudy(baseframe, regions=regions, count_categories='deaths_new_dma_per_1M')

label_offsets = {'Germany': {'x_offset': 8, 'y_offset': 8}, 'Spain': {'x_offset': 5, 'y_offset': 5}}  
p = casestudy.comp_chart.make(comp_type='multiline', label_offsets=label_offsets, width=750)

show(p)
```








<div class="bk-root" id="2cb9eebd-d32e-4a12-9471-f8c7f7499755" data-root-id="1100"></div>





![Bokeh](README_files/bokeh.png)


```python
%matplotlib inline
```


```python
regions = list(baseframe[baseframe['country'] == 'Brazil'] \
    .sort_values(by='population', ascending=False) \
    .region_name.unique())[:20]

casestudy = CaseStudy(
    baseframe, count_dma=5, 
    factors=['temp'],
    regions=regions, start_hurdle=10, start_factor='cases', lognat=True,
)
kwargs = {
    'color_factor': 'temp',
    'fs_xticks': 16, 'fs_yticks': 12, 'fs_zticks': 12,
    'fs_xlabel': 12, 'fs_ylabel': 18, 'fs_zlabel': 18,
    'title': 'Daily Deaths in Brazil as of May 2',
    'x_title': 0.499, 'y_title': 0.738, 'fs_title': 22, 'rot_title': -9.5,
    'x_colorbar': 0.09, 'y_colorbar': .225, 'h_colorbar': 20, 'w_colorbar': .01, 
    'a_colorbar': 'vertical', 'cb_labelpad': -57,
    'tight': True, 'abbreviate': 'first', 'comp_size': 10,
}
p = casestudy.comp_chart4d.make(comp_category='deaths_new_dma_per_1M', **kwargs)
```


![png](output_14_0.png)

