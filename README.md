# see19

**An aggregation dataset and interface for visualizing and analyzing the epidemiology of Coronavirus Disease 2019 aka SARS-CoV-2 aka COVID19 aka C19**

***
# Latest Analysis
* [How Effective Is Social Distancing?](https://nbviewer.jupyter.org/github/ryanskene/see19/blob/master/notebooks/analysis/See19%20-%20How%20Effective%20Is%20Social%20Distancing%3F.ipynb)
* [What Factors Are Really Impacting C19 Virulence?]()

***
# The Dataset
The dataset is in `csv` format and can be found [here](https://github.com/ryanskene/see19/tree/master/dataset)

You can find relevant statistics and detailed sourcing in the **[Guide](https://nbviewer.jupyter.org/github/ryanskene/see19/blob/master/notebooks/guide/See19%20Guide.ipynb)**

# The Package

the `see19` package is available on [pypi](https://pypi.org/project/see19/) and can be installed as follows:

`pip install see19`

The package provides a helpful `pandas`-based interface for working with the dataset. It also provides several visualization tools 

# The Guide
The **[Guide](https://nbviewer.jupyter.org/github/ryanskene/see19/blob/master/notebooks/guide/See19%20Guide.ipynb)** details data sources, structure, functionality, and visualization tools.

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
<div align="center"><b>I AM NOT A PROFESSIONAL OR AN ACADEMIC. CALL ME AN AMATEUR "ENTHUSIAST". THIS IS A SOLO PROJECT UNTIL NOW. <br/>I AM SURE THERE ARE MISTAKES. PLEASE FLAG ANY ISSUES YOU SEE!</b></div>

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

Then simple code:


```python
# For displaying bokeh in markdown
from pweave.bokeh import output_pweave, show
```


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



<script type="text/javascript">
    (function() {
          var fn = function() {
            Bokeh.safely(function() {
              (function(root) {
                function embed_document(root) {

                var docs_json = '{&quot;2b642d41-9169-4666-8981-6afed81d0623&quot;:{&quot;roots&quot;:{&quot;references&quot;:[{&quot;attributes&quot;:{&quot;text&quot;:&quot;Comparison of Daily Fatalities as of May 03&quot;},&quot;id&quot;:&quot;1083&quot;,&quot;type&quot;:&quot;Title&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1105&quot;,&quot;type&quot;:&quot;HelpTool&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1127&quot;,&quot;type&quot;:&quot;UnionRenderers&quot;},{&quot;attributes&quot;:{&quot;below&quot;:[{&quot;id&quot;:&quot;1092&quot;}],&quot;center&quot;:[{&quot;id&quot;:&quot;1095&quot;},{&quot;id&quot;:&quot;1099&quot;},{&quot;id&quot;:&quot;1119&quot;},{&quot;id&quot;:&quot;1120&quot;}],&quot;left&quot;:[{&quot;id&quot;:&quot;1096&quot;}],&quot;min_border&quot;:0,&quot;plot_height&quot;:500,&quot;plot_width&quot;:750,&quot;renderers&quot;:[{&quot;id&quot;:&quot;1117&quot;}],&quot;title&quot;:{&quot;id&quot;:&quot;1083&quot;},&quot;toolbar&quot;:{&quot;id&quot;:&quot;1106&quot;},&quot;toolbar_location&quot;:null,&quot;x_range&quot;:{&quot;id&quot;:&quot;1085&quot;},&quot;x_scale&quot;:{&quot;id&quot;:&quot;1088&quot;},&quot;y_range&quot;:{&quot;id&quot;:&quot;1081&quot;},&quot;y_scale&quot;:{&quot;id&quot;:&quot;1090&quot;}},&quot;id&quot;:&quot;1082&quot;,&quot;subtype&quot;:&quot;Figure&quot;,&quot;type&quot;:&quot;Plot&quot;},{&quot;attributes&quot;:{&quot;bottom_units&quot;:&quot;screen&quot;,&quot;fill_alpha&quot;:0.5,&quot;fill_color&quot;:&quot;lightgrey&quot;,&quot;left_units&quot;:&quot;screen&quot;,&quot;level&quot;:&quot;overlay&quot;,&quot;line_alpha&quot;:1.0,&quot;line_color&quot;:&quot;black&quot;,&quot;line_dash&quot;:[4,4],&quot;line_width&quot;:2,&quot;render_mode&quot;:&quot;css&quot;,&quot;right_units&quot;:&quot;screen&quot;,&quot;top_units&quot;:&quot;screen&quot;},&quot;id&quot;:&quot;1126&quot;,&quot;type&quot;:&quot;BoxAnnotation&quot;},{&quot;attributes&quot;:{&quot;text&quot;:&quot;Spain&quot;,&quot;text_alpha&quot;:0.6,&quot;text_color&quot;:&quot;#20908C&quot;,&quot;text_font_size&quot;:&quot;8pt&quot;,&quot;x&quot;:60,&quot;x_offset&quot;:-15,&quot;y&quot;:7.1402848663214495,&quot;y_offset&quot;:10},&quot;id&quot;:&quot;1120&quot;,&quot;type&quot;:&quot;Label&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1128&quot;,&quot;type&quot;:&quot;Selection&quot;},{&quot;attributes&quot;:{&quot;axis&quot;:{&quot;id&quot;:&quot;1096&quot;},&quot;dimension&quot;:1,&quot;ticker&quot;:null},&quot;id&quot;:&quot;1099&quot;,&quot;type&quot;:&quot;Grid&quot;},{&quot;attributes&quot;:{&quot;line_alpha&quot;:{&quot;value&quot;:0.1},&quot;line_color&quot;:{&quot;field&quot;:&quot;color&quot;},&quot;line_width&quot;:{&quot;value&quot;:5},&quot;xs&quot;:{&quot;field&quot;:&quot;x&quot;},&quot;ys&quot;:{&quot;field&quot;:&quot;y&quot;}},&quot;id&quot;:&quot;1116&quot;,&quot;type&quot;:&quot;MultiLine&quot;},{&quot;attributes&quot;:{&quot;source&quot;:{&quot;id&quot;:&quot;1113&quot;}},&quot;id&quot;:&quot;1118&quot;,&quot;type&quot;:&quot;CDSView&quot;},{&quot;attributes&quot;:{&quot;axis_label&quot;:&quot;Daily Deaths per 1M (3DMA)&quot;,&quot;formatter&quot;:{&quot;id&quot;:&quot;1125&quot;},&quot;ticker&quot;:{&quot;id&quot;:&quot;1097&quot;}},&quot;id&quot;:&quot;1096&quot;,&quot;type&quot;:&quot;LinearAxis&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1093&quot;,&quot;type&quot;:&quot;BasicTicker&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1101&quot;,&quot;type&quot;:&quot;WheelZoomTool&quot;},{&quot;attributes&quot;:{&quot;data_source&quot;:{&quot;id&quot;:&quot;1113&quot;},&quot;glyph&quot;:{&quot;id&quot;:&quot;1115&quot;},&quot;hover_glyph&quot;:null,&quot;muted_glyph&quot;:null,&quot;nonselection_glyph&quot;:{&quot;id&quot;:&quot;1116&quot;},&quot;selection_glyph&quot;:null,&quot;view&quot;:{&quot;id&quot;:&quot;1118&quot;}},&quot;id&quot;:&quot;1117&quot;,&quot;type&quot;:&quot;GlyphRenderer&quot;},{&quot;attributes&quot;:{&quot;axis&quot;:{&quot;id&quot;:&quot;1092&quot;},&quot;grid_line_color&quot;:null,&quot;ticker&quot;:null},&quot;id&quot;:&quot;1095&quot;,&quot;type&quot;:&quot;Grid&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1125&quot;,&quot;type&quot;:&quot;BasicTickFormatter&quot;},{&quot;attributes&quot;:{&quot;active_drag&quot;:&quot;auto&quot;,&quot;active_inspect&quot;:&quot;auto&quot;,&quot;active_multi&quot;:null,&quot;active_scroll&quot;:&quot;auto&quot;,&quot;active_tap&quot;:&quot;auto&quot;,&quot;tools&quot;:[{&quot;id&quot;:&quot;1100&quot;},{&quot;id&quot;:&quot;1101&quot;},{&quot;id&quot;:&quot;1102&quot;},{&quot;id&quot;:&quot;1103&quot;},{&quot;id&quot;:&quot;1104&quot;},{&quot;id&quot;:&quot;1105&quot;}]},&quot;id&quot;:&quot;1106&quot;,&quot;type&quot;:&quot;Toolbar&quot;},{&quot;attributes&quot;:{&quot;overlay&quot;:{&quot;id&quot;:&quot;1126&quot;}},&quot;id&quot;:&quot;1102&quot;,&quot;type&quot;:&quot;BoxZoomTool&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1088&quot;,&quot;type&quot;:&quot;LinearScale&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1085&quot;,&quot;type&quot;:&quot;DataRange1d&quot;},{&quot;attributes&quot;:{&quot;data&quot;:{&quot;color&quot;:[&quot;#440154&quot;,&quot;#20908C&quot;],&quot;regions&quot;:[&quot;Germany&quot;,&quot;Spain&quot;],&quot;x&quot;:[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]],&quot;y&quot;:[[0.008275176941384957,0.008275176941384957,0.012412765412077434,0.004137588470692478,0.020687942353462392,0.024825530824154868,0.033100707765539826,0.041375884706924784,0.062063827060387176,0.07033900400177213,0.1117148887086969,0.17791630423977656,0.2317049543587788,0.2068794235346239,0.2317049543587788,0.30204395836055087,0.4634099087175576,0.5958127397797168,0.7654538670781085,0.9392325828471926,1.1005985332041992,1.253689306619821,1.4150552569768275,1.6012467381579891,1.9115658734599248,2.068794235346239,2.1680963586428583,1.9736297005203123,2.213609831820476,2.3667006052360975,3.1652551800797455,3.2976580111419054,3.107328941490051,1.6012467381579891,1.7170992153373785,1.7667502769856882,2.3087743666464027,3.235594184081518,3.5500509078541462,4.3775686019926425,2.7101204483035732,2.2094722433497833,2.110170120053164,2.374975782177483,2.8673488101898874,2.9501005796037365,3.0080268181934318,2.474277905474102,1.6591729767476835,1.5143573802734471,1.808126161692613,2.0315559391100066,2.056381469934162,1.7460623346322257,1.427468022388905,1.0054339983782723],[0.007219701583742618,0.014439403167485236,0.021659104751227856,0.02887880633497047,0.05775761266994094,0.10107582217239666,0.16605313642608024,0.18049253959356548,0.26712895859847685,0.1949319427610507,0.7075307552067764,1.017977923307709,1.6894101705957727,1.5089176310022074,2.440259135305005,2.4113803289700346,3.5232143728663976,3.6820478077087353,5.429215590974448,6.800958891885546,9.154581608185639,10.345832369503173,13.536940469517408,14.829267053007337,16.8219046901203,16.858003198039015,17.601632461164503,18.61239068288847,17.919299330849178,18.655708892390926,19.002254568410574,19.73866412995232,18.482436054381104,16.554775731521826,15.471820493960431,15.146933922692012,15.529578106630373,15.204691535361954,14.69931242449997,13.096538672909109,12.721114190554495,12.093000152768886,10.468567296426796,10.822332674030186,11.255514769054741,14.049539281963135,9.638301614296395,8.2160204022991,6.1367463461812255,8.945210262257104,9.125702801850668,9.421710566784116,8.966869367008332,8.555346376735002,7.457951736006124,7.19804247899139,6.64212545704321,7.833376218360741,7.378535018584956,5.205404841878428,7.1402848663214495]]},&quot;selected&quot;:{&quot;id&quot;:&quot;1128&quot;},&quot;selection_policy&quot;:{&quot;id&quot;:&quot;1127&quot;}},&quot;id&quot;:&quot;1113&quot;,&quot;type&quot;:&quot;ColumnDataSource&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1097&quot;,&quot;type&quot;:&quot;BasicTicker&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1103&quot;,&quot;type&quot;:&quot;SaveTool&quot;},{&quot;attributes&quot;:{&quot;axis_label&quot;:&quot;Days Since 1 Death&quot;,&quot;formatter&quot;:{&quot;id&quot;:&quot;1123&quot;},&quot;major_tick_line_color&quot;:null,&quot;ticker&quot;:{&quot;id&quot;:&quot;1093&quot;}},&quot;id&quot;:&quot;1092&quot;,&quot;type&quot;:&quot;LinearAxis&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1100&quot;,&quot;type&quot;:&quot;PanTool&quot;},{&quot;attributes&quot;:{&quot;text&quot;:&quot;Germany&quot;,&quot;text_alpha&quot;:0.6,&quot;text_color&quot;:&quot;#440154&quot;,&quot;text_font_size&quot;:&quot;8pt&quot;,&quot;x&quot;:55,&quot;x_offset&quot;:-12,&quot;y&quot;:1.0054339983782723,&quot;y_offset&quot;:13},&quot;id&quot;:&quot;1119&quot;,&quot;type&quot;:&quot;Label&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1104&quot;,&quot;type&quot;:&quot;ResetTool&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1090&quot;,&quot;type&quot;:&quot;LinearScale&quot;},{&quot;attributes&quot;:{&quot;line_color&quot;:{&quot;field&quot;:&quot;color&quot;},&quot;line_width&quot;:{&quot;value&quot;:5},&quot;xs&quot;:{&quot;field&quot;:&quot;x&quot;},&quot;ys&quot;:{&quot;field&quot;:&quot;y&quot;}},&quot;id&quot;:&quot;1115&quot;,&quot;type&quot;:&quot;MultiLine&quot;},{&quot;attributes&quot;:{&quot;end&quot;:20.725597336449937,&quot;start&quot;:0.004137588470692478},&quot;id&quot;:&quot;1081&quot;,&quot;type&quot;:&quot;Range1d&quot;},{&quot;attributes&quot;:{},&quot;id&quot;:&quot;1123&quot;,&quot;type&quot;:&quot;BasicTickFormatter&quot;}],&quot;root_ids&quot;:[&quot;1082&quot;]},&quot;title&quot;:&quot;Bokeh Application&quot;,&quot;version&quot;:&quot;2.0.0&quot;}}';
                var render_items = [{"docid":"2b642d41-9169-4666-8981-6afed81d0623","root_ids":["1082"],"roots":{"1082":"4e93fb9d-8fa0-4f22-a1b9-68e0fb36af40"}}];
                root.Bokeh.embed.embed_items(docs_json, render_items);

                }
                if (root.Bokeh !== undefined) {
                  embed_document(root);
                } else {
                  var attempts = 0;
                  var timer = setInterval(function(root) {
                    if (root.Bokeh !== undefined) {
                      clearInterval(timer);
                      embed_document(root);
                    } else {
                      attempts++;
                      if (attempts > 100) {
                        clearInterval(timer);
                        console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                      }
                    }
                  }, 10, root)
                }
              })(window);
            });
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
</script>
<div class="bk-root" id="4e93fb9d-8fa0-4f22-a1b9-68e0fb36af40" data-root-id="1082"></div>



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


![png](README_files/README_12_0.png)

