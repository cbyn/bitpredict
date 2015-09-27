import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, output_server, push
from bokeh.models.formatters import DatetimeTickFormatter, PrintfTickFormatter
from bokeh import embed
from json import load
from urllib2 import urlopen
import time

client = pymongo.MongoClient()
db = client['bitmicro']
collection = db['btc_performance']


def get_data():
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    timestamps = pd.to_datetime(data.index, unit='s').to_series()
    returns = data.returns.cumsum()*100
    return timestamps, returns

timestamps, returns = get_data()
output_server('bitpredict_performance')

background = '#f2f2f2'
ylabel_standoff = 0
xformatter = DatetimeTickFormatter(formats=dict(hours=["%H:%M"]))
yformatter = PrintfTickFormatter(format="%8.1f")

p = figure(title=None,
           plot_width=750,
           plot_height=500,
           x_axis_type='datetime',
           min_border_top=5,
           min_border_bottom=10,
           background_fill=background,
           x_axis_label='Date',
           tools='',
           toolbar_location=None)
p.line(x=timestamps,
       y=returns,
       name='returns',
       color='#8959a8',
       line_width=1,
       legend='Cumulative Return',
       line_cap='round',
       line_join='round')
p.legend.orientation = 'top_left'
p.legend.border_line_color = background
p.outline_line_color = None
p.xgrid.grid_line_color = 'white'
p.ygrid.grid_line_color = 'white'
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.minor_tick_line_color = None
p.yaxis.axis_label = 'Percent'
p.yaxis.axis_label_standoff = ylabel_standoff
p.xaxis.formatter = xformatter
p.yaxis.formatter = yformatter
p.xaxis.axis_label_standoff = 12
p.yaxis.major_label_text_font = 'courier'
p.xaxis.major_label_text_font = 'courier'

push()
ip = load(urlopen('http://jsonip.com'))['ip']
ssn = cursession()
ssn.publish()
tag = embed.autoload_server(p, ssn, public=True).replace('localhost', ip)
html = """
{%% extends "layout.html" %%}
{%% block bokeh %%}
%s
{%% endblock %%}
""" % tag

with open('templates/performance.html', 'w+') as f:
    f.write(html)

renderer = p.select(dict(name='returns'))
ds_returns = renderer[0].data_source

while True:
    timestamps, returns = get_data()
    ds_returns.data['x'] = timestamps
    ds_returns.data['y'] = returns
    ssn.store_objects(ds_returns)
    time.sleep(300)
