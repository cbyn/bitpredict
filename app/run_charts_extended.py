import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, output_server, push
from bokeh.models.formatters import DatetimeTickFormatter, PrintfTickFormatter
from bokeh.io import vplot
from bokeh import embed
from json import load
from urllib2 import urlopen
import time

client = pymongo.MongoClient()
db = client['bitmicro']
collection = db['btc_predictions']


def get_data():
    cursor = collection.find().limit(3*60*60).sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    timestamps = pd.to_datetime(data.index, unit='s').to_series()
    prices = data.price
    predictions = data.prediction*10000
    returns = (data.position*data.change).cumsum()*10000
    return timestamps, prices, predictions, returns

timestamps, prices, predictions, returns = get_data()
output_server('bitpredict_extended')

background = '#f2f2f2'
ylabel_standoff = 0
xformatter = DatetimeTickFormatter(formats=dict(hours=["%H:%M"]))
yformatter = PrintfTickFormatter(format="%8.1f")
p1 = figure(title=None,
            plot_width=750,
            plot_height=300,
            x_axis_type='datetime',
            min_border_top=10,
            min_border_bottom=33,
            background_fill=background,
            tools='',
            toolbar_location=None)
p1.line(x=timestamps,
        y=prices,
        name='prices',
        color='#4271ae',
        line_width=1,
        legend='Bitcoin Bid/Ask Midpoint',
        line_cap='round',
        line_join='round')
p1.legend.orientation = 'top_left'
p1.legend.border_line_color = background
p1.outline_line_color = None
p1.xgrid.grid_line_color = 'white'
p1.ygrid.grid_line_color = 'white'
p1.axis.axis_line_color = None
p1.axis.major_tick_line_color = None
p1.axis.minor_tick_line_color = None
p1.yaxis.axis_label = 'Price'
p1.yaxis.axis_label_standoff = ylabel_standoff
p1.xaxis.formatter = xformatter
p1.yaxis.formatter = PrintfTickFormatter(format='%8.2f')
p1.yaxis.major_label_text_font = 'courier'
p1.xaxis.major_label_text_font = 'courier'

p2 = figure(title=None,
            plot_width=750,
            plot_height=295,
            x_axis_type='datetime',
            min_border_top=5,
            min_border_bottom=33,
            background_fill=background,
            tools='',
            toolbar_location=None)
p2.line(x=timestamps,
        y=predictions,
        name='predictions',
        color='#c82829',
        line_width=1,
        legend='30 Second Prediction',
        line_cap='round',
        line_join='round')
p2.legend.orientation = 'top_left'
p2.legend.border_line_color = background
p2.outline_line_color = None
p2.xgrid.grid_line_color = 'white'
p2.ygrid.grid_line_color = 'white'
p2.axis.axis_line_color = None
p2.axis.major_tick_line_color = None
p2.axis.minor_tick_line_color = None
p2.yaxis.axis_label = 'Basis Points'
p2.yaxis.axis_label_standoff = ylabel_standoff
p2.xaxis.formatter = xformatter
p2.yaxis.formatter = yformatter
p2.yaxis.major_label_text_font = 'courier'
p2.xaxis.major_label_text_font = 'courier'
p2.x_range = p1.x_range

p3 = figure(title=None,
            plot_width=750,
            plot_height=320,
            x_axis_type='datetime',
            min_border_top=5,
            min_border_bottom=10,
            background_fill=background,
            x_axis_label='Greenwich Mean Time',
            tools='',
            toolbar_location=None)
p3.line(x=timestamps,
        y=returns,
        name='returns',
        color='#8959a8',
        line_width=1,
        legend='Cumulative Return',
        line_cap='round',
        line_join='round')
p3.legend.orientation = 'top_left'
p3.legend.border_line_color = background
p3.outline_line_color = None
p3.xgrid.grid_line_color = 'white'
p3.ygrid.grid_line_color = 'white'
p3.axis.axis_line_color = None
p3.axis.major_tick_line_color = None
p3.axis.minor_tick_line_color = None
p3.yaxis.axis_label = 'Basis Points'
p3.yaxis.axis_label_standoff = ylabel_standoff
p3.xaxis.formatter = xformatter
p3.yaxis.formatter = yformatter
p3.xaxis.axis_label_standoff = 12
p3.yaxis.major_label_text_font = 'courier'
p3.xaxis.major_label_text_font = 'courier'
p3.x_range = p1.x_range

vp = vplot(p1, p2, p3)
push()
ip = load(urlopen('http://jsonip.com'))['ip']
ssn = cursession()
ssn.publish()
tag = embed.autoload_server(vp, ssn, public=True).replace('localhost', ip)
html = """
{%% extends "layout.html" %%}
{%% block bokeh %%}
%s
{%% endblock %%}
""" % tag
with open('templates/extended.html', 'w+') as f:
    f.write(html)

renderer = p1.select(dict(name='prices'))
ds_prices = renderer[0].data_source
renderer = p2.select(dict(name='predictions'))
ds_predictions = renderer[0].data_source
renderer = p3.select(dict(name='returns'))
ds_returns = renderer[0].data_source

while True:
    timestamps, prices, predictions, returns = get_data()
    ds_prices.data['x'] = timestamps
    ds_predictions.data['x'] = timestamps
    ds_returns.data['x'] = timestamps
    ds_prices.data['y'] = prices
    ds_predictions.data['y'] = predictions
    ds_returns.data['y'] = returns
    ssn.store_objects(ds_prices)
    ssn.store_objects(ds_predictions)
    ssn.store_objects(ds_returns)
    time.sleep(60)
