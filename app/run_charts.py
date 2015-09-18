import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, output_server, push
from bokeh.models.formatters import DatetimeTickFormatter, PrintfTickFormatter
# from bokeh.models import LinearAxis, Range1d
# from bokeh.models.widgets import layouts
from bokeh.io import vplot
from bokeh import embed
from json import load
from urllib2 import urlopen
import time
import re
from math import log

client = pymongo.MongoClient()
db = client['bitmicro']
collection = db['btc_predictions']


def get_data():
    cursor = collection.find().limit(10*60).sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    timestamps = pd.to_datetime(data.index, unit='s').to_series()
    prices = data.current_price
    predictions = data.prediction*10000
    data.future_price.replace(0, data.current_price.iloc[-1], inplace=True)
    actual = (data.future_price/data.current_price).apply(log)
    returns = actual*data.position*10000
    return timestamps, prices, predictions, returns.cumsum()

timestamps, prices, predictions, returns = get_data()
output_server('short_charts')

background = '#f2f2f2'
label_standoff = 17
xformatter = DatetimeTickFormatter(formats=dict(minutes=["%H:%M"]))
p1 = figure(title=None,
            plot_width=750,
            plot_height=300,
            x_axis_type='datetime',
            min_border_top=10,
            min_border_bottom=0,
            background_fill=background,
            tools='',
            toolbar_location=None)
# p1.line(x=timestamps,
#         y=(1+predictions)*prices,
#         name='predictions',
#         color='green',
#         line_width=1,
#         legend='30-Second Forecast',
#         line_cap='round',
#         line_join='round')
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
p1.yaxis.axis_label_standoff = 10
p1.xaxis.formatter = xformatter
p1.yaxis.formatter = PrintfTickFormatter(format='%6.2f')

p2 = figure(title=None,
            plot_width=750,
            plot_height=200,
            x_axis_type='datetime',
            min_border_top=20,
            min_border_bottom=0,
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
p2.yaxis.axis_label_standoff = label_standoff
p2.xaxis.formatter = xformatter
p2.yaxis.formatter = PrintfTickFormatter(format='%+6.1f')
p2.x_range = p1.x_range

p3 = figure(title=None,
            plot_width=750,
            plot_height=200,
            x_axis_type='datetime',
            min_border_top=20,
            min_border_bottom=0,
            background_fill=background,
            x_axis_label='Greenwich Time',
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
p3.yaxis.axis_label_standoff = label_standoff
p3.xaxis.formatter = xformatter
p3.yaxis.formatter = PrintfTickFormatter(format='%+6.1f')
p3.x_range = p1.x_range

vp = vplot(p1, p2, p3)
push()

ip = load(urlopen('http://jsonip.com'))['ip']
ssn = cursession()
tag = embed.autoload_server(vp, ssn, public=True).replace('localhost', ip)

with open('templates/index.html', 'r') as f:
    html = f.read()
match = r"<script\s*src=\"http://52.16.234.158:5006/bokeh.*?</script>"
html = re.sub(match, tag, html, 1, re.DOTALL)
with open('templates/index.html', 'w+') as f:
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
    time.sleep(1)
