import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, output_server, push
# from bokeh.models.widgets import layouts
from bokeh.io import gridplot
from bokeh import embed
from json import load
from urllib2 import urlopen
import time
import re

client = pymongo.MongoClient()
db = client['bitmicro']
collection = db['btc_predictions']

cursor = collection.find().limit(60*5).sort('_id', pymongo.DESCENDING)
data = pd.DataFrame(list(cursor))
data = data.set_index('_id')
data = data.sort_index(ascending=True)
timestamps = pd.to_datetime(data.index, unit='s').to_series()
prices = data.current_price
signals = data.prediction*100*100

output_server('price')
p1 = figure(title='Bitcoin Bid/Ask Midpoint',
            plot_width=800,
            plot_height=400,
            x_axis_type='datetime',
            y_axis_label='',
            tools='')
p1.toolbar_location = None
p1.line(timestamps, prices, name='price')

output_server('signal')
p2 = figure(title=None,
            plot_width=800,
            plot_height=200,
            x_axis_type='datetime',
            y_axis_label='Basis Points',
            tools='')
p2.toolbar_location = None
p2.line(timestamps, signals, name='signal')

gp = gridplot([[p1], [p2]])
gp.toolbar_location = None

# vbox = layouts.VBox()
# vbox.children.append(p1)
# vbox.children.append(p2)

push()

ip = load(urlopen('http://jsonip.com'))['ip']
ssn = cursession()
tag = embed.autoload_server(gp, ssn, public=True).replace('localhost', ip)

with open('templates/index.html', 'r') as f:
    html = f.read()
html = re.sub(r'<script.*?></script>', tag, html, 1, re.DOTALL)
with open('templates/index.html', 'w+') as f:
    f.write(html)

renderer2 = p2.select(dict(name='signal'))
ds2 = renderer2[0].data_source

while True:
    cursor = collection.find().limit(60*5).sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    ds2.data['x'] = pd.to_datetime(data.index, unit='s').to_series()
    ds2.data['y'] = data.prediction*100*100
    ssn.store_objects(ds2)
    time.sleep(1)
