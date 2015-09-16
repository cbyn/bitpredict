import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, output_server, push
from bokeh import embed
from json import load
from urllib2 import urlopen
import time
import re

client = pymongo.MongoClient()
db = client['bitmicro']
predictions = db['btc_predictions']

cursor = predictions.find().limit(60*5).sort('_id', pymongo.DESCENDING)
data = pd.DataFrame(list(cursor))
data = data.set_index('_id')
data = data.sort_index(ascending=True)
x = pd.to_datetime(data.index, unit='s').to_series()
y = data.prediction*100*100

output_server('model_signal')
p = figure(title='30-second Prediction',
           plot_width=800,
           plot_height=250,
           x_axis_type='datetime',
           x_axis_label='Greenwich Time',
           y_axis_label='Basis Points',
           tools='')
p.toolbar_location = None
p.line(x, y, name='signal')
push()

ip = load(urlopen('http://jsonip.com'))['ip']
ssn = cursession()
tag = embed.autoload_server(p, ssn, public=True).replace('localhost', ip)

with open('templates/index.html', 'r') as f:
    html = f.read()
html = re.sub(r'<script.*?></script>', tag, html, 1, re.DOTALL)
with open('templates/index.html', 'w+') as f:
    f.write(html)

renderer = p.select(dict(name='signal'))
ds = renderer[0].data_source

while True:
    cursor = predictions.find().limit(60*5).sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    ds.data['x'] = pd.to_datetime(data.index, unit='s').to_series()
    ds.data['y'] = data.prediction*100*100
    ssn.store_objects(ds)
    time.sleep(1)
