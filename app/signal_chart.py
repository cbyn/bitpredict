import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, show, output_server
import time

client = pymongo.MongoClient()
db = client['bitmicro']
predictions = db['btc_predictions']

cursor = predictions.find().limit(60*60).sort('_id', pymongo.DESCENDING)
data = pd.DataFrame(list(cursor))
data = data.set_index('_id')
data = data.sort_index(ascending=True)
x = pd.to_datetime(data.index, unit='s').to_series()
y = data.prediction*100

output_server('model_signal')
p = figure(title='30-second Prediction',
           x_axis_type='datetime',
           y_axis_label='Basis Points')
p.line(x, y, name='signal')
show(p)

renderer = p.select(dict(name='signal'))
ds = renderer[0].data_source

while True:
    cursor = predictions.find().limit(60*60).sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    ds.data['x'] = pd.to_datetime(data.index, unit='s').to_series()
    ds.data['y'] = data.prediction*100
    cursession().store_objects(ds)
    time.sleep(1)
