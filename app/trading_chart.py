import pandas as pd
import pymongo
from bokeh.plotting import cursession, figure, output_server, show
import time
from math import log

client = pymongo.MongoClient()
db = client['bitmicro']
predictions = db['btc_predictions2']

cursor = predictions.find().sort('_id', pymongo.DESCENDING)
data = pd.DataFrame(list(cursor))
data = data[data.future_price != 0]
data = data.set_index('_id')
data = data.sort_index(ascending=True)
times = pd.to_datetime(data.index, unit='s').to_series()
actual = (data.future_price/data.current_price).apply(log)
returns = actual*data.position*100*100

output_server('all_returns')
p1 = figure(title='Phony Returns',
            x_axis_type='datetime',
            x_axis_label='Greenwich Time',
            y_axis_label='Basis Points')
p1.line(times, returns.cumsum(), name='all_returns')
show(p1)

renderer1 = p1.select(dict(name='all_returns'))
ds1 = renderer1[0].data_source

while True:
    cursor = predictions.find().sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    data = data[data.future_price != 0]
    data = data.set_index('_id')
    data = data.sort_index(ascending=True)
    times = pd.to_datetime(data.index, unit='s').to_series()
    actual = (data.future_price/data.current_price).apply(log)
    returns = actual*data.position*100*100

    ds1.data['x'] = times
    ds1.data['y'] = returns.cumsum()
    cursession().store_objects(ds1)

    time.sleep(1)
