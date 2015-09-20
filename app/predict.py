from bitpredict.model import features as f
import pymongo
import time
import sys
import pickle
import numpy as np
from math import log

client = pymongo.MongoClient()
db = client['bitmicro']
symbol = sys.argv[1]
duration = int(sys.argv[2])
threshold = float(sys.argv[3])
predictions = db[symbol+'_predictions']

with open('cols.pkl', 'r') as file1:
    cols = pickle.load(file1)
with open('model.pkl', 'r') as file2:
    model = pickle.load(file2)

print 'Running...'
trade = 0
position = 0
trade_time = 0
change = 0
previous_price = None
while True:
    start = time.time()
    try:
        data = f.make_features(symbol,
                               1,
                               [duration],
                               [30, 60, 120, 180],
                               [2, 4, 8],
                               True)
        pred = model.predict(data[cols].values)[0]
    except Exception as e:
        print e
        sys.exc_clear()
    else:
        if data.width.iloc[0] > 0:
            # If a trade happened in the previous second
            if trade != 0:
                position = trade
                trade = 0
            # If an open position has expired
            if position != 0 and (start - trade_time) >= duration+1:
                position = 0
            # If we can execute a new trade
            if position == 0 and abs(pred) >= threshold:
                trade_time = time.time()
                trade = np.sign(pred)
            price = data.mid.iloc[0]
            if price and previous_price:
                change = log(price/previous_price)
            else:
                change = 0
            entry = {'prediction': pred,
                     'price': price,
                     'change': change,
                     'trade': trade,
                     'position': position,
                     'future_price': 0}
            # Set as mongo update so it doesn't blow up if data is not updating
            predictions.update_one({'_id': data.index[0]},
                                   {'$setOnInsert': entry},
                                   upsert=True)
            previous_price = price
            # Put in future price to use for simple (non live) calculations
            predictions.update_many({'_id': {'$gt': start-duration-.5,
                                             '$lt': start-duration+.5}},
                                    {'$set': {'future_price': price}})
    time_delta = time.time()-start
    if time_delta < 1.0:
        time.sleep(1-time_delta)
