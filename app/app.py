from bitmicro.model import features as f
import pymongo
import time
import sys

client = pymongo.MongoClient()
db = client['bitmicro']
symbol = sys.argv[1]
duration = int(sys.argv[2])
predictions = db[symbol+'_predictions']

# Need to import a pickled model
while True:
    start = time.time()
    data = f.make_features(symbol, 1, [duration], [10, 30, 120, 300], True)
    current_price = data.pop('mid').iloc[0]
    pred = 0
    entry = {'prediction': pred,
             'current_price': current_price,
             'future_price': current_price}
    predictions.update_one({'_id': data.index[0]},
                           {'$setOnInsert': entry},
                           upsert=True)
    predictions.update_many({'_id': {'$lt': start-duration+1,
                                     '$gt': start-duration-1}},
                            {'$set': {'future_price': current_price}})
    time_delta = time.time()-start
    if time_delta < 1.0:
        time.sleep(1-time_delta)
