import pymongo
import time
import sys
import pandas as pd

client = pymongo.MongoClient()
db = client['bitmicro']
symbol = sys.argv[1]
frequency = int(sys.argv[2])
predictions = db[symbol+'_predictions']
performance = db[symbol+'_performance']

print 'Running...'
last_timestamp = time.time() - frequency
while True:
    start = time.time()

    query = {'_id': {'$gt': last_timestamp}}
    cursor = predictions.find(query).sort('_id', pymongo.DESCENDING)
    data = pd.DataFrame(list(cursor))
    if not data.empty:
        data = data.set_index('_id')
        data = data.sort_index(ascending=True)
        returns = (data.position*data.change).sum()
        last_timestamp = data.index[-1]
        # Set as mongo update so it doesn't blow up if data is not updating
        performance.update_one({'_id': last_timestamp},
                               {'$setOnInsert': {'returns': returns}},
                               upsert=True)

    time_delta = time.time()-start
    time.sleep(frequency-time_delta)
