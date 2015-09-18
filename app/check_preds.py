import pandas as pd
import pymongo
from sklearn.metrics import r2_score
from math import log
import sys

client = pymongo.MongoClient()
db = client['bitmicro']
predictions = db['btc_predictions']

if len(sys.argv) > 1:
    limit = int(sys.argv[1])
    cursor = predictions.find().limit(limit).sort('_id', pymongo.DESCENDING)
else:
    cursor = predictions.find().sort('_id', pymongo.DESCENDING)

df = pd.DataFrame(list(cursor))
df = df[df.future_price != 0]
df['actual'] = (df.future_price/df.price).apply(log)
score = r2_score(df.actual.values, df.prediction.values)
print 'observations:', len(df)
print 'r^2:', score
