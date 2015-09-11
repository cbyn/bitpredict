import pandas as pd
import pymongo
import itertools

client = pymongo.MongoClient()
db = client['bitmicro']
predictions = db['btc_predictions']
cursor = predictions.find()


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

while cursor.alive:
    df = pd.DataFrame(grouper(100, cursor))
    print len(df)
    df.head(10), '/n'

# for chunk in pd.read_table(cursor, chunksize=100):
#     print len(chunk)
#     print chunk.head()
