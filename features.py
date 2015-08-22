# TODO: use cmongo and monary instead of pymongo?
from pymongo import MongoClient
import pandas as pd
import itertools

client = MongoClient()
db = client['bitmicro']
trades_db = db['grouped_ltc_trades']
books_db = db['ltc_books']

trades = pd.DataFrame(list(trades_db.find()))
trades = trades.set_index('_id')

min_ts = trades.index[0] - 10
max_ts = min_ts+100  # trades.index[-1]
books_query = {'_id': {'$gt': min_ts, '$lt': max_ts}}
books = pd.DataFrame(list(books_db.find(books_query)))
books = books.set_index('_id')


def book_to_df(book):
    bids = pd.DataFrame(book['bids'])
    asks = pd.DataFrame(book['asks'])

    return pd.concat((bids, asks), axis=1, keys=('bids', 'asks'))

book = book_to_df(books_db.find_one())


def check_times():
    time_diff = []
    for i in range(len(books)):
        book = books.iloc[i]
        ask_ts = max([ask['timestamp'] for ask in book.asks])
        bid_ts = max([bid['timestamp'] for bid in book.bids])
        ts = max(ask_ts, bid_ts)
        time_diff.append(book.name-ts)

    print max(time_diff)
    print min(time_diff)


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
