import pymongo
import pandas as pd

client = pymongo.MongoClient()
db = client['bitmicro']


def get_book_df(symbol, limit, convert_timestamps=False):
    '''
    Returns a DataFrame of book data for symbol
    '''
    books_db = db[symbol+'_books']
    cursor = books_db.find().limit(limit).sort('_id', pymongo.ASCENDING)
    books = pd.DataFrame(list(cursor))
    books = books.set_index('_id')
    if convert_timestamps:
        books.index = pd.to_datetime(books.index, unit='s')
    return books.applymap(pd.DataFrame)


def get_width_and_mid(books):
    '''
    Returns width of best market and midpoint for DataFrame of book data
    '''
    best_bid = books.bids.apply(lambda x: x.price[0])
    best_ask = books.asks.apply(lambda x: x.price[0])
    return best_ask-best_bid, (best_bid + best_ask)/2


def get_future_mid(books, offset, sensitivity):
    '''
    Returns future midpoints for DataFrame of book data
    '''
    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity:
            return books.mid.iloc[i]
    return books.index.map(future)


def get_imbalance(books):
    '''
    Returns imbalances between bids and offers for DataFrame of book data
    '''
    # TODO: account for distance from mid
    total_bid_size = books.bids.apply(lambda x: x.amount.sum())
    total_ask_size = books.asks.apply(lambda x: x.amount.sum())
    return total_bid_size - total_ask_size


def get_trade_df(symbol, min_ts, max_ts, convert_timestamps=False):
    '''
    Returns a DataFrame of trades for symbol in time range
    '''
    trades_db = db[symbol+'_trades']
    query = {'timestamp': {'$gt': min_ts}, 'timestamp': {'$lt': max_ts}}
    cursor = trades_db.find(query).sort('_id', pymongo.ASCENDING)
    trades = pd.DataFrame(list(cursor))
    trades = trades.set_index('timestamp')
    if convert_timestamps:
        trades.index = pd.to_datetime(trades.index, unit='s')
    return trades


def check_times(books):
    '''
    Returns list of differeces between collection time and max book timestamps
    '''
    time_diff = []
    for i in range(len(books)):
        book = books.iloc[i]
        ask_ts = max(book.asks.timestamp)
        bid_ts = max(book.bids.timestamp)
        ts = max(ask_ts, bid_ts)
        time_diff.append(book.name-ts)
    return time_diff


if __name__ == '__main__':
    ltc_books = get_book_df('ltc', 1000)
    ltc_books['width'], ltc_books['mid'] = get_width_and_mid(ltc_books)
    ltc_books['future_mid'] = get_future_mid(ltc_books, 5, 1)
    ltc_books['imbalance'] = get_imbalance(ltc_books)
    min_ts = ltc_books.index[0] - 30
    max_ts = ltc_books.index[-1]
    ltc_trades = get_trade_df('ltc', min_ts, max_ts)
