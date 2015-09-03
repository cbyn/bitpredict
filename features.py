import pymongo
import pandas as pd
from math import log
from time import time

client = pymongo.MongoClient()
db = client['bitmicro']


def get_book_df(symbol, limit, convert_timestamps=False):
    '''
    Returns a DataFrame of book data for symbol
    '''
    start = time()
    books_db = db[symbol+'_books']
    cursor = books_db.find().limit(limit).sort('_id', pymongo.ASCENDING)
    books = pd.DataFrame(list(cursor))
    books = books.set_index('_id')
    if convert_timestamps:
        books.index = pd.to_datetime(books.index, unit='s')
    print 'get_book_df', (time()-start)/60
    return books.applymap(pd.DataFrame)


def get_width_and_mid(books):
    '''
    Returns width of best market and midpoint for each data point in
    DataFrame of book data
    '''
    start = time()
    best_bid = books.bids.apply(lambda x: x.price[0])
    best_ask = books.asks.apply(lambda x: x.price[0])
    print 'get_width_and_mid', (time()-start)/60
    return best_ask-best_bid, (best_bid + best_ask)/2


def get_future_mid(books, offset, sensitivity):
    '''
    Returns future midpoints for each data point in DataFrame of book data
    '''
    start = time()

    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity:
            return books.mid.iloc[i]
    print 'get_future_mid', (time()-start)/60
    return books.index.map(future)


def get_imbalance(books):
    # TODO: account for distance from mid
    '''
    Returns imbalances between bids and offers for each data point in
    DataFrame of book data
    '''
    start = time()
    total_bid_size = books.bids.apply(lambda x: x.amount.sum())
    total_ask_size = books.asks.apply(lambda x: x.amount.sum())
    print 'get_imbalance', (time()-start)/60
    return total_bid_size - total_ask_size


def get_trade_df(symbol, min_ts, max_ts, convert_timestamps=False):
    '''
    Returns a DataFrame of trades for symbol in time range
    '''
    start = time()
    trades_db = db[symbol+'_trades']
    query = {'timestamp': {'$gt': min_ts, '$lt': max_ts}}
    cursor = trades_db.find(query).sort('_id', pymongo.ASCENDING)
    trades = pd.DataFrame(list(cursor))
    trades = trades.set_index('_id')
    if convert_timestamps:
        trades.index = pd.to_datetime(trades.index, unit='s')
    print 'get_trade_df', (time()-start)/60
    return trades


def get_trades_in_range(trades, min_ts, max_ts):
    return trades[(trades['timestamp'] >= min_ts)
                  & (trades['timestamp'] < max_ts)]


def get_trades_average(books, trades, offset):
    # TODO weight by volume and time
    '''
    Returns an average of trades for each data point in DataFrame of book data
    '''
    start = time()

    def mean_trades(ts):
        return trades[(trades['timestamp'] >= ts-offset)
                      & (trades['timestamp'] < ts)].price.mean()
    print 'get_trades_average', (time()-start)/60
    return books.index.map(mean_trades)

# TODO:
# trades buy/sell ratio
# trades total volume
# book total volume


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


def make_features(symbol, sample, y_offset, trades_offset):
    '''
    Returns a dataframe with y and all features
    '''
    start = time()
    ltc_books = get_book_df(symbol, sample)
    ltc_books['width'], ltc_books['mid'] = get_width_and_mid(ltc_books)
    ltc_books['future_mid'] = get_future_mid(ltc_books, y_offset, sensitivity=5)
    ltc_books['imbalance'] = get_imbalance(ltc_books)
    min_ts = ltc_books.index[0] - trades_offset
    max_ts = ltc_books.index[-1]
    ltc_trades = get_trade_df(symbol, min_ts, max_ts)
    ltc_books['trades_avg'] = get_trades_average(ltc_books,
                                                 ltc_trades, trades_offset)
    ltc_books['trades_avg'] = (ltc_books['mid'] /
                               ltc_books['trades_avg']).apply(log).fillna(0)
    ltc_books['y'] = (ltc_books['future_mid']/ltc_books['mid']).apply(log)
    print 'make_features', (time()-start)/60
    return ltc_books[['y', 'width', 'mid', 'imbalance', 'trades_avg']].dropna()


def fit_model(X, y, symbol):
    start = time()
    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    y_binary = np.zeros(len(y))
    y_binary[y.values > 0] = 1
    y_binary[y.values < 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(X, y_binary)

    model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    model.fit(x_train, y_train)
    print symbol, 'random forest score:', model.score(x_test, y_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    print symbol, 'logit score:', model.score(x_test, y_test)
    print 'fit_model', (time()-start)/60

if __name__ == '__main__':
    symbol = 'ltc'
    data = make_features(symbol, sample=1000000, y_offset=60, trades_offset=300)
    y = data.pop('y')
    fit_model(data.values, y, symbol)
