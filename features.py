import pymongo
import pandas as pd
import numpy as np
from math import log
from time import time

client = pymongo.MongoClient()
db = client['bitmicro']

# TODO:
# fix get_imbalance to weight by distance from mid
# volume (and time?) weight trades
# trades buy/sell ratio
# trades total volume
# book total volume


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
    print 'get_book_df run time:', (time()-start)/60, 'minutes'
    return books.applymap(pd.DataFrame)


def get_width_and_mid(books):
    '''
    Returns width of best market and midpoint for each data point in
    DataFrame of book data
    '''
    start = time()
    best_bid = books.bids.apply(lambda x: x.price[0])
    best_ask = books.asks.apply(lambda x: x.price[0])
    print 'get_width_and_mid run time:', (time()-start)/60, 'minutes'
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
    print 'get_future_mid run time:', (time()-start)/60, 'minutes'
    return books.index.map(future)


# def get_imbalance(books):
#     '''
#     Returns imbalances between bids and offers for each data point in
#     DataFrame of book data
#     '''
#     start = time()
#     total_bid_size = books.bids.apply(lambda x: x.amount.sum())
#     total_ask_size = books.asks.apply(lambda x: x.amount.sum())
#     print 'get_imbalance run time:', (time()-start)/60, 'minutes'
#     return total_bid_size - total_ask_size


def calc_imbalance(book):
    def calc(x):
        return x.amount*(.5*book.width/(x.price-book.mid))**6
    bid_imbalance = book.bids.apply(calc, axis=1)
    ask_imbalance = book.asks.apply(calc, axis=1)
    return (bid_imbalance-ask_imbalance).sum()


def get_imbalance(books):
    '''
    Returns imbalances between bids and offers for each data point in
    DataFrame of book data
    '''
    start = time()
    out = books.apply(calc_imbalance, axis=1)
    print 'get_imbalance run time:', (time()-start)/60, 'minutes'
    return out


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
    print 'get_trade_df run time:', (time()-start)/60, 'minutes'
    return trades


# def get_trades_in_range(trades, min_ts, max_ts):
#     return trades[(trades['timestamp'] >= min_ts)
#                   & (trades['timestamp'] < max_ts)]


def get_trades_average(books, trades, offset):
    # TODO use search sorted?
    '''
    Returns an average of trades for each data point in DataFrame of book data
    '''
    start = time()

    def mean_trades(ts):
        ts = int(ts)
        return trades[(trades['timestamp'] >= ts-offset)
                      & (trades['timestamp'] < ts)].price.mean()
    print 'get_trades_average run time:', (time()-start)/60, 'minutes'
    return books.index.map(mean_trades)


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


def make_features(symbol, sample, mid_offsets, trades_offsets):
    '''
    Returns a dataframe with mid targets and features
    '''
    start = time()
    books = get_book_df(symbol, sample)
    books['width'], books['mid'] = get_width_and_mid(books)
    for n in mid_offsets:
        books['mid{}'.format(n)] = \
            get_future_mid(books, n, sensitivity=5)
        books['mid{}'.format(n)] = \
            (books['mid{}'.format(n)]/books['mid']).apply(log)
    books['imbalance'] = get_imbalance(books)
    min_ts = books.index[0] - trades_offsets[-1]
    max_ts = books.index[-1]
    ltc_trades = get_trade_df(symbol, min_ts, max_ts)
    # Fill trades NaNs with zero (there are no trades in range)
    for n in trades_offsets:
        books['trades{}'.format(n)] = \
            get_trades_average(books, ltc_trades, n)
        books['trades{}'.format(n)] = \
            (books['mid'] / books['trades{}'.format(n)]).apply(log).fillna(0)
    print 'make_features run time:', (time()-start)/60, 'minutes'
    # Drop observations where y is NaN
    return books.drop(['bids', 'asks'], axis=1).dropna()


def fit_model(X, y, chunk):
    '''
    Fits a model to data
    '''
    from sklearn.ensemble import RandomForestClassifier
    y_binary = np.zeros(len(y))
    y_binary[y > 0] = 1
    y_binary[y < 0] = -1
    X_train, X_test = train_test_split(X, chunk)
    y_train, y_test = train_test_split(y_binary, chunk)

    model = RandomForestClassifier(max_depth=5, n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def train_test_split(data, chunk):
    '''
    Returns a chunked split of data
    '''
    splits = np.array([[0]*chunk+[1]*chunk
                       for _ in range(len(data)/chunk/2+1)]).ravel()
    splits = splits[:len(data)]
    train = np.compress(splits == 0, data, axis=0)
    test = np.compress(splits == 1, data, axis=0)
    return train, test


def run_models(data, chunk):
    '''
    Runs a grid search on mid and trades offsets
    '''
    mid_offsets = [col for col in data.columns if 'mid' in col]
    trades_offsets = [col for col in data.columns if 'trades' in col]
    scores = {}
    for m in mid_offsets:
        for t in trades_offsets:
            y = data[m].values
            X = data[['width', 'imbalance', t]].values
            _, score = fit_model(X, y, chunk)
            scores[score] = '{}_{}'.format(m, t)
    for score in sorted(scores):
        print scores[score], score


def make_data(symbol, sample, filename):
    data = make_features(symbol,
                         sample=sample,
                         mid_offsets=[5, 10, 30, 60, 120, 300, 600],
                         trades_offsets=[5, 10, 30, 60, 120, 300, 600])
    import pickle
    with open(filename, 'w+') as f:
        pickle.dump(data, f)
    return data
