import pymongo
import pandas as pd
import numpy as np
from math import log
from time import time
import sys

client = pymongo.MongoClient()
db = client['bitmicro']

# TODO:
# volume on best orders as a direct feature
# trades total volume
# book total volume
# time-weight trades?
# experiment with exponent in get_imbalance


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


def get_imbalance(books):
    '''
    Returns a measure of the imbalance between bids and offers for each data
    point in DataFrame of book data
    '''
    start = time()

    def calc_imbalance(book):
        def calc(x):
            return x.amount*(.5*book.width/(x.price-book.mid))**2
        bid_imbalance = book.bids.apply(calc, axis=1)
        ask_imbalance = book.asks.apply(calc, axis=1)
        return bid_imbalance.sum() - ask_imbalance.sum()
        # return (bid_imbalance-ask_imbalance).sum()
    books = books.apply(calc_imbalance, axis=1)
    print 'get_imbalance run time:', (time()-start)/60, 'minutes'
    return books


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


def get_trades_in_range(trades, ts, offset):
    '''
    Returns trades in a given timestamp range
    '''
    ts = int(ts)
    i_0 = trades.timestamp.searchsorted([ts-offset], side='left')[0]
    i_n = trades.timestamp.searchsorted([ts-1], side='right')[0]
    return trades.iloc[i_0:i_n]


def get_trades_average(books, trades, offset):
    '''
    Returns a volume-weighted average of trades for each data point in
    DataFrame of book data
    '''
    start = time()

    def mean_trades(ts):
        trades_n = get_trades_in_range(trades, ts, offset)
        if not trades_n.empty:
            return (trades_n.price*trades_n.amount/trades_n.amount).sum()
        # return trades.iloc[i_0:i_n].price.mean()
    print 'get_trades_average run time:', (time()-start)/60, 'minutes'
    return books.index.map(mean_trades)


def get_aggressor(books, trades, offset):
    '''
    Returns a measure of whether trade aggressors were buyers or sellers for
    each data point in DataFrame of book data
    '''
    start = time()

    def aggressor(ts):
        trades_n = get_trades_in_range(trades, ts, offset)
        buys = trades_n['type'] == 'buy'
        buy_vol = trades_n[buys].amount.sum()
        sell_vol = trades_n[~buys].amount.sum()
        return buy_vol - sell_vol
    print 'get_aggressor run time:', (time()-start)/60, 'minutes'
    return books.index.map(aggressor)


def check_times(books):
    '''
    Returns list of differences between collection time and max book timestamps
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
    Returns a DataFrame with mid targets and features
    '''
    start = time()
    books = get_book_df(symbol, sample)
    books['width'], books['mid'] = get_width_and_mid(books)
    for n in mid_offsets:
        books['mid{}'.format(n)] = \
            get_future_mid(books, n, sensitivity=5)
        books['mid{}'.format(n)] = \
            (books['mid{}'.format(n)]/books.mid).apply(log)
    books['imbalance'] = get_imbalance(books)
    min_ts = books.index[0] - trades_offsets[-1]
    max_ts = books.index[-1]
    trades = get_trade_df(symbol, min_ts, max_ts)
    # Fill trade NaNs with zero (there are no trades in range)
    for n in trades_offsets:
        books['trades{}'.format(n)] = get_trades_average(books, trades, n)
        books['trades{}'.format(n)] = \
            (books.mid / books['trades{}'.format(n)]).apply(log).fillna(0)
        books['aggressor{}'.format(n)] = get_aggressor(books, trades, n)
    print 'make_features run time:', (time()-start)/60, 'minutes'
    # Drop observations where y is NaN
    return books.drop(['bids', 'asks'], axis=1).dropna()


def fit_classifier(X, y, chunk):
    '''
    Fits a model to data
    '''
    from sklearn.ensemble import RandomForestClassifier
    y_binary = np.zeros(len(y))
    y_binary[y > 0] = 1
    y_binary[y < 0] = -1
    X_train, X_test = train_test_split(X, chunk)
    y_train, y_test = train_test_split(y_binary, chunk)

    model = RandomForestClassifier(n_estimators=100,
                                   min_samples_leaf=500,
                                   max_depth=10,
                                   random_state=42,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def fit_regressor(X, y, chunk):
    '''
    Fits a model to data
    '''
    from sklearn.ensemble import RandomForestRegressor
    X_train, X_test = train_test_split(X, chunk)
    y_train, y_test = train_test_split(y, chunk)

    model = RandomForestRegressor(n_estimators=100,
                                  min_samples_leaf=500,
                                  max_depth=10,
                                  random_state=42,
                                  n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def train_test_split(data, chunk):
    # '''
    # Returns a chunked split of data
    # '''
    # splits = np.array([[0]*chunk+[1]*chunk
    #                    for _ in range(len(data)/chunk/2+1)]).ravel()
    # splits = splits[:len(data)]
    # train = np.compress(splits == 0, data, axis=0)
    # test = np.compress(splits == 1, data, axis=0)
    # return train, test
    i = int(chunk*len(data))
    return data[:i], data[i:]


def run_models(data, chunk):
    '''
    Runs model with a range of target offsets
    '''
    mids = [col for col in data.columns if 'mid' in col]
    trades = [col for col in data.columns if 'trades' in col]
    aggressors = [col for col in data.columns if 'aggressor' in col]
    classifier_scores = {}
    regressor_scores = {}
    for m in mids:
        y = data[m].values
        X = data[['width', 'imbalance']+trades+aggressors].values
        _, classifier_score = fit_classifier(X, y, chunk)
        classifier_scores[classifier_score] = m
        _, regressor_score = fit_regressor(X, y, chunk)
        regressor_scores[regressor_score] = m
    print 'classifier accuracies:'
    for score in sorted(classifier_scores):
        print classifier_scores[score], score
    print 'regressor r^2:'
    for score in sorted(regressor_scores):
        print regressor_scores[score], score


def make_data(symbol, sample):
    data = make_features(symbol,
                         sample=sample,
                         mid_offsets=[5, 10, 30, 60],
                         trades_offsets=[30, 60, 120, 300, 600])
    return data

if __name__ == '__main__' and len(sys.argv) == 4:
    import pickle
    data = make_data(sys.argv[1], int(sys.argv[2]))
    with open(sys.argv[3], 'w+') as f:
        pickle.dump(data, f)
