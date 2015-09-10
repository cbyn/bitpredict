import pymongo
import pandas as pd
from math import log
from time import time
import sys
from scipy.stats import linregress
import pickle

# TODO
# time-weight trades

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


def get_imbalance(books, n=5):
    '''
    Returns a measure of the imbalance between bids and offers for each data
    point in DataFrame of book data
    '''
    start = time()

    def calc_imbalance(book):
        return (book.bids.amount.iloc[:n] - book.asks.amount.iloc[:n]).sum()
    imbalance = books.apply(calc_imbalance, axis=1)
    print 'get_imbalance run time:', (time()-start)/60, 'minutes'
    return imbalance


def get_power_imbalance(books, n=10, power=2):
    '''
    Returns a measure of the imbalance between bids and offers for each data
    point in DataFrame of book data; volumes are additionally weighed by
    closeness to the midpoint

    '''
    start = time()

    def calc_imbalance(book):
        def calc(x):
            return x.amount*(.5*book.width/(x.price-book.mid))**power
        bid_imbalance = book.bids.iloc[:n].apply(calc, axis=1)
        ask_imbalance = book.asks.iloc[:n].apply(calc, axis=1)
        return (bid_imbalance-ask_imbalance).sum()
    imbalance = books.apply(calc_imbalance, axis=1)
    print 'get_imbalance run time:', (time()-start)/60, 'minutes'
    return imbalance


def get_adjusted_price(books, n=5):
    '''
    Returns an average of price weighted by inverse volume for each data point
    in DataFrame of book data
    '''
    start = time()

    def calc_adjusted_price(book):
        bid_inv = 1/book.bids.amount.iloc[:n]
        ask_inv = 1/book.asks.amount.iloc[:n]
        bid_price = book.bids.price.iloc[:n]
        ask_price = book.asks.price.iloc[:n]
        return (bid_price*bid_inv + ask_price*ask_inv).sum() /\
            (bid_inv + ask_inv).sum()
    adjusted = books.apply(calc_adjusted_price, axis=1)
    print 'get_adjusted_price run time:', (time()-start)/60, 'minutes'
    return adjusted


def get_power_adjusted_price(books, n=10, power=2):
    '''
    Returns an average of price weighted by inverse volume for each data point
    in DataFrame of book data; volumes are additionally weighed by closeness
    to the midpoint
    '''
    start = time()

    def calc_adjusted_price(book):
        def calc(x):
            return x.amount*(.5*book.width/(x.price-book.mid))**power
        bid_inv = 1/book.bids.iloc[:n].apply(calc, axis=1)
        ask_inv = 1/book.asks.iloc[:n].apply(calc, axis=1)
        bid_price = book.bids.price.iloc[:n]
        ask_price = book.asks.price.iloc[:n]
        return (bid_price*bid_inv + ask_price*ask_inv).sum() /\
            (bid_inv + ask_inv).sum()
    adjusted = books.apply(calc_adjusted_price, axis=1)
    print 'get_adjusted_price run time:', (time()-start)/60, 'minutes'
    return adjusted


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
            return (trades_n.price*trades_n.amount).sum()/trades_n.amount.sum()
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


def get_trend(books, trades, offset):
    '''
    Returns the linear trend in previous trades for each data point in
    DataFrame of book data
    '''
    start = time()

    def trend(ts):
        trades_n = get_trades_in_range(trades, ts, offset)
        if len(trades_n) < 3:
            return 0
        else:
            return linregress(trades_n.index.values, trades_n.price.values)[0]
    print 'get_trend run time:', (time()-start)/60, 'minutes'
    return books.index.map(trend)


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
    Returns a DataFrame with targets and features
    '''
    start = time()
    # Book related features:
    books = get_book_df(symbol, sample)
    books['width'], books['mid'] = get_width_and_mid(books)
    for n in mid_offsets:
        books['mid{}'.format(n)] = \
            get_future_mid(books, n, sensitivity=1)
        books['mid{}'.format(n)] = \
            (books['mid{}'.format(n)]/books.mid).apply(log)
        # books['prev{}'.format(n)] = get_future_mid(books, -n, sensitivity=1)
        # books['prev{}'.format(n)] = (books.mid/books['prev{}'.format(n)])\
        #     .apply(log).fillna(0)  # Fill prev NaNs with zero (assume no change)
    # Drop observations where y is NaN
    books = books.dropna()
    books['imbalance2'] = get_power_imbalance(books, 10, 2)
    books['adjusted_price2'] = get_power_adjusted_price(books, 10, 2)
    books['adjusted_price2'] = (books.adjusted_price2/books.mid).apply(log)
    books['imbalance8'] = get_power_imbalance(books, 10, 8)
    books['adjusted_price8'] = get_power_adjusted_price(books, 10, 8)
    books['adjusted_price8'] = (books.adjusted_price8/books.mid).apply(log)

    # # Trade related features:
    # min_ts = books.index[0] - trades_offsets[-1]
    # max_ts = books.index[-1]
    # trades = get_trade_df(symbol, min_ts, max_ts)
    # # Fill trade NaNs with zero (there are no trades in range)
    # for n in trades_offsets:
    #     books['trades{}'.format(n)] = get_trades_average(books, trades, n)
    #     books['trades{}'.format(n)] = \
    #         (books.mid / books['trades{}'.format(n)]).apply(log).fillna(0)
    #     books['aggressor{}'.format(n)] = get_aggressor(books, trades, n)
    #     books['trend{}'.format(n)] = get_trend(books, trades, n)
    print 'make_features run time:', (time()-start)/60, 'minutes'

    return books.drop(['bids', 'asks'], axis=1)


def make_data(symbol, sample):
    data = make_features(symbol,
                         sample=sample,
                         mid_offsets=[30],
                         trades_offsets=[10, 30, 120, 300])
    return data

if __name__ == '__main__' and len(sys.argv) == 4:
    data = make_data(sys.argv[1], int(sys.argv[2]))
    with open(sys.argv[3], 'w+') as f:
        pickle.dump(data, f)
