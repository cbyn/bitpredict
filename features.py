import pymongo
import pandas as pd
from math import log

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
    Returns width of best market and midpoint for each data point in
    DataFrame of book data
    '''
    best_bid = books.bids.apply(lambda x: x.price[0])
    best_ask = books.asks.apply(lambda x: x.price[0])
    return best_ask-best_bid, (best_bid + best_ask)/2


def get_future_mid(books, offset, sensitivity):
    '''
    Returns future midpoints for each data point in DataFrame of book data
    '''
    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity:
            return books.mid.iloc[i]
    return books.index.map(future)


def get_imbalance(books):
    # TODO: account for distance from mid
    '''
    Returns imbalances between bids and offers for each data point in
    DataFrame of book data
    '''
    total_bid_size = books.bids.apply(lambda x: x.amount.sum())
    total_ask_size = books.asks.apply(lambda x: x.amount.sum())
    return total_bid_size - total_ask_size


def get_trade_df(symbol, min_ts, max_ts, convert_timestamps=False):
    '''
    Returns a DataFrame of trades for symbol in time range
    '''
    trades_db = db[symbol+'_trades']
    query = {'timestamp': {'$gt': min_ts, '$lt': max_ts}}
    cursor = trades_db.find(query).sort('_id', pymongo.ASCENDING)
    trades = pd.DataFrame(list(cursor))
    trades = trades.set_index('_id')
    if convert_timestamps:
        trades.index = pd.to_datetime(trades.index, unit='s')
    return trades


def get_trades_in_range(trades, min_ts, max_ts):
    return trades[(trades['timestamp'] >= min_ts)
                  & (trades['timestamp'] < max_ts)]


def get_trades_average(books, trades, offset):
    # TODO weight by volume and time
    '''
    Returns an average of trades for each data point in DataFrame of book data
    '''
    def mean_trades(ts):
        return trades[(trades['timestamp'] >= ts-offset)
                      & (trades['timestamp'] < ts)].price.mean()
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
    ltc_books = get_book_df(symbol, sample)
    ltc_books['width'], ltc_books['mid'] = get_width_and_mid(ltc_books)
    ltc_books['future_mid'] = get_future_mid(ltc_books, y_offset, 5)
    ltc_books['imbalance'] = get_imbalance(ltc_books)
    min_ts = ltc_books.index[0] - trades_offset
    max_ts = ltc_books.index[-1]
    ltc_trades = get_trade_df(symbol, min_ts, max_ts)
    ltc_books['trades_avg'] = get_trades_average(ltc_books,
                                                 ltc_trades, trades_offset)
    ltc_books['trades_avg'] = (ltc_books['mid'] /
                               ltc_books['trades_avg']).apply(log).fillna(0)
    ltc_books['y'] = (ltc_books['future_mid']/ltc_books['mid']).apply(log)
    return ltc_books[['y', 'width', 'mid', 'imbalance', 'trades_avg']].dropna()


if __name__ == '__main__':
    symbol = 'ltc'
    data = make_features(symbol, 10000, 60, 300)

    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    y = data.pop('y')
    y_binary = np.zeros(len(y))
    y_binary[y.values > 0] = 1
    y_binary[y.values < 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(data.values, y_binary)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    print symbol, 'tree score:', model.score(x_test, y_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    print symbol, 'logit score:', model.score(x_test, y_test)
