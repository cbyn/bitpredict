import urllib2
import time
import json
from pymongo import MongoClient
import sys

api = 'https://api.bitfinex.com/v1'
symbol = sys.argv[1]
limit = 1000

client = MongoClient()
db = client['bitmicro']
ltc_trades = db[symbol+'_trades']


def format_trade(trade):
    if all(key in trade for key in ('tid', 'amount', 'price', 'timestamp')):
        trade['_id'] = trade.pop('tid')
        trade['amount'] = float(trade['amount'])
        trade['price'] = float(trade['price'])
        trade['timestamp'] = float(trade['timestamp'])

    return trade


def get_json(url):
    resp = urllib2.urlopen(url)
    return json.load(resp, object_hook=format_trade), resp.getcode()


last = 0
while True:
    url = '{0}/trades/{1}usd?timestamp={2}&limit_trades={3}'.format(api, symbol,
                                                                    last, limit)

    trades, code = get_json(url)
    if code != 200:
        print code
    else:
        for trade in trades:
            ltc_trades.update_one({'_id': trade['_id']},
                                  {'$setOnInsert': trade}, upsert=True)

    last = trades[0]['timestamp'] - 5
    time.sleep(60)
