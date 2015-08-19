import urllib2
import time
import json
from pymongo import MongoClient

api = 'https://api.bitfinex.com/v1'
symbol = 'ltcusd'
limit = 10000

client = MongoClient()
db = client['bitmicro']
ltc_trades = db['ltc_trades']


def get_json(url):
    resp = urllib2.urlopen(url)
    return json.load(resp), resp.getcode()

last = 0
while True:
    url = '{0}/trades/{1}?timestamp={2}&limit_trades={2}'.format(api, symbol,
                                                                 last, limit)

    trades, code = get_json(url)
    if code != 200:
        print code
    else:
        for trade in trades:
            trade['_id'] = trade.pop('tid')
            ltc_trades.update_one({'_id': trade['_id']},
                                  {'$set': trade}, upsert=True)

    last = trades[0]['timestamp'] - 5
    time.sleep(300)
