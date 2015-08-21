// group ltc trades by second, with volume-weigted price
db.ltc_trades.aggregate([
	{ $match: { timestamp: { $gt: 1440117467} } },	
	{ $group: {
		_id: "$timestamp",
		total_price: { $sum: { $multiply: [ "$price", "$amount" ] } },
		total_quantity: { $sum: "$amount" }
	} },
       	{ $project: {
		price: { $divide: [ "$total_price", "$total_quantity" ] },
		amount: "$total_quantity"
	} },
	{ $sort: {_id: 1} },
	{ $out: "grouped_ltc_trades" }
])
