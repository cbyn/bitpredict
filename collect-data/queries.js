db.ltc_trades.aggregate([
	{ $group: {
		_id: '$timestamp',
		total_price: { $sum: { $multiply: [ "$price", "$amount" ] } },
		total_quantity: { $sum: "$amount" }
	} },
       	{ $project: {
		price: { $divide: [ "$total_price", "$total_quantity" ] },
		amount: "$total_quantity"
	} },
	{ $sort: {_id: 1} }
])
