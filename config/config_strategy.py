
config_strategy = {
"Momentum_Strategy":{"Description": "Update the 20-trade and 50-trade moving average lines with the mid-price of the latest 20 and 50 best ask and best bid respectively. Buy when the 20-trade moving average line crosses above the 50-trade moving average line, and sell when the 20-trade moving average line crosses below the 50-trade moving average line.",
                    "name":"Momentum_Strategy","min_size":1,"max_size":10,"wake_up_freq":'20'},
"Value_Strategy":{"Description": "Give what you think is the true price of the current stock. Buy when the stock price is lower than that price and sell when it is higher than that price",
                    "r_bar":22319,"sigma_n": 22319/10,"kappa":1.67e-15,"lambda_a":7e-11,"sigma_s":10000,
                    "name":"Value_Strategy","min_size":1,"max_size":10,"wake_up_freq":'20'},
"MACD_Strategy":{"Description":"Calculate the MACD bar using the median price between on best ask and best bid in the most recent 26 updated order books. Buy when DIFF crosses above DEA and sell when DIFF crosses below DEA",
                    "fast_period":12,"slow_period":26,"signal_period":9,
                    "name":"MACD_Strategy","min_size":1,"max_size":15,"wake_up_freq":'20'},
"OrderBookImbalance_Strategy":{"Description":"Calculate the order book imbalance using the ratio of the total volume on the bid side to the total volume on the ask side.  When the seller has the upper hand in liquidity, they believe that the price has entered an oversold state and expect it to rise.When the buyer's liquidity is dominant, they believe that the price has entered an overbought state and expect the price to fall",
                    "levels":10,"entry_threshold":0.17,"trail_dist":0.085,
                    "name":"OrderBookImbalance_Strategy","min_size":1,"max_size":20,"wake_up_freq":'20'},
"RSI_Strategy":{"Description":"Judge market sentiment by calculating the relative strength of gains and losses over the past 14 price cycles. Each time a transaction is made, the strategy calculates the RSI value by taking the median price of the buy and sell orders. When the RSI drops below 30 and enters the oversold zone, it is considered that the price decline is too large and a rebound may occur, and a purchase will be made. When the RSI exceeds 70 and enters the overbought zone, it is believed that the price increase is too large and may fall back, and one will sell. This is a counter-transformation strategy, suitable for seizing the opportunities of excessive price hikes and drops in a volatile market.",
                    "levels":10,"entry_threshold":0.17,"trail_dist":0.085,
                    "name":"RSI_Strategy","min_size":1,"max_size":10,"wake_up_freq":'20'}
}