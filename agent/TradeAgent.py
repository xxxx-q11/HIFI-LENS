from agent.TradingAgent import TradingAgent
from util.util import log_print

from math import sqrt
import numpy as np
import pandas as pd
from config.config_strategy import config_strategy

class TradeAgent(TradingAgent):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000,
                     log_orders=False, log_to_file=True, random_state=None):

        # Base class init.
        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, log_to_file=log_to_file, random_state=random_state)

        self.symbol = symbol  # symbol to trade
        self.mid_list, self.avg_20_list, self.avg_50_list = [], [], []
        self.ema12_list, self.ema26_list = [], []
        self.diff_list = []
        self.dea_list = []
        self.macd_list = []
        self.trades_times = 0
        self.indicators_signals = None
        # Add RSI related attributes
        self.price_changes = []  # Price change
        self.rsi_period = 14  # RSI calculation period
        self.rsi_list = []  # RSI value list

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state = 'AWAITING_WAKEUP'

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()

        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        #marked to fundamental
        #rT = self.oracle.observePrice(self.symbol, self.currentTime, self.id, sigma_n=0, random_state=self.random_state)
        # Get the current market quote
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        # Use the mid price to calculate the holding value
        if bid and ask:
            rT = int(bid + ask)/2
        else:
            rT = self.last_trade[self.symbol]
        # final (real) fundamental value times shares held.
        surplus = rT * H

        log_print("surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings['CASH'] - self.starting_cash
        surplus = float(surplus)/self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, surplus {}",
            self.name, H, self.holdings['CASH'], self.starting_cash, rT, surplus)

        #print("Final surplus", self.name, surplus)

    def wakeup(self, currentTime):
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                log_print("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        #delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        #self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))
        self.setWakeup(currentTime + pd.Timedelta(seconds=self.random_state.randint(10, self.wake_up_freq)))

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        self.cancelOrders()

        if type(self) == TradeAgent:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def updateEstimates(self):
        r_T = self.oracle.observePrice(self.symbol, self.currentTime,self.id, sigma_n=100,
                                     random_state=self.random_state)
        log_print("{} estimates r_T = {} as of {}", self.name, r_T, self.currentTime)

        return r_T

    def placeOrder_Value(self):
        #estimate final value of the fundamental price
        #used for surplus calculation
        r_T = self.updateEstimates()
        size = self.random_state.randint(self.min_size, self.max_size)

        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        if bid and ask:
            mid = int((ask+bid)/2)
            spread = abs(ask - bid)

            if self.random_state.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = self.random_state.randint(1, 6)
                #adjustment to the limit price, allowed to post inside the spread
                #or deeper in the book as a passive order to maximize surplus

            if r_T < mid:
                #fundamental belief that price will go down, place a sell order
                buy = False
                p = bid + adjust_int #submit a market order to sell, limit order inside the spread or deeper in the book
            elif r_T >= mid:
                #fundamental belief that price will go up, buy order
                buy = True
                p = ask - adjust_int #submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = self.random_state.randint(0, 1 + 1)
            p = r_T

        # Place the order
        self.placeLimitOrder(self.symbol,size, buy, p)

    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk = True, tag = None):
        super().placeLimitOrder(symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk = True, tag = None)
        self.trades_times +=1

    def random_placeLimitOrder(self,bid,ask,size,buy):
        # self.percent_aggr chance of market order, otherwise place limit order
        if bid and ask:
            mid = int((ask+bid)/2)
            spread = abs(ask - bid)

            if self.random_state.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = self.random_state.randint( 1, 6)
                #adjust_int = np.random.randint( 0, self.depth_spread*spread )
                #adjustment to the limit price, allowed to post inside the spread
                #or deeper in the book as a passive order to maximize surplus

            if buy == False:
                #fundamental belief that price will go down, place a sell order
                p = bid + adjust_int #submit a market order to sell, limit order inside the spread or deeper in the book
            elif buy == True:
                #fundamental belief that price will go up, buy order
                p = ask - adjust_int #submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = self.random_state.randint(0, 1 + 1)
            p = self.oracle.observePrice(self.symbol, self.currentTime,self.id, sigma_n=100,
                                     random_state=self.random_state)
        # Place the order
        self.placeLimitOrder(self.symbol,size, buy, p)

    def placeOrders_Momentum(self):
        """ Momentum Agent actions logic """
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
        size = self.random_state.randint(self.min_size, self.max_size)
        if bid and ask:
            if self.indicators_signals:
                if self.indicators_signals["avg_20"] and self.indicators_signals["avg_50"]:
                    if self.indicators_signals["avg_20"] >= self.indicators_signals["avg_50"]:
                        self.random_placeLimitOrder(bid,ask,size,buy=True)
                    else:
                        self.random_placeLimitOrder(bid,ask,size,buy=False) 

    def placeOrders_MACD(self):
        """Trading Strategies Based on MACD Signals"""
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
            
        signals = self.indicators_signals
        if self.indicators_signals:
            if signals is None:
                return
            size = self.random_state.randint(self.min_size, self.max_size)
            # buy when MACD crosses above DEA
            if signals['is_golden_cross']:
                self.random_placeLimitOrder(bid,ask,size,buy=True)
        
            # sell when MACD crosses below DEA
            elif signals['is_death_cross']:
                self.random_placeLimitOrder(bid,ask,size,buy=False) 

    def placeOrders_OrderBookImbalance(self):
        """Simplified Order Book Imbalance Strategy"""
        # Get the order book data
        bids, asks, last_trade = self.getKnowBiDAsK_all(self.symbol)
        if not (bids and asks):
            log_print("OBI agent inactive: zero bid or ask liquidity")
            return
            
        # Calculate the liquidity of the bid and ask sides
        bid_liq = sum(x[1] for x in bids)
        ask_liq = sum(x[1] for x in asks)
        bid_pct = bid_liq / (bid_liq + ask_liq)
        
        # Place orders based on the imbalance
        self._place_orders_by_imbalance(bid_pct, bids[0][0], asks[0][0])

    def placeOrders_RSI(self):
        """Trading Strategies Based on RSI Signals"""
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
        if not (bid and ask):
            size = self.random_state.randint(self.min_size, self.max_size)
            buy = self.random_state.randint(0, 1 + 1)
            p = self.oracle.observePrice(self.symbol, self.currentTime, self.id, 
                                    sigma_n=100, random_state=self.random_state)
            self.placeLimitOrder(self.symbol, size, buy, p)
            return
            
        signals = self.get_rsi_signals()
        if signals is None:
            return
            
        size = self.random_state.randint(self.min_size, self.max_size)
        
        # buy when RSI < 30
        if signals['is_oversold']:
            self.random_placeLimitOrder(bid, ask, size, buy=True)
            
        #  sell when RSI > 70
        elif signals['is_overbought']:
            self.random_placeLimitOrder(bid, ask, size, buy=False)
                
    def receiveMessage(self, currentTime, msg):
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receiveMessage(currentTime, msg)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == 'AWAITING_SPREAD':
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if msg.body['msg'] == 'QUERY_SPREAD':
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed: return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                #self.Technical_indicators() # Record technical indicators such as moving averages
                self.indicators_signals = msg.body['indicators_signals']
                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=0, high=100), unit='ns')
    
    def calculateSurplus(self):
        """Calculate the surplus of the agent.
        Surplus = Holding Value + Cash Change
        Return the surplus ratio relative to the initial capital.
        """
        # Get the number of holdings (in 100 shares)
        H = int(round(self.getHoldings(self.symbol), -2) / 100)

        # Get the current market quote
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        # Use the mid price to calculate the holding value
        if bid and ask:
            rT = int(bid + ask)/2
        else:
            rT = self.last_trade[self.symbol]

        # Calculate the holding value
        surplus = rT * H
        # Add the cash change
        surplus += self.holdings['CASH'] - self.starting_cash
        # Calculate the surplus ratio relative to the initial capital
        #surplus = float(surplus) / self.starting_cash
        #print("Final relative surplus", self.name, surplus)
        return surplus

    def Technical_indicators(self):
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
        if bid and ask:
            self.mid_list.append((bid + ask) / 2)
            if len(self.mid_list) > 20: self.avg_20_list.append(TradeAgent.ma(self.mid_list, n=20)[-1].round(2))
            if len(self.mid_list) > 50: self.avg_50_list.append(TradeAgent.ma(self.mid_list, n=50)[-1].round(2))
            # Add RSI calculation
            self._calculate_rsi()
            # MACD calculation
            self._calculate_macd()
    def _calculate_macd(self):
        """Calculate MACD indicators"""
        # Ensure there is enough data
        if len(self.mid_list) < 26:
            return
            
        # Calculate 12-day EMA
        if len(self.ema12_list) == 0:
            # First calculation, use simple average
            self.ema12_list.append(sum(self.mid_list[:12]) / 12)
        else:
            # EMA calculation
            ema12 = self.mid_list[-1] * 2 / 13 + self.ema12_list[-1] * 11 / 13
            self.ema12_list.append(ema12)
        
        # Calculate 26-day EMA
        if len(self.ema26_list) == 0:
            # First calculation, use simple average
            self.ema26_list.append(sum(self.mid_list[:26]) / 26)
        else:
            # EMA calculation
            ema26 = self.mid_list[-1] * 2 / 27 + self.ema26_list[-1] * 25 / 27
            self.ema26_list.append(ema26)
        
        # Ensure there is EMA data before calculating DIFF
        if len(self.ema12_list) > 0 and len(self.ema26_list) > 0:
            diff = self.ema12_list[-1] - self.ema26_list[-1]
            self.diff_list.append(diff)
            
            # Calculate DEA (9-day EMA of DIFF)
            if len(self.diff_list) >= 9:
                if len(self.dea_list) == 0:
                    # First calculation of DEA
                    self.dea_list.append(sum(self.diff_list[:9]) / 9)
                else:
                    # DEA EMA calculation
                    dea = self.diff_list[-1] * 2 / 10 + self.dea_list[-1] * 8 / 10
                    self.dea_list.append(dea)
                
                # Calculate MACD column
                macd = 2 * (self.diff_list[-1] - self.dea_list[-1])
                self.macd_list.append(macd)

    def get_macd_signals(self):
        """Get MACD signals"""
        if len(self.macd_list) < 2:
            return None
            
        # MACD(DIFF crosses above DEA)
        is_golden_cross = (self.diff_list[-2] <= self.dea_list[-2] and 
                        self.diff_list[-1] > self.dea_list[-1])
        
        # MACD(DIFF crosses below DEA)
        is_death_cross = (self.diff_list[-2] >= self.dea_list[-2] and 
                        self.diff_list[-1] < self.dea_list[-1])
        
        return {
            'diff': self.diff_list[-1],
            'dea': self.dea_list[-1],
            'macd': self.macd_list[-1],
            'is_golden_cross': is_golden_cross,
            'is_death_cross': is_death_cross
        }
    def _place_orders_by_imbalance(self, bid_pct, bid_price, ask_price):
        """Place orders based on the order book imbalance"""
        size = self.random_state.randint(self.min_size, self.max_size)
        # Buy side is too strong, sell
        if bid_pct > (0.5 + self.entry_threshold):
            self.random_placeLimitOrder(bid_price, ask_price, size, buy=False)
            
        # Sell side is too strong, buy
        elif bid_pct < (0.5 - self.entry_threshold):
            self.random_placeLimitOrder(bid_price, ask_price, size, buy=True)
    # RSI
    def _calculate_rsi(self):
        """Calculate RSI indicators"""
        if len(self.mid_list) < 2:
            return
            
        # Calculate price change
        price_change = self.mid_list[-1] - self.mid_list[-2]
        self.price_changes.append(price_change)
        
        # Ensure there is enough data to calculate RSI
        if len(self.price_changes) >= self.rsi_period:
            gains = []
            losses = []
            
            # Get the recent period price changes
            recent_changes = self.price_changes[-self.rsi_period:]
            
            # Separate gains and losses
            for change in recent_changes:
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            # Calculate average gains and losses
            avg_gain = sum(gains) / self.rsi_period
            avg_loss = sum(losses) / self.rsi_period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            self.rsi_list.append(rsi)

    def get_rsi_signals(self):
        """Get RSI trading signals"""
        if len(self.rsi_list) < 1:
            return None
            
        current_rsi = self.rsi_list[-1]
        
        return {
            'rsi': current_rsi,
            'is_oversold': current_rsi < 30,  # RSI < 30 
            'is_overbought': current_rsi > 70  # RSI > 70 
        }            
    def strategy_chose(self,strategy_type):

        if strategy_type == "strategy_2":
            strategy_type = "Value_Strategy"
            self.sigma_n=config_strategy[strategy_type]["sigma_n"]
            self.r_bar=config_strategy[strategy_type]["r_bar"]
            self.kappa=config_strategy[strategy_type]["kappa"]
            self.lambda_a=config_strategy[strategy_type]["lambda_a"]
            self.sigma_s = config_strategy[strategy_type]["sigma_s"]
            self.min_size=config_strategy[strategy_type]["min_size"]
            self.max_size=config_strategy[strategy_type]["max_size"]
            self.wake_up_freq=config_strategy[strategy_type]["wake_up_freq"]
            self.r_t = self.r_bar
            self.sigma_t = 0
            self.percent_aggr = 0.1                 #percent of time that the agent will aggress the spread
            #self.size = np.random.randint(20, 50)   #size that the agent will be placing
            self.depth_spread = 2
            self.placeOrder = self.placeOrder_Value
        elif strategy_type == "strategy_1":
            strategy_type = "Momentum_Strategy"
            self.min_size=config_strategy[strategy_type]["min_size"]
            self.max_size=config_strategy[strategy_type]["max_size"]
            self.wake_up_freq=config_strategy[strategy_type]["wake_up_freq"]
            self.placeOrder = self.placeOrders_Momentum
            self.percent_aggr = 0.7
            self.depth_spread = 2
        elif strategy_type == "strategy_3":
            strategy_type = "MACD_Strategy"
            self.min_size = config_strategy[strategy_type]["min_size"]
            self.max_size = config_strategy[strategy_type]["max_size"]
            self.wake_up_freq = config_strategy[strategy_type]["wake_up_freq"]
            self.placeOrder = self.placeOrders_MACD
            self.percent_aggr = 0.7
            self.depth_spread = 2
        elif strategy_type == "strategy_4":
            strategy_type = "OrderBookImbalance_Strategy"
            self.min_size=config_strategy[strategy_type]["min_size"]
            self.max_size=config_strategy[strategy_type]["max_size"]
            self.levels = config_strategy[strategy_type]["levels"]
            self.entry_threshold = config_strategy[strategy_type]["entry_threshold"]
            self.wake_up_freq = config_strategy[strategy_type]["wake_up_freq"]
            self.placeOrder = self.placeOrders_OrderBookImbalance
            self.percent_aggr = 0.5
        elif strategy_type == "strategy_5":     # In fact, this strategy has not been enabled
            strategy_type = "RSI_Strategy"
            self.min_size = config_strategy[strategy_type]["min_size"]
            self.max_size = config_strategy[strategy_type]["max_size"]
            self.wake_up_freq = config_strategy[strategy_type]["wake_up_freq"]
            self.placeOrder = self.placeOrders_RSI
            self.percent_aggr = 0.7
            self.depth_spread = 2
        
    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n