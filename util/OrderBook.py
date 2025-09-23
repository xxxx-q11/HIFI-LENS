# This implementation is optimized for performance using sortedcontainers.
# The core data structures self.bids and self.asks are replaced with
# SortedDict to achieve O(log N) for insertions, deletions, and lookups.
# To maintain backward compatibility with any code that might access
# the book directly, the original list-of-lists structure is returned
# via @property getters. All internal methods are modified to use the
# efficient SortedDict implementation.

import sys
from collections import deque
from sortedcontainers import SortedDict

from message.Message import Message
from util.order.LimitOrder import LimitOrder
from util.util import log_print, be_silent

from copy import deepcopy
import pandas as pd
from pandas import json_normalize
from functools import reduce
from scipy.sparse import dok_matrix
from tqdm import tqdm
import time
from itertools import islice


class OrderBook:

    def __init__(self, owner, symbol):
        self.owner = owner
        self.symbol = symbol

        # Optimized Data Structure.
        # use sortedcontainers. SortedDict to store orders, in order to realize the efficient operation of O (log N).
        # _bids key is price, value is the order queue (deque) at that price.
        # In order to make the buy orders sorted in descending order (best offer first), we pass a lambda function, so that SortedDict sorts by the negative value of the price.
        self._bids = SortedDict(lambda price: -price)
        # _asks sorted in ascending order (best offer first), using the default sorting of SortedDict.
        self._asks = SortedDict()

        self.last_trade = None
        self.last_trade_info = []
        self.highest_trade = float('-inf')
        self.lowest_trade = float('inf')

        self.book_log = []
        self.quotes_seen = set()
        self.history = [{}]
        self.last_update_ts = None
        self.sma20 = None
        self.sma50 = None
        self.diff = None # MACD line (DIF)
        self.dea = None  # Signal line (DEA)
        self.macd = None  # MACD column (Histogram)
        self.minute_volume = {} # New: used to record the volume of each minute

        # --- (New) Internal state for high-performance incremental calculation ---
        # SMA calculation queue
        self.price_history_for_sma20 = deque(maxlen=20)
        self.price_history_for_sma50 = deque(maxlen=50)
            
        # Store the latest EMA value
        self.current_ema12 = None
        self.current_ema26 = None

        # Store the latest two DIFF and DEA values, used to judge the crossover
        self.diff_history = deque(maxlen=2)
        self.dea_history = deque(maxlen=2)
            
        # Store the complete DIFF history, only used to initialize DEA calculation
        self._internal_diff_list_for_init = []
        self._transacted_volume = {
            "unrolled_transactions": None,
            "self.history_previous_length": 0,
            # --- (New) Technical indicator attributes ---
        }
        # Use queue to implement efficient minute price recording
        self.minute_prices = deque(maxlen=360)  # Price queue, automatically limit length
        self.minute_times = deque(maxlen=360)   # Time queue, automatically limit length
        self.last_minute = None  # Last recorded minute      

    # --- Compatibility attributes ---
    # The following attributes ensure that any external code accessing .bids or .asks
    # will receive the old version of the data format (list of lists) they expect,
    # ensuring backward compatibility.

    @property
    def bids(self):
        # Optimization: This is a property wrapper, used to ensure backward compatibility.
        # It converts the internal efficient SortedDict structure into the old nested list format in real time.
        # When external code reads self.bids, it feels no change in the internal implementation.
        return [list(orders) for price, orders in self._bids.items()]

    @property
    def asks(self):
        # Optimization: Same as above, this is a compatibility attribute wrapper for asks.
        return [list(orders) for price, orders in self._asks.items()]


    def handleLimitOrder(self, order):
        # This high-level function remains unchanged as it relies on the now-optimized
        # internal methods like executeOrder and enterOrder.
        if order.symbol != self.symbol:
            log_print("{} order discarded.  Does not match OrderBook symbol: {}", order.symbol, self.symbol)
            return

        if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
            log_print("{} order discarded.  Quantity ({}) must be a positive integer.", order.symbol, order.quantity)
            return

        self.history[0][order.order_id] = {'entry_time': self.owner.currentTime,
                                           'quantity': order.quantity, 'is_buy_order': order.is_buy_order,
                                           'limit_price': order.limit_price, 'transactions': [],
                                           'modifications': [],
                                           'cancellations': []}

        matching = True
        self.prettyPrint()
        executed = []

        while matching:
            matched_order = deepcopy(self.executeOrder(order))

            if matched_order:
                filled_order = deepcopy(order)
                filled_order.quantity = matched_order.quantity
                filled_order.fill_price = matched_order.fill_price
                order.quantity -= filled_order.quantity

                log_print("MATCHED: new order {} vs old order {}", filled_order, matched_order)
                log_print("SENT: notifications of order execution to agents {} and {} for orders {} and {}",
                          filled_order.agent_id, matched_order.agent_id, filled_order.order_id, matched_order.order_id)

                self.owner.sendMessage(order.agent_id, Message({"msg": "ORDER_EXECUTED", "order": filled_order}))
                self.owner.sendMessage(matched_order.agent_id,
                                       Message({"msg": "ORDER_EXECUTED", "order": matched_order}))

                executed.append((filled_order.quantity, filled_order.fill_price))

                if order.quantity <= 0:
                    matching = False
            else:
                self.enterOrder(deepcopy(order))
                log_print("ACCEPTED: new order {}", order)
                log_print("SENT: notifications of order acceptance to agent {} for order {}",
                          order.agent_id, order.order_id)
                self.owner.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))
                matching = False

        if not matching:
            if self._bids:
                best_bid_price = self._bids.peekitem(0)[0]
                best_bid_orders = self._bids[best_bid_price]
                self.owner.logEvent('BEST_BID', f"{self.symbol},{best_bid_price},{sum([o.quantity for o in best_bid_orders])}")

            if self._asks:
                best_ask_price = self._asks.peekitem(0)[0]
                best_ask_orders = self._asks[best_ask_price]
                self.owner.logEvent('BEST_ASK', f"{self.symbol},{best_ask_price},{sum([o.quantity for o in best_ask_orders])}")

            if executed:
                trade_qty = sum(q for q, p in executed)
                trade_price = sum(p * q for q, p in executed)
                avg_price = int(round(trade_price / trade_qty))
                log_print("Avg: {} @ ${:0.4f}", trade_qty, avg_price)
                self.owner.logEvent('LAST_TRADE', f"{trade_qty},${avg_price:0.4f}")

                self.last_trade = avg_price
                self.update_trade_extremes()
                self.last_trade_info.append({
                    "time": str(self.owner.currentTime),
                    "price": self.last_trade,
                    "quantity": trade_qty
                })

                # --- (New) Update the volume of each minute ---
                current_minute = self.owner.currentTime.replace(second=0, microsecond=0, nanosecond=0)
                self.minute_volume[current_minute] = self.minute_volume.get(current_minute, 0) + trade_qty

                # --- (New) Update the technical indicators after each transaction ---
                self._update_indicators(avg_price,self.owner.currentTime)
                self.history.insert(0, {})
                self.history = self.history[:self.owner.stream_history + 1]

            if self.owner.book_freq is not None:
                row = {'QuoteTime': self.owner.currentTime}
                for quote, volume in self.getInsideBids():
                    row[quote] = -volume
                    self.quotes_seen.add(quote)
                for quote, volume in self.getInsideAsks():
                    if quote in row and row[quote] is not None:
                        print("WARNING: THIS IS A REAL PROBLEM: an order book contains bids and asks at the same quote price!")
                    row[quote] = volume
                    self.quotes_seen.add(quote)
                self.book_log.append(row)
        self.last_update_ts = self.owner.currentTime
        self.prettyPrint()

    def handleMarketOrder(self, order):
        # This high-level function remains unchanged.
        if order.symbol != self.symbol:
            log_print("{} order discarded.  Does not match OrderBook symbol: {}", order.symbol, self.symbol)
            return

        if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
            log_print("{} order discarded.  Quantity ({}) must be a positive integer.", order.symbol, order.quantity)
            return

        orderbook_side = self.getInsideAsks() if order.is_buy_order else self.getInsideBids()
        limit_orders = {}
        order_quantity = order.quantity
        for price, size in orderbook_side:
            if order_quantity <= size:
                limit_orders[price] = order_quantity
                break
            else:
                limit_orders[price] = size
                order_quantity -= size
        
        log_print("{} placing market order as multiple limit orders", order.symbol, order.quantity)
        for p, q in limit_orders.items():
            limit_order = LimitOrder(order.agent_id, order.time_placed, order.symbol, q, order.is_buy_order, p)
            self.handleLimitOrder(limit_order)

    def executeOrder(self, order):
        # Optimization: Use SortedDict, directly get the best quote through peekitem(0) with O(log N) complexity, without traversal.
        book = self._asks if order.is_buy_order else self._bids
        
        if not book:
            print("WARNING: executeOrder() called on empty order book!")
            print(f"CurrentTime: {self.owner.currentTime}, LastTrade: {self.last_trade}")
            return None

        best_price_level = book.peekitem(0)
        best_price = best_price_level[0]
        best_orders = best_price_level[1]

        if not self.isMatch(order, best_orders[0]):
            return None
        
        matched_order = None
        if order.quantity >= best_orders[0].quantity:
            matched_order = best_orders.popleft()
            if not best_orders:
                del book[best_price]
        else:
            matched_order = deepcopy(best_orders[0])
            matched_order.quantity = order.quantity
            best_orders[0].quantity -= matched_order.quantity

        matched_order.fill_price = matched_order.limit_price

        self.history[0][order.order_id]['transactions'].append((self.owner.currentTime, order.quantity))
        for idx, orders_in_history in enumerate(self.history):
            if matched_order.order_id in orders_in_history:
                self.history[idx][matched_order.order_id]['transactions'].append(
                    (self.owner.currentTime, matched_order.quantity))
                break
        
        return matched_order

    def isMatch(self, order, o):
        # Unchanged.
        if order.is_buy_order == o.is_buy_order:
            print(f"WARNING: isMatch() called on orders of same type: {order} vs {o}")
            return False
        if order.is_buy_order and (order.limit_price >= o.limit_price):
            return True
        if not order.is_buy_order and (order.limit_price <= o.limit_price):
            return True
        return False

    def enterOrder(self, order):
        # Optimization: Use SortedDict, the time complexity of inserting a new order is reduced from O(N) to O(log N).
        # No longer need to manually traverse the list to find the insertion position.
        book = self._bids if order.is_buy_order else self._asks
        price = order.limit_price

        if price not in book:
            book[price] = deque()
        
        book[price].append(order)

    def cancelOrder(self, order):
        # Optimization: Use SortedDict, directly find the order through the price key, the time complexity is reduced from O(N) to O(log N).
        book = self._bids if order.is_buy_order else self._asks
        price = order.limit_price

        if price in book:
            orders_at_price = book[price]
            for i, o in enumerate(orders_at_price):
                if o.order_id == order.order_id:
                    #cancelled_order = orders_at_price.pop(i)
                    cancelled_order = o 
                    del orders_at_price[i] 
                    if not orders_at_price:
                        del book[price]

                    for idx, orders_in_history in enumerate(self.history):
                        if cancelled_order.order_id in orders_in_history:
                            self.history[idx][cancelled_order.order_id]['cancellations'].append(
                                (self.owner.currentTime, cancelled_order.quantity))
                            break
                    
                    log_print("CANCELLED: order {}", order)
                    log_print("SENT: notifications of order cancellation to agent {} for order {}",
                              cancelled_order.agent_id, cancelled_order.order_id)
                    self.owner.sendMessage(order.agent_id,
                                           Message({"msg": "ORDER_CANCELLED", "order": cancelled_order}))
                    self.last_update_ts = self.owner.currentTime
                    return

    def modifyOrder(self, order, new_order):
        # Optimization: Use SortedDict, directly find the order through the price key, the time complexity is reduced from O(N) to O(log N).
        if not self.isSameOrder(order, new_order): return
        book = self._bids if order.is_buy_order else self._asks
        price = order.limit_price

        if price in book:
            orders_at_price = book[price]
            for i, o in enumerate(orders_at_price):
                if o.order_id == order.order_id:
                    orders_at_price[i] = new_order
                    for idx, orders in enumerate(self.history):
                        if new_order.order_id in orders:
                            self.history[idx][new_order.order_id]['modifications'].append(
                                (self.owner.currentTime, new_order.quantity))
                            log_print("MODIFIED: order {}", order)
                            log_print("SENT: notifications of order modification to agent {} for order {}",
                                      new_order.agent_id, new_order.order_id)
                            self.owner.sendMessage(order.agent_id,
                                                   Message({"msg": "ORDER_MODIFIED", "new_order": new_order}))
                    self.last_update_ts = self.owner.currentTime
                    return

    def getInsideBids(self, depth=sys.maxsize):
        # Optimization: Directly traverse the sorted SortedDict, no longer need to manually sort and slice.
        book = []
        # Use itertools.islice efficiently to handle the depth parameter, avoiding complete traversal.

        for price, orders in islice(self._bids.items(), depth):
            qty = sum(o.quantity for o in orders)
            book.append((price, qty))
        return book

    def getInsideAsks(self, depth=sys.maxsize):
        # Optimization: Same as above, directly traverse the sorted SortedDict.
        book = []
        from itertools import islice
        for price, orders in islice(self._asks.items(), depth):
            qty = sum(o.quantity for o in orders)
            book.append((price, qty))
        return book

    # --- Unchanged utility, history, and logging methods below ---

    def _get_recent_history(self):
        if self._transacted_volume["self.history_previous_length"] == 0:
            self._transacted_volume["self.history_previous_length"] = len(self.history)
            return self.history
        elif self._transacted_volume["self.history_previous_length"] == len(self.history):
            return {}
        else:
            idx = len(self.history) - self._transacted_volume["self.history_previous_length"] - 1
            recent_history = self.history[0:idx]
            self._transacted_volume["self.history_previous_length"] = len(self.history)
            return recent_history

    def _update_unrolled_transactions(self, recent_history):
        new_unrolled_txn = self._unrolled_transactions_from_order_history(recent_history)
        old_unrolled_txn = self._transacted_volume["unrolled_transactions"]
        total_unrolled_txn = pd.concat([old_unrolled_txn, new_unrolled_txn], ignore_index=True)
        self._transacted_volume["unrolled_transactions"] = total_unrolled_txn

    def _unrolled_transactions_from_order_history(self, history):
        unrolled_history = []
        for elem in history:
            for _, val in elem.items():
                unrolled_history.append(val)

        unrolled_history_df = pd.DataFrame(unrolled_history, columns=[
            'entry_time', 'quantity', 'is_buy_order', 'limit_price', 'transactions', 'modifications', 'cancellations'
        ])

        if unrolled_history_df.empty:
            return pd.DataFrame(columns=['execution_time', 'quantity'])

        executed_transactions = unrolled_history_df[unrolled_history_df['transactions'].map(lambda d: len(d)) > 0]
        transaction_list = [element for list_ in executed_transactions['transactions'].values for element in list_]
        unrolled_transactions = pd.DataFrame(transaction_list, columns=['execution_time', 'quantity'])
        unrolled_transactions = unrolled_transactions.sort_values(by=['execution_time'])
        unrolled_transactions = unrolled_transactions.drop_duplicates(keep='last')
        return unrolled_transactions

    def get_transacted_volume(self, lookback_period='10min'):
        recent_history = self._get_recent_history()
        self._update_unrolled_transactions(recent_history)
        unrolled_transactions = self._transacted_volume["unrolled_transactions"]
        lookback_pd = pd.to_timedelta(lookback_period)
        window_start = self.owner.currentTime - lookback_pd
        executed_within_lookback_period = unrolled_transactions[unrolled_transactions['execution_time'] >= window_start]
        transacted_volume = executed_within_lookback_period['quantity'].sum()
        return transacted_volume

    def isBetterPrice(self, order, o):
        if order.is_buy_order != o.is_buy_order:
            print(f"WARNING: isBetterPrice() called on orders of different type: {order} vs {o}")
            return False
        if order.is_buy_order and (order.limit_price > o.limit_price):
            return True
        if not order.is_buy_order and (order.limit_price < o.limit_price):
            return True
        return False

    def isEqualPrice(self, order, o):
        return order.limit_price == o.limit_price

    def isSameOrder(self, order, new_order):
        return order.order_id == new_order.order_id

    def book_log_to_df(self):
        quotes = sorted(list(self.quotes_seen))
        log_len = len(self.book_log)
        quote_idx_dict = {quote: idx for idx, quote in enumerate(quotes)}
        quotes_times = []
        S = dok_matrix((log_len, len(quotes)), dtype=int)
        for i, row in enumerate(tqdm(self.book_log, desc="Processing orderbook log")):
            quotes_times.append(row['QuoteTime'])
            for quote, vol in row.items():
                if quote == "QuoteTime":
                    continue
                S[i, quote_idx_dict[quote]] = vol
        S = S.tocsc()
        df = pd.DataFrame.sparse.from_spmatrix(S, columns=quotes)
        df.insert(0, 'QuoteTime', quotes_times, allow_duplicates=True)
        return df

    def prettyPrint(self, silent=False):
        if be_silent: return ''
        book = f"{self.symbol} order book as of {self.owner.currentTime}\n"
        try:
            oracle_price = self.owner.oracle.observePrice(self.symbol, self.owner.currentTime, sigma_n=0, random_state=self.owner.random_state)
        except (AttributeError, TypeError):
            oracle_price = "N/A" # Handle cases where oracle is not available or arguments are incorrect.
        
        book += f"Last trades: simulated {self.last_trade}, historical {oracle_price}\n"
        book += f"{'BID':>10s}{'PRICE':>10s}{'ASK':>10s}\n"
        book += f"{'---':>10s}{'-----':>10s}{'---':>10s}\n"

        # The getInsideAsks returns asks sorted best to worst (lowest to highest price).
        # To print them from highest price down, we reverse the list.
        for quote, volume in reversed(self.getInsideAsks()):
            book += f"{'':10s}{quote:<10d}{volume:<10d}\n"
        
        # The getInsideBids returns bids sorted best to worst (highest to lowest price).
        # We print them in this natural order.
        for quote, volume in self.getInsideBids():
            book += f"{volume:>10d}{quote:>10d}{'':10s}\n"

        if silent: return book
        log_print(book)

    def update_trade_extremes(self):
        if self.last_trade is not None:
            self.highest_trade = max(self.highest_trade, self.last_trade)
            self.lowest_trade = min(self.lowest_trade, self.last_trade)

    def cancel_all_orders(self):
        # Optimization: Directly traverse the efficient SortedDict data structure.
        for book in [self._bids, self._asks]:
            for price_level in book.values():
                for order in price_level:
                    self.owner.sendMessage(order.agent_id,
                        Message({"msg": "ORDER_CANCELLED", "order": order}))
        self._bids.clear()
        self._asks.clear()

    def _update_indicators(self, new_price,cuurentTime):
        """
         Combine the calculation logic of TradeAgent and the high-performance data structure to update the indicators.
        """
        # --- 1. Update SMA ---
        self.price_history_for_sma20.append(new_price)
        self.price_history_for_sma50.append(new_price)

        if len(self.price_history_for_sma20) == 20:
            self.sma20 = sum(self.price_history_for_sma20) / 20
        
        if len(self.price_history_for_sma50) == 50:
            self.sma50 = sum(self.price_history_for_sma50) / 50

        # --- 2. Update MACD related indicators ---

        self.record_minute_price(cuurentTime,new_price)

        # Calculate EMA12
        if self.current_ema12 is None:
            all_prices = list(self.price_history_for_sma50)
            if len(all_prices) >= 12:
                self.current_ema12 = sum(all_prices[-12:]) / 12
        else:
            self.current_ema12 = new_price * 2 / 13 + self.current_ema12 * 11 / 13

        # Calculate EMA26
        if self.current_ema26 is None:
            all_prices = list(self.price_history_for_sma50)
            if len(all_prices) == 26:
                self.current_ema26 = sum(all_prices) / 26
        else:
            self.current_ema26 = new_price * 2 / 27 + self.current_ema26 * 25 / 27

        # Calculate DIFF (MACD line)
        if self.current_ema12 is not None and self.current_ema26 is not None:
            self.diff = self.current_ema12 - self.current_ema26
            self.diff_history.append(self.diff)
            self._internal_diff_list_for_init.append(self.diff)

            # Calculate DEA (signal line)
            if self.diff is not None:
                if self.dea is None: # First calculation of DEA
                    if len(self._internal_diff_list_for_init) == 9:
                        self.dea = sum(self._internal_diff_list_for_init) / 9
                        self.dea_history.append(self.dea)
                else: # Subsequent incremental calculation of DEA
                    self.dea = self.diff * 2 / 10 + self.dea * 8 / 10
                    self.dea_history.append(self.dea)

            # Calculate MACD column
            if self.diff is not None and self.dea is not None:
                self.macd = 2 * (self.diff - self.dea)

        log_print(f"Indicators updated for {self.symbol}: SMA20={self.sma20}, SMA50={self.sma50}, DIFF={self.diff}, DEA={self.dea}, MACD={self.macd}")

    def get_indicators_signals(self):
        """
        Get MACD signal, follow the logic of TradeAgent, but use an efficient deque.
        """
        if len(self.diff_history) < 2 or len(self.dea_history) < 2:
            return None

        # MACD golden cross judgment (DIFF crosses DEA)
        is_golden_cross = (self.diff_history[0] <= self.dea_history[0] and \
                           self.diff_history[1] > self.dea_history[1])
        
        # MACD death cross judgment (DIFF crosses DEA)
        is_death_cross = (self.diff_history[0] >= self.dea_history[0] and \
                          self.diff_history[1] < self.dea_history[1])
        recent_minute_prices = self.get_recent_minute_prices(30)
        return {
            'diff': self.diff,
            'dea': self.dea,
            'macd': self.macd,
            'is_golden_cross': is_golden_cross,
            'is_death_cross': is_death_cross,
            "avg_20":self.sma20,
            "avg_50":self.sma50,
            "recent_minute_prices":recent_minute_prices
        }
    def record_minute_price(self, currentTime, price):
        """Use queue to efficiently record minute price - automatically maintain 360 records"""
        
        # Extract minute time (remove seconds and microseconds)
        minute_time = currentTime.replace(second=0, microsecond=0, nanosecond=0)
        
        # If the same minute has been recorded, skip
        if minute_time == self.last_minute:
            return
        
        # Use queue to automatically add, automatically delete the oldest data when maxlen is exceeded
        self.minute_prices.append(price)
        self.minute_times.append(minute_time)
        self.last_minute = minute_time
        
        # No need to manually check length, deque will automatically handle it
    def get_recent_minute_prices(self, n=30):
        """Get the price of the last N minutes - queue efficient slicing"""
        if not self.minute_prices:
            return []
        
        length = len(self.minute_prices)
        if n >= length:
            return list(self.minute_prices)
        
        # Use itertools.islice more efficiently (need to import itertools)
        
        return list(islice(self.minute_prices, length - n, length))