# The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp to open and close trading,
# a list of equity symbols for which it should create order books, a frequency at which to archive snapshots
# of its order books, a pipeline delay (in ns) for order activity, the exchange computation delay (in ns),
# the levels of order stream history to maintain per symbol (maintains all orders that led to the last N trades),
# whether to log all order activity to the agent log, and a random state object (already seeded) to use
# for stochasticity.
from agent.FinancialAgent import FinancialAgent
from message.Message import Message
from util.OrderBook import OrderBook
from util.util import log_print
import time
import datetime as dt
import json
import numpy as np
from util.last_day_trade import last_day_trade_history
from run import all_parallel_process_message,all_parallel_chose_agent_strategy
import os
import orjson 
# from util.simple_llm_cache import get_cache
# from run import serialize_agents_state, deserialize_agents_state


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)

from copy import deepcopy
from util.News import News_schedule

class ExchangeAgent(FinancialAgent):

  def __init__(self, id, name, type, mkt_open, mkt_close, symbols,num_example = None, book_freq='S', wide_book=False, pipeline_delay = 40000,
               computation_delay = 1, stream_history = 0, log_orders = False, random_state = None):

    super().__init__(id, name, type, random_state)

    # Do not request repeated wakeup calls.
    self.reschedule = False

    # Store this exchange's open and close times.
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close

    # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
    # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
    # hours, etc).
    self.pipeline_delay = pipeline_delay

    # Computation delay is applied on every wakeup call or message received.
    self.computation_delay = computation_delay

    # The exchange maintains an order stream of all orders leading to the last L trades
    # to support certain agents from the auction literature (GD, HBL, etc).
    self.stream_history = stream_history

    # Log all order activity?
    self.log_orders = log_orders

    # Create an order book for each symbol.
    self.order_books = {}

    for symbol in symbols:
      self.order_books[symbol] = OrderBook(self, symbol)

    # At what frequency will we archive the order books for visualization and analysis?
    self.book_freq = book_freq

    # Store orderbook in wide format? ONLY WORKS with book_freq == 0
    self.wide_book = wide_book


    # The subscription dict is a dictionary with the key = agent ID,
    # value = dict (key = symbol, value = list [levels (no of levels to recieve updates for),
    # frequency (min number of ns between messages), last agent update timestamp]
    # e.g. {101 : {'AAPL' : [1, 10, pd.Timestamp(10:00:00)}}
    self.subscription_dict = {}
    # Market news dictionary
    self.market_subscribe_news_dict = {}
    # News update count
    self.news_update_count = 0
    self.market_news = {}
    # LLM_Agent number
    self.num_example = num_example
    # Yesterday's trading High, Low, Open, Close four prices
    self.last_day_trade = last_day_trade_history
    # Yesterday's trading count
    self.last_day_trade_price_len = 0
  # The exchange agent overrides this to obtain a reference to an oracle.
  # This is needed to establish a "last trade price" at open (i.e. an opening
  # price) in case agents query last trade before any simulated trades are made.
  # This can probably go away once we code the opening cross auction.
  def kernelInitializing (self, kernel):
    super().kernelInitializing(kernel)

    self.oracle = self.kernel.oracle

    # Obtain opening prices (in integer cents).  These are not noisy right now.
    for symbol in self.order_books:
      try:
        self.order_books[symbol].last_trade = self.oracle.getDailyOpenPrice(symbol, self.mkt_open)
        log_print ("Opening price for {} is {}", symbol, self.order_books[symbol].last_trade)
      except AttributeError as e:
        log_print(str(e))


  # The exchange agent overrides this to additionally log the full depth of its
  # order books for the entire day.
  def kernelTerminating (self):
    super().kernelTerminating()

    # Use the optimized data exporter, significantly reducing data saving time
    self.save_orderbook_data_to_json()
    self._fallback_kernel_terminating()

  def _fallback_kernel_terminating(self):
    """
  The original kernelTerminating method is used as a backup solution
    """
    # If the oracle supports writing the fundamental value series for its
    # symbols, write them to disk.
    if hasattr(self.oracle, 'f_log'):
      for symbol in self.oracle.f_log:
        dfFund = pd.DataFrame(self.oracle.f_log[symbol])
        if not dfFund.empty:
          dfFund.set_index('FundamentalTime', inplace=True)
          self.writeLog(dfFund, filename='fundamental_{}'.format(symbol))
          log_print("Fundamental archival complete.")
    if self.book_freq is None: 
      print("No order book log to export.")
      return
    else:
      # Iterate over the order books controlled by this exchange.
      for symbol in self.order_books:
        start_time = dt.datetime.now()
        self.logOrderBookSnapshots(symbol)
        end_time = dt.datetime.now()
        print("Time taken to log the order book: {}".format(end_time - start_time))
        print("Order book archival complete.")
  def save_orderbook_data_to_json(self):
      """
      Save orderbook data to JSON file, using orjson to improve performance
      """
      try:
          # Ensure the directory exists
          os.makedirs("Experimental_data/result_data", exist_ok=True)
          for symbol in self.order_books:
            # 1. Save last_trade_info
            last_trade_info_json_path = f"Experimental_data/result_data/orderbook_last_trade_info_{symbol}.json"
            with open(last_trade_info_json_path, "wb") as f:
                f.write(orjson.dumps(
                    self.order_books[symbol].last_trade_info, 
                    option=orjson.OPT_SERIALIZE_NUMPY | 
                          orjson.OPT_NAIVE_UTC |
                          orjson.OPT_INDENT_2
                ))
            print(f"Order book last_trade_info for {symbol} saved to {last_trade_info_json_path}")

            # 2. Save minute_volume
            minute_volume_json_path = f"Experimental_data/result_data/orderbook_minute_volume_{symbol}.json"
            minute_volume_str_keys = {str(k): v for k, v in self.order_books[symbol].minute_volume.items()}
            with open(minute_volume_json_path, "wb") as f:
                f.write(orjson.dumps(
                    minute_volume_str_keys, 
                    option=orjson.OPT_SERIALIZE_NUMPY | 
                          orjson.OPT_NAIVE_UTC |
                          orjson.OPT_INDENT_2
                ))
            print(f"orderbook_minute_volume for {symbol} saved to {minute_volume_json_path}")
            
            # 3. Save book_log
            book_log_json_path = f"Experimental_data/result_data/orderbook_book_log_{symbol}.json"
            with open(book_log_json_path, "wb") as f:
                f.write(orjson.dumps(
                    self.order_books[symbol].book_log, 
                    option=orjson.OPT_SERIALIZE_NUMPY | 
                          orjson.OPT_NAIVE_UTC |
                          orjson.OPT_INDENT_2
                ))
            print(f"Order book book_log for {symbol} saved to {book_log_json_path}")
            
            print(f"All orderbook data has been saved: {symbol}")
          
      except Exception as e:
          print(f"Error saving orderbook data {symbol}: {e}")

          
  def receiveMessage(self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Unless the intent of an experiment is to examine computational issues within an Exchange,
    # it will typically have either 1 ns delay (near instant but cannot process multiple orders
    # in the same atomic time unit) or 0 ns delay (can process any number of orders, always in
    # the atomic time unit in which they are received).  This is separate from, and additional
    # to, any parallel pipeline delay imposed for order book activity.

    # Note that computation delay MUST be updated before any calls to sendMessage.
    self.setComputationDelay(self.computation_delay)

    # Whether the news is updated at the current time
    self.publishMarketNews(currentTime,News_schedule)
    # Is the exchange closed?  (This block only affects post-close, not pre-open.)
    if currentTime > self.mkt_close:
      # Most messages after close will receive a 'MKT_CLOSED' message in response.  A few things
      # might still be processed, like requests for final trade prices or such.
      if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
        log_print("{} received {}: {}", self.name, msg.body['msg'], msg.body['order'])
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return
      elif 'QUERY' in msg.body['msg']:
        # Specifically do allow querying after market close, so agents can get the
        # final trade of the day as their "daily close" price for a symbol.
        pass
      else:
        log_print("{} received {}, discarded: market is closed.", self.name, msg.body['msg'])
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return
    

    # Log order messages only if that option is configured.  Log all other messages.
    if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
      if self.log_orders: self.logEvent(msg.body['msg'], msg.body['order'].to_dict())
    else:
      self.logEvent(msg.body['msg'], msg.body['sender'])

    # Handle the DATA SUBSCRIPTION request and cancellation messages from the agents.
    if msg.body['msg'] in ["MARKET_DATA_SUBSCRIPTION_REQUEST", "MARKET_DATA_SUBSCRIPTION_CANCELLATION"]:
      log_print("{} received {} request from agent {}", self.name, msg.body['msg'], msg.body['sender'])
      self.updateSubscriptionDict(msg, currentTime)
    # Process the market news request
    if msg.body['msg'] == "MARKET_NEWS_REQUEST":
      log_print("{} received {} request from agent {}", self.name, msg.body['msg'], msg.body['sender'])
      self.updateMarketSubscribeNewsDict(msg, currentTime)

    # Handle all message types understood by this exchange.
    if msg.body['msg'] == "WHEN_MKT_OPEN":
      log_print("{} received WHEN_MKT_OPEN request from agent {}", self.name, msg.body['sender'])

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_OPEN", "data": self.mkt_open}))
    elif msg.body['msg'] == "WHEN_MKT_CLOSE":
      log_print("{} received WHEN_MKT_CLOSE request from agent {}", self.name, msg.body['sender'])

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_CLOSE", "data": self.mkt_close}))
    elif msg.body['msg'] == "QUERY_LAST_TRADE":
      symbol = msg.body['symbol']
      if symbol not in self.order_books:
        log_print("Last trade request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_LAST_TRADE ({}) request from agent {}", self.name, symbol, msg.body['sender'])

        # Return the single last executed trade price (currently not volume) for the requested symbol.
        # This will return the average share price if multiple executions resulted from a single order.
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_LAST_TRADE", "symbol": symbol,
                                                      "data": self.order_books[symbol].last_trade,
                                                      "mkt_closed": True if currentTime > self.mkt_close else False}))
        #print()
        
    elif msg.body['msg'] == "QUERY_SPREAD":
      symbol = msg.body['symbol']
      depth = msg.body['depth']
      if symbol not in self.order_books:
        log_print("Bid-ask spread request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_SPREAD ({}:{}) request from agent {}", self.name, symbol, depth,
                  msg.body['sender'])

        # Return the requested depth on both sides of the order book for the requested symbol.
        # Returns price levels and aggregated volume at each level (not individual orders).
        if msg.body['sender'] in range(1,1+self.num_example):
          extracted_values = self.extract_evenly_spaced_trade_info(symbol, num_points=20)
          self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
                                              "bids": self.order_books[symbol].getInsideBids(depth),
                                              "asks": self.order_books[symbol].getInsideAsks(depth),
                                              "data": self.order_books[symbol].last_trade,
                                              "trade_history":extracted_values,
                                              "mkt_closed": True if currentTime > self.mkt_close else False,
                                              "book": ''}))
        else:
          indicators_signals = self.order_books[symbol].get_indicators_signals()
          self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
                                                        "bids": self.order_books[symbol].getInsideBids(depth),
                                                        "asks": self.order_books[symbol].getInsideAsks(depth),
                                                        "data": self.order_books[symbol].last_trade,
                                                        "indicators_signals": indicators_signals,
                                                        "mkt_closed": True if currentTime > self.mkt_close else False,
                                                        "book": ''}))

        # It is possible to also send the pretty-printed order book to the agent for logging, but forcing pretty-printing
        # of a large order book is very slow, so we should only do it with good reason.  We don't currently
        # have a configurable option for it.
        # "book": self.order_books[symbol].prettyPrint(silent=True) }))
    elif msg.body['msg'] == "QUERY_ORDER_STREAM":
      symbol = msg.body['symbol']
      length = msg.body['length']

      if symbol not in self.order_books:
        log_print("Order stream request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_ORDER_STREAM ({}:{}) request from agent {}", self.name, symbol, length,
                  msg.body['sender'])

      # We return indices [1:length] inclusive because the agent will want "orders leading up to the last
      # L trades", and the items under index 0 are more recent than the last trade.
      self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_ORDER_STREAM", "symbol": symbol, "length": length,
                                                    "mkt_closed": True if currentTime > self.mkt_close else False,
                                                    "orders": self.order_books[symbol].history[1:length + 1]
                                                    }))
    elif msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME':
      symbol = msg.body['symbol']
      lookback_period = msg.body['lookback_period']
      if symbol not in self.order_books:
        log_print("Order stream request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_TRANSACTED_VOLUME ({}:{}) request from agent {}", self.name, symbol, lookback_period,
                  msg.body['sender'])
      self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_TRANSACTED_VOLUME", "symbol": symbol,
                                                    "transacted_volume": self.order_books[symbol].get_transacted_volume(lookback_period),
                                                    "mkt_closed": True if currentTime > self.mkt_close else False
                                                    }))
    elif msg.body['msg'] == "LIMIT_ORDER":
      order = msg.body['order']
      log_print("{} received LIMIT_ORDER: {}", self.name, order)
      if order.symbol not in self.order_books:
        log_print("Limit Order discarded.  Unknown symbol: {}", order.symbol)
      else:
        # Hand the order to the order book for processing.
        self.order_books[order.symbol].handleLimitOrder(deepcopy(order))
        self.publishOrderBookData()
    elif msg.body['msg'] == "MARKET_ORDER":
      order = msg.body['order']
      log_print("{} received MARKET_ORDER: {}", self.name, order)
      if order.symbol not in self.order_books:
        log_print("Market Order discarded.  Unknown symbol: {}", order.symbol)
      else:
        # Hand the market order to the order book for processing.
        self.order_books[order.symbol].handleMarketOrder(deepcopy(order))
        self.publishOrderBookData()
    elif msg.body['msg'] == "CANCEL_ORDER":
      # Note: this is somewhat open to abuse, as in theory agents could cancel other agents' orders.
      # An agent could also become confused if they receive a (partial) execution on an order they
      # then successfully cancel, but receive the cancel confirmation first.  Things to think about
      # for later...
      order = msg.body['order']
      log_print("{} received CANCEL_ORDER: {}", self.name, order)
      if order.symbol not in self.order_books:
        log_print("Cancellation request discarded.  Unknown symbol: {}", order.symbol)
      else:
        # Hand the order to the order book for processing.
        self.order_books[order.symbol].cancelOrder(deepcopy(order))
        self.publishOrderBookData()
    elif msg.body['msg'] == 'MODIFY_ORDER':
      # Replace an existing order with a modified order.  There could be some timing issues
      # here.  What if an order is partially executed, but the submitting agent has not
      # yet received the norification, and submits a modification to the quantity of the
      # (already partially executed) order?  I guess it is okay if we just think of this
      # as "delete and then add new" and make it the agent's problem if anything weird
      # happens.
      order = msg.body['order']
      new_order = msg.body['new_order']
      log_print("{} received MODIFY_ORDER: {}, new order: {}".format(self.name, order, new_order))
      if order.symbol not in self.order_books:
        log_print("Modification request discarded.  Unknown symbol: {}".format(order.symbol))
      else:
        self.order_books[order.symbol].modifyOrder(deepcopy(order), deepcopy(new_order))
        self.publishOrderBookData()

  def updateSubscriptionDict(self, msg, currentTime):
    # The subscription dict is a dictionary with the key = agent ID,
    # value = dict (key = symbol, value = list [levels (no of levels to recieve updates for),
    # frequency (min number of ns between messages), last agent update timestamp]
    # e.g. {101 : {'AAPL' : [1, 10, pd.Timestamp(10:00:00)}}
    if msg.body['msg'] == "MARKET_DATA_SUBSCRIPTION_REQUEST":
      agent_id, symbol, levels, freq = msg.body['sender'], msg.body['symbol'], msg.body['levels'], msg.body['freq']
      self.subscription_dict[agent_id] = {symbol: [levels, freq, currentTime]}
    elif msg.body['msg'] == "MARKET_DATA_SUBSCRIPTION_CANCELLATION":
      agent_id, symbol = msg.body['sender'], msg.body['symbol']
      del self.subscription_dict[agent_id][symbol]
  # Process the market news request
  def updateMarketSubscribeNewsDict(self, msg,currentTime):
    agent_id, symbol = msg.body['sender'], msg.body['symbol']
    self.market_subscribe_news_dict[agent_id] = symbol

  def publishOrderBookData(self):
    '''
    The exchange agents sends an order book update to the agents using the subscription API if one of the following
    conditions are met:
    1) agent requests ALL order book updates (freq == 0)
    2) order book update timestamp > last time agent was updated AND the orderbook update time stamp is greater than
    the last agent update time stamp by a period more than that specified in the freq parameter.
    '''
    for agent_id, params in self.subscription_dict.items():
      for symbol, values in params.items():
        levels, freq, last_agent_update = values[0], values[1], values[2]
        orderbook_last_update = self.order_books[symbol].last_update_ts
        if (freq == 0) or \
           ((orderbook_last_update > last_agent_update) and ((orderbook_last_update - last_agent_update).total_seconds() * 1e9 >= freq)):
          self.sendMessage(agent_id, Message({"msg": "MARKET_DATA",
                                              "symbol": symbol,
                                              "bids": self.order_books[symbol].getInsideBids(levels),
                                              "asks": self.order_books[symbol].getInsideAsks(levels),
                                              "last_transaction": self.order_books[symbol].last_trade,
                                              "exchange_ts": self.currentTime}))
          self.subscription_dict[agent_id][symbol][2] = orderbook_last_update
          
  def publishMarketNews(self,currentTime,News_schedule):
    if self.news_update_count < len(News_schedule["times"]):
      if currentTime > News_schedule["times"][self.news_update_count]:
        print("It's time for the news update")
        if self.market_subscribe_news_dict:
          print("There are news subscriptions.")
          index_time = News_schedule["times"][self.news_update_count]
          for agent_id in self.market_subscribe_news_dict:
            symbol = self.market_subscribe_news_dict[agent_id]
            extracted_values = self.extract_evenly_spaced_trade_info(symbol, num_points=20)
            self.sendMessage(agent_id, Message({"msg": "MARKET_NEWS",
                                                "symbol": symbol,
                                                "bids": self.order_books[symbol].getInsideBids(5),
                                                "asks": self.order_books[symbol].getInsideAsks(5),
                                                "last_transaction": self.order_books[symbol].last_trade,
                                                "News": News_schedule["message"][index_time],
                                                "trade_history":extracted_values,
                                                "exchange_ts": self.currentTime}))
            self.market_news.update({str(index_time):News_schedule["message"][index_time]})
        else:
          print("没There are news subscriptions.")
          # time.sleep(10)
        self.news_update_count += 1
        News_schedule["current_time"] = self.news_update_count

  def logOrderBookSnapshots(self, symbol):
    """
    Log full depth quotes (price, volume) from this order book at some pre-determined frequency. Here we are looking at
    the actual log for this order book (i.e. are there snapshots to export, independent of the requested frequency).
    """
    def get_quote_range_iterator(s):
      """ Helper method for order book logging. Takes pandas Series and returns python range() from first to last
          element.
      """
      forbidden_values = [0, 19999900] # TODO: Put constant value in more sensible place!
      quotes = sorted(s)
      for val in forbidden_values:
        try: quotes.remove(val)
        except ValueError:
          pass
      return quotes

    book = self.order_books[symbol]

    if book.book_log:

      print("Logging order book to file...")
      dfLog = book.book_log_to_df()
      dfLog.set_index('QuoteTime', inplace=True)
      dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
      dfLog.sort_index(inplace=True)

      if str(self.book_freq).isdigit() and int(self.book_freq) == 0:  # Save all possible information
        # Get the full range of quotes at the finest possible resolution.
        quotes = get_quote_range_iterator(dfLog.columns.unique())

        # Restructure the log to have multi-level rows of all possible pairs of time and quote
        # with volume as the only column.
        if not self.wide_book:
          filledIndex = pd.MultiIndex.from_product([dfLog.index, quotes], names=['time', 'quote'])
          dfLog = dfLog.stack()
          dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FULL'

      else:  # Sample at frequency self.book_freq
        # With multiple quotes in a nanosecond, use the last one, then resample to the requested freq.
        dfLog = dfLog.resample(self.book_freq).ffill()
        dfLog.sort_index(inplace=True)

        # Create a fully populated index at the desired frequency from market open to close.
        # Then project the logged data into this complete index.
        time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
        dfLog = dfLog.reindex(time_idx, method='ffill')
        dfLog.sort_index(inplace=True)

        if not self.wide_book:
          dfLog = dfLog.stack()
          dfLog.sort_index(inplace=True)

          # Get the full range of quotes at the finest possible resolution.
          quotes = get_quote_range_iterator(dfLog.index.get_level_values(1).unique())

          # Restructure the log to have multi-level rows of all possible pairs of time and quote
          # with volume as the only column.
          filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time', 'quote'])
          dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FREQ_{self.book_freq}'

      # Final cleanup
      if not self.wide_book:
        dfLog.rename('Volume')
        df = pd.DataFrame(index=dfLog.index)
        df['Volume'] = dfLog
      else:
        df = dfLog
        df = df.reindex(sorted(df.columns), axis=1)

      # Archive the order book snapshots directly to a file named with the symbol, rather than
      # to the exchange agent log.
      self.writeLog(df, filename=filename)
      print("Order book logging complete!")
    else:
        print("No order book log to export.")

  def sendMessage (self, recipientID, msg):
    # The ExchangeAgent automatically applies appropriate parallel processing pipeline delay
    # to those message types which require it.
    # TODO: probably organize the order types into categories once there are more, so we can
    # take action by category (e.g. ORDER-related messages) instead of enumerating all message
    # types to be affected.
    if msg.body['msg'] in ['ORDER_ACCEPTED', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
      # Messages that require order book modification (not simple queries) incur the additional
      # parallel processing delay as configured.
      super().sendMessage(recipientID, msg, delay = self.pipeline_delay)
      if self.log_orders: self.logEvent(msg.body['msg'], msg.body['order'].to_dict())
    else:
      # Other message types incur only the currently-configured computation delay for this agent.
      super().sendMessage(recipientID, msg)

  # Simple accessor methods for the market open and close times.
  def getMarketOpen(self):
    return self.__mkt_open

  def getMarketClose(self):
    return self.__mkt_close
  
  def update_last_day_trade_history(self):
    for symbol in self.order_books:
      close_price = self.order_books[symbol].last_trade_info[-1]["price"]
      open_price = self.order_books[symbol].last_trade_info[0]["price"]
      high_price = self.order_books[symbol].highest_trade
      low_price = self.order_books[symbol].lowest_trade
      last_day_trade = {
        "Closing price": close_price,
        "Opening price": open_price,
        "highest price": high_price,
        "lowest price": low_price
        }
    return last_day_trade

  def extract_evenly_spaced_trade_info(self, symbol, num_points=20):
    """
    Extract num_points elements from the last_trade_info of the specified symbol (including the first and last elements).
    If the number of elements is less than num_points, return all elements.
    """
    num_len = int(self.last_day_trade_price_len)
    last_trade_info_list = self.order_books[symbol].last_trade_info[num_len:]
    num_elements = len(last_trade_info_list)
    if num_elements == 0:
        return []
    if num_elements <= num_points:
        return last_trade_info_list
    # Generate equidistant indices
    indices = np.round(np.linspace(0, num_elements - 1, num=num_points)).astype(int)
    # Use the original indices to ensure the number of elements
    return [last_trade_info_list[i] for i in indices]
  
  def cancel_all_open_orders(self):
    """Cancel all open orders when the market closes, and notify the agents"""
    for symbol, order_book in self.order_books.items():
        order_book.cancel_all_orders()
  
  def return_len_trade_price(self):
    num = 0
    for symbol in self.order_books:
      num = len(self.order_books[symbol].last_trade_info)
    return num