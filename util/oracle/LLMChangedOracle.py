import datetime as dt
import numpy as np
import pandas as pd
import os, random, sys

from math import sqrt
from util.util import log_print


class LLMChangedOracle:

  def __init__(self, mkt_open, mkt_close, symbol):
    # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
    # inner keys: r_bar, kappa, sigma_s.
    #self.symbols = symbols
    #self.llm_changed_price = 100000 # Manually set the Open price to 0
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close
    self.symbol = symbol
    self.llm_changed_price = None # Manually set the Open price to 0
    self.TradeAgent_id_start = None
    self.TradeAgent_num = 0
    # Store the division ratio of each group, with the default being None, indicating equal division
    self.group_divisions = None
    self.subgroup_divisions = None
    # The dictionary r holds the fundamenal value series for each symbol.
    self.r = {}

    then = dt.datetime.now()

    # for symbol in symbols:
    #   s = symbols[symbol]
    #   log_print ("LLMChangedOracle computing fundamental value series for {}", symbol)
    #   self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)
    
    now = dt.datetime.now()
    log_print ("LLMChangedOracle initialized for symbol {}", symbol)
    #log_print ("LLMChangedOracle initialized for symbols {}", symbols)
    log_print ("LLMChangedOracle initialization took {}", now - then)

  def generate_fundamental_value_series(self, symbol, r_bar, kappa, sigma_s):
    # Generates the fundamental value series for a single stock symbol.  r_bar is the
    # mean fundamental value, kappa is the mean reversion coefficient, and sigma_s
    # is the shock variance.  (Note: NOT STANDARD DEVIATION.)

    # Because the oracle uses the global np.random PRNG to create the fundamental value
    # series, it is important to create the oracle BEFORE the agents.  In this way the
    # addition of a new agent will not affect the sequence created.  (Observations using
    # the oracle will use an agent's PRNG and thus not cause a problem.)

    # Turn variance into std.
    sigma_s = sqrt(sigma_s)

    # Create the time series into which values will be projected and initialize the first value.
    date_range = pd.date_range(self.mkt_open, self.mkt_close, closed='left', freq='N')

    s = pd.Series(index=date_range)
    r = np.zeros(len(s.index))
    r[0] = r_bar

    # Predetermine the random shocks for all time steps (at once, for computation speed).
    shock = np.random.normal(scale=sigma_s, size=(r.shape[0]))

    # Compute the mean reverting fundamental value series.
    for t in range(1, r.shape[0]):
      r[t] = max(0, (kappa * r_bar) + ( (1 - kappa) * r[t-1] ) + shock[t])

    # Replace the series values with the fundamental value series.  Round and convert to
    # integer cents.
    s[:] = np.round(r)
    s = s.astype(int)
    
    return (s)


  # Return the daily open price for the symbol given.  In the case of the MeanRevertingOracle,
  # this will simply be the first fundamental value, which is also the fundamental mean.
  # We will use the mkt_open time as given, however, even if it disagrees with this.
  def getDailyOpenPrice (self, symbol, mkt_open=None):
  
    # If we did not already know mkt_open, we should remember it.
    if (mkt_open is not None) and (self.mkt_open is None):
      self.mkt_open = mkt_open
  
    log_print ("Oracle: client requested {} at market open: {}", symbol, self.mkt_open)
  
    #open = self.r[symbol].loc[self.mkt_open]
    open = self.llm_changed_price[0][0] # Manually set the Open price to 0
    log_print ("Oracle: market open price was was {}", open)
  
    return open


  # Return a noisy observation of the current fundamental value.  While the fundamental
  # value for a given equity at a given time step does not change, multiple agents
  # observing that value will receive different observations.
  #
  # Only the Exchange or other privileged agents should use noisy=False.
  #
  # sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.
  #
  # Each agent must pass its RandomState object to observePrice.  This ensures that
  # each agent will receive the same answers across multiple same-seed simulations
  # even if a new agent has been added to the experiment.

  def observePrice(self, symbol, currentTime, id, sigma_n = 10000, random_state = None):
 
    n = len(self.llm_changed_price)
    interval = self.TradeAgent_num // n
    #print(interval)
    for i in range(n):
        start = self.TradeAgent_id_start + i * interval
        end = self.TradeAgent_id_start + (i + 1) * interval
        #print(start, end)
        if id in range(start, end):
          k =len(self.llm_changed_price[0])
          interval_small = (end-start) // k
          # print(k)
          # print(interval_small)
          for j in range(k):
            start_small = start + j * interval_small
            end_small = start + (j + 1) * interval_small
            if id in range(start_small, end_small):
              if sigma_n == 0:
                obs = self.llm_changed_price[i][j]
              else:
                obs = int(round(random_state.normal(loc=self.llm_changed_price[i][j], scale=sqrt(sigma_n))))
                
              break
          break
    log_print ("Oracle: current fundamental value is {} at {}", self.llm_changed_price, currentTime)
    log_print ("Oracle: giving client value observation {}", obs)
 
    # Reminder: all simulator prices are specified in integer cents.
    return obs

  def base_observePrice(self, symbol, currentTime, id, sigma_n = 10000, random_state = None):
    base_price = int(sum(self.llm_changed_price[0]) / len(self.llm_changed_price[0]))
    obs = int(round(random_state.normal(loc=base_price, scale=sqrt(sigma_n))))
    return obs