from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np
from run import parallel_process_message, parallel_chose_agent_strategy
from run import create_parallel_agents
import time
import threading
import json
from copy import deepcopy
from util.simple_llm_cache import get_cache
from run import serialize_agents_state, deserialize_agents_state

class ExampleLLAgent(TradingAgent):
    def __init__(self, id, name, type, symbol, starting_cash, invest_company,
                 wake_up_freq='4000s',
                 subscribe=False, log_orders=False, random_state=None, LLM_agent=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.subscribe = subscribe
        self.subscription_requested = False
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.base_price = 0
        self.base_price_dict = {}
        self.invest_company = invest_company
        self.parellel_num = len(invest_company)
        self.json_mesage = {}
        self.LLM_agent = create_parallel_agents(self.parellel_num,self.invest_company)
        self.agent_surplus = []
        self.agent_surplus_dict = {}
        self.first_wakeup_time = None
        
        self.agent_strategty = None
        self.TradeAgent_strategy = []

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        self.requestMarketNews(self.symbol)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=1, freq=10e9)
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol, depth=5)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'MARKET_NEWS' or (self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD'):
            self.handle_message_and_update_strategy(currentTime, msg)

    def handle_message(self, currentTime, msg):
        if msg['msg'] == 'MARKET_NEWS':
            print(f"LLMagent{self.id} accept the news")
            json_message_base = self.convert_message_to_json(msg, currentTime)
        elif self.state == 'AWAITING_SPREAD' and msg['msg'] == 'QUERY_SPREAD':
            bids, asks, last_trade = self.getKnowBiDAsK_all(self.symbol)
            print(f"LLMagent{self.id}Timed wake-up")
            json_message_base = self.convert_messageData_to_json(msg, bids, asks, last_trade, currentTime)
        else:
            return

        self.json_mesage.update({currentTime: json_message_base})
        if self.first_wakeup_time is None:
            self.first_wakeup_time = currentTime
            print(f"Agent {self.id} The first wake-up time: {self.first_wakeup_time}")
        agent_surpluses = self.calculate_surplus(currentTime)

        return json_message_base,agent_surpluses
    
    def update_price_and_strategy(self, currentTime, price_list,strategy_list):
        self.strategy_chose(strategy_list, currentTime)
        self.base_price = [float(item['price']) for item in price_list]
        self.oracle.llm_changed_price[self.id - 1] = self.base_price
        self.base_price_dict.update({currentTime: price_list})
        print(self.base_price)


    # Modify the handle_message_and_update_strategy method
    def handle_message_and_update_strategy(self, currentTime, msg):
        if msg.body['msg'] == 'MARKET_NEWS':
            print(f"LLMagent{self.id} accept the news")
            json_message_base = self.convert_message_to_json(msg.body, currentTime)
        elif self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bids, asks, last_trade = self.getKnowBiDAsK_all(self.symbol)
            print(f"LLMagent{self.id}Timed wake-up")
            json_message_base = self.convert_messageData_to_json(msg.body, bids, asks, last_trade, currentTime)
        else:
            return

        self.json_mesage.update({currentTime: json_message_base})
        if self.first_wakeup_time is None:
            self.first_wakeup_time = currentTime
            print(f"Agent {self.id} The first wake-up time: {self.first_wakeup_time}")
        agent_surpluses = self.calculate_surplus(currentTime)
        
        # Check the cache (modified part - using Ray Agent state management)
        cache = get_cache()
        if cache and cache.cache_mode:
            cached_results = cache.get_cached_results_with_state()
            if cached_results:
                result_list, TradeAgent_strategy_list, agent_state = cached_results
                print(f"Use the cached LLM results, skip the actual call")
                
                # During trading: agent_state is empty, because the state has been restored in Open
                # Here we do not need to restore the state, because the Agent has been restored to the last state in Open
                if agent_state:  # This condition is usually not met, because agent_state is empty during trading
                    deserialize_agents_state(self.LLM_agent, agent_state)
                    from util.simple_ray_agent_manager import SimpleRayAgentManager
                    summary = SimpleRayAgentManager.get_state_summary(agent_state)
                    print(f"The Ray Agent state has been restored: {summary}")
            else:
                print("The cache has been used up, switch to the actual LLM call")
                result_list = parallel_process_message(self.LLM_agent, currentTime, json_message_base, agent_surpluses)
                TradeAgent_strategy_list = parallel_chose_agent_strategy(self.LLM_agent)
                
                # Save to the cache (including Ray Agent state)
                if cache and not cache.cache_mode:
                    agent_state = serialize_agents_state(self.LLM_agent)
                    cache.add_results_with_state(result_list, TradeAgent_strategy_list, agent_state)
        else:
            # Normal call LLM
            result_list = parallel_process_message(self.LLM_agent, currentTime, json_message_base, agent_surpluses)
            TradeAgent_strategy_list = parallel_chose_agent_strategy(self.LLM_agent)
            
            # Save to the cache (including Ray Agent state)
            if cache and not cache.cache_mode:
                agent_state = serialize_agents_state(self.LLM_agent)
                cache.add_results_with_state(result_list, TradeAgent_strategy_list, agent_state)
        
        # ... The subsequent processing remains unchanged ...
        self.strategy_chose(TradeAgent_strategy_list, currentTime)
        self.base_price = [float(item['price']) for item in result_list]
        self.oracle.llm_changed_price[self.id - 1] = self.base_price
        self.base_price_dict.update({currentTime: result_list})
        print(f"The returned price: {self.base_price}")

        if self.state == 'AWAITING_SPREAD':
            self.setWakeup(currentTime + self.getWakeFrequency())
            print("Set the next wake-up time")
            self.state = 'AWAITING_WAKEUP'


    def kernelStopping(self):
        super().kernelStopping()
        with open(f"Experimental_data/result_data/base_price_dict_{self.id}_.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.base_price_dict.items()}, f, ensure_ascii=False, indent=2)
        with open(f"Experimental_data/result_data/json_mesage_{self.id}_.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.json_mesage.items()}, f, ensure_ascii=False, indent=2)
        with open(f"Experimental_data/result_data/agent_surplus_dict_{self.id}_.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.agent_surplus_dict.items()}, f, ensure_ascii=False, indent=2)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def convert_message_to_json(self, msg_body, currentTime):
        if msg_body['msg'] != 'MARKET_NEWS':
            return None
        json_message = {
            "msg": msg_body['msg'],
            "symbol": msg_body['symbol'],
            "bids": msg_body['bids'],
            "asks": msg_body['asks'],
            "last_transaction": msg_body['last_transaction'],
            "News": msg_body['News'],
            'exchange_ts': str(msg_body['exchange_ts']),
            "trade_history": msg_body["trade_history"],
            "currentTime": str(currentTime)
        }
        return json_message

    def convert_messageData_to_json(self, msg_body, bids, asks, last_trade, currentTime):
        json_message = {
            "msg": msg_body['msg'],
            "symbol": msg_body['symbol'],
            "bids": bids,
            "asks": asks,
            "last_transaction": last_trade,
            "News": "None",
            'exchange_ts': str(currentTime),
            "trade_history": msg_body["trade_history"],
            "currentTime": str(currentTime)
        }
        return json_message

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def calculate_surplus(self, currentTime):
        all_agents = self.kernel.agents
        n = len(self.oracle.llm_changed_price)
        interval = self.oracle.TradeAgent_num // n
        start = self.oracle.TradeAgent_id_start + (self.id - 1) * interval
        end = self.oracle.TradeAgent_id_start + self.id * interval
        k = len(self.oracle.llm_changed_price[0])
        interval_small = (end - start) // k
        self.agent_surplus = [0] * k
        for j in range(k):
            start_small = start + j * interval_small
            end_small = start + (j + 1) * interval_small
            origin_cash = 0
            for id in range(start_small, end_small):
                surplus = all_agents[id].calculateSurplus()
                origin_cash += all_agents[id].starting_cash
                self.agent_surplus[j] += surplus
            surplus_rate = self.agent_surplus[j] / origin_cash
            self.agent_surplus[j] = surplus_rate
            agent_surplus_snapshot = [self.id, j, deepcopy(self.agent_surplus)]
            self.agent_surplus_dict.update({currentTime: agent_surplus_snapshot})
        return self.agent_surplus

    def strategy_chose(self, strategy_list, currentTime=0):
        all_agents = self.kernel.agents
        n = len(self.oracle.llm_changed_price)
        interval = self.oracle.TradeAgent_num // n
        start = self.oracle.TradeAgent_id_start + (self.id - 1) * interval
        end = self.oracle.TradeAgent_id_start + self.id * interval
        k = len(self.oracle.llm_changed_price[0])
        interval_small = (end - start) // k
        for j in range(k):
            start_small = start + j * interval_small
            end_small = start + (j + 1) * interval_small
            for id in range(start_small, end_small):
                all_agents[id].strategy_chose(strategy_list[j])
                #print(f"Agent {id} Choose the strategy: {strategy_list[j]}")
            print({self.id: f"LLMAgent Choose the strategy: {strategy_list[j]}"})
            self.TradeAgent_strategy.append({self.id: f"LLMAgent{self.invest_company[j]} Choose the strategy: {strategy_list[j]}"})
        strategy_snapshot = deepcopy(self.TradeAgent_strategy)
        self.kernel.LLM_data.update({currentTime: strategy_snapshot})
        self.TradeAgent_strategy.clear()