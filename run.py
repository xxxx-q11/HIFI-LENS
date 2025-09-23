
from Agent_FLLM.manager_agent import ManagerAgent
from config_LLM import base_config
import copy
import os
import sys
import json
import pandas as pd
import datetime
import ray
from numpy import double

# Make sure Ray has been initialized
if not ray.is_initialized():
    ray.init(include_dashboard=False)

# Define remote functions for creating StrategyAgents
@ray.remote
class RayManagerAgent(ManagerAgent):
    def __init__(self, invest_company, config, agent_id):
        super().__init__(invest_company, config=config)
        self.agent_id = agent_id
    
    def get_agent_state(self):
        """Get the Agent state information for caching"""
        cacheable_attributes = [
            # Basic state
            'state', 'currentTime', 'current_time',
            # Investment related state
            'investment_style', 'next_goal', 'last_surplus_rate', 'surplus_rate',
            # News and data state
            'last_finance_fundamental_information', 'last_news_fundamental_information',
            'break_news', 'summary_news', 'datetime',
            # Historical record state
            'last_self_evaluation_list', 'market_sentiment_list', 'technical_indicator',
            'institutional_policy', 'trade_history',
            # Current state
            'last_trade_history', 'last_technical_indicator', 'last_institutional_policy',
            'last_self_evaluation', 'market_sentiment',
            # Long-term memory state
            'previous_long_term_memory', 'previous_self_reflection', 'previous_institutional_policy',
            'previous_market_sentiment', 'previous_technical_indicator', 'previous_trade_history',
            # Market data state
            'market_data', 'last_market_data', 'initial_data', 'market_news',
            # ManagerAgent specific state
            'available_strategy', 'strategy', 'last_price', 'last_transaction', 'one_news',
            # Company and investor information
            'company', 'invest_company', 'event',
            # Short-term memory (for long-term memory compression)
            'short_term_memory',
            # News Summary Related (for market sentiment analysis)
            'intraday_news'
        ]
        
        state = {}
        for attr in cacheable_attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    try:
                        import json
                        json.dumps(value, ensure_ascii=False)
                        state[attr] = value
                    except (TypeError, ValueError):
                        state[attr] = str(value)
                else:
                    state[attr] = None
        
        import pandas as pd
        state['_timestamp'] = str(pd.Timestamp.now())
        state['_agent_type'] = 'RayManagerAgent'
        
        return state
    
    def set_agent_state(self, state):
        # "Set the status information of the Agent for recovery"
        restored_count = 0
        for attr, value in state.items():
            if attr.startswith('_'):  # Skip metadata
                continue
            if hasattr(self, attr):
                try:
                    setattr(self, attr, value)
                    restored_count += 1
                except Exception as e:
                    print(f"Failed to restore the status attribute {attr}: {e}")
        
        print(f"Agent {self.agent_id} has restored {restored_count} status attributes")
        return restored_count

# Add convenient functions
def serialize_agents_state(ray_agents):
    """Serialize the state of multiple Agents into a single dictionary"""
    from util.simple_ray_agent_manager import get_agent_states
    states = get_agent_states(ray_agents)
    if not states:
        return {}
    
    if len(states) == 1:
        return states[0]
    
    combined_state = {
        '_timestamp': str(pd.Timestamp.now()),
        '_agent_type': 'MultipleRayAgents',
        '_agent_count': len(states),
        'agents_states': states
    }
    
    return combined_state

def deserialize_agents_state(ray_agents, combined_state):
    """Restore the state of multiple Agents from a single dictionary"""
    if not combined_state or '_error' in combined_state:
        print("Skip status recovery: status data is invalid")
        return
    
    try:
        from util.simple_ray_agent_manager import set_agent_states, SimpleRayAgentManager
        if 'agents_states' in combined_state:
            states = combined_state['agents_states']
            set_agent_states(ray_agents, states)
        else:
            if ray_agents:
                SimpleRayAgentManager.deserialize_ray_agent_state(ray_agents[0], combined_state)
    except Exception as e:
        print(f"Failed to restore the Agent state: {e}")

# Create n parallel Strategy Agents
def create_parallel_agents(n=4 , invest_company_name=None):
    agents = []
    for i in range(n):
        # Create a deep copy of the configuration for each agent to avoid sharing the same configuration object
        agent_config = copy.deepcopy(base_config)
        # Create a Ray remote StrategyAgent
        agent = RayManagerAgent.remote(config=agent_config, invest_company = invest_company_name[i], agent_id=i)
        agents.append(agent)
    return agents

# Example: parallel call the ReMessage method
def parallel_process_message(llm_agents, current_time, base_message, per_agent_surpluses):
    """Parallel calls the ReMessage method on a list of Ray agent handles, with agent-specific surplus."""
    if not llm_agents:
        return []

    tasks = []
    for i, agent in enumerate(llm_agents):
        # Create a copy of the base message and add the specific surplus for the agent
        message = base_message.copy()
        message['surplus_rate'] = per_agent_surpluses[i]
        tasks.append(agent.ReMessage.remote(currentTime=current_time, msg=message))
    
    results = ray.get(tasks)
    return results

def parallel_process_opening_price(LLM_agents, message):
    results = ray.get([agent.open.remote(msg=message) for agent in LLM_agents])
    print(results)
    
    # Initialize an empty list to store the processed price
    prices = []
    
    for result in results:
        if isinstance(result[0], str):
            # Remove markdown code block marker
            result_price = re.sub(r"^```json\s*|^```python\s*|^```[\s]*|```$", "", result[0].strip(), flags=re.MULTILINE)
            try:
                result_json = json.loads(result_price)
            except Exception:
                # If it is not a standard json, try to fix the common problems and then parse
                try:
                    # Replace single quotes with double quotes, this is the common reason for JSON parsing errors
                    fixed_result = result_price.replace("'", "\"")
                    result_json = json.loads(fixed_result)
                except Exception as e:
                    print("The result format cannot be parsed:", e)
                    # If the parsing fails, try to extract the price information
                    price_match = re.search(r"'price':\s*'(\d+)'", result_price)
                    if price_match:
                        result_json = {"price": price_match.group(1)}
                    else:
                        result_json = {"price": "0"}
        else:
            result_json = result[0]
        
        # Get the price and add it to the list
        price = double(result_json.get('price', 0))
        integer_part = int(abs(price))
        if  integer_part <= 999:
            price *= 100
        prices.append(price)
        print("{}given price".format(result[1]),price)
    print("The processed price list", prices)
    return prices

def parallel_chose_agent_strategy(LLM_agents):
    results = ray.get([agent.Strategy_chose.remote() for agent in LLM_agents])
    agent_strategy_list = [item["name"] for item in results]
    return agent_strategy_list

def all_parallel_process_message(llm_agents, current_time, base_message, per_agent_surpluses,num):
    """Parallel calls the ReMessage method on a list of Ray agent handles, with agent-specific surplus."""
    if not llm_agents:
        return []

    tasks = []
    for k in range(num):
        for i, agent in enumerate(llm_agents[k]):
            # Create a copy of the base message and add the specific surplus for the agent
            message = base_message[k].copy()
            message['surplus_rate'] = per_agent_surpluses[k][i]
            tasks.append(agent.ReMessage.remote(currentTime=current_time, msg=message))
    
    results = ray.get(tasks)
    n = len(results) // num
    results = [results[i:i+n] for i in range(0, len(results), n)]
    return results

def all_parallel_chose_agent_strategy(llm_agents,num):
    if not llm_agents:
        return []

    tasks = []
    for k in range(num):
        for agent in llm_agents[k]:
            tasks.append(agent.Strategy_chose.remote())
    results = ray.get(tasks)
    agent_strategy_list = [item["name"] for item in results]
    n = len(agent_strategy_list) // num
    agent_strategy_list = [agent_strategy_list[i:i+n] for i in range(0, len(agent_strategy_list), n)]

    return agent_strategy_list