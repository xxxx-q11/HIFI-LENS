import json
import pandas as pd
from typing import Dict, Any, List

class AgentStateManager:
    """Agent state manager, responsible for serializing and deserializing the state"""
    
    # Define the variables that need to be cached
    CACHEABLE_ATTRIBUTES = [
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
        'available_strategy', 'strategy', 'last_price', 'last_transaction', 'one_news'
    ]
    
    @staticmethod
    def serialize_agent_state(agent) -> Dict[str, Any]:
        """Serialize the Agent state to a JSON-able dictionary"""
        state = {}
        for attr in AgentStateManager.CACHEABLE_ATTRIBUTES:
            if hasattr(agent, attr):
                value = getattr(agent, attr)
                # Handle special type serialization
                if value is not None:
                    try:
                        # Test if it can be JSON serialized
                        json.dumps(value, ensure_ascii=False)
                        state[attr] = value
                    except (TypeError, ValueError):
                        # If it cannot be serialized directly, convert to a string
                        state[attr] = str(value)
                else:
                    state[attr] = None
        
        # Add timestamp and Agent type information
        state['_timestamp'] = str(pd.Timestamp.now())
        state['_agent_type'] = agent.__class__.__name__
        
        return state
    
    @staticmethod
    def deserialize_agent_state(agent, state: Dict[str, Any]) -> None:
        """Restore the Agent state from the dictionary"""
        restored_count = 0
        for attr, value in state.items():
            if attr.startswith('_'):  # Skip metadata
                continue
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, value)
                    restored_count += 1
                except Exception as e:
                    print(f"Failed to restore the status attribute {attr}: {e}")
                    # Continue processing other attributes
        
        print(f"Agent {agent.agent_id} has restored {restored_count} status attributes")
    
    @staticmethod
    def get_state_summary(state: Dict[str, Any]) -> str:
        """Get the status summary information"""
        summary = []
        if 'current_time' in state and state['current_time']:
            summary.append(f"Time: {state['current_time']}")
        if 'surplus_rate' in state and state['surplus_rate'] is not None:
            summary.append(f"Surplus rate: {state['surplus_rate']}")
        if 'market_sentiment' in state and state['market_sentiment']:
            summary.append("Market sentiment: set")
        if 'last_self_evaluation' in state and state['last_self_evaluation']:
            summary.append("Self evaluation: set")
        if 'investment_style' in state and state['investment_style']:
            summary.append(f"Investment style: {state['investment_style']}")
        if 'next_goal' in state and state['next_goal']:
            summary.append("Next goal: set")
        
        return " | ".join(summary) if summary else "No status information"
    
    @staticmethod
    def validate_state(state: Dict[str, Any]) -> bool:
        """Validate the completeness of the state data"""
        if not isinstance(state, dict):
            return False
        
        # Check if there is basic metadata
        if '_timestamp' not in state or '_agent_type' not in state:
            return False
        
        return True
    
    @staticmethod
    def get_state_size(state: Dict[str, Any]) -> int:
        """Get the size of the state data (bytes)"""
        try:
            return len(json.dumps(state, ensure_ascii=False).encode('utf-8'))
        except Exception:
            return 0
    
    @staticmethod
    def compress_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Compress the state data (remove empty values and unnecessary data)"""
        compressed = {}
        for key, value in state.items():
            # Keep metadata
            if key.startswith('_'):
                compressed[key] = value
                continue
            
            # Skip empty values
            if value is None:
                continue
            
            # Skip empty lists and empty dictionaries
            if isinstance(value, (list, dict)) and len(value) == 0:
                continue
            
            # Skip empty strings
            if isinstance(value, str) and value.strip() == '':
                continue
            
            compressed[key] = value
        
        return compressed