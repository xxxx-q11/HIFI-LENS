import json
import pandas as pd
import ray
from typing import Dict, Any, List, Optional

class SimpleRayAgentManager:
    """Simplified Ray Agent state manager, specifically for Ray remote objects"""
    
    @staticmethod
    def serialize_ray_agent_state(ray_agent) -> Dict[str, Any]:
        """Serialize Ray remote Agent state"""
        try:
            # Get state through Ray remote call
            state_future = ray_agent.get_agent_state.remote()
            state = ray.get(state_future)
            return state
        except Exception as e:
            print(f"Serialize Ray Agent state failed: {e}")
            return {
                '_timestamp': str(pd.Timestamp.now()),
                '_agent_type': 'RayAgent',
                '_error': str(e)
            }
    
    @staticmethod
    def deserialize_ray_agent_state(ray_agent, state: Dict[str, Any]) -> None:
        """Restore Ray remote Agent state"""
        if not state or '_error' in state:
            print("Skip status recovery: status data is invalid or contains errors")
            return
        
        try:
            # Set state through Ray remote call
            restore_future = ray_agent.set_agent_state.remote(state)
            restored_count = ray.get(restore_future)
            print(f"Ray Agent state has been restored, {restored_count} attributes")
        except Exception as e:
            print(f"Restore Ray Agent state failed: {e}")
            
    @staticmethod
    def get_state_summary(state: Dict[str, Any]) -> str:
        """Get status summary information"""
        if '_error' in state:
            return f"Status error: {state['_error']}"
        
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

def get_agent_states(ray_agents):
    """Get multiple Ray Agent states"""
    if not ray_agents:
        return []
    
    try:
        state_futures = [agent.get_agent_state.remote() for agent in ray_agents]
        states = ray.get(state_futures)
        return states
    except Exception as e:
        print(f"Get Agent state failed: {e}")
        return [{'_error': str(e)} for _ in ray_agents]

def set_agent_states(ray_agents, states):
    """Set multiple Ray Agent states"""
    if not ray_agents or not states:
        return
    
    try:
        restore_futures = []
        for agent, state in zip(ray_agents, states):
            if state and '_error' not in state:
                restore_futures.append(agent.set_agent_state.remote(state))
        
        if restore_futures:
            restored_counts = ray.get(restore_futures)
            total_restored = sum(restored_counts)
            print(f"Total restored {total_restored} Agent state attributes")
    except Exception as e:
        print(f"Set Agent state failed: {e}")