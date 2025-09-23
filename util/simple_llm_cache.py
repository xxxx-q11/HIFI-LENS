import json
import os
import pandas as pd
from typing import List, Any, Optional, Dict
import orjson 
import time

class SimpleLLMCache:
    def __init__(self, cache_file: str = "llm_cache.json"):
        self.cache_file = cache_file
        self.result_list_cache = []
        self.strategy_list_cache = []
        self.opening_price_cache = []
        self.opening_strategy_cache = []
        self.agent_state_cache = []
        self.opening_agent_state_cache = []
        
        self.current_index = 0
        self.current_opening_index = 0
        self.cache_mode = False  # False= Save mode, True= Read mode
        
    def load_cache(self) -> bool:
        """Load cache file, if it exists, enter read mode"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.result_list_cache = data.get('result_list_cache', [])
                self.strategy_list_cache = data.get('strategy_list_cache', [])
                self.opening_price_cache = data.get('opening_price_cache', [])
                self.opening_strategy_cache = data.get('opening_strategy_cache', [])
                # Load the Agent status cache
                self.agent_state_cache = data.get('agent_state_cache', [])
                self.opening_agent_state_cache = data.get('opening_agent_state_cache', [])
                
                
                self.current_opening_index = 0
                self.cache_mode = True
                print(f"Load cache file: {self.cache_file}")
                print(f"Intraday record: {len(self.result_list_cache)}, Status record: {len(self.agent_state_cache)}")
                print(f"Openprice record: {len(self.opening_price_cache)}, OpenStatus record: {len(self.opening_agent_state_cache)}")
                return True
            except Exception as e:
                print(f"Load cache file failed: {e}")
                return False
        else:
            print(f"Cache file does not exist: {self.cache_file}, will create a new cache file")
            # Ensure the cache file directory exists
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                print(f"Create cache directory: {cache_dir}")
        return False
    
    def save_cache(self):
        """Save cache to file"""
        if not self.cache_mode:  # Only save in save mode
            data = {
                "metadata": {
                    "total_calls": len(self.result_list_cache),
                    "total_opening_calls": len(self.opening_price_cache),
                    "created_at": str(pd.Timestamp.now()),
                    "simulation_config": "rsmtry_LLM3"
                },
                "result_list_cache": self.result_list_cache,
                "strategy_list_cache": self.strategy_list_cache,
                "opening_price_cache": self.opening_price_cache,
                "opening_strategy_cache": self.opening_strategy_cache,
                "agent_state_cache": self.agent_state_cache,
                "opening_agent_state_cache": self.opening_agent_state_cache
            }
            # 3. Use orjson (if you need JSON format, it's 5-10 times faster than standard json)
            json_file = self.cache_file
            with open(json_file, 'wb') as f:
              f.write(orjson.dumps(data, 
                                   option=orjson.OPT_SERIALIZE_NUMPY | 
                                          orjson.OPT_NAIVE_UTC |
                                          orjson.OPT_INDENT_2))
            print(f"JSON cache has been saved: {json_file}")
            total_time = time.time() - start_time
            print(f"Cache saved, total time: {total_time:.2f} seconds")
    
    def get_cached_results_with_state(self) -> Optional[tuple]:
        """Get cached results and Agent state"""
        print(f"current_index: {self.current_index}, len(self.result_list_cache): {len(self.result_list_cache)}")
        if self.cache_mode and self.current_index < len(self.result_list_cache):
            result_list = self.result_list_cache[self.current_index]
            strategy_list = self.strategy_list_cache[self.current_index]
            
            # Intraday trading: Do not restore the state, because it has been restored at the opening
            agent_state = {}
            
            self.current_index += 1
            print(f"Use cached result [{self.current_index}/{len(self.result_list_cache)}]")
            return result_list, strategy_list, agent_state
        elif self.cache_mode and self.current_index >= len(self.result_list_cache):
            print("Cache is used up, switch to append mode, new LLM call results and states will be appended to the cache")
            self.cache_mode = False
        return None
    
    def add_results_with_state(self, result_list: List[Any], strategy_list: List[str], agent_state: Dict[str, Any]):
        """Add new results and Agent state to cache"""
        self.result_list_cache.append(result_list)
        self.strategy_list_cache.append(strategy_list)
        self.agent_state_cache.append(agent_state)
        print(f"{'Append' if self.current_index > 0 else 'Save'}LLM call results and states [{len(self.result_list_cache)}]")
        # Immediately append to checkpoints file folder
        self._append_to_checkpoint()
        # Update the main cache file
        self._update_main_cache()
    
    def get_cached_opening_results_with_state(self) -> Optional[tuple]:
        """Get cached Open results and Agent state"""
        if self.cache_mode and self.current_opening_index < len(self.opening_price_cache):
            opening_price = self.opening_price_cache[self.current_opening_index]
            opening_strategy = self.opening_strategy_cache[self.current_opening_index]
            
            # Open state recovery: restore the last Agent state
            agent_state = {}
            if self.current_opening_index == 0 and self.agent_state_cache:
                # Restore the last Agent state of the last LLM call at the opening
                agent_state = self.agent_state_cache[-1]
                print(f"Restore the last Agent state of the last LLM call at the opening (index: {len(self.agent_state_cache)-1})")
            
            self.current_opening_index += 1
            print(f"Use cached Open result [{self.current_opening_index}/{len(self.opening_price_cache)}]")
            return opening_price, opening_strategy, agent_state
        elif self.cache_mode and self.current_opening_index >= len(self.opening_price_cache):
            print("Open cache is used up, switch to append mode")
            self.cache_mode = False
        return None
    
    def add_opening_results_with_state(self, opening_price: List[float], opening_strategy: List[str], agent_state: Dict[str, Any]):
        """Add Open results and Agent state to cache"""
        self.opening_price_cache.append(opening_price)
        self.opening_strategy_cache.append(opening_strategy)
        self.opening_agent_state_cache.append(agent_state)
        print(f"{'Append' if self.current_opening_index > 0 else 'Save'}OpenLLM call results and states [{len(self.opening_price_cache)}]")
        # Immediately append to checkpoints file folder
        self._append_opening_to_checkpoint()
        # Update the main cache file
        self._update_main_cache()
    
    def _update_main_cache(self):
        """Update the main cache file"""
        data = {
            "metadata": {
                "total_calls": len(self.result_list_cache),
                "total_opening_calls": len(self.opening_price_cache),
                "created_at": str(pd.Timestamp.now()),
                "simulation_config": "rsmtry_LLM3"
            },
            "result_list_cache": self.result_list_cache,
            "strategy_list_cache": self.strategy_list_cache,
            "opening_price_cache": self.opening_price_cache,
            "opening_strategy_cache": self.opening_strategy_cache,
            # Save Agent status cache
            "agent_state_cache": self.agent_state_cache,
            "opening_agent_state_cache": self.opening_agent_state_cache
        }
        
        # Ensure the main cache file directory exists
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Create main cache directory: {cache_dir}")
            
        # Use orjson to save
        with open(self.cache_file, 'wb') as f:
            f.write(orjson.dumps(data, 
                               option=orjson.OPT_SERIALIZE_NUMPY | 
                                      orjson.OPT_NAIVE_UTC |
                                      orjson.OPT_INDENT_2))
        print(f"Main cache file has been updated: {self.cache_file}")

    
    def _append_to_checkpoint(self):
        """Append to checkpoints file folder in fast append mode"""
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = os.path.join(checkpoint_dir, f"llm_cache_checkpoint_{len(self.result_list_cache)}.json")
        
        # Only save the latest record
        latest_data = {
            "index": len(self.result_list_cache) - 1,
            "result_list": self.result_list_cache[-1],
            "strategy_list": self.strategy_list_cache[-1],
            "agent_state": self.agent_state_cache[-1] if self.agent_state_cache else {},
            "timestamp": str(pd.Timestamp.now())
        }
        
        # Use orjson to save
        with open(checkpoint_file, 'wb') as f:
            f.write(orjson.dumps(latest_data, 
                               option=orjson.OPT_SERIALIZE_NUMPY | 
                                      orjson.OPT_NAIVE_UTC |
                                      orjson.OPT_INDENT_2))
        print(f"Append save checkpoint: {checkpoint_file}")
    
    def _append_opening_to_checkpoint(self):
        """Append to checkpoints file folder in fast append mode"""
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = os.path.join(checkpoint_dir, f"opening_cache_checkpoint_{len(self.opening_price_cache)}.json")
        
        # Only save the latest Open record
        latest_data = {
            "index": len(self.opening_price_cache) - 1,
            "opening_price": self.opening_price_cache[-1],
            "opening_strategy": self.opening_strategy_cache[-1],
            "agent_state": self.opening_agent_state_cache[-1] if self.opening_agent_state_cache else {},
            "timestamp": str(pd.Timestamp.now())
        }
        
        # Use orjson to save
        with open(checkpoint_file, 'wb') as f:
            f.write(orjson.dumps(latest_data, 
                               option=orjson.OPT_SERIALIZE_NUMPY | 
                                      orjson.OPT_NAIVE_UTC |
                                      orjson.OPT_INDENT_2))
        print(f"Append save Open checkpoint: {checkpoint_file}")

    # Keep the compatibility of the original method
    def get_cached_results(self) -> Optional[tuple]:
        """Get cached results (compatibility method)"""
        result = self.get_cached_results_with_state()
        if result:
            return result[0], result[1]  # Only return result_list and strategy_list
        return None
    
    def add_results(self, result_list: List[Any], strategy_list: List[str]):
        """Add new results to cache (compatibility method)"""
        self.add_results_with_state(result_list, strategy_list, {})
    
    def get_cached_opening_results(self) -> Optional[tuple]:
        """Get cached Open results (compatibility method)"""
        result = self.get_cached_opening_results_with_state()
        if result:
            return result[0], result[1]  # Only return opening_price and opening_strategy
        return None
    
    def add_opening_results(self, opening_price: List[float], opening_strategy: List[str]):
        """Add Open results to cache (compatibility method)"""
        self.add_opening_results_with_state(opening_price, opening_strategy, {})
    
    def has_more_cache(self) -> bool:
        """Check if there is more cache"""
        return self.cache_mode and self.current_index < len(self.result_list_cache)
    
    def has_more_opening_cache(self) -> bool:
        """Check if there is more Open cache"""
        return self.cache_mode and self.current_opening_index < len(self.opening_price_cache)


# Global cache instance
_global_cache = None

def initialize_cache(cache_file: str = "llm_cache.json", enable_cache: bool = False):
    """Initialize cache"""
    global _global_cache
    if enable_cache:
        _global_cache = SimpleLLMCache(cache_file)
        _global_cache.load_cache()
    else:
        _global_cache = None

def get_cache() -> Optional[SimpleLLMCache]:
    """Get global cache instance"""
    return _global_cache