import random
import sys,os
from openai import OpenAI
import openai
from typing import Any, Dict
# import tiktoken
class BaseAgent:
    def __init__(self, invest_company: str, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI(
            api_key=self.config['apikeys']['deepseek_api_key'],
            base_url="https://api.deepseek.com"
        )
        
        # Basic Status
        self.state = 'INACTIVE'
        self.news_sources = ["CBS", "CNN", "Foxnews", "yahoo", "Reuters"]
        self.news_agent = None
        self.market_sentiment = None
        self.next_goal = None
        self.currentTime = None
        self.current_time = None
        self.investment_style = None
        self.news = None
        
        # Fundamental information
        self.last_finance_fundamental_information = None
        self.last_news_fundamental_information = None
        
        # Strategy related 
        self.available_strategy = None
        self.strategy = None
        
        # Yield related
        self.last_surplus_rate = 0
        self.surplus_rate = 0
        
        # News and events
        self.break_news = []
        self.datetime = None
        
        # Model configuration
        self.think_model = self.config['apikeys']['think_model_name']
        self.generate_model = self.config['apikeys']['generate_model_name']
        
        # Historical data
        self.last_data = None
        self.last_self_evaluation_list = []
        self.market_sentiment_list = []
        self.short_term_memory = []
        
        # Long-term memory
        self.previous_long_term_memory = None
        self.previous_self_reflection = None
        self.previous_institutional_policy = None
        self.previous_market_sentiment = None
        self.previous_technical_indicator = None
        self.previous_trade_history = None
        
        # Current state record
        self.technical_indicator = []
        self.institutional_policy = []
        self.trade_history = []
        self.last_trade_history = None
        self.last_technical_indicator = None
        self.last_institutional_policy = None
        self.last_self_evaluation = None
        
        # Market data
        self.market_data = None
        self.one_news = None
        self.summary_news = []
        self.last_market_data = []
        self.initial_data = None
        self.market_reporter = None
        self.market_news = None
        
        # Company information
        self.event = self.config['agent_config']['event']
        self.company = self.config['agent_config']['company']
        self.invest_company = invest_company
        
        # New: Attributes used in ManagerAgent but missing in BaseAgent
        self.last_price = None          # Used in self_evaluation
        self.last_transaction = None    # Used in thought_price
        self.intraday_news = None      # Used for market sentiment analysis
    
    