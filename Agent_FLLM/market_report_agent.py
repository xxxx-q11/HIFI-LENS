import json
import random
from typing import Any, Dict, List
from numpy import double
from openai import OpenAI
import openai
import Agent_FLLM.prompts.market_report_agent as PROMPT
from config_LLM import base_config
from Agent_FLLM.utils import extract_query,stream_to_json, read_json_file, extract_random_records

class MarketReporterAgent:
    def __init__(self, config: Dict[str, Any]):
        
        self.time = None
        self.model = config['apikeys']['generate_model_name']  # Use the fixed model name set in the script
        self.temple = None
        self.market_data_list = None
        self.technical_indicator = None
        self.trade_history = None
        self
        self.config = config
        self.client = OpenAI(
            api_key=self.config['apikeys']['deepseek_api_key'],  # Use the fixed API Key set in the script
            base_url="https://api.deepseek.com"  # Caddy Proxy address
        )  
        self.temple = extract_random_records(f'./Data/Template/Market/market_news_template.json',1)
        
    def market_report(self, generate_time, market_data_list, technical_indicator, trade_history):
        self.time = generate_time
        self.market_data_list = market_data_list
        self.technical_indicator = technical_indicator
        self.trade_history = trade_history
        
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.market_report.format(
                                context=PROMPT.context,
                                generate_time=self.time,
                                temple=self.temple,
                                market_data_list=self.market_data_list,
                                trade_history=self.trade_history,
                                technical_indicator=self.technical_indicator,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please generate a report in JSON format based on the template and market data list, and market technical indicators, and trading history. And strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Report, format example: {'title':'', 'datetime': '', 'content': ''}, don't begin with any title like 'json'."
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                content = response_stream.choices[0].message.content
                parsed = json.loads(content) 
                return parsed  
                
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise Exception("Even when the maximum retry count is reached, the returned content is still not in valid JSON format")
                
        raise Exception("No API request was made (logical error)")  # Prevent the loop from not being executed