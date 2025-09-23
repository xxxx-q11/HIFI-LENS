import json
import random
import re
import time
from typing import Any, Dict, List
from numpy import double
from openai import OpenAI
import openai
import Agent_FLLM.prompts.manager_agent as PROMPT
from Agent_FLLM.news_agent import NewsAgent
from Agent_FLLM.market_report_agent import MarketReporterAgent
from config_LLM import base_config
from Agent_FLLM.utils import extract_query,stream_to_json, read_json_file, extract_random_records,format_trade_history_advanced, get_investment_style
from Agent_FLLM import agent
import pandas
class ManagerAgent(agent.BaseAgent):
    def __init__(self, invest_company, config: Dict[str, Any]):
        super().__init__(invest_company, config)
        # self.execute_search(tool_name="yfinance", query="AAPL", params={})
        # self.execute_search(tool_name="FinnhubNewsFetcher", query="AAPL", params={})
        # self.execute_search(tool_name="RedditExtractor", query="AAPL", params={})
        # self.set_profile()
        
        self.set_profile()
        self.available_strategy = self._parse_strategy_descriptions()
        last_news_fundamental_information =  extract_random_records(f'./Data/{self.event}/{self.company}/News_Fundamental/news.json',15)
        self.last_news_fundamental_information = self.Update_fundamental_information_news(result=last_news_fundamental_information)
        last_financial_fundamental_information =  read_json_file(f'./Data/{self.event}/{self.company}/Financial_Fundamental/data.json')
        self.last_finance_fundamental_information = self.Update_fundamental_information_finance(result=last_financial_fundamental_information)
        self.initial_data = read_json_file(f'./Data/{self.event}/{self.company}/Initial/initial.json')
        
    
    def set_profile(self):
        """Set the personal information of the agent"""
        data = read_json_file(f'./Data/{self.event}/{self.company}/Company/company.json')
        self.investment_style = get_investment_style(self.invest_company, data)
    
    def _parse_strategy_descriptions(self) -> List[str]:
        """Convert the strategy configuration to natural language description"""
        strategy_descriptions = []
        for strategy in PROMPT.config_strategy:
            desc = f"{strategy['name']}: {strategy['Description']}"
            strategy_descriptions.append(desc)
        # print(tool_descriptions)
        return strategy_descriptions
    
    def select_strategy(self) -> Dict:
        max_retries = 10  # Maximum retry count
        retry_delay = 1  # Retry delay(seconds)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.think_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": PROMPT.select_strategy.format(
                                context=PROMPT.context.format(invest_company=self.invest_company),
                                last_surplus_rate=self.last_surplus_rate,
                                Investment_style=self.investment_style,
                                next_goal=self.next_goal,
                                market_sentiment=self.market_sentiment,
                                institutional_policy=self.last_institutional_policy,
                                technical_indicator=self.last_technical_indicator,
                                self_reflection=self.last_self_evaluation,
                                strategy_descriptions="\n".join(self.available_strategy)
                            )
                        },
                        {
                            "role": "user", 
                            "content": "Please select the most appropriate investment strategy based on the profit situation and strategy descriptions. Do not return in markdown format! Return format example: {{'name': 'strategy name', 'reason': 'Reasons for choice'}}, don't begin with any title like 'json'. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                result = json.loads(response.choices[0].message.content)
                
                # Verify the basic structure (optional)
                if "name" not in result or "reason" not in result:
                    raise json.JSONDecodeError("Missing required keys", "", 0)
                    
                return result
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"JSON parsing failed(attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}seconds later retry...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"After {max_retries} attempts, an effective JSON response still could not be obtained") from e
    
    
    def summary_break_news(self, result):
        """Summarize breaking news (with JSON retry mechanism)）"""
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.summary_break_news.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company
                                ),
                                datetime = self.datetime,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please summarize this breaking news based on my personal information(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Summary, format example: {{'summary': 'My summary of breaking news for AAPL ...', 'datetime': 'Input News datetime'}}. Return strictly in JSON format!"
                            f"{result if isinstance(result, str) else json.dumps(result.to_dict(orient='records') if hasattr(result, 'to_dict') else (result if isinstance(result, (dict, list)) else str(result)), ensure_ascii=False)}"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}  # Force the JSON output format
                )
                
                # Try to parse JSON，An exception will be thrown if it fails
                return json.loads(response.choices[0].message.content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("JSON parsing failed,the maximum number of retries has been reached")
    
    def thought_price(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.think_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.thought_price.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                Investment_style = self.investment_style,
                                next_goal = self.next_goal,
                                institutional_policy = self.institutional_policy,
                                market_sentiment = self.market_sentiment,
                                technical_indicator = self.last_technical_indicator,
                                last_transaction = self.last_transaction,
                                surplus_rate = self.surplus_rate,
                                trade_history = self.last_trade_history,
                                last_news_fundamental_information = self.last_news_fundamental_information,
                                last_finance_fundamental_information = self.last_finance_fundamental_information,
                                self_reflection=self.last_self_evaluation,
                                previous_market_sentiment = self.previous_market_sentiment,
                                previous_institutional_policy = self.previous_institutional_policy,
                                previous_self_reflection = self.previous_self_reflection,
                                previous_technical_indicator = self.previous_technical_indicator,
                                previous_trade_history = self.previous_trade_history,
                                
                            )
                        },
                        {
                            "role": "user",
                            "content": "The current stock price may not truly reflect the real value of the company at present. Please comprehensively consider the above information, give your opinion on the current stock price of the company, and provide the reason strictly in JSON format. Do not return in markdown format ! Return format example: {{'price': '2.33' , 'reason':'The reason for AAPL price is ...'}}, don't begin with any title like 'json'. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("JSON parsing failed,the maximum number of retries has been reached")
        
        # Never execute here (for completeness)
        return None
    
    def opening_price(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.think_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.opening_price.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                Investment_style = self.investment_style,
                                previous_institutional_policy = self.previous_institutional_policy,
                                institutional_policy = self.institutional_policy,
                                previous_market_sentiment = self.previous_market_sentiment,
                                market_sentiment = self.market_sentiment,
                                previous_technical_indicator = self.previous_technical_indicator,
                                previous_trade_history = self.previous_trade_history,
                                previous_self_reflection = self.previous_self_reflection,
                                last_news_fundamental_information = self.last_news_fundamental_information,
                                last_finance_fundamental_information = self.last_finance_fundamental_information,
                                next_goal = self.next_goal,
                            )
                        },
                        {
                            "role": "user",
                            "content": "The current stock price may not truly reflect the real value of the company at present. Please comprehensively consider the above information, give your opinion on the opening price, and provide the reason strictly in JSON format. Do not return in markdown format ! Return format example: {{'price': '2.33' , 'reason':'The reason for opening price is ...'}}, don't begin with any title like 'json'. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON And maintain the original return format
                content = response_stream.choices[0].message.content
                parsed_content = json.loads(content)
                print(parsed_content)
                return [parsed_content, self.invest_company]
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("Open price prediction JSON parsing failed,the maximum number of retries has been reached")
        


    def ReMessage(self, currentTime, msg):
        self.market_reporter = MarketReporterAgent(config = base_config)
        self.state = 'ACTIVE'
        self.current_time = currentTime
        self.last_transaction = msg['last_transaction']
        # self.news = msg['News']
        # self.news.append(news)
        
        text = f"\n【Time:{self.current_time} Order Book Data 】\n"
        text += "(Bids) Price x Quantity:\n" + \
                "\n".join([f"{price}*{qty}" for price, qty in msg["bids"]]) + \
                "\n\n(Asks) Price x Quantity:\n" + \
                "\n".join([f"{price}*{qty}" for price, qty in msg["asks"]])
        self.last_market_data.append(text)
        self.last_technical_indicator = self.Technical_Indicators()
        self.technical_indicator.append(self.last_technical_indicator)
        print(self.last_technical_indicator)
        
        text += "\n【2025 3-26 ~ 2025 4~1 Market Data】\n"
        text +=  str(self.initial_data) + "\n"
        self.market_data = text
        
        self.surplus_rate = msg['surplus_rate']
        if  msg["surplus_rate"] != None:
            self.last_self_evaluation = self.self_evaluation()
            self.last_self_evaluation_list.append(self.last_self_evaluation)
            print(self.last_self_evaluation)
            self.next_goal = self.updata_next_goal()
            print(self.next_goal)
        self.last_trade_history = format_trade_history_advanced(msg['trade_history'])
        self.trade_history.append(self.last_trade_history)

        self.market_news = self.market_reporter.market_report(generate_time = self.current_time, market_data_list = self.last_market_data, technical_indicator = self.last_technical_indicator
                                                         , trade_history = self.last_trade_history)
        if msg['News'] != 'None':
            self.news_agent = NewsAgent(config = base_config, source = random.choice(self.news_sources))
            news = self.news_agent.Generate_rumor_news(information=msg['News'])
            self.one_news = news
            summary = self.summary_break_news(news)
            self.break_news.append(summary)
            self.summary_news.append(news)
            self.last_institutional_policy = self.Policy_Indicators()
            self.institutional_policy.append(self.last_institutional_policy)
            self.market_sentiment = self.Update_market_sentiment()
            self.market_sentiment_list.append(self.market_sentiment)
            print(self.market_sentiment)
        else:
            self.market_sentiment = self.Market_Sentiment_None()
            self.market_sentiment_list.append(self.market_sentiment)
        self.strategy = self.select_strategy()
        result = self.thought_price()
        print(result)
        # result = eval(result)
        # price = double(result['price'])
        if isinstance(result, str):
            # Remove markdown code block markers
            result = re.sub(r"^```json\s*|^```python\s*|^```[\s]*|```$", "", result.strip(), flags=re.MULTILINE)
            try:
                result_json = json.loads(result)
            except Exception:
                # If it is not a standard json, attempt to fix the common problems and then parse
                try:
                    # Replace single quotes with double quotes, this is the common reason for JSON parsing errors
                    fixed_result = result.replace("'", "\"")
                    result_json = json.loads(fixed_result)
                except Exception as e:
                    print("The result format cannot be parsed:", e)
                    # If parsing fails, attempt to extract price information
                    price_match = re.search(r"'price':\s*'(\d+)'", result)
                    if price_match:
                        result_json = {"price": price_match.group(1)}
                    # else:
                    #     result_json = {"price": "0"}
        else:
            result_json = result
        price = double(result_json.get('price', 0))
        integer_part = int(abs(price))
        if  integer_part <= 999:
            price = self.transform_price(price)
            result_json['price'] = str(price)
        print("The processed price",price)
        self.last_price = price
        self.last_surplus_rate = self.surplus_rate
        self.summary_news.clear()
        return result_json
    
    def Strategy_chose(self):
        return self.strategy
    
    def transform_price(self,price):
        # Convert to string, separate the integer and decimal parts
        price_str = str(price)
        if '.' not in price_str:
            price_str += '.0'
        integer_part, decimal_part = price_str.split('.')
        
        # Calculate the number of decimal places to move
        shift = 5 - len(integer_part)  # The number of decimal places to move for the target 5-digit integer
        
        if shift >= 0: 
            # The integer part is less than 5: move the decimal point to the right
            new_integer = integer_part + decimal_part[:shift]  # Take the decimal part to fill
            new_decimal = decimal_part[shift:]  # The remaining decimal places
            if len(new_integer) < 5:  # If it is still less than 5, add zeros to the end
                new_integer = new_integer.ljust(5, '0')
        else:  
            # The integer part is more than 5: move the decimal point to the left
            new_integer = integer_part[:5]  # Take the first 5 integers
            new_decimal = integer_part[5:] + decimal_part  # The remaining integers are converted to decimals
        
        # Add zeros to the decimal places
        if new_decimal == '':
            new_decimal = '0'
        result = f"{new_integer}.{new_decimal}"
        result = result if '.' in result else result + '.0'
        return float(result)
    
    def open(self,msg):
        # self.news = msg['News']
        self.market_news = None
        self.current_time = msg['date']
        self.market_data = msg['last_day_trade']
        if len(self.break_news) > 3:
                self.last_news_fundamental_information = self.Update_fundamental_information_news(result=self.break_news)
                self.break_news.clear()
        if self.previous_long_term_memory != None:
            self.short_term_memory = self.last_self_evaluation_list
            self.previous_long_term_memory = self.previous_self_reflection
            self.previous_self_reflection = self.Long_Memory()
            
            self.last_self_evaluation_list.clear()
            self.technical_indicator.append(self.market_data)
            self.short_term_memory = self.technical_indicator
            self.previous_long_term_memory = self.previous_technical_indicator
            self.previous_technical_indicator = self.Long_Memory()
            self.technical_indicator.clear()
            self.short_term_memory = self.trade_history
            self.previous_long_term_memory = self.previous_trade_history 
            self.previous_trade_history = self.Long_Memory()
            self.trade_history.clear()
            self.short_term_memory = self.market_sentiment_list
            self.previous_long_term_memory = self.previous_market_sentiment
            self.previous_market_sentiment = self.Long_Memory()
            self.market_sentiment_list.clear()
            self.short_term_memory = self.institutional_policy
            self.previous_long_term_memory = self.previous_institutional_policy
            self.previous_institutional_policy = self.Long_Memory()
            self.institutional_policy.clear()
            self.last_market_data.clear()
        else:
            self.previous_technical_indicator = self.market_data
        self.last_technical_indicator = self.previous_technical_indicator

        if msg['policy']:
            for policy_item in msg['policy']:
                self.one_news = policy_item
                self.last_institutional_policy = self.Policy_Indicators()
                self.institutional_policy.append(self.last_institutional_policy)
        
        if msg['News']:  # Check if the list is not empty
            for news_item in msg['News']:  # Traverse the news list
                self.news_agent = NewsAgent(config = base_config, source = random.choice(self.news_sources))
                news = self.news_agent.Generate_rumor_news(information=news_item)
                
                self.datetime = news['datetime']
                summary = self.summary_break_news(news)  # Process single news
                self.break_news.append(summary)
                self.summary_news.append(summary)
                self.one_news = news
                self.last_institutional_policy = self.Policy_Indicators()
                self.institutional_policy.append(self.last_institutional_policy)
            # print(self.summary_news)
            self.market_sentiment = self.Update_market_sentiment()
            self.market_sentiment_list.append(self.market_sentiment)
            print(self.market_sentiment)    
            
        else:
           self.market_sentiment = self.Market_Sentiment_None()
           self.market_sentiment_list.append(self.market_sentiment)
           print(self.market_sentiment)
        self.last_institutional_policy = self.institutional_policy
        print(self.last_institutional_policy)
        self.strategy = self.select_strategy()   
        result = self.opening_price()
        res = result
        if isinstance(result[0], str):
            result_price = re.sub(r"^```json\s*|^```python\s*|^```[\s]*|```$", "", result[0].strip(), flags=re.MULTILINE)
            try:
                result_json = json.loads(result_price)
            except Exception:
                try:
                    fixed_result = result_price.replace("'", "\"")
                    result_json = json.loads(fixed_result)
                except Exception as e:
                    print("The result format cannot be parsed:", e)
                    price_match = re.search(r"'price':\s*'(\d+)'", result_price)
                    if price_match:
                        result_json = {"price": price_match.group(1)}
                    else:
                        result_json = {"price": "0"}
        else:
            result_json = result[0]
        price = double(result_json.get('price', 0))
        integer_part = int(abs(price))
        if  integer_part <= 999:
            price = self.transform_price(price)
        print("The processed price",price)
        self.last_price = price
        self.summary_news.clear()
        return res
    
    def self_evaluation(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.self_evaluation.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                last_surplus_rate=self.last_surplus_rate,
                                Investment_style=self.investment_style,
                                last_price=self.last_price,
                                strategy=self.strategy,
                                datetime=self.current_time,
                                surplus_rate=self.surplus_rate,
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Please analyze the profit or deficit information of the this transaction above information, then give the self evaluation for this trade(combine personal internal information and personal investment personality, return strictly in JSON format), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Do not return in markdown format ! Format example: {{'self_evaluation': 'My self evaluation for this trade ...'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("Self-assessment: JSON parsing failed, reaching the maximum retry count")
        
    
    def Update_fundamental_information_news(self, result):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Update_fundamental_information_news.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                
                                last_news_fundamental_information=self.last_news_fundamental_information,
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Please update the news fundamental information of the company according to the information provided(combine your personal internal information and personal investment personality, return strictly in JSON format). Format example: {{'news_fundamental_information': 'My new news fundamental information for AAPL ...'}}, strictly in JSON format!\n"
                            f"{result if isinstance(result, str) else json.dumps(result.to_dict(orient='records') if hasattr(result, 'to_dict') else (result if isinstance(result, (dict, list)) else str(result)), ensure_ascii=False)}"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("News fundamental information update JSON parsing failed, reached the maximum number of retries")
        
        
    
    def Update_fundamental_information_finance(self, result):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Update_fundamental_information_finance.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                
                                last_finance_fundamental_information=self.last_finance_fundamental_information,
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Please update the financial fundamental information of the company according to the information provided(combine your personal internal information and personal investment personality, return strictly in JSON format). Format example: {{'finance_fundamental_information': 'My new financial fundamental information for AAPL ...'}}, strictly in JSON format!\n"
                            f"{result if isinstance(result, str) else json.dumps(result.to_dict(orient='records') if hasattr(result, 'to_dict') else (result if isinstance(result, (dict, list)) else str(result)), ensure_ascii=False)}"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("Financial fundamental information update JSON parsing failed, reached the maximum number of retries")
        
        
    
    def updata_next_goal(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.updata_next_goal.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                self_reflection=self.last_self_evaluation,
                                Investment_style=self.investment_style,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please give the next goal for next trade above information(combine personal internal information and personal investment personality, return strictly in JSON format), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Next GoalFormat example: {{'next_goal': 'My next goal for next trade ...'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("The next target update JSON parsing failed,reached the maximum number of retries")
        
    
    def Investment_Style(self):
        response_stream = self.client.chat.completions.create(
            model=self.generate_model,
             
            messages=[
                {
                    "role": "system",
                    "content": PROMPT.Investment_Style.format(
                        context=PROMPT.context.format(
                            invest_company = self.invest_company,
                        ),
                        Company = self.company,
                        Investment_style = self.investment_style,
                        previous_self_reflection = self.previous_self_reflection,
                    )
                },
                {
                    "role": "user",
                    # Fix the method: convert the DataFrame to a dictionary and then serialize
                    "content": "Please analyse the above information, then give my new investment style strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Investment Style. Format example: {{'investment_style': 'My investment style is ...'}}. Return strictly in JSON format!"
                }
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response_stream.choices[0].message.content
        return json.loads(content)
    
    def Long_Memory(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Long_Memory.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                previous_long_term_memory=self.previous_long_term_memory,
                                short_term_memory=self.short_term_memory,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please give long-term memory based on my short-term memory list and previous long-term memory, which can only be compressed and cannot delete or ignore any information. Return strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Long-Term Memory. Format example: {{'long_term_memory': 'My long-term memory about XXX is ...'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("Long-term memory update JSON parsing failed, reaching the maximum retry count")
    
    def Update_market_sentiment(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Market_Sentiment.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Investment_style=self.investment_style,
                                datetime=self.current_time,
                                intraday_news=self.summary_news,
                                market_data=self.market_data,
                                market_sentiment=self.market_sentiment,
                                market_news = self.market_news,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please analyse the market sentiment above information, then give your market sentiment(combine your personal internal information and personal investment personality, return strictly in JSON format). Format example: {{'external_market_sentiment': 'External market sentiment for AAPL ...', 'stock_market_sentiment': 'Stock market sentiment for AAPL ...','datetime': 'Input datetime'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("Market sentiment update JSON parsing failed, reaching the maximum retry count")
        

    def Market_Sentiment_None(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Market_Sentiment_None.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                company=self.company,
                                Investment_style=self.investment_style,
                                datetime=self.current_time,
                                market_data=self.market_data,
                                market_sentiment=self.market_sentiment,
                                market_news = self.market_news,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please analyse the stock market sentiment above information, then give stock market sentiment strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Market Sentiment. Format example: {{'stock_market_sentiment': 'Stock market sentiment for AAPL ...','datetime': 'Input datetime'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("News market sentiment analysis JSON parsing failed, reaching the maximum retry count")
        
    
    def Policy_Indicators(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Policy_Indicators.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                intraday_news=self.one_news,
                                datetime=self.current_time,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please analyse the policy indicators above information, then give your policy indicators(combine your personal internal information and personal investment personality, return strictly in JSON format). Format example: Format example: {{'policy_indicators': 'Policy indicators for XXX......', 'datetime': 'Input datetime'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("Policy indicators analysis JSON parsing failed, reaching the maximum retry count")
        
    
    def Technical_Indicators(self):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=self.generate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.Technical_Indicators.format(
                                context=PROMPT.context.format(
                                    invest_company = self.invest_company,
                                ),
                                Company=self.company,
                                Investment_style=self.investment_style,
                                datetime=self.current_time,
                                market_data=self.last_market_data,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please analyse the market technical indicators above information, then give market trend strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Technical Indicators. Format example: {{'technical_indicators': 'The technical indicators in the market are respectively ... ', 'market_trend':'The market trend I think is ...' ,'datetime': 'Input datetime'}}. Return strictly in JSON format!"
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Try to parse JSON
                content = response_stream.choices[0].message.content
                return json.loads(content)
                
            except (json.JSONDecodeError, TypeError):
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError("echnical indicators analysis JSON parsing failed, reaching the maximum retry count")
        
    def test_1(self):
        print("Manager Agent")
        return "Manager_Agent"
    
