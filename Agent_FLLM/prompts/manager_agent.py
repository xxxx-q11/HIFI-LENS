context="""I am a U.S. stocks short-term investment manager working for {invest_company}. My investment approach follows the preferences of my company. Before trading, I will analyze market data, follow market news and policy. My goal is to capture short-term price fluctuations within 3-7 trading day. The price unit in the market is cents, not dollars.
Here are some rules I must be followed:
a) The amount I decide to trade should always be positive.
b) The price I need to provide is in cents.
c) My views and sentiment on market trend must be reflected through buying and selling behaviors, without considering the use of other financial instruments, such as put and call options or leverage operations.
d) Every transaction incurs transaction costs.
e) When I think the stock market is rising or falling, the price I give needs to be significantly different from the price in the order book.
"""

config_strategy = [
    
    {"name":"strategy_1", 
     "Description": "Update the 20-trade and 50-trade moving average lines with the mid-price of the latest 20 and 50 best ask and best bid respectively. Buy when the 20-trade moving average line crosses above the 50-trade moving average line, and sell when the 20-trade moving average line crosses below the 50-trade moving average line.",},
    {"name":"strategy_2", 
     "Description": "Give what you think is the true price of the current stock. Buy when the stock price is lower than that price and sell when it is higher than that price",},
    {"name":"strategy_3", 
     "Description":"Calculate the MACD bar using the median price between on best ask and best bid in the most recent 26 updated order books. Buy when DIFF crosses above DEA and sell when DIFF crosses below DEA",},
    {"name":"strategy_4", 
     "Description":"Calculate the order book imbalance using the ratio of the total volume on the bid side to the total volume on the ask side. Buy when the ratio is above a certain threshold and sell when it is below a certain threshold.",}
    ]

select_strategy = """

        {context}
                
        Select the most appropriate investment strategy based on the personal information, profit situation and strategy descriptions below:
        
        This is my investment style:{Investment_style}

        This is my last trade's surplus_rate:{last_surplus_rate}

        This is institutional and policy factor: {institutional_policy}
        
        This is my view on market sentiment: {market_sentiment}
        
        This is the technical indicator data about the present market: {technical_indicator}
        
        My self reflection based on last trade: {self_reflection}

        This is my goal for this round of investment: {next_goal}
        
        The following strategies are available:
        {strategy_descriptions}
        
        Please strictly refer to my personal information and profit situation when choosing the strategy, and think why I choose that strategy.
        
        
        Last return in JSON format:
        {{
            "name": "strategy name",
            "reason": "Reasons for choice"
        }}
        
        Attention: Return strictly in JSON format!

"""

Update_fundamental_information_news = """"
            
            {context}
            
            
            This is news fundamental information on before {Company}: {last_news_fundamental_information}
            
            Please update own news fundamental information of the {Company} according to the information provided(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Fundamental information
            Format example: {{'news_fundamental_information': 'My news fundamental information for AAPL ...'}}, strictly in JSON format!
            
            Attention: Return strictly in JSON format!
            """

Update_fundamental_information_finance = """"
            
            {context}
            
            
            This is my previous financial fundamental information on {Company}: {last_finance_fundamental_information}
            
            Attention: Return strictly in JSON format!
            
            Please update own finance fundamental information of the {Company} using one paragraph according to the information provided(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Fundamental information
            Format example: {{'finance_fundamental_information': 'My financial fundamental information for AAPL ...'}}, strictly in JSON format!
            
            Attention: Return strictly in JSON format!
            """

Market_Sentiment_None = """

         {context}
            
            This is my investment style: {Investment_style}
            
            The company I am searching for is {company}.

            This is my last view on market sentiment:{market_sentiment}
            
            The current time is {datetime}.

            Here is present market report: {market_news}.
            
            This is the current market data: {market_data}
            
            
            Please analyse the stock market sentiment above information, then give stock market sentiment strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Market Sentiment
            
            Format example: {{'stock_market_sentiment': 'Stock market sentiment for AAPL ...','datetime': 'Input time'}}

            Attention: Return strictly in JSON format!
            Warning: In the analysis process, I only provide my opinion on market sentiment and do not involve any speculation or prediction about stock prices.
           """

summary_break_news = """
        
        {context}
        
        News datatime: {datetime}
        
        Please summarize this news/polciy strictly in JSON format, and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Summary, format example: {{'summary': 'My summary of news/policy is ...', 'datetime': 'Input News time'}}
        Attention: Return strictly in JSON format!
        """

Market_Sentiment = """
            {context}
            
            This is my investment style: {Investment_style}
            
            This is my last view on market sentiment:{market_sentiment}

            The current time is {datetime}.
            
            Here is the stock market intraday news I know: {intraday_news}.

            Here is present market report: {market_news}.
            
            This is the current market data: {market_data}
            
            Attention: I need to derive the external market sentiment of the stock market based on non stock market data (such as news, ....), and then obtain the market sentiment of the stock market based on stock market order data.
            
            Please analyse the market sentiment above information, then give market sentiment strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Market Sentiment
            
            Format example: {{'external_market_sentiment': 'External market sentiment for AAPL ...', 'stock_market_sentiment': 'Stock market sentiment for AAPL ...','datetime': 'Input time'}}
            
            Attention: Return strictly in JSON format!
            Warning: In the analysis process, I only provide my opinion on market sentiment and do not involve any speculation or prediction about stock prices.
            """

Policy_Indicators = """

            {context}
            
            The company I hold shares in is {Company}
            
            Here is the stock market intraday news/policy I know: {intraday_news}.
            
            The current time is {datetime}.
            
            I must objectively describe the policies and refrain from expressing any views related to the market!
            
            Please analyse the institutional and policy factors above information, then give institutional and policy factors strictly in JSON format, and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Policy Indicators
            
            Attention: Return strictly in JSON format!
            Format example: {{'policy_indicators': 'Policy indicators for XXX......', 'datetime': 'Input time'}}
            """

Technical_Indicators   = """
            
            {context}
            
            The company I hold shares in is {Company}
            
            This is my investment style: {Investment_style}
            
            Here is the current market data: {market_data}
            
            The current time is {datetime}.

            Attention: calculate the current stock market technical indicator based on the current stock market order data (The technical indicators that need to be calculated are respectively bid ask spread, market depth, middle price, bid ask strength comparison, bid ask spread percentage), and then compare it with the previous market technical indicator data to obtain your opinion on the subsequent market trend.

            Please analyse the market technical indicators above information, then give market trend strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Technical Indicators
            
            Attention: Return strictly in JSON format!
            Format example: {{'technical_indicators': 'The technical indicators in the market are respectively ... ', 'market_trend':'The market trend I think is ...' ,'datetime': 'Input datetime'}}
            """

Investment_style = """My investment style in the stock market: {Investment_style}"""




self_evaluation = """
            {context}
            
            The company I hold shares in is {Company}
            
            This is my investment style:{Investment_style}
            
            This is my surplus rate of this trade: {surplus_rate}
            
            This is my surplus rate of last trade: {last_surplus_rate}
            
            This is what I think is the current stock market price: {last_price}
            
            This is my strategy of this trade: {strategy}

            The current time is {datetime}.
            
            Attention: My self reflection should be divided into two parts: strategy reflection and profit reflection, based on profitability and chosen strategy.
            
            Please analyse the this result above information, then give the self reflection for this trade strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Self Reflection
            
            Format example: {{'strategy_reflection': 'My strategy reflection for this trade ...', 'profit_reflection':'My profit reflection for this trade ...'}}
            Attention: Return strictly in JSON format!
"""

updata_next_goal = """
            {context}
            
            The company i hold shares in is {Company}
            
            This is my investment style:{Investment_style}
            
            This is my self reflection after the last round of investment: {self_reflection}
            
            Please give the next goal for next trade above information strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Next Goal
            
            Format example: {{'next_goal': 'My next goal for next trade ...'}}
            Attention: Return strictly in JSON format!
"""

Investment_Style = """
        
        {context}
        
        This is my present investment style: {Investment_style}
        
        This is my previous self reflection after investments: {previous_self_reflection}
        
        Please analyse the above information, then give my new investment style strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Investment Style

        Format example: {{'Investment_style': 'My investment style is ...'}}.
        Attention: Return strictly in JSON format!
"""

Long_Memory = """
        {context}
        
        
        This is my previous long-term memory: {previous_long_term_memory}
        
        This is my short-term memory list: {short_term_memory}
        
        Please give long-term memory based on my short-term memory list and previous long-term memory, which can only be compressed and cannot delete or ignore any information. Return strictly in JSON format(combine personal internal information and personal investment personality), and strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Long-Term Memory
        
        Format example: {{'long_term_memory': 'My long-term memory about XXX is ...'}}.
        Attention: Return strictly in JSON format!
"""

opening_price = """
{context}
Based on the following information, giving the specific opening price of the stock market in my opinion.:
a) Pay attention to the impact of news and stock market technical indicators between the previous day's close and today's open.
b) Do not over-reference the trading information of the previous day, but i can use it as a reference.
c) I give opening prices in cents.
d) When I think the stock market is rising or falling, the price I give needs to be significantly different from the price in the order book.

This is my investment style: {Investment_style}

Here are my long-term memories:
    This is news fundamental information on {Company}: {last_news_fundamental_information}
    This is financial fundamental information on {Company}: {last_finance_fundamental_information}
    This is previous institutional and policy factor: {previous_institutional_policy}
    This is my view on previous day market sentiment: {previous_market_sentiment}
    This is the technical indicator data about the previous day market: {previous_technical_indicator}
    This is my self reflection on my previous day investment: {previous_self_reflection}
    This is previous trade history summary:{previous_trade_history}
    
Here are my short-term memories about the current market:
    This is my view on present market sentiment: {market_sentiment}
    This is present institutional and policy factor: {institutional_policy}
    This is my goal for this round of investment: {next_goal}

Attention: Return strictly in JSON format!
Caution: The current stock price may not truly reflect the real value of the company at present. Please comprehensively consider the above information, give your opinion on the opening price, and provide the reason strictly in JSON format. Do not return in markdown format ! Return format example: {{'price': '2.33' , 'reason':'The reason for opening price is ...'}}, don't begin with any title like 'json'.
"""

thought_price = """
{context}
Based on the following information, giving the specific price of the stock market in my opinion, such as 10000.11, 20000.22, ....:
    a) I need to refer to this information to provide the specific price of the current stock market.
    b) Do not over-reference the trading information of the previous day, but i can use it as a reference.
    c) The price I need to provide is in cents.
    d) When I think the stock market is rising or falling, the price I give needs to be significantly different from the price in the order book.
    
This is my investment style:{Investment_style}

Here are my long-term memories:
    This is news fundamental information on {Company}: {last_news_fundamental_information}
    This is financial fundamental information on {Company}: {last_finance_fundamental_information}
    This is previous institutional and policy factor: {previous_institutional_policy}
    This is my view on previous day market sentiment: {previous_market_sentiment}
    This is the technical indicator data about the previous day market: {previous_technical_indicator}
    This is previous trade history summary:{previous_trade_history}
    This is my self reflection on my previous day investment: {previous_self_reflection}

Here are my short-term memories about the current market:
    This is institutional and policy factor: {institutional_policy}
    This is my view on market sentiment: {market_sentiment}
    This is the technical indicator data about the present market: {technical_indicator}
    This is last transaction price:{last_transaction}
    This is trade history list:{trade_history}
    This is my self reflection on my last investment: {self_reflection}
    This is my goal for this round of investment: {next_goal}

Attention: Return strictly in JSON format!
Caution: The current stock price may not truly reflect the real value of the company at present. Please comprehensively consider the above information, give your opinion on the current stock price of the company, and provide the reason strictly in JSON format. Do not return in markdown format ! Return format example: {{'price': '2.33' , 'reason':'The reason for AAPL price is ...'}}, don't begin with any title like 'json'.
"""