import json
import random
from typing import Any, Dict, List
from numpy import double
from openai import OpenAI
import openai
import time
import Agent_FLLM.prompts.news_agent as PROMPT
from config_LLM import base_config
from Agent_FLLM.utils import extract_query,stream_to_json, read_json_file, extract_random_records
import re

class NewsAgent:
    def __init__(self, source,config: Dict[str, Any]):
        self.config = config
        self.keywords = None
        self.time = None
        self.model = self.config['apikeys']['generate_model_name']  # Use the fixed model name set in the script
        self.temple = None
        self.source = source
        self.summary = None
        self.config = config
        self.client = OpenAI(
            api_key=self.config['apikeys']['deepseek_api_key'],  # Use the fixed API Key set in the script
            base_url="https://api.deepseek.com"  # Caddy Proxy address
        )
     
        self.temple = extract_random_records(f'./Data/Template/{self.source}/generated_template.json',1)

    def Generate_rumor_news(self, information):
        """
        Generate a rumor news.
        This function will first determine whether the content returned by the API is in the correct JSON format.
        If so, return the parsed dictionary directly.
        If not, it will automatically re-run, with a maximum of 3 re-runs.
        """
        self.keywords = information['title']
        self.time = information['datetime']
        self.summary = information['summary']
        
        # Define the maximum number of retries
        max_retries = 10
        # The total number of attempts is 1 initial attempt + max_retries retries
        total_attempts = 1 + max_retries
        
        # Use for loop to control the number of attempts
        for attempt in range(total_attempts):
            try:
                # --- Step 1: Call the API to get data ---
                # This part of the code needs to be placed inside the loop, because each retry needs to re-call the API
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": PROMPT.rumor_news.format(
                                context=PROMPT.context,
                                generate_time=self.time,
                                keywords=self.keywords,
                                temple=self.temple,
                                summary=self.summary,
                            )
                        },
                        {
                            "role": "user",
                            "content": "Please generate news in JSON format based on the given template, keywords and time and summary. Strictly adhere to the set character portraits without any warnings or reminders and are not allowed to add any explanatory text. Then give: 1) Rumor News, format example: {{'title':'', 'souce':'','datetime': '', 'content': ''}}, don't begin with any title like 'json'."
                        }
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                content = response_stream.choices[0].message.content
                
                # --- step 2: Try to parse JSON ---
                # This is the critical decision step. If json.loads() succeeds, the code will continue to execute downward.
                # If content is not a valid JSON string, it will throw a json.JSONDecodeError exception.
                parsed_data = json.loads(content)
                
                # --- step 3: Successful handling ---
                # If the code can execute here, it means the JSON format is correct and parsing is successful.
                print(f"The valid JSON was successfully obtained on the {attempt + 1} th attempt")
                return parsed_data  # Directly return the result and exit the function and loop

            except json.JSONDecodeError as e:
                # --- Step 4: Failure handling ---
                # If the code in the try block throws a JSONDecodeError, the program will jump here.
                print(f"attempt the {attempt + 1}/{total_attempts} fails: the returned JSON is not valid. Error: {e}")
                # Print out the non-JSON content received, which helps with debugging
                print(f"Received original content (first 500 characters): {content}...")

                # Check if there is a retry opportunity
                if attempt < max_retries:
                    print("The next attempt will be made in 1 second...")
                    time.sleep(1)  # Wait for a short period of time before retrying, which is a good practice
                else:
                    fixed_content = self._fix_json_missing_comma(content)
                    try:
                        parsed_data = json.loads(fixed_content)
                        print(f"The JSON repair was successful and it was successfully parsed in the {attempt + 1} th attempt!")
                        return parsed_data
                    except json.JSONDecodeError as fix_e:
                        # --- Step 4: After repair, still failed, prepare to retry ---
                        print(f"JSON repair failed. The error after repair: {fix_e}")
                    # If this is the last attempt (i.e., the 3rd attempt in the 0, 1, 2, 3 attempts), then do not retry
                    print("The maximum number of retries has been reached, the task failed.")
                    # Explicitly throw an exception to notify the upper layer caller that the operation failed
                    # This is more effective than returning None or {} to prevent unexpected errors in subsequent code
                    raise ValueError(f"After the {total_attempts} attempt, the valid JSON was still not obtained from the API.")
    
    def _fix_json_missing_comma(self, json_string: str) -> str:
        """
        A private method, attempt to fix the JSON string with a missing comma.
        """
        # Regular expressions look for the missing comma between the end of a value and the beginning of the next key and insert it.

        # Pattern Matching: (right parenthesis/quotation mark/number/the end of true/false/null)(blank)(the starting quotation mark of the next key)
        regex_pattern = r'([}\]\d"e|l])\s+(")'
        fixed_string = re.sub(regex_pattern, r'\1, \2', json_string)
        return fixed_string