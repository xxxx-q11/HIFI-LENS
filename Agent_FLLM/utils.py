import json
import re
import random
from datetime import date, datetime

def extract_query(original_query):
    # Regularly match all parentheses and their internal content, replace with an empty string
    clean_query = re.sub(r'\s*$[^)]*$', '', original_query)
    # Remove leading and trailing spaces (optional)
    return clean_query.strip()

def convert_to_json(data):
    """Safe conversion of dictionary to JSON format"""
    def custom_serializer(obj):
        """Handle special type serialization"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()  # Convert to ISO8601 string
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()    # Process Pandas objects
        else:
            raise TypeError(f"不可序列化类型: {type(obj)}")

    try:
        return json.dumps(
            data,
            ensure_ascii=False,  # Allow non-ASCII characters such as Chinese
            indent=2,           # Beautify output
            default=custom_serializer
        )
    except TypeError as e:
        print(f"JSON conversion failed: {str(e)}")
        return None

def safe_json_parse(input_str):
    """Safe parsing of JSON strings containing non-standard formats"""
    try:
        # attempt directly parse
        return json.loads(input_str)
    except json.JSONDecodeError as e:
        print(f"Initial parsing failed: {str(e)}，attempt repair format...")
        # attempt replace single quotes and escape internal double quotes
        fixed_str = (
            input_str
            .replace("'", "\"")  # Replace single quotes with double quotes
            .replace('\\"', '\\\\"')  # Escape existing double quotes
        )
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            print(f"After repair, still failed: {str(e)}")
            return None
        
import json


def stream_to_json(stream):
    full_content = ""
    created = None
    try:
        for chunk in stream:
            full_content += chunk.choices[0].delta.content or ""
            role = chunk.choices[0].delta.role or role
            created = chunk.created

        # Find the position of the </think> tag
        think_index = full_content.find("</think>")
        if think_index != -1:
            # Extract the content after </think>
            content_after_think = full_content[think_index + len("</think>"):].strip()
            # Remove the line break
            content_after_think = content_after_think.replace('\n', '')

            try:
                
                return content_after_think
            except json.JSONDecodeError:
                print("The content after </think> is not a valid JSON format")
                return None
        else:
            print("</think> tag not found")
            return None

    except Exception as e:
        print(f"Error parsing stream data: {str(e)}")
        return None

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_random_records(file_path, n):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure the data is a list format
    if not isinstance(data, list):
        raise ValueError("The JSON file content must be a list format")
    
    # Calculate the actual sampling quantity (handle the case where N exceeds the data quantity)
    sample_size = min(n, len(data))
    
    # Randomly extract data
    random_records = random.sample(data, sample_size)
    
    # Convert to a JSON format string
    return json.dumps(random_records, ensure_ascii=False, indent=2)

def format_trade_history_advanced(trade_history):
    if not trade_history:
        return "No transaction history"
    
    # Create title
    text = "Transaction history (including trend statistics):\n"
    text += "{:<20} {:<10} {}\n".format("Time", "Transaction Price", "Trend of Changes")
    text += "-" * 40 + "\n"
    
    prev_price = None
    price_changes = []
    
    # Calculate price changes
    for i, trade in enumerate(trade_history):
        # Simplify and beautify the time format
        time_str = trade["time"]
        price = trade["price"]
        
        # Calculate price changes
        trend_symbol = ""
        if prev_price is not None:
            change = price - prev_price
            price_changes.append(change)
            
            if change > 0:
                trend_symbol = "↑ +{}".format(change)
            elif change < 0:
                trend_symbol = "↓ {}".format(change)  # Negative numbers will automatically have a minus sign
            else:
                trend_symbol = "→ 0"
        
        # Add line
        text += "{:<20} {:<10} {}\n".format(time_str, price, trend_symbol)
        prev_price = price
    
    # Add statistical summary
    text += "\nTransaction statistics:\n"
    text += "-" * 40 + "\n"
    
    if price_changes:
        total_changes = sum(price_changes)
        avg_change = total_changes / len(price_changes)
        num_up = sum(1 for c in price_changes if c > 0)
        num_down = sum(1 for c in price_changes if c < 0)
        num_flat = sum(1 for c in price_changes if c == 0)
        
        text += "Total changes: {:.2f}\n".format(total_changes)
        text += "Average change: {:.2f}\n".format(avg_change)
        text += "Number of price increases: {}/{} ({:.1%})\n".format(num_up, len(price_changes), num_up/len(price_changes))
        text += "Number of price declines: {}/{} ({:.1%})\n".format(num_down, len(price_changes), num_down/len(price_changes))
        text += "Number of price flattenings: {}/{} ({:.1%})\n".format(num_flat, len(price_changes), num_flat/len(price_changes))
        
    start_price = trade_history[0]["price"]
    end_price = trade_history[-1]["price"]
    total_change = end_price - start_price
    text += "\nStarting price: {}\n".format(start_price)
    text += "Closing price: {}\n".format(end_price)
    text += "Total price change: {}\n".format(total_change)
    text += "Percentage change: {:.2%}\n".format(total_change/start_price)
    
    return text

def get_investment_style(company_name, companies_list):
    """
    Obtain the corresponding investment style based on the company name

    Parameter
    company_name (str): The name of the company to be queried
    companies_list (list): A dictionary list containing company information, with each dictionary including the keys "company" and "investment_style"

    Return
    str: The investment style of the company. If not found, a prompt message will be returned
    """
    for company_info in companies_list:
        # Ignore the case and possible punctuation differences in the name, accurately match the company name
        if company_info.get("company") == company_name:
            return company_info.get("investment_style")
    return f"The investment style information for the company '{company_name}' was not found"