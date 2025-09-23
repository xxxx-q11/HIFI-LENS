import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os 
import matplotlib.ticker as ticker 
import pandas as pd

# -- User-configurable parameters --
start_date_str = '2025-04-02'  # Set the start date
number_of_days = 3           # Set the number of consecutive drawing days required
company = "JNJ"  # Set the company name for file names and chart titles
data_directory = 'Data/Trump/JNJ/json_data_JNJ'
#data_directory = os.path.join(base_dir, f'json_data_{company}')

# -- End of User-configurable parameters --

# // ... existing code ...
# A list for storing all times and opening prices
all_timestamps_data = [] # Store the tuple (datetime_object, opening_price)

# Generate file path list
file_paths = []
try:
    current_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    for i in range(number_of_days):
        date_to_process = current_date + timedelta(days=i)
        file_name = f"{date_to_process.strftime('%Y_%m_%d')}_trading_day.json"
        full_file_path = os.path.join(data_directory, file_name)
        file_paths.append(full_file_path)
except ValueError:
    print(f"Error: The start date '{start_date_str}' is not in the correct format, please use the YYYY-MM-DD format.")
    exit()

print(f"The file paths to be processed: {file_paths}")

# Traverse each file and read the data
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"Warning: The file {file_path} was not found, this file will be skipped.")
        continue
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract data
        for ts_str, values in data.items():
            # Convert the string timestamp to a datetime object for better plotting
            try:
                dt_object = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
                opening_price = values['Open']
                all_timestamps_data.append((dt_object, opening_price))
            except ValueError as e:
                print(f"Warning: The timestamp '{ts_str}' or its corresponding data cannot be parsed in the file {file_path}: {e}")
                continue # Skip entries that cannot be parsed
            except KeyError:
                print(f"Warning: The timestamp '{ts_str}' is missing the 'Open' key in the file {file_path}.")
                continue
    except FileNotFoundError: # Theoretically caught by the above os.path.exists, but retained for safety
        print(f"Error: The file {file_path} was not found.")
        continue # Continue processing the next file
    except json.JSONDecodeError:
        print(f"Error: The JSON file {file_path} cannot be decoded.")
        continue # Continue processing the next file
    except Exception as e:
        print(f"Error: An unknown error occurred while processing the file {file_path}: {e}")
        continue # Continue processing the next file

# Sort all data by timestamp
all_timestamps_data.sort(key=lambda x: x[0])

# Separate sorted timestamps and Open prices
# timestamps = [item[0]-timedelta(hours=12) for item in all_timestamps_data]
timestamps = [item[0] for item in all_timestamps_data]
opening_prices = [item[1]*100 for item in all_timestamps_data]

# Read data
with open(f"Experimental_data/result_data/orderbook_last_trade_info_{company}.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame for easier processing
df = pd.DataFrame(data) #
df["time"] = pd.to_datetime(df["time"])
if not timestamps or not opening_prices:
    print("Unable to extract valid data for plotting.")
else:
    # Create chart
    plt.figure(figsize=(18, 8)) # Adjust the chart size to accommodate more data

    # Use the index of the data points as the X axis
    x_indices = range(len(timestamps))
    plt.plot(x_indices, opening_prices, marker='o', linestyle='-', markersize=3)
    # Map the trade price to the X axis index of the Open price
 # Map the trade price to the X axis index of the Open price
    trade_x = []
    trade_prices = []
    for i, t in enumerate(df["time"]):
        if i % 20 == 0:  # Take one every 20 points
            # Find the closest time point index
            closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - t))
            trade_x.append(closest_idx)
            trade_prices.append(df["price"][i])
    
    plt.plot(trade_x, trade_prices, color="black", label="Trade Price")

    # Custom X axis tick format function
    def format_fn(value, tick_number):
        # 'value' is the index value on the X axis
        actual_index = int(round(value)) # Ensure it is an integer index
        if 0 <= actual_index < len(timestamps):
            # Return the timestamp formatted string corresponding to the index
            return timestamps[actual_index].strftime('%Y-%m-%d %H:%M')
        return ''

    # Use MaxNLocator to automatically select the appropriate number of ticks (based on the index)
    # nbins parameter can be adjusted to control the approximate number of ticks
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
    # Use FuncFormatter to convert the index ticks to date time labels
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_fn))
    
    plt.xticks(rotation=45, ha="right") # Rotate the X axis labels to prevent overlap, and right align

    # Set the chart title and axis labels
    plt.title(f'The opening price of consecutive {number_of_days} days (excluding trading hours, starting from {start_date_str})')
    plt.xlabel('trading time') # X axis label indicates that the time is not completely continuous
    plt.ylabel('open price')

    plt.grid(True) # Display grid
    plt.tight_layout() # Adjust the layout to accommodate the labels
    plt.show()
    plt.savefig(f'Experimental_data/result_data/{company}_price_comparison.png')  # Change to English file name