import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.ticker as ticker

company  = "JNJ"
# Read data
with open(f"Experimental_data/result_data/orderbook_last_trade_info_{company}.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame for easier processing
df = pd.DataFrame(data) #
all_timestamps_data = [ts_str for ts_str in pd.to_datetime(df["time"])]
# df["price"] = pd.to_datetime(df["price"])
# Sort all data by timestamp
#all_timestamps_data.sort()



# Plot
plt.figure(figsize=(16, 6))
# Use the index of the data points as the X axis
x_indices = range(len(all_timestamps_data))
plt.plot(x_indices, df["price"], label="Trade Price")

# Add a red line connecting the benchmark price and the first trade price
# plt.plot([0, 1], [22389, 20554], color='red', linestyle='-', 
#          linewidth=2, label='Initial Price Change')
# Custom X axis tick format function
def format_fn(value, tick_number):
    # 'value' is the index value on the X axis
    actual_index = int(round(value)) # Make sure it is an integer index
    if 0 <= actual_index < len(all_timestamps_data):
        # Return the timestamp formatted string corresponding to the index
        return all_timestamps_data[actual_index].strftime('%Y-%m-%d %H:%M')
    return ''

# Use MaxNLocator to automatically select the appropriate number of ticks (based on the index)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
# Use FuncFormatter to convert the index ticks to date time labels
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_fn))

plt.xticks(rotation=45, ha="right") # Rotate the X axis labels to prevent overlap, and right align
plt.xlabel("Time")
plt.ylabel("Price")
plt.title(f"The chart of {company} trading price changes over time")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"Experimental_data/result_data/{company}Chart of changes in transaction prices over time.png")

import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# Read the transaction price data
try:
    with open(f"Experimental_data/result_data/orderbook_last_trade_info_{company}.json", "r", encoding="utf-8") as f:
        trade_data = json.load(f)
except FileNotFoundError:
    print("The transaction price data file was not found")
    exit(1)

# Read the base price data file
base_price_files = [
    "Experimental_data/result_data/base_price_dict_1_.json",
]

# Set colors
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

# Create a graph
plt.figure(figsize=(16, 10))

# Plot the transaction price
df = pd.DataFrame(trade_data)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')
all_timestamps_data = df['time'].tolist()
x_indices = range(len(all_timestamps_data))
plt.plot(x_indices, df["price"], label="Trade Price", color='black', linewidth=1)

for i, file_path in enumerate(base_price_files):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    times = []
    prices = []  # Outer list, storing the set of all price lists
    
    # First traversal: determine the maximum length of price_list at each time point (i.e., n)
    max_prices_count = 0
    for price_list in data.values():
        if len(price_list) > max_prices_count:
            max_prices_count = len(price_list)
    
    # Initialize prices: create max_prices_count empty lists [1,3](@ref)
    prices = [[] for _ in range(max_prices_count)]
    
    # Secondary traversal: fill in the time and price data
    for time_str, price_list in data.items():
        time_obj = pd.to_datetime(time_str)
        times.append(time_obj)
        
        # Traverse each price element at the current time point
        for idx in range(len(price_list)):
            if idx < max_prices_count:  # Prevent index out of bounds
                prices[idx].append(float(price_list[idx]["price"]))
    
    # Sort by time (using zip to combine all lists)
    sorted_data = sorted(zip(times, *prices))  
    
    # Unpack the sorted data
    times = [item[0] for item in sorted_data]
    # Dynamically update each price list [1,7](@ref)
    for idx in range(max_prices_count):
        prices[idx] = [item[idx+1] for item in sorted_data]  # Index offset+1 (0th position is time)

    # Find the position of each time point in all_timestamps_data in times
    x_indices_base = []
    for t in times:
        # Find the closest time point
        closest_idx = min(range(len(all_timestamps_data)), 
                         key=lambda i: abs(all_timestamps_data[i] - t))
        x_indices_base.append(closest_idx)

    file_name = file_path.split('/')[-1].replace('.json', '')
    for k, price_list in enumerate(prices):
        plt.plot(x_indices_base, price_list, linestyle='', marker='o', markersize=4,
                color=colors[0], alpha=0.5)
    # plt.plot(x_indices_base, prices_2, linestyle='', marker='^', markersize=4,
    #          color=colors[i], label=f'{file_name} - Price 2', alpha=0.5)

# Set x axis formatting
def format_fn(value, tick_number):
    actual_index = int(round(value))
    if 0 <= actual_index < len(all_timestamps_data):
        return all_timestamps_data[actual_index].strftime('%Y-%m-%d %H:%M')
    return ''

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_fn))

# Improve the graph
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title(f"A comparison chart of {company} transaction price and benchmark price")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.margins(x=0.02)

# Add price range annotation
price_min = df['price'].min()
price_max = df['price'].max()
plt.text(0.02, 0.98, f'Price range: {price_min:.2f} - {price_max:.2f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

# Save and display
plt.tight_layout()
plt.savefig(f"Experimental_data/result_data/{company}Chart of changes in transaction prices over time.png", bbox_inches='tight')
plt.show()