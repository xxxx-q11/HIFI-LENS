import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

company = "JNJ"  # Set the company name for file names and chart titles

# Trading Time Filter Switch - Set to True to enable filtering and avoid displaying non-trading timelines
ENABLE_TRADING_HOURS_FILTER = True

def set_trading_hours_filter(enable):
    """Set the trading time filter switch"""
    global ENABLE_TRADING_HOURS_FILTER
    ENABLE_TRADING_HOURS_FILTER = enable


# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variable, used to store the latest order book and transaction data
latest_orderbook_data = {
    'bids': [],
    'asks': [],
    'last_trade': None
}
latest_trade_history = []
latest_news = []
latest_minute_volume = []

# Global variable, used to store the reference to the OrderBook object
exchange_agent = None
order_books = {}
news = []

def is_trading_hours(dt):
    """
    Determine if the given time is within the NASDAQ trading hours
    Nasdaq trading hours (Eastern Time of the United States)
    From Monday to Friday
    - 9:30 a.m. - 4:00 p.m. (4:00 p.m.
    """
    # Check if it is a weekday (Monday to Friday)
    if dt.weekday() >= 5:  # Saturday =5, Sunday =6
        return False
    
    # Get the time part
    current_time = dt.time()
    
    # Define the NASDAQ trading hours
    market_open = dt_time(9, 30)   # 9:30 a.m.
    market_close = dt_time(16, 0)  # 4:00 p.m. (4:00 p.m.)
    
    # Check if it is within the trading hours
    return market_open <= current_time <= market_close

def filter_trading_hours_data(data_list, time_key='time'):
    """
    Filter data, only keep the data within the NASDAQ trading hours
    """
    filtered_data = []
    for item in data_list:
        if not item or time_key not in item:
            continue
        
        try:
            time_str = item[time_key]
            if isinstance(time_str, str):
                time_str = time_str.replace(' ', 'T')
                dot_index = time_str.find('.')
                if dot_index != -1:
                    time_str = time_str[:dot_index + 4] 
                dt = datetime.fromisoformat(time_str)
            else:
                dt = time_str
            
            # Only retain data during trading hours
            if is_trading_hours(dt):
                filtered_data.append(item)
        except (ValueError, TypeError) as e:
            print(f"Error parsing time: {e}, time string: {item.get(time_key)}")
            continue
    
    return filtered_data

def initialize(exchange_agent_instance):
    """Initialization function, called by the abides main program, passing in the ExchangeAgent instance"""
    global exchange_agent, order_books, news
    exchange_agent = exchange_agent_instance
    # Get all order books
    order_books = exchange_agent.order_books
    # # Get symbol
    # company = list(order_books.keys())[0]
    # Get news
    if exchange_agent.market_news:
        news = exchange_agent.market_news
    
    # Print news data, for debugging
    print(f"News data at the time of initialization: {news}")
    
    # Start Flask application
    start_web_server()

def get_latest_orderbook_data(symbol=company):
    """Get the latest order book data from the OrderBook object"""
    global order_books
    
    if symbol not in order_books:
        return {'bids': [], 'asks': [], 'last_trade': None}
    
    orderbook = order_books[symbol]
    
    # Get bids and asks
    bids = []
    asks = []
    
    # Get the first 5 bids
    for i, price_level in enumerate(orderbook.getInsideBids(5)):
        price, volume = price_level
        bids.append({'price': price, 'volume': volume})
    
    # Get the first 5 asks
    for i, price_level in enumerate(orderbook.getInsideAsks(5)):
        price, volume = price_level
        asks.append({'price': price, 'volume': volume})
    
    return {
        'bids': bids,
        'asks': asks,
        'last_trade': orderbook.last_trade
    }

def get_trade_history(symbol=company, max_points=5000):
    """
    Get the transaction history data from the OrderBook object.
    If transaction time filtering is enabled, only data within the transaction time period will be returned.
    If there are too many data points, downsampling will be performed to ensure the performance of the front end.
    """
    global order_books
    
    if symbol not in order_books:
        return []
    
    orderbook = order_books[symbol]
    trade_history = orderbook.last_trade_info.copy()
    

    # According to the switch, decide whether to filter the data within the transaction time period
    if ENABLE_TRADING_HOURS_FILTER:
        filtered_history = filter_trading_hours_data(trade_history, 'time')
    else:
        filtered_history = trade_history
        #print("Transaction time filtering is disabled, return all data")
    
    num_trades = len(filtered_history)
    
    if num_trades <= max_points:
        # If the data points are not many, return all
        return filtered_history
    else:
        # If the data points are too many, perform equidistant sampling
        # We use np.linspace to generate uniformly distributed indices
        indices = np.linspace(0, num_trades - 1, num=max_points, dtype=int)
        # Ensure the index is unique, and extract the data
        sampled_history = [filtered_history[i] for i in np.unique(indices)]
        return sampled_history

def get_latest_news():
    """Obtain the latest news data from the ExchangeAgent object"""
    global news
    
# If news is of dictionary type, convert it to the format expected by the front end
    formatted_news = []
    if isinstance(news, dict) and news:
        for timestamp, content in news.items():
            formatted_news.append({
                'headline': f'News - {timestamp} - {content["title"]}',
                'body': content["summary"]
            })
    
    return formatted_news

def get_minute_volume_history(symbol= company):
    """Get the minute volume history data from the OrderBook object"""
    global order_books
    
    if symbol not in order_books:
        return []
    
    orderbook = order_books[symbol]
    minute_volume_copy = orderbook.minute_volume.copy()
    # Convert the dictionary to a list, and sort by time
    minute_volume_list = [
        {'time': dt.isoformat(), 'volume': vol} 
        for dt, vol in minute_volume_copy.items()
    ]
    minute_volume_list.sort(key=lambda x: x['time'])
    
    
    # According to the switch, decide whether to filter the data within the transaction time period
    if ENABLE_TRADING_HOURS_FILTER:
        filtered_volume = filter_trading_hours_data(minute_volume_list, 'time')
        return filtered_volume
    else:
        #print("Transaction time filtering is disabled, return all minute volume data")
        return minute_volume_list

def update_data_thread():
    """Periodically update the data thread function"""
    global latest_orderbook_data, latest_trade_history, latest_news, news, exchange_agent, latest_minute_volume
    
    while True:
        try:
            # Get the latest news data from the ExchangeAgent
            if exchange_agent and hasattr(exchange_agent, 'market_news'):
                news = exchange_agent.market_news
            
            # Update the order book data
            latest_orderbook_data = get_latest_orderbook_data()
            
            # Update the transaction history data
            latest_trade_history = get_trade_history()

            # Update the minute volume data
            latest_minute_volume = get_minute_volume_history()

            # Update the news data
            latest_news = get_latest_news()
            
            # Send the updated data through WebSocket
            socketio.emit('orderbook_update', latest_orderbook_data)
            socketio.emit('trade_history_update', latest_trade_history)
            socketio.emit('minute_volume_update', latest_minute_volume)
            socketio.emit('news_update', latest_news)
            
            # Updated once per second
            time.sleep(1)
        except Exception as e:
            print(f"The data update thread has an error: {e}")
            time.sleep(5)  # Wait for 5 seconds after an error occurs and then try again

# Flask Routing
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index.html template: {e}")
        return f"<h1>Error</h1><p>Error rendering template: {e}</p>"

@app.route('/test')
def test():
    try:
        return render_template('test.html')
    except Exception as e:
        print(f"Error rendering test.html template: {e}")
        return f"<h1>Error</h1><p>Error rendering template: {e}</p>"

@app.route('/api/orderbook')
def get_orderbook():
    return jsonify(latest_orderbook_data)

@app.route('/api/trade_history')
def get_trades():
    return jsonify(latest_trade_history)

@app.route('/api/minute_volume')
def get_minute_volume():
    return jsonify(latest_minute_volume)

@app.route('/api/news')
def get_news():
    return jsonify(get_latest_news())

@app.route('/api/toggle_filter/<enable>')
def toggle_filter(enable):
    """Switch the trading time filter switch"""
    enable_filter = enable.lower() in ['true', '1', 'on', 'yes']
    set_trading_hours_filter(enable_filter)
    return jsonify({
        'success': True,
        'filter_enabled': ENABLE_TRADING_HOURS_FILTER,
        'message': f"Transaction time filtering has {'enabled' if ENABLE_TRADING_HOURS_FILTER else 'disabled'}"
    })

@app.route('/api/status')
def get_status():
    """Get the current status"""
    return jsonify({
        'filter_enabled': ENABLE_TRADING_HOURS_FILTER,
        'trade_history_count': len(latest_trade_history),
        'minute_volume_count': len(latest_minute_volume),
        'orderbook_bids': len(latest_orderbook_data.get('bids', [])),
        'orderbook_asks': len(latest_orderbook_data.get('asks', []))
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send the latest data when connecting
    socketio.emit('orderbook_update', latest_orderbook_data)
    socketio.emit('trade_history_update', latest_trade_history)
    socketio.emit('minute_volume_update', latest_minute_volume)
    socketio.emit('news_update', get_latest_news())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def start_web_server():
    """Start the Web server function"""
    try:
        # Start data update thread
        data_thread = threading.Thread(target=update_data_thread)
        data_thread.daemon = True
        data_thread.start()
        
        # Start Flask application in a new thread
        web_thread = threading.Thread(target=lambda: socketio.run(app, host='0.0.0.0', port=8504, debug=False))
        web_thread.daemon = True
        web_thread.start()
        
        print("The real-time visualization Web server has been started. Please visit: http://localhost:8504")
        print(f"Template Directory: {app.template_folder}")
        print(f"Static file directory: {app.static_folder}")
    except Exception as e:
        print(f"Error starting Web server: {e}")

# If directly running this script, display the error information
if __name__ == '__main__':
    print(" Error: This script should not run directly but should be imported by the abides main program and the initialize function should be called." )
    print(" Please modify the abides main program, import this module and call the initialize function at the appropriate position." )
