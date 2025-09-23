import pandas as pd
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np
from util.util import log_print

class OptimizedDataExporter:
    """
    Optimized data exporter, specifically designed to solve the problem of data saving taking too long in kernelTerminating()
    """
    
    @staticmethod
    def export_orderbook_data_parallel(exchange_agent):
        """
        Parallel export of order book data, significantly reducing saving time
        """
        start_time = time.time()
        
        # Get all symbols that need to be exported
        symbols = list(exchange_agent.order_books.keys())
        
        if not symbols:
            log_print("No order book data needs to be exported")
            return
        
        # Use thread pool to parallel process data export for different symbols
        max_workers = min(len(symbols), mp.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all export tasks
            futures = []
            
            for symbol in symbols:
                # Order book snapshot export
                future_orderbook = executor.submit(
                    OptimizedDataExporter._export_orderbook_snapshots_optimized,
                    exchange_agent, symbol
                )
                futures.append(('orderbook', symbol, future_orderbook))
                
                # JSON data export
                future_json = executor.submit(
                    OptimizedDataExporter._export_json_data_optimized,
                    exchange_agent, symbol
                )
                futures.append(('json', symbol, future_json))
            
            # Wait for all tasks to complete
            for task_type, symbol, future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minutes timeout
                    log_print(f"{task_type} data exported: {symbol}, time: {result:.2f} seconds")
                except Exception as e:
                    log_print(f"{task_type} data export failed: {symbol}, error: {e}")
        
        total_time = time.time() - start_time
        log_print(f"All order book data exported, total time: {total_time:.2f} seconds")
        
        return total_time
    
    @staticmethod
    def _export_orderbook_snapshots_optimized(exchange_agent, symbol):
        """
        Optimized order book snapshot export
        """
        start_time = time.time()
        
        try:
            order_book = exchange_agent.order_books[symbol]
            
            # Check if there is data to export
            if not order_book.book_log:
                log_print(f"Order book {symbol} has no snapshot data to export")
                return 0
            
            # Use optimized data processing method
            log_print(f"Start exporting order book snapshot: {symbol}")
            
            # Batch process large data set, avoid memory overflow
            batch_size = 10000
            book_log = order_book.book_log
            
            if len(book_log) > batch_size:
                # Batch process large data set
                OptimizedDataExporter._export_large_orderbook_batched(
                    exchange_agent, symbol, book_log, batch_size
                )
            else:
                # Directly process small data set
                exchange_agent.logOrderBookSnapshots(symbol)
            
        except Exception as e:
            log_print(f"Export order book snapshot failed: {symbol}, error: {e}")
            raise
        
        return time.time() - start_time
    
    @staticmethod
    def _export_large_orderbook_batched(exchange_agent, symbol, book_log, batch_size):
        """
        Batch process large order book data
        """
        total_batches = (len(book_log) + batch_size - 1) // batch_size
        
        for i in range(0, len(book_log), batch_size):
            batch_num = i // batch_size + 1
            log_print(f"Process order book {symbol} batch {batch_num}/{total_batches}")
            
            # Temporarily replace book_log with current batch
            original_log = exchange_agent.order_books[symbol].book_log
            exchange_agent.order_books[symbol].book_log = book_log[i:i+batch_size]
            
            try:
                # Export current batch
                filename_suffix = f"_batch_{batch_num}" if total_batches > 1 else ""
                exchange_agent.logOrderBookSnapshots(symbol)
            finally:
                # Restore original data
                exchange_agent.order_books[symbol].book_log = original_log
    
    @staticmethod
    def _export_json_data_optimized(exchange_agent, symbol):
        """
        Optimized JSON data export
        """
        start_time = time.time()
        
        try:
            order_book = exchange_agent.order_books[symbol]
            
            # Parallel export multiple JSON files
            json_tasks = [
                ('last_trade_info', order_book.last_trade_info),
                ('minute_volume', {str(k): v for k, v in order_book.minute_volume.items()}),
                ('book_log', order_book.book_log)
            ]
            
            # Use thread pool to parallel write JSON files
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                for data_type, data in json_tasks:
                    future = executor.submit(
                        OptimizedDataExporter._write_json_file,
                        symbol, data_type, data
                    )
                    futures.append(future)
                
                # Wait for all JSON files to be written
                for future in futures:
                    future.result()
            
        except Exception as e:
            log_print(f"Export JSON data failed: {symbol}, error: {e}")
            raise
        
        return time.time() - start_time
    
    @staticmethod
    def _write_json_file(symbol, data_type, data):
        """
        Optimized JSON file write
        """
        try:
            # Ensure directory exists
            os.makedirs("Experimental_data/result_data", exist_ok=True)
            
            file_path = f"Experimental_data/result_data/orderbook_{data_type}_{symbol}.json"
            
            # Use more efficient JSON serialization parameters
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, 
                         ensure_ascii=False, 
                         default=str, 
                         indent=None,  # No indentation, reduce file size
                         separators=(',', ':'))  
            
            log_print(f"JSON file saved: {file_path}")
            
        except Exception as e:
            log_print(f"Write JSON file failed: {data_type}_{symbol}, error: {e}")
            raise
    
    @staticmethod
    def export_oracle_data_optimized(exchange_agent):
        """
        Optimized Oracle data export
        """
        start_time = time.time()
        
        try:
            if not hasattr(exchange_agent.oracle, 'f_log'):
                return 0
            
            # Parallel process Oracle data
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = []
                
                for symbol in exchange_agent.oracle.f_log:
                    future = executor.submit(
                        OptimizedDataExporter._export_single_oracle_symbol,
                        exchange_agent, symbol
                    )
                    futures.append((symbol, future))
                
                # Wait for all tasks to complete
                for symbol, future in futures:
                    try:
                        future.result(timeout=60)
                        log_print(f"Oracle data exported: {symbol}")
                    except Exception as e:
                        log_print(f"Oracle data export failed: {symbol}, error: {e}")
        
        except Exception as e:
            log_print(f"Oracle data export process error: {e}")
        
        return time.time() - start_time
    
    @staticmethod
    def _export_single_oracle_symbol(exchange_agent, symbol):
        """
        Export single symbol Oracle data
        """
        try:
            dfFund = pd.DataFrame(exchange_agent.oracle.f_log[symbol])
            if not dfFund.empty:
                dfFund.set_index('FundamentalTime', inplace=True)
                exchange_agent.writeLog(dfFund, filename=f'fundamental_{symbol}')
        except Exception as e:
            log_print(f"Export Oracle data failed: {symbol}, error: {e}")
            raise

class FastDataFrameProcessor:
    """
    Fast DataFrame processor, optimize pandas operations
    """
    
    @staticmethod
    def optimize_dataframe_operations():
        """
        Optimize pandas settings to improve performance
        """
        # Set pandas options to improve performance
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        
        # If available, use a faster engine
        try:
            import pyarrow
            pd.set_option('string_storage', 'pyarrow')
        except ImportError:
            pass
    
    @staticmethod
    def efficient_dataframe_creation(data_list, columns):
        """
        Efficient DataFrame creation
        """
        if not data_list:
            return pd.DataFrame(columns=columns)
        
        # Use numpy array to create DataFrame, faster than creating directly from list
        if isinstance(data_list[0], dict):
            # Dictionary list to DataFrame
            return pd.DataFrame(data_list)
        else:
            # Normal list to DataFrame
            np_array = np.array(data_list)
            return pd.DataFrame(np_array, columns=columns)

# Use example and integration function
def optimize_kernel_terminating(exchange_agent):
    """
    Optimize kernelTerminating method main entry function
    """
    log_print("Start optimized data export process...")
    
    # 1. Optimize pandas settings
    FastDataFrameProcessor.optimize_dataframe_operations()
    
    # 2. Parallel export Oracle data
    oracle_time = OptimizedDataExporter.export_oracle_data_optimized(exchange_agent)
    
    # 3. Parallel export order book data
    orderbook_time = OptimizedDataExporter.export_orderbook_data_parallel(exchange_agent)
    
    total_time = oracle_time + orderbook_time
    log_print(f"Optimized data export completed, total time: {total_time:.2f} seconds")
    
    return total_time
