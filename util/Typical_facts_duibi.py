import pandas as pd
import numpy as np
# ... existing imports ...
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import json
import glob
from datetime import datetime, time
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import logging
import sys
import matplotlib
import matplotlib as mpl

sns.set_style("whitegrid") 
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    
    'figure.figsize': (8, 4.5),  
    'figure.dpi': 300,
    'savefig.dpi': 300,

    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'grid.color': 'lightgrey',
    'grid.linewidth': 0.5,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    
    'axes.prop_cycle': plt.cycler('color', ['#336699', '#CC6677', '#44AA99', '#882255', '#AA4499', 
                                          '#117733', '#999933', '#88CCEE', '#CC6677', '#AA4499'])
})


# ==================== Path configuration ====================
# Here you can change the input and output paths
INPUT_DIR = # Enter the folder path
BASE_OUTPUT_DIR = INPUT_DIR # Output folder path
SINGLE_FILE = None             # Single file path (if needed to process a single file)
REAL_DATA_FILE = # Real data file path
# ==================================================


def setup_logger(output_dir, file_name):
    """Set the logger"""
    log_file = os.path.join(output_dir, f"{file_name}_analysis.log")
    
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def detect_json_format(file_path):
    """Detect JSON file format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            return "dict_format", data
        elif isinstance(data, list):
            return "list_format", data
        else:
            return None, None
    except Exception as e:
        print(f"Read file failed: {e}")
        return None, None

def convert_dict_format(data):
    """Convert a dictionary with timestamp as the key to a list containing the Time field"""
    result = []
    for timestamp, values in data.items():
        new_item = {"time": timestamp}
        new_item.update(values)
        result.append(new_item)
    return result

def load_trade_data(file_path, logger):
    """Load trading data from JSON file, support two formats"""
    try:
        logger.info(f"Read file: {file_path}")
        
        format_type, data = detect_json_format(file_path)
        
        if format_type is None:
            logger.error(f"Cannot recognize file format: {file_path}")
            return None
        
        if format_type == "dict_format":
            data = convert_dict_format(data)
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            
            if 'Close' in df.columns:
                df['price'] = df['Close'].astype(float)
            elif 'close' in df.columns:
                df['price'] = df['close'].astype(float)
            elif 'price' in df.columns:
                df['price'] = df['price'].astype(float)
            else:
                logger.warning(f"Price field not found, use the first numeric column")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['price'] = df[numeric_cols[0]].astype(float)
                else:
                    logger.error(f"No numeric column found")
                    return None
        
        elif format_type == "list_format":
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df['price'] = df['price'].astype(float)
        
        df['date'] = df['time'].dt.date
        df = df.sort_values('time')
        
        logger.info(f"Loaded {len(df)} records, time range from {df['time'].min()} to {df['time'].max()}")
        logger.info(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Read file {file_path}: {str(e)}")
        return None

def calculate_returns(df, sampling_interval='1T', logger=None):
    """Calculate log returns, excluding overnight returns"""
    if logger:
        logger.info(f"Calculate log returns with {sampling_interval} interval")
    
    df = df.sort_values('time').copy()
    
    if 'date' not in df.columns:
        df['date'] = df['time'].dt.date
    
    all_returns = []
    
    for date, day_group in df.groupby('date'):
        day_data = day_group.set_index('time')['price']
        resampled = day_data.resample(sampling_interval).last()
        
        market_open = time(9, 30)
        market_close = time(16, 0)
        mask = (resampled.index.time >= market_open) & (resampled.index.time <= market_close)
        resampled = resampled[mask]
        
        if len(resampled) > 1:
            # Calculate log returns
            day_returns = np.log(resampled / resampled.shift(1)).dropna()
            all_returns.append(day_returns)
    if all_returns:
        returns = pd.concat(all_returns)
        if logger:
            logger.info(f"Calculated {len(returns)} log returns")
        return returns
    else:
        if logger:
            logger.warning("Not enough data to calculate returns")
        return pd.Series()

def calculate_acf(returns, max_lag=30):
    """Calculate the ACF of the returns"""
    acf_values = []
    n = len(returns)
    
    returns_array = returns.values if hasattr(returns, 'values') else np.array(returns)
    
    for lag in range(1, min(max_lag + 1, n)):
        if lag < n:
            r_t = returns_array[:-lag]
            r_t_lag = returns_array[lag:]
            
            if len(r_t) > 0 and len(r_t_lag) > 0:
                corr_matrix = np.corrcoef(r_t, r_t_lag)
                acf = corr_matrix[0, 1]
                acf_values.append(acf)
            else:
                acf_values.append(np.nan)
        else:
            acf_values.append(np.nan)
    
    return acf_values

def calculate_abs_acf(returns, max_lag=20):
    """Calculate the ACF of the absolute returns"""
    abs_returns = np.abs(returns)
    acf_values = []
    n = len(abs_returns)
    
    abs_returns_array = abs_returns.values if hasattr(abs_returns, 'values') else np.array(abs_returns)
    
    for lag in range(1, min(max_lag + 1, n)):
        if lag < n:
            abs_r_t = abs_returns_array[:-lag]
            abs_r_t_lag = abs_returns_array[lag:]
            
            if len(abs_r_t) > 0 and len(abs_r_t_lag) > 0:
                corr_matrix = np.corrcoef(abs_r_t, abs_r_t_lag)
                acf = corr_matrix[0, 1]
                acf_values.append(acf)
            else:
                acf_values.append(np.nan)
        else:
            acf_values.append(np.nan)
    
    return acf_values

def calculate_leverage_corr(returns, max_lag=5):
    """Calculate the correlation coefficient of the leverage effect"""
    leverage_values = []
    n = len(returns)
    
    returns_array = returns.values if hasattr(returns, 'values') else np.array(returns)
    
    for lag in range(1, min(max_lag + 1, n)):
        if lag < n:
            r_t = returns_array[:-lag]
            abs_r_t_lag = np.abs(returns_array[lag:])
            
            if len(r_t) > 0 and len(abs_r_t_lag) > 0:
                corr_matrix = np.corrcoef(r_t, abs_r_t_lag)
                corr = corr_matrix[0, 1]
                leverage_values.append(corr)
            else:
                leverage_values.append(np.nan)
        else:
            leverage_values.append(np.nan)
    
    return leverage_values

def calculate_kurtosis(returns):
    """Calculate the excess kurtosis according to the formula"""
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    
    numerator = np.mean((returns - mean_r) ** 4)
    kurtosis = numerator / (std_r ** 4) - 3
    
    return kurtosis

def test_unit_root(df, logger):
    """6. Unit root test (ADF test)"""
    logger.info("\n==== 6. Unit root test (stationarity test) ====")
    
    try:
        # Use the same sampling method as the returns calculation: sample every minute, consider the opening time
        df = df.sort_values('time').copy()
        
        if 'date' not in df.columns:
            df['date'] = df['time'].dt.date
        
        all_prices = []
        
        # Group by date, process each day separately
        for date, day_group in df.groupby('date'):
            day_data = day_group.set_index('time')['price']
            # Resample every minute
            resampled = day_data.resample('1T').last()
            
            # Only keep the data within the opening time (9:30-16:00)
            market_open = time(9, 30)
            market_close = time(16, 0)
            mask = (resampled.index.time >= market_open) & (resampled.index.time <= market_close)
            resampled = resampled[mask]
            
            if len(resampled) > 0:
                all_prices.extend(resampled.dropna().values)
        
        if len(all_prices) == 0:
            logger.warning("After sampling, there is no valid price data")
            return {
                "Features": "Unit root test",
                "Satisfied": False,
                "ADF_p": np.nan,
                "Conclusion": "Data insufficient"
            }
        
        prices = np.array(all_prices)
        logger.info(f"After sampling, the price sequence length: {len(prices)}")
        
        # If the data is too large, resample
        max_samples = 100
        if len(prices) > max_samples:
            logger.info(f"The sampling data amount {len(prices)} is too large, resample to {max_samples} data points")
            step = len(prices) // max_samples
            prices = prices[::step][:max_samples]
        
        if len(prices) < 100:
            logger.warning(f"Price data is insufficient ({len(prices)} points)")
            return {
                "Features": "Unit root test",
                "Satisfied": False,
                "ADF_p": np.nan,
                "Conclusion": "Data insufficient"
            }
        
        prices = prices[~np.isnan(prices)]
        
        # ADF test
        try:
            maxlag = min(int(12 * (len(prices)/100)**(1/4)), len(prices)//3)
            adf_result = adfuller(prices, maxlag=maxlag, autolag='AIC')
        except:
            logger.info("Automatic lag selection failed, use fixed lag order")
            adf_result = adfuller(prices, maxlag=5, autolag=None)
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        
        logger.info(f"ADF statistic: {adf_statistic:.4f}")
        logger.info(f"p value: {p_value:.4f}")
        
        has_unit_root = p_value > 0.05
        
        logger.info(f"Conclusion: The price sequence {'is not stationary (has a unit root)' if has_unit_root else 'is stationary (does not have a unit root)'}")
        logger.info(f"Unit root test: {'Passed' if has_unit_root else 'Failed'}")
        
        return {
            "Features": "Unit root test",
            "Satisfied": has_unit_root,
            "ADF_p": p_value
        }
        
    except Exception as e:
        logger.error(f"Unit root test failed: {str(e)}")
        return {
            "Features": "Unit root test",
            "Satisfied": False,
            "ADF_p": np.nan
        }

def calculate_acf_confidence_interval(returns, max_lag=20):
    """
    Calculate the confidence interval of the ACF, using the Bartlett formula (considering the impact of previous lag terms)
    
    Args:
        returns: Returns sequence
        max_lag: Maximum lag period
    
    Returns:
        confidence_intervals: Confidence interval boundaries for each lag period
        acf_values: ACF值
    """
    n = len(returns)
    returns_array = returns.values if hasattr(returns, 'values') else np.array(returns)
    
    # Use the original rate of return (non-standardized) because we need to consider the actual variance
    confidence_intervals = []
    acf_values = []
    
    for lag in range(1, min(max_lag + 1, n)):
        if lag >= n:
            confidence_intervals.append(np.nan)
            acf_values.append(np.nan)
            continue
            
        # Calculate ACF
        r_t = returns_array[:-lag]
        r_t_lag = returns_array[lag:]
        
        if len(r_t) > 0 and len(r_t_lag) > 0:
            acf = np.corrcoef(r_t, r_t_lag)[0, 1]
        else:
            acf = np.nan
            
        acf_values.append(acf)
        
        # Use Bartlett formula to calculate the standard error
        if lag == 1:
            # For lag=1, use the basic formula
            se = 1.0 / np.sqrt(n)
        else:
            # For lag>1, consider the impact of previous lag terms
            sum_acf_squared = sum([acf_values[i]**2 for i in range(lag-1) if not np.isnan(acf_values[i])])
            se = np.sqrt((1 + 2 * sum_acf_squared) / n)
            
        ci_bound = 1.96 * se
        confidence_intervals.append(ci_bound)
    
    return confidence_intervals, acf_values

def test_linear_autocorrelation_absence_count(returns, output_dir, logger, real_returns=None):
    """1. Linear autocorrelation absence test (statistic 1-30 periods insignificant number)"""
    logger.info("\n==== 1. Linear autocorrelation absence test (statistic 1-30 periods insignificant number) ====")
    
    max_lag = 30
    # Use the improved confidence interval calculation method (Bartlett formula)
    confidence_intervals, acf_values = calculate_acf_confidence_interval(returns, max_lag)
    
    # To display a unified confidence interval, use the confidence interval of the first lag period as a reference
    ci_95 = confidence_intervals[0] if confidence_intervals else 1.96 / np.sqrt(n)
    print(ci_95)
    #acf_values = calculate_acf(returns, max_lag)
    n = len(returns)
    total_count = 30
    se = 1.0 / np.sqrt(n)
    #ci_95 = 1.96 * se
    
    logger.info(f"Sample size n = {n}")
    logger.info(f"95% confidence interval = ±{ci_95:.4f}")
    
    insignificant_count = sum(1 for acf in acf_values[:30] if abs(acf) <= ci_95)
    significant_count = 30 - insignificant_count
    
    # If real data is provided, calculate its ACF
    real_acf_values = None
    real_ci_95 = None
    real_insignificant_count = None
    if real_returns is not None:
        real_confidence_intervals,real_acf_values = calculate_acf_confidence_interval(real_returns, max_lag)
        real_n = len(real_returns)
        real_se = 1.0 / np.sqrt(real_n)
        real_ci_95 = real_confidence_intervals[0] if real_confidence_intervals else 1.96 / np.sqrt(real_n)
        print(real_ci_95)
        real_insignificant_count = sum(1 for acf in real_acf_values[:30] if abs(acf) <= real_ci_95)
        logger.info(f"Real data sample size n = {real_n}")
        logger.info(f"Real data insignificant ACF number: {real_insignificant_count}/30")
    
    # Draw ACF graph
    fig, ax = plt.subplots(figsize=(6, 5))
    lags = range(1, len(acf_values) + 1)
    
    # Draw the confidence interval of the simulated data - use a dashed line
    ax.axhline(y=ci_95, color='#336699', linestyle='--', linewidth=1.5, 
            alpha=0.8, label='95% CI (Simulated)', zorder=2)
    ax.axhline(y=-ci_95, color='#336699', linestyle='--', linewidth=1.5, 
            alpha=0.8, zorder=2)

    # Draw the ACF of the simulated data
    ax.plot(lags, acf_values, color='#336699', marker='o', markersize=5,
            linestyle='-', linewidth=1.5, label='ACF (Simulated)', zorder=3)

    # If real data is provided, add its ACF and confidence interval
    if real_acf_values is not None:
        # Draw the confidence interval of the real data - use a dashed line
        ax.axhline(y=real_ci_95, color='#CC6677', linestyle=':', linewidth=1.5,
                alpha=0.8, label='95% CI (Real)', zorder=2)
        ax.axhline(y=-real_ci_95, color='#CC6677', linestyle=':', linewidth=1.5,
                alpha=0.8, zorder=2)
        
        # Draw the ACF of the real data
        ax.plot(lags, real_acf_values, color='#CC6677', marker='s', markersize=5,
                linestyle='-', linewidth=1.5, label='ACF (Real)', zorder=4)

    # Draw the center line
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=1)
    # Set the title and labels
    #ax.set_title('Autocorrelation Function (ACF) Comparison', pad=15)
    ax.set_xlabel('Lag', fontsize=17, fontweight='bold')
    ax.set_ylabel('Autocorrelation (ACF)', fontsize=17, fontweight='bold')
    
    # Set the coordinate axis range and ticks
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', which='major', direction='in')
    
    # Set the border color
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Add legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, 
                      edgecolor='black', fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'linear_autocorrelation.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Insignificant ACF number: {insignificant_count}/30")
    
    has_feature = insignificant_count > 20
    
    logger.info(f"Judgment standard: Insignificant ACF number > 20")
    logger.info(f"Conclusion: {'Satisfied' if has_feature else 'Not satisfied'}Linear autocorrelation absence feature")
    
    return {
        "Features": "Linear autocorrelation absence",
        "Satisfied": has_feature,
        "ACF insignificant number": insignificant_count
    }

def test_leptokurtosis(returns, output_dir, logger,real_returns=None):
    """2. Test leptokurtosis feature"""
    logger.info("\n==== 2. Leptokurtosis feature test ====")
    
    kurtosis = calculate_kurtosis(returns)
    has_feature = kurtosis > 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left graph: probability density distribution
    returns = (returns - np.mean(returns)) / np.std(returns)
    if real_returns is not None:
        real_returns = (real_returns - np.mean(real_returns)) / np.std(real_returns)
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 100)
    # Calculate data range
    data_min = min(returns.min(), real_returns.min() if real_returns is not None else returns.min())
    data_max = max(returns.max(), real_returns.max() if real_returns is not None else returns.max())

    # Use np.linspace to create bin boundaries
    n, bins, patches = ax1.hist(returns, bins=np.linspace(data_min, data_max, 51), density=True, 
                            color='#336699', label='Simulated', zorder=2,alpha=0.8)

    if real_returns is not None:
        # Use the same bin boundaries
        real_n, real_bins, real_patches = ax1.hist(real_returns, bins=np.linspace(data_min, data_max, 51), density=True,
                                color='#CC6677', label='Real', zorder=2,alpha=0.8)
        real_mean_r = np.mean(real_returns)
        real_std_r = np.std(real_returns)
        real_x = np.linspace(real_returns.min(), real_returns.max(), 100)

        ax1.plot(real_x,stats.norm.pdf(real_x,real_mean_r,real_std_r),
            color='#CC6677',linestyle='--',linewidth=2,label='Normal',zorder=3)                        
    ax1.set_xlabel('Log(Returns)')
    ax1.set_ylabel('Density')
    #ax1.set_title(f'Return Distribution (K={kurtosis:.2f})')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')

    # Add all 4 borders to the left graph
    ax1.spines['left'].set_color('grey')
    ax1.spines['bottom'].set_color('grey')
    ax1.spines['top'].set_color('grey')
    ax1.spines['right'].set_color('grey')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    # Add all 4 borders to the right graph
    ax2.spines['left'].set_color('grey')
    ax2.spines['bottom'].set_color('grey')
    ax2.spines['top'].set_color('grey')
    ax2.spines['right'].set_color('grey')
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    # Draw the Q-Q plot of the simulated data
    returns_std = (returns - np.mean(returns)) / np.std(returns)
    stats.probplot(returns_std, dist="norm", plot=ax2)

    # Set the Q-Q plot style of the simulated data
    ax2.get_lines()[0].set_markerfacecolor('#336699') 
    ax2.get_lines()[0].set_markeredgecolor('#336699')
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[0].set_alpha(0.8)
    ax2.get_lines()[0].set_marker('o')
    ax2.get_lines()[1].set_alpha(0)


    if real_returns is not None:
        # Draw the Q-Q plot of the real data
        real_returns_std = (real_returns - np.mean(real_returns)) / np.std(real_returns)
        stats.probplot(real_returns_std, dist="norm", plot=ax2)
        
        # Set the Q-Q plot style of the real data
        ax2.get_lines()[2].set_markerfacecolor('#CC6677')  #
        ax2.get_lines()[2].set_markeredgecolor('#CC6677')
        ax2.get_lines()[2].set_markersize(4)
        ax2.get_lines()[2].set_alpha(0.8)
        ax2.get_lines()[2].set_marker('s')
        ax2.get_lines()[3].set_alpha(0)

    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, 
            alpha=0.8, label='y=x', zorder=0)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#336699', label='Simulated', 
            markersize=6, linestyle='None',alpha=0.8),
        Line2D([0], [0], marker='s', color='#CC6677', label='Real', 
            markersize=6, linestyle='None',alpha=0.8),
        Line2D([0], [0], color='black', linestyle='--', label='y=x', linewidth=1)
    ]
    ax2.legend(handles=legend_elements, loc='upper left', frameon=True, 
            fancybox=False, edgecolor='black', fontsize=9)
    ax2.set_title('')  
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Data Quantiles')
    
    # Adjust layout
    plt.tight_layout()
    
    #fig_path = os.path.join(output_dir, 'leptokurtosis.png')
    fig_path = os.path.join(output_dir, 'leptokurtosis.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Excess kurtosis K: {kurtosis:.4f}")
    logger.info(f"Judgment standard: K > 0")
    logger.info(f"Conclusion: {'Satisfied' if has_feature else 'Not satisfied'}Leptokurtosis feature")
    
    return {
        "Features": "Leptokurtosis",
        "Satisfied": has_feature,
        "Kurtosis K": kurtosis
    }

def test_aggregational_gaussianity(df, output_dir, logger,real_df=None):
    """3. Test aggregational gaussianity"""
    logger.info("\n==== 3. Aggregational gaussianity test ====")
    
    time_scales = ['1T', '5T', '15T', '30T', '60T']
    kurtosis_values = []
    scale_minutes = []
    
    for scale in time_scales:
        returns = calculate_returns(df, scale)
        if len(returns) > 30:
            kurt = calculate_kurtosis(returns)
            kurtosis_values.append(kurt)
            scale_minutes.append(int(scale[:-1]))
            logger.info(f"  {scale}: Kurtosis={kurt:.4f}")
    
    if len(kurtosis_values) < 2:
        logger.warning("Data insufficient")
        return {"Features": "Aggregational gaussianity", "Satisfied": False}
    real_kurtosis_values = []
    real_scale_minutes = []
    if real_df is not None:
        for scale in time_scales:
            real_returns = calculate_returns(real_df, scale)
            if len(real_returns) > 30:
                real_kurt = calculate_kurtosis(real_returns)
                real_kurtosis_values.append(real_kurt)
                real_scale_minutes.append(int(scale[:-1]))
                logger.info(f"  {scale}: Kurtosis={real_kurt:.4f}")
    
    if len(kurtosis_values) < 2:
        logger.warning("Data insufficient")
        return {"Features": "Aggregational gaussianity", "Satisfied": False}     
    
    # Modify this part of the code
    fig, ax = plt.subplots(figsize=(7, 5))  # Create fig and ax objects

    ax.plot(scale_minutes, kurtosis_values, marker='o', label='Simulated')
    if real_df is not None:
        ax.plot(real_scale_minutes, real_kurtosis_values, marker='s', label='Real')

    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    ax.set_xlabel('Time Scale (minutes)')
    ax.set_ylabel('Excess Kurtosis K(Δt)')
    ax.set_title('Aggregational Gaussianity')
    ax.grid(True, alpha=0.3)

    # Set the border
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()

    
    fig_path = os.path.join(output_dir, 'aggregational_gaussianity.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    slope, _ = np.polyfit(scale_minutes, kurtosis_values, 1)
    has_feature = slope < 0
    
    logger.info(f"Kurtosis trend slope: {slope:.6f}")
    logger.info(f"Conclusion: {'Satisfied' if has_feature else 'Not satisfied'}Aggregational gaussianity")
    
    return {
        "Features": "Aggregational gaussianity",
        "Satisfied": has_feature
    }

def test_volatility_clustering(returns, output_dir, logger, max_lag=10, real_returns=None, first_date=None):
    """Test volatility clustering"""
    logger.info("Test volatility clustering...")
    
    # Calculate rolling volatility (using 30-minute window)
    window_size = 30  # 30-minute window
    rolling_vol_sim = returns.rolling(window=window_size).std()
    
    # If there is real data, calculate its rolling volatility
    rolling_vol_real = None
    if real_returns is not None:
        rolling_vol_real = real_returns.rolling(window=window_size).std()
    
    # Draw rolling volatility graph
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def create_continuous_time_index(time_series, values):
        """Create a continuous time index, connecting the trading time of each day"""
        dates = time_series.date
        unique_dates = sorted(set(dates))
        
        continuous_times = []
        continuous_values = []
        minutes_elapsed = 0
        trading_minutes = 390  # 6.5 hours = 390 minutes
        
        for date in unique_dates:
            # Get the data of the day
            mask = dates == date
            day_times = time_series[mask]
            day_values = values[mask]
            
            # Create a continuous time index
            day_minutes = np.arange(len(day_values)) + minutes_elapsed
            continuous_times.extend(day_minutes)
            continuous_values.extend(day_values)
            
            # Update the cumulative minutes
            minutes_elapsed += trading_minutes
        
        return np.array(continuous_times), np.array(continuous_values)
    
    # Filter out NaN values, only keep valid rolling volatility data
    valid_mask = ~rolling_vol_sim.isna()
    time_index = returns.index[valid_mask]
    rolling_vol_valid = rolling_vol_sim[valid_mask]
    
    # Create a continuous time index and draw the simulated data
    cont_times_sim, cont_values_sim = create_continuous_time_index(time_index, rolling_vol_valid)
    ax.plot(cont_times_sim, cont_values_sim, color='#336699',
            linewidth=1, label='Simulated Rolling Volatility')
    
    # If there is real data, add comparison
    if rolling_vol_real is not None:
        real_valid_mask = ~rolling_vol_real.isna()
        real_time_index = real_returns.index[real_valid_mask]
        rolling_vol_real_valid = rolling_vol_real[real_valid_mask]
        
        # Create a continuous time index and draw the real data
        cont_times_real, cont_values_real = create_continuous_time_index(real_time_index, rolling_vol_real_valid)
        ax.plot(cont_times_real, cont_values_real, color='#CC6677', 
                linewidth=1, label='Real Rolling Volatility')
    
    # Set the chart properties
    ax.set_xlabel('Trading Minutes')
    ax.set_ylabel('Rolling Volatility (30-min window)')
    ax.set_title('Rolling Volatility Time Series')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    
    # Set the X-axis ticks
    trading_minutes = 390  # 6.5 hours = 390 minutes
    dates = sorted(set(time_index.date))  # Get all trading dates
    days = len(dates)
    
    # Set the X-axis ticks
    trading_minutes = 390  # 6.5 hours = 390 minutes
    dates = sorted(set(time_index.date))  # Get all trading dates
    days = len(dates)
    
    # Create tick positions and labels
    tick_positions = []
    tick_labels = []
    for i, date in enumerate(dates):
        if i == 0:
            # The first day displays opening, noon, and closing
            tick_positions.extend([
                i * trading_minutes,  # Opening
                i * trading_minutes + trading_minutes // 2,  # Noon
            ])
            date_str = date.strftime('%m-%d')
            tick_labels.extend([
                f'{date_str}\n09:30',
                f'{date_str}\n13:00',
            ])
        elif i == days - 1:
            # The last day displays opening, noon, and closing
            tick_positions.extend([
                i * trading_minutes,  # Opening
                i * trading_minutes + trading_minutes // 2,  # Noon
                (i + 1) * trading_minutes - 1  # Closing
            ])
            date_str = date.strftime('%m-%d')
            tick_labels.extend([
                f'{date_str}\n09:30',
                f'{date_str}\n13:00',
                f'{date_str}\n16:00'
            ])
        else:
            # The middle days only display opening and noon, because the closing will coincide with the opening of the next day
            tick_positions.extend([
                i * trading_minutes,  # Opening
                i * trading_minutes + trading_minutes // 2,  # Noon
            ])
            date_str = date.strftime('%m-%d')
            tick_labels.extend([
                f'{date_str}\n09:30',
                f'{date_str}\n13:00',
            ])
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    plt.xticks(rotation=0)
    
    # Add borders
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'rolling_volatility.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rolling volatility graph saved to: {fig_path}")
    
    # ... The rest of the code remains the same (GARCH model and long memory analysis)
    
    # 2. Long memory drawing (only drawing, no judgment)
    print(f"\nDrawing long memory analysis graph...")
    max_lag_long = min(100, len(returns)//2)
    abs_acf_long = calculate_abs_acf(returns, max_lag=max_lag_long)

    # Calculate the long memory ACF of the real data
    real_abs_acf_long = None
    if real_returns is not None:
        real_max_lag_long = min(100, len(real_returns)//2)
        real_abs_acf_long = calculate_abs_acf(real_returns, max_lag=real_max_lag_long)
        print(f"Long memory ACF of real data calculation completed, lag period: {len(real_abs_acf_long)}")

    if len(abs_acf_long) >= 20:
        # Draw the long memory image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left graph: absolute return autocorrelation function
        lags = range(1, len(abs_acf_long) + 1)
        ax1.plot(lags, abs_acf_long, color='#336699', linestyle='-', alpha=0.8, 
                linewidth=1.5, label='Simulated')
        
        # If there is real data, add comparison
        if real_abs_acf_long is not None:
            real_lags = range(1, len(real_abs_acf_long) + 1)
            ax1.plot(real_lags, real_abs_acf_long, color='#CC6677', linestyle='--', 
                    alpha=0.8, linewidth=1.5, label='Real')
        
        ax1.set_xlabel('Lag period (Lag)')
        ax1.set_ylabel('Absolute rate of return')
        ax1.set_title('Absolute rate of return autocorrelation function (long-term)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
        
        # Add borders to the left graph
        ax1.spines['left'].set_color('grey')
        ax1.spines['bottom'].set_color('grey')
        ax1.spines['top'].set_color('grey')
        ax1.spines['right'].set_color('grey')
        ax1.spines['top'].set_visible(True)
        ax1.spines['right'].set_visible(True)
        
        # Right figure: ACF attenuation on the logarithmic scale
        positive_acf = [(i+1, v) for i, v in enumerate(abs_acf_long) if v > 0]
        if positive_acf:
            lags_pos, acf_pos = zip(*positive_acf)
            ax2.loglog(lags_pos, acf_pos, color='#336699', linestyle='-', 
                    alpha=0.8, linewidth=1.5, label='Simulated')
            
            # If there is real data, add comparison
            if real_abs_acf_long is not None:
                real_positive_acf = [(i+1, v) for i, v in enumerate(real_abs_acf_long) if v > 0]
                if real_positive_acf:
                    real_lags_pos, real_acf_pos = zip(*real_positive_acf)
                    ax2.loglog(real_lags_pos, real_acf_pos, color='#CC6677', 
                            linestyle='--', alpha=0.8, linewidth=1.5, label='Real')
            
            ax2.set_xlabel('Log scale')
            ax2.set_ylabel('Absolute rate of return ACF (Log scale)')
            ax2.set_title('ACF decay under log scale')
            ax2.grid(True, alpha=0.3)
            ax2.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
        
        # Add borders to the right graph
        ax2.spines['left'].set_color('grey')
        ax2.spines['bottom'].set_color('grey')
        ax2.spines['top'].set_color('grey')
        ax2.spines['right'].set_color('grey')
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        
        plt.tight_layout()
        
        # Save the image
        fig_path = os.path.join(output_dir, 'long_memory.png')
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Long memory analysis graph saved to: {fig_path}")
    
    # 3. The original volatility clustering test logic
    n = len(returns)
    se = 1.0 / np.sqrt(n)
    ci_95 = 1.96 * se
    
    abs_acf = calculate_abs_acf(returns, max_lag=20)
    acf = calculate_acf(returns, max_lag=20)
    
    acf_condition = False
    lag1_abs_acf = np.nan
    
    if len(abs_acf) > 0 and len(acf) > 0:
        lag1_abs_acf = abs_acf[0]
        lag1_acf = acf[0]
        
        is_significant = abs(lag1_abs_acf) > ci_95
        
        logger.info(f"Absolute return ACF(1): {lag1_abs_acf:.4f}")
        logger.info(f"Return ACF(1): {lag1_acf:.4f}")
        
        condition1 = lag1_abs_acf > abs(lag1_acf)
        condition2 = is_significant and lag1_abs_acf > 0
        acf_condition = condition1 and condition2
    
    garch_condition = False
    alpha = beta = persistence = np.nan
    
    try:
        model = arch_model(returns, vol='GARCH', p=1, q=1, mean='constant')
        res = model.fit(disp='off', show_warning=False)
        
        alpha = res.params['alpha[1]']
        beta = res.params['beta[1]']
        alpha_pval = res.pvalues['alpha[1]']
        beta_pval = res.pvalues['beta[1]']
        persistence = alpha + beta
        
        logger.info(f"GARCH α: {alpha:.4f} (p={alpha_pval:.4f})")
        logger.info(f"GARCH β: {beta:.4f} (p={beta_pval:.4f})")
        logger.info(f"Persistence α+β: {persistence:.4f}")
        
        garch_condition = (alpha_pval < 0.05 and alpha > 0 and 
                          beta_pval < 0.05 and beta > 0 and 
                          0.7 <= persistence <= 1)
        
    except Exception as e:
        logger.error(f"GARCH model failed: {e}")
    
    has_clustering = garch_condition
    
    logger.info(f"Conclusion: {'exists' if has_clustering else 'does not exist'}volatility clustering")
    
    return {
        "Features": "Volatility clustering",
        "Satisfied": has_clustering,
        "GARCH_αβ": persistence if not np.isnan(persistence) else np.nan,
        "Absolute ACF(1)": lag1_abs_acf
    }

def test_leverage_effect(returns, output_dir, logger):
    """5. Test leverage effect"""
    logger.info("\n==== 5. Test leverage effect ====")
    
    leverage_corr = calculate_leverage_corr(returns, max_lag=5)
    
    if len(leverage_corr) < 1 or np.isnan(leverage_corr[0]):
        logger.warning("Data insufficient")
        return {"Features": "Leverage effect", "Satisfied": False, "L(1)": np.nan}
    
    lag1_leverage = leverage_corr[0]
    has_feature = lag1_leverage < 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    lags = range(1, len(leverage_corr) + 1)
    colors = ['#D55E00' if v < 0 else '#0173B2' for v in leverage_corr]
    
    # Draw the bar chart
    bars = ax.bar(lags, leverage_corr, alpha=0.8, color=colors, 
                  width=0.7, zorder=3)
    
    # Add the zero line
    ax.axhline(y=0, color='#666666', linestyle='-', 
               linewidth=0.5, zorder=2)
    
    # Set the coordinate axis
    ax.set_xlabel('Lag (τ)')
    ax.set_ylabel('Leverage Correlation L(τ)')
    ax.set_title('Leverage Effect')
    
    # Add the grid
    ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
    
    # Add the numerical label
    for i, v in enumerate(leverage_corr):
        if abs(v) > 0.05:  # Only display significant values
            ax.text(i+1, v + (0.02 if v >= 0 else -0.02),
                   f'{v:.2f}',
                   ha='center', va='bottom' if v >= 0 else 'top',
                   fontsize=8)
    
    # Set the y-axis range, ensure symmetry
    y_max = max(abs(min(leverage_corr)), abs(max(leverage_corr)))
    ax.set_ylim(-y_max*1.2, y_max*1.2)
    
    # Adjust the layout
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'leverage_effect.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"L(1): {lag1_leverage:.4f}")
    logger.info(f"Conclusion: {'Satisfied' if has_feature else 'does not satisfy'}leverage effect")
    
    return {
        "Features": "Leverage effect",
        "Satisfied": has_feature,
        "L(1)": lag1_leverage
    }

def analyze_all_features(df, output_dir, logger, real_df=None,first_date=None):
    """Analyze all features"""
    returns = calculate_returns(df, '1T', logger)
    
    # If real data is provided, calculate its returns
    real_returns = None
    if real_df is not None:
        real_returns = calculate_returns(real_df, '1T', logger)
        logger.info(f"Loading real data returns: {len(real_returns)} data points")
    
    if len(returns) < 100:
        logger.warning("Data insufficient")
        return None
    
    results = []
    
    # 1. Linear autocorrelation absence
    # result1 = test_linear_autocorrelation_absence_count(returns, output_dir, logger, real_returns)
    # results.append(result1)
    
    # # 2. Leptokurtosis
    # result2 = test_leptokurtosis(returns, output_dir, logger,real_returns)
    # results.append(result2)
    test_combined_features(returns, output_dir, logger, real_returns,real_df)
    # 3. Aggregational Gaussianity
    result3 = test_aggregational_gaussianity(df, output_dir, logger,real_df)
    results.append(result3)
    
    # 4. Volatility clustering
    result4 = test_volatility_clustering(returns, output_dir, logger,real_returns = real_returns,first_date=first_date)
    results.append(result4)
    
    # 5. Leverage effect
    result5 = test_leverage_effect(returns, output_dir, logger)
    results.append(result5)
    
    # 6. Unit root test
    result6 = test_unit_root(df, logger)
    results.append(result6)
    
    return results

def process_single_file(file_path, base_output_dir, real_df=None,first_date=None):
    """Process a single file, support comparison with real data"""
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(base_output_dir, file_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = setup_logger(output_dir, file_name)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing file: {file_name}")
    logger.info(f"{'='*60}")
    
    df = load_trade_data(file_path, logger)
    
    if df is None or len(df) == 0:
        logger.error(f"Cannot load data: {file_path}")
        return None
    
    results = analyze_all_features(df, output_dir, logger, real_df,first_date)
    
    if results:
        # Summarize key parameters
        key_params = {
            "File": file_name,
            "ACF insignificant count": "",
            "Kurtosis K": "",
            "GARCH_αβ": "",
            "Absolute ACF(1)": "",
            "L(1)": "",
            "ADF_p": "",
            "Satisfied features": 0
        }
        
        satisfied_count = 0
        for result in results:
            if result["Satisfied"]:
                satisfied_count += 1
            
            # 提取关键参数
            if result["Features"] == "Linear autocorrelation absence":
                key_params["ACF insignificant count"] = f"{result.get('ACF insignificant count', '')}/30"
            elif result["Features"] == "Leptokurtosis":
                key_params["Kurtosis K"] = f"{result.get('Kurtosis K', 0):.3f}"
            elif result["Features"] == "Volatility clustering":
                key_params["GARCH_αβ"] = f"{result.get('GARCH_αβ', 0):.3f}" if not np.isnan(result.get('GARCH_αβ', np.nan)) else "N/A"
                key_params["Absolute ACF(1)"] = f"{result.get('Absolute ACF(1)', 0):.3f}" if not np.isnan(result.get('Absolute ACF(1)', np.nan)) else "N/A"
            elif result["Features"] == "Leverage effect":
                key_params["L(1)"] = f"{result.get('L(1)', 0):.3f}" if not np.isnan(result.get('L(1)', np.nan)) else "N/A"
            elif result["Features"] == "Unit root test":
                key_params["ADF_p"] = f"{result.get('ADF_p', 0):.3f}" if not np.isnan(result.get('ADF_p', np.nan)) else "N/A"
        
        key_params["Satisfied features"] = f"{satisfied_count}/6"
        
        # Save summary results
        summary = []
        for result in results:
            summary.append({
                "Features": result["Features"],
                "Satisfied": "Yes" if result["Satisfied"] else "No"
            })
        
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(output_dir, "summary.xlsx")
        summary_df.to_excel(summary_file, index=False, engine='openpyxl')
        
        logger.info(f"\nResults saved to: {output_dir}")
        
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        
        return {
            "File": file_name,
            "Results": summary,
            "Key parameters": key_params
        }
    
    return None

def test_combined_features(returns, output_dir, logger, real_returns=None, real_df=None):
    """Merge and plot the linear autocorrelation and leptokurtosis test"""
    logger.info("\n==== 合并测试：线性自相关和尖峰厚尾特性 ====")
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ==================== 1. Linear autocorrelation absence test ====================
    max_lag = 30
    # Calculate ACF and confidence interval
    from statsmodels.tsa.stattools import acf
    n = len(returns)
    se = 1.0 / np.sqrt(n)
    ci_95 = 1.96 * se
    
    acf_values = []
    for lag in range(1, min(max_lag + 1, n)):
        r_t = returns[:-lag]
        r_t_lag = returns[lag:]
        if len(r_t) > 0 and len(r_t_lag) > 0:
            corr_matrix = np.corrcoef(r_t, r_t_lag)
            acf_values.append(corr_matrix[0, 1])
    
    # Calculate real data ACF
    real_acf_values = None
    real_ci_95 = ci_95
    real_n = n
    if real_returns is not None:
        real_n = len(real_returns)
        real_se = 1.0 / np.sqrt(real_n)
        real_ci_95 = 1.96 * real_se
        real_acf_values = []
        for lag in range(1, min(max_lag + 1, real_n)):
            real_r_t = real_returns[:-lag]
            real_r_t_lag = real_returns[lag:]
            if len(real_r_t) > 0 and len(real_r_t_lag) > 0:
                real_corr_matrix = np.corrcoef(real_r_t, real_r_t_lag)
                real_acf_values.append(real_corr_matrix[0, 1])
    
    # Draw ACF graph
    lags = range(1, len(acf_values) + 1)
    if real_acf_values is not None:
        real_lags = range(1, len(real_acf_values) + 1)
        # Real data confidence interval
        ax1.axhline(y=real_ci_95, color='#CC6677', linestyle=':', linewidth=1.5,
                     label='95% CI (Real)', zorder=1)
        ax1.axhline(y=-real_ci_95, color='#CC6677', linestyle=':', linewidth=1.5,
                     zorder=1)
        # Real data ACF
        ax1.plot(real_lags, real_acf_values, color='#CC6677', marker='s', markersize=5,
                 linestyle='-', linewidth=1.5, label='ACF (Real)', zorder=2)
    # Simulated data confidence interval
    ax1.axhline(y=ci_95, color='#336699', linestyle='--', linewidth=1.5, 
                 label='95% CI (Simulated)', zorder=3)
    ax1.axhline(y=-ci_95, color='#336699', linestyle='--', linewidth=1.5, 
                 zorder=3)
    # Simulated data ACF
    ax1.plot(lags, acf_values, color='#336699', marker='o', markersize=5,
             linestyle='-', linewidth=1.5, label='ACF (Simulated)', zorder=4)
    
    # Draw center line
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=1)
    
    # Set title and labels
    ax1.set_xlabel('Lag', fontsize=23, fontweight='bold')
    ax1.set_ylabel('Autocorrelation (ACF)', fontsize=23, fontweight='bold')
    ax1.set_xlim(0, 31)
    ax1.set_ylim(-0.5, 0.5)
    ax1.tick_params(axis='both', which='major', direction='in')
    
    # Set borders
    for spine in ax1.spines.values():
        spine.set_color('grey')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    
    # Add legend
    ax1.legend(loc='upper right',  fancybox=False, frameon=True,
               edgecolor='black', fontsize=14)
    
    # Add label a) in the top left corner outside the borders
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=23, 
             fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    # ==================== 2. Leptokurtosis test ====================
    # Calculate kurtosis
    kurtosis = calculate_kurtosis(returns)
    has_feature = kurtosis > 0
    
    # Left plot: probability density distribution
    returns = (returns - np.mean(returns)) / np.std(returns)
    if real_returns is not None:
        real_returns = (real_returns - np.mean(real_returns)) / np.std(real_returns)
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 100)
    # Calculate data range
    data_min = min(returns.min(), real_returns.min() if real_returns is not None else returns.min())
    data_max = max(returns.max(), real_returns.max() if real_returns is not None else returns.max())

    if real_returns is not None:
        # Use the same bin boundaries
        real_n, real_bins, real_patches = ax2.hist(real_returns, bins=np.linspace(data_min, data_max, 51), density=True,
                                color='#CC6677', label='Real', zorder=1,alpha=0.7)
        real_mean_r = np.mean(real_returns)
        real_std_r = np.std(real_returns)
        real_x = np.linspace(real_returns.min(), real_returns.max(), 100)
    # ax1.plot(x, stats.norm.pdf(x, mean_r, std_r), 
    #         color='#D55E00', linestyle='--', linewidth=2, 
    #         label='Normal', zorder=3)    
        ax2.plot(real_x,stats.norm.pdf(real_x,real_mean_r,real_std_r),
            color='#CC6677',linestyle='--',linewidth=2,label='Normal',zorder=3)     
    # Use np.linspace to create bin boundaries
    n, bins, patches = ax2.hist(returns, bins=np.linspace(data_min, data_max, 51), density=True, 
                            color='#336699', label='Simulated', zorder=2,alpha=0.7)
    ax2.set_xlabel('Log(Returns)', fontsize=23, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=23, fontweight='bold')
    #ax1.set_title(f'Return Distribution (K={kurtosis:.2f})')
    # Manually create legend, specify order
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    
    legend_elements = [
        Rectangle((0, 0), 1, 0.5, facecolor='#CC6677', alpha=0.7, label='Real'),
        Rectangle((0, 0), 1, 0.5, facecolor='#336699', alpha=0.7, label='Simulated'),
        Line2D([0], [0], color='#CC6677', linestyle='--', linewidth=2, label='Normal')
    ]
    ax2.legend(handles=legend_elements, fancybox=False, frameon=True, edgecolor='black', fontsize=14)
    ax2.text(-0.14, 1.05, '(b)', transform=ax2.transAxes, fontsize=23, 
            fontweight='bold', verticalalignment='top', horizontalalignment='left')

    # Add all 4 borders to the left plot
    ax2.spines['left'].set_color('grey')
    ax2.spines['bottom'].set_color('grey')
    ax2.spines['top'].set_color('grey')
    ax2.spines['right'].set_color('grey')
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    # Add all 4 borders to the right plot
    ax3.spines['left'].set_color('grey')
    ax3.spines['bottom'].set_color('grey')
    ax3.spines['top'].set_color('grey')
    ax3.spines['right'].set_color('grey')
    ax3.spines['top'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    
    # Right plot: Q-Q plot

    
    if real_returns is not None:
        real_returns_std = (real_returns - np.mean(real_returns)) / np.std(real_returns)
        stats.probplot(real_returns_std, dist="norm", plot=ax3)
        # Set real data Q-Q plot style
        ax3.get_lines()[0].set_markerfacecolor('#CC6677')
        ax3.get_lines()[0].set_markeredgecolor('#CC6677')
        ax3.get_lines()[0].set_markersize(4)
        ax3.get_lines()[0].set_marker('s')
        ax3.get_lines()[1].set_alpha(0)
    returns_std = (returns - np.mean(returns)) / np.std(returns)
    stats.probplot(returns_std, dist="norm", plot=ax3)
    # Set simulated data Q-Q plot style
    ax3.get_lines()[2].set_markerfacecolor('#336699')
    ax3.get_lines()[2].set_markeredgecolor('#336699')
    ax3.get_lines()[2].set_markersize(4)
    ax3.get_lines()[2].set_marker('o')
    ax3.get_lines()[3].set_alpha(0)
    # Add y=x baseline
    x_min, x_max = ax3.get_xlim()
    y_min, y_max = ax3.get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, 
             label='y=x', zorder=0)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='#CC6677', label='Real', 
               markersize=6, linestyle='None'),
        Line2D([0], [0], marker='o', color='#336699', label='Simulated', 
               markersize=6, linestyle='None'),
        Line2D([0], [0], color='black', linestyle='--', label='y=x', linewidth=1)
    ]
    ax3.legend(handles=legend_elements, loc='upper left', frameon=True, 
               fancybox=False, edgecolor='black', fontsize=14)
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=23, 
            fontweight='bold', verticalalignment='top', horizontalalignment='left')
    ax3.set_title('')  # Set to empty title
    ax3.set_xlabel('Theoretical Quantiles', fontsize=23, fontweight='bold')
    ax3.set_ylabel('Data Quantiles', fontsize=23, fontweight='bold')
    
    
    # Set borders
    for spine in ax3.spines.values():
        spine.set_color('grey')
    ax3.spines['top'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Save picture
    fig_path = os.path.join(output_dir, 'combined_features.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined features graph saved to: {fig_path}")
    
    return fig_path

def main():
    """Main function"""
    
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
    
    main_logger = setup_logger(BASE_OUTPUT_DIR, "main")
    
    # Load real data
    real_df = None
    if os.path.exists(REAL_DATA_FILE):
        print(f"Load real data: {REAL_DATA_FILE}")
        main_logger.info(f"Load real data: {REAL_DATA_FILE}")
        real_df = load_trade_data(REAL_DATA_FILE, main_logger)
        if real_df is not None:
            # Get the date of the first data
            first_date = real_df['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        else:
            main_logger.error("Cannot load real data, will continue to process but not compare")
    
    if SINGLE_FILE and os.path.exists(SINGLE_FILE):
        print(f"Single file processing mode: {SINGLE_FILE}")
        main_logger.info("Single file processing mode")
        result = process_single_file(SINGLE_FILE, BASE_OUTPUT_DIR, real_df,first_date)
        if result:
            main_logger.info("\nProcessing completed!")
    else:
        print(f"Batch processing mode - scan directory: {INPUT_DIR}")
        main_logger.info(f"Batch processing mode - scan directory: {INPUT_DIR}")
        
        json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
        
        if not json_files:
            main_logger.error(f"No JSON files found in {INPUT_DIR}")
            return
        
        main_logger.info(f"Found {len(json_files)} JSON files")
        
        all_results = []
        
        for file_path in json_files:
            result = process_single_file(file_path, BASE_OUTPUT_DIR,real_df,first_date)
            if result:
                all_results.append(result)
        
        if all_results:
            main_logger.info("\n" + "="*60)
            main_logger.info("Batch processing summary")
            main_logger.info("="*60)
            
            # Create key parameter summary table
            key_params_list = []
            for file_result in all_results:
                key_params_list.append(file_result["Key parameters"])
            
            # Create DataFrame
            summary_df = pd.DataFrame(key_params_list)
            
            # Add the satisfaction of each feature
            for file_result in all_results:
                row_idx = summary_df[summary_df["File"] == file_result["File"]].index[0]
                for item in file_result["Results"]:
                    summary_df.loc[row_idx, item["Features"]] = item["Satisfied"]
            
            # Adjust the column order
            key_cols = ["File", "Satisfied features", "ACF insignificant count", "Kurtosis K", "GARCH_αβ", 
                       "绝对ACF(1)", "L(1)", "ADF_p"]
            other_cols = [col for col in summary_df.columns if col not in key_cols]
            summary_df = summary_df[key_cols + other_cols]
            
            summary_file = os.path.join(BASE_OUTPUT_DIR, "batch_summary.xlsx")
            summary_df.to_excel(summary_file, index=False, engine='openpyxl')
      
            main_logger.info(f"Processed {len(all_results)} files")
            main_logger.info(f"Batch summary results saved to: {summary_file}")
            
            # Print statistics
            main_logger.info("\nFeature satisfaction statistics:")
            for col in other_cols:
                if col in summary_df.columns:
                    satisfied = (summary_df[col] == "Yes").sum()
                    main_logger.info(f"  {col}: {satisfied}/{len(all_results)} files satisfied")
    
    main_logger.info("\nAll processing completed!")

if __name__ == "__main__":
    main()
