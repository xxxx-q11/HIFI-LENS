import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.TradeAgent import TradeAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from model.LatencyModel import LatencyModel
from agent.examples.ExampleLLMagent import ExampleLLAgent
from util.oracle.LLMChangedOracle import LLMChangedOracle
from util.invest_company import invest_company_dict
import ray
import random

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for RMSC03 config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='16:00:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
                    help='Volatility of fundamental time series.'
                    )
parser.add_argument('--enable-llm-cache',
                    action='store_true',
                    help='Enable LLM result caching')
parser.add_argument('--llm-cache-file',
                    default='llm_cache.json',
                    help='LLM cache file path')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

if not ray.is_initialized():
    ray.init(include_dashboard=False)
# Initialize the cache at the beginning of the main program
if args.enable_llm_cache:
    from util.simple_llm_cache import initialize_cache
    initialize_cache(cache_file=args.llm_cache_file, enable_cache=True)
    print(f"LLM cache has been enabled: {args.llm_cache_file}")
    
log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = None
book_freq = 0

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

open_price_all = 22319
r_bar = open_price_all
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

#oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)
oracle = LLMChangedOracle(mkt_open,mkt_close,symbol)
#Generate and match the number of Open prices with LLM
num_example = 1
# trading_days = ["2025-01-27","2025-01-28"]
trading_days = ["2025-04-02","2025-04-03","2025-04-04"]
invest_company = invest_company_dict[symbol]
invest_company = invest_company[0:10]
# invest_company = random.sample(invest_company, 10)  # Randomly select num_example investment companies
oracle.llm_changed_price = [[0]*len(invest_company)]*num_example # Manually set the Open price to 0
print("LLM Changed Price:", oracle.llm_changed_price)
# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             num_example=num_example,
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) ExampleLLAgent
agents.extend([ExampleLLAgent(id=j,
                             name="ExampleLLAgent {}".format(1),
                             type="ExampleLLAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             invest_company = invest_company,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                             for j in range(agent_count, agent_count + num_example)])
agent_types.extend("ExampleLLAgent")
agent_count += num_example

# 2) Noise Agents
num_noise = 12000
noise_mkt_open = historical_date + pd.to_timedelta("09:30:00")  # These times needed for distribution of arrival times
                                                                # of Noise Agents
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

# 3) Value Agents
num_value = 100
agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          r_bar=r_bar,
                          kappa=kappa,
                          log_orders=log_orders,
                          lambda_a=lambda_a,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# 3) Trade Agents
TradeAgent_id_start = agent_count
oracle.TradeAgent_id_start = TradeAgent_id_start
num_Trade = 2950
oracle.TradeAgent_num = num_Trade
agents.extend([TradeAgent(id=j,
                          name="Trade Agent {}".format(j),
                          type="TradeAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_Trade)])
agent_count += num_Trade
agent_types.extend(['TradeAgent'])

# 4) Market Maker Agents

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [(args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
            (args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
            (args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
             (args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size)
             ]

num_mm_agents = len(mm_params)
mm_cancel_limit_delay = 50  # 50 nanoseconds

agents.extend([AdaptiveMarketMakerAgent(id=j,
                                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                type='AdaptivePOVMarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                pov=mm_params[idx][1],
                                min_order_size=mm_params[idx][4],
                                window_size=mm_params[idx][0],
                                num_ticks=mm_params[idx][2],
                                wake_up_freq=mm_params[idx][3],
                                cancel_limit_delay=mm_cancel_limit_delay,
                                skew_beta=args.mm_skew_beta,
                                level_spacing=args.mm_level_spacing,
                                spread_alpha=args.mm_spread_alpha,
                                backstop_quantity=args.mm_backstop_quantity,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('POVMarketMakerAgent')


#5) Momentum Agents
num_momentum_agents = 50

agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=log_orders,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("RMSC03 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')
defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY .

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**31))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              symbol = symbol,
              log_dir=args.log_dir,
              num_example=num_example,
              trading_days=trading_days)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
ray.shutdown()
