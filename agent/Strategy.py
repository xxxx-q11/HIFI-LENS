
class TradingStrategy:
    def place_order(self, agent):
        pass
    
class ValueStrategy(TradingStrategy):
    def __init__(self, config):
        self.sigma_n = config["sigma_n"] 
        self.r_bar = config["r_bar"]
        
    def place_order(self, agent):

        pass

class MomentumStrategy(TradingStrategy):
    def __init__(self, config):
        self.min_size = config["min_size"]
        self.max_size = config["max_size"]
        
    def place_order(self, agent):
        pass

class TradeAgent:
    def __init__(self, strategy):
        self.strategy = strategy
        
    def place_order(self):
        self.strategy.place_order(self)