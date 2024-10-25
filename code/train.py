from ray.tune.registry import register_env
import ray

from model import AAPLTradingEnv

# Initialize Ray
ray.init()

# Register the custom environment
def env_creator(env_config):
    return AAPLTradingEnv(order_book=env_config["order_book"], V=env_config["V"], H=env_config["H"], I=env_config["I"], T=env_config["T"])

register_env("aapl_trading_env", env_creator)