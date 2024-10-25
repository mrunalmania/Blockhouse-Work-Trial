import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import gymnasium as gym  # Change from gym to gymnasium
from gymnasium import spaces  # Import spaces from gymnasium
import numpy as np
from ray.tune.registry import register_env

# Load the AAPL order book CSV
file_path = "./data/AAPL_Quotes_Data.csv"  # Replace with your actual file path
order_book = pd.read_csv(file_path)

# Define the custom AAPL Trading environment
class AAPLTradingEnv(gym.Env):
    def __init__(self, order_book, V=10000, H=100, I=10, T=10):
        super(AAPLTradingEnv, self).__init__()
        self.order_book = order_book
        self.V = V  # Total volume to trade
        self.H = H  # Time horizon
        self.I = I  # Number of inventory divisions
        self.T = T  # Number of time divisions

        # Observation space now contains top 5 bid and ask prices, along with time and remaining inventory
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(10 + 2,),  # 10 prices + time + inventory
            dtype=np.float32
        )
        
        # Action space (Hold, Cross Spread, Place in Own Book)
        self.action_space = spaces.Discrete(3)

        self.reset()

    # def reset(self):
    #     self.current_step = 0
    #     self.remaining_volume = self.V
    #     self.state = self._get_observation()
    #     return self.state


    def reset(self):
        self.current_step = 0
        self.remaining_volume = self.V
        self.state = self._get_observation()
        
        # Return the observation and an empty dictionary for the 'info' field
        return self.state, {}

    def _get_observation(self):
        """Returns normalized observation including top 5 bid and ask prices"""
        row = self.order_book.iloc[self.current_step]
        bid_prices = row[['bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5']].values
        ask_prices = row[['ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5']].values

        normalized_time = self.current_step / self.T
        normalized_inventory = self.remaining_volume / self.V

        # Concatenate prices with normalized time and inventory
        obs = np.concatenate([bid_prices, ask_prices, [normalized_time, normalized_inventory]])
        return obs

    def step(self, action):
        # Retrieve current bid and ask prices
        row = self.order_book.iloc[self.current_step]
        ask_price = row['ask_price_1']
        bid_price = row['bid_price_1']

        reward = 0
        done = False

        # Simulate the action effects
        if action == 0:  # Hold
            reward = 0
        elif action == 1:  # Cross the spread
            reward = -(ask_price - bid_price) * self.remaining_volume / self.V
            self.remaining_volume = 0  # All shares sold
        elif action == 2:  # Place in own book (wait for market takers)
            if np.random.random() < 0.5:  # Random fill condition
                reward = (ask_price - bid_price) * (self.remaining_volume / 2) / self.V
                self.remaining_volume /= 2

        # Update step and check if done
        self.current_step += 1
        if self.current_step >= self.T or self.remaining_volume <= 0:
            done = True
        
        return self._get_observation(), reward, done, {}

##############################  Init Mode #######################################

# Initialize Ray
ray.init()

# Register the custom environment
def env_creator(env_config):
    return AAPLTradingEnv(order_book=env_config["order_book"], V=env_config["V"], H=env_config["H"], I=env_config["I"], T=env_config["T"])

register_env("aapl_trading_env", env_creator)

# PPO configuration for Ray 2.x
# config = (
#     PPOConfig()
#     .environment(env="AAPLTradingEnv", env_config={"order_book": order_book, "V": 10000, "H": 100, "I": 10, "T": 10})
#     .env_runners(num_env_runners=1)  # Use env_runners instead of rollouts
#     .framework("torch")  # We will use PyTorch for this
#     .training(lr=1e-4, train_batch_size=1000, num_sgd_iter=10)  # Removed sgd_minibatch_size
#     .model(fcnet_hiddens=[128, 128], fcnet_activation="relu")
# )


config = PPOConfig()

config.api_stack(
    enable_rl_module_and_learner= True,
    enable_env_runner_and_connector_v2= True
)

config.environment(env = "aapl_trading_env", env_config={"order_book": order_book, "V": 10000, "H": 100, "I": 10, "T": 10})
config.env_runners(num_env_runners=1)
config.training(
    gamma=0.9, lr=0.1, train_batch_size_per_learner = 128
)
# Create PPO trainer instance
algo = config.build()

# Training loop
for i in range(10):  # Run multiple training iterations
    result = algo.train()
    print(f"Iteration: {i}, Reward: {result['episode_reward_mean']}")

# Save the trained model
checkpoint = algo.save()
print(f"Checkpoint saved at: {checkpoint}")

# Shutdown Ray
ray.shutdown()