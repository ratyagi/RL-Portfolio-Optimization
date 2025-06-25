import gym
from gym import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    """
    A simple RL environment for SPY portfolio management using regime-labeled data.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """

    def __init__(self, data_path):
        super(PortfolioEnv, self).__init__()

        # Load regime-labeled dataset
        self.df = pd.read_csv(data_path)
        self.n_steps = len(self.df)

        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Observation space: 3 continuous features + 1 discrete regime + 1 position state
        # Features: Close price, volatility, momentum, regime, position (holding or not)
        low = np.array([0, 0, -np.inf, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, 2, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.current_step = 0
        self.position = 0  # 0 = no position, 1 = holding stock
        self.initial_balance = 1.0  # Starting portfolio value
        self.portfolio_value = self.initial_balance

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.portfolio_value = self.initial_balance
        return self._get_observation()

    def step(self, action):
        done = False
        reward = 0

        # Calculate reward based on daily return and position
        current_return = self.df.loc[self.current_step, 'returns']
        if action == 1:  # Buy
            self.position = 1
        elif action == 2:  # Sell
            self.position = 0

        reward = self.position * current_return  # Reward = return if holding

        # Update portfolio value
        self.portfolio_value *= (1 + reward)

        # Move to next day
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        next_state = self._get_observation()
        info = {'portfolio_value': self.portfolio_value}

        return next_state, reward, done, info

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Close'],
            row['volatility'],
            row['momentum'],
            row['regime'],
            self.position
        ], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}, Position: {self.position}')