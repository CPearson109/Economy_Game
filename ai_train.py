import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from economyGame import Game
import pygame
import time

class EconomySimEnv(gym.Env):
    def __init__(self):
        super(EconomySimEnv, self).__init__()

        # Define action space
        # Actions: [action_food, action_fuel, action_clothes, advance_day]
        # Each action is between -1 and 1
        # Negative values indicate selling, positive values indicate buying
        # For advance_day, values >= 0.5 indicate advancing the day
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Observation space: [player_money, food_price, food_inventory,
        #                     fuel_price, fuel_inventory, clothes_price, clothes_inventory]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Initialize the game
        self.game = Game()

        # Initialize previous net worth
        self.previous_net_worth = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        observation = self.game.get_observation()

        # Calculate initial net worth
        self.previous_net_worth = self.calculate_net_worth(observation)

        return observation, {}

    def calculate_net_worth(self, observation):
        player_money = observation[0]
        prices = observation[1::2]
        inventories = observation[2::2]
        net_worth = player_money + np.sum(prices * inventories)
        return net_worth

    def step(self, action):
        # Clip actions to be within the action space limits
        action = np.clip(action, self.action_space.low, self.action_space.high)

        pygame.event.pump()

        # Map the action to buying/selling amounts
        # Positive values indicate buying, negative values indicate selling
        resources = ['Food', 'Fuel', 'Clothes']
        for i, resource in enumerate(resources):
            amount = action[i]

            if amount > 0:
                # Buy
                self.game.selected_resource = resource
                self.game.process_action(1, buy=True)
            elif amount < 0:
                # Sell
                self.game.selected_resource = resource
                self.game.process_action(1, buy=False)
            # If amount is zero, do nothing

        # Check if the AI wants to advance the day
        if action[3] >= 0.5:
            # Advance the game by one day
            self.game.advance_day()

        # Get the observation
        observation = self.game.get_observation()

        # Calculate the current net worth
        current_net_worth = self.calculate_net_worth(observation)

        # Calculate the reward as the change in net worth
        reward = current_net_worth - self.previous_net_worth

        # Update previous net worth
        self.previous_net_worth = current_net_worth

        # The episode continues indefinitely until a certain condition is met (you can define it)
        done = False

        info = {
            'reward': reward,
            'player_money': self.game.player.money,
            'net_worth': current_net_worth,
        }

        return observation, reward, done, False, info

    def render(self, mode='human'):
        # Call the existing render function from economyGame.py
        self.game.render()

    def close(self):
        pass

# Modify the Game class to handle variable amounts
def modify_game_class():
    # Modify the Game class to accept variable amounts in process_action
    def process_action(self, amount, buy=True):
        # Ensure amount is positive
        amount = abs(amount)

        # Define the maximum amount the player can buy/sell based on their money/inventory
        if buy:
            price = self.market.prices[self.selected_resource]
            total_cost = price * amount

            if total_cost > self.player.money:
                amount = self.player.money / price  # Adjust amount to max affordable

            # Buy the maximum possible amount
            self.player.buy(self.market, self.selected_resource, amount)
        else:
            inventory_amount = self.player.inventory[self.selected_resource]
            if amount > inventory_amount:
                amount = inventory_amount  # Adjust amount to what's available

            # Sell the maximum possible amount
            self.player.sell(self.market, self.selected_resource, amount)

    # Replace the original process_action method with the new one
    setattr(Game, 'process_action', process_action)

modify_game_class()

import time

if __name__ == "__main__":
    # Create the environment
    env = EconomySimEnv()
    env = DummyVecEnv([lambda: env])

    # Create the agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Training setup
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)

    # Test the trained agent
    obs, _ = env.reset()
    frame_count = 0
    render_interval = 5  # Render every 5 frames, for instance

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _, info = env.step(action)

        if frame_count % render_interval == 0:
            env.render()  # Render only every few frames

        # Process pygame events to avoid freezing or crashing
        pygame.event.pump()

        frame_count += 1
        if dones:
            obs, _ = env.reset()

        # Add a small sleep to reduce load on the CPU
        time.sleep(0.05)
