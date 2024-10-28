import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import csv
from economyGame import Game
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Initialize Pygame
pygame.init()

# Define the custom Gym environment class
class EconomySimEnv(gym.Env):
    def __init__(self):
        super(EconomySimEnv, self).__init__()

        # Initialize the game
        self.game = Game()

        # Define action space: [buy/sell amount for Food, Fuel, Clothes, advance_day]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Define observation space with relevant player and market state
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Initialize previous net worth, day counter, and buy/sell counters
        self.previous_net_worth = None
        self.day_counter = 0  # Counter for tracking in-game days
        self.items_bought = 0  # Track total items bought in a term
        self.items_sold = 0    # Track total items sold in a term

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.day_counter = 0  # Reset the day counter at the start of each term
        self.items_bought = 0  # Reset items bought counter
        self.items_sold = 0    # Reset items sold counter
        observation = self.get_observation()

        # Calculate initial net worth
        self.previous_net_worth = self.calculate_net_worth(observation)

        return observation, {}

    def get_observation(self):
        # Observation includes player money and inventory levels
        player_money = self.game.player.money
        food_inventory = self.game.player.inventory['Food']
        fuel_inventory = self.game.player.inventory['Fuel']
        clothes_inventory = self.game.player.inventory['Clothes']

        return np.array([
            player_money, food_inventory, fuel_inventory, clothes_inventory
        ], dtype=np.float32)

    def calculate_net_worth(self, observation):
        # Net worth calculation: money + value of each inventory item at market price
        player_money = observation[0]
        net_worth = player_money + sum(
            self.game.market.prices[item] * observation[i + 1] for i, item in enumerate(["Food", "Fuel", "Clothes"])
        )
        return net_worth

    def step(self, action):
        # Clip actions within the action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Process each resource's buy/sell action based on the action values
        resources = ['Food', 'Fuel', 'Clothes']
        for i, resource in enumerate(resources):
            amount = action[i]
            quantity = int(np.round(abs(amount)))  # Convert to integer quantity

            self.game.selected_resource = resource  # Set resource for the action
            if amount > 0:  # Agent decides to buy the resource
                self.game.process_action(quantity)  # Pass the positive integer quantity to buy
                self.items_bought += quantity       # Increment total items bought by integer count
            elif amount < 0:  # Agent decides to sell the resource
                self.game.process_action(-quantity) # Pass the positive integer quantity to sell
                self.items_sold += quantity         # Increment total items sold by integer count

        # Advance the day if the action for advancing is above 0.5
        if action[3] >= 0.5:
            self.game.advance_day()
            self.day_counter += 1  # Increment the day counter

        # Update the observation
        observation = self.get_observation()

        # Calculate the current net worth
        current_net_worth = self.calculate_net_worth(observation)

        # Reward is based on the change in net worth
        reward = current_net_worth - self.previous_net_worth

        # Update previous net worth
        self.previous_net_worth = current_net_worth

        # Check if term (365 days) is complete or if the player is bankrupt
        done = self.day_counter >= 365 or current_net_worth <= 0

        # Information for logging
        info = {
            'reward': reward,
            'player_money': self.game.player.money,
            'net_worth': current_net_worth,
            'days': self.day_counter,
            'items_bought': self.items_bought,
            'items_sold': self.items_sold
        }

        return observation, reward, done, False, info

    def render(self, mode='human'):
        self.game.render()
        pygame.display.flip()

    def close(self):
        pygame.quit()


class CSVLoggerCallback(BaseCallback):
    def __init__(self, csv_path, verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        self.csv_path = csv_path
        self.term_count = 0        # Track the number of completed terms
        self.cumulative_reward = 0  # Track cumulative reward for each term

        # Initialize CSV file with headers
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Term', 'Cumulative Reward', 'Final Net Worth',
                'Food Inventory', 'Fuel Inventory', 'Clothes Inventory',
                'Items Bought', 'Items Sold'
            ])

    def _on_step(self) -> bool:
        # Accumulate rewards at each step
        self.cumulative_reward += self.locals["rewards"][0]  # Add the step reward to cumulative reward
        return True

    def _on_rollout_end(self) -> None:
        # Log the details to CSV at the end of each term
        self.term_count += 1

        # Gather current data to log
        player_money = self.training_env.envs[0].game.player.money
        inventories = self.training_env.envs[0].game.player.inventory
        net_worth = player_money + sum(self.training_env.envs[0].game.market.prices[item] * inventories[item] for item in inventories)
        items_bought = self.training_env.envs[0].items_bought
        items_sold = self.training_env.envs[0].items_sold

        # Write data to CSV
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.term_count,
                self.cumulative_reward,
                net_worth,
                inventories['Food'],
                inventories['Fuel'],
                inventories['Clothes'],
                items_bought,
                items_sold
            ])

        # Reset cumulative reward, items bought, and items sold for the next term
        self.cumulative_reward = 0
        self.training_env.envs[0].items_bought = 0
        self.training_env.envs[0].items_sold = 0


if __name__ == "__main__":
    env = DummyVecEnv([lambda: EconomySimEnv()])

    model = PPO("MlpPolicy", env, verbose=1)

    # CSV path to store term results
    csv_path = "economy_sim_results.csv"

    # Create the CSV logger callback
    csv_logger_callback = CSVLoggerCallback(csv_path=csv_path)

    total_timesteps = 10000000
    model.learn(total_timesteps=total_timesteps, callback=csv_logger_callback)

    model.save("economy_sim_ppo_model")

    env.close()
    pygame.quit()
