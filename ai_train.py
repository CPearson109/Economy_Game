import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from economyGame import Game  # Import the Game class from economyGame.py

render_training = False

class EconomySimEnv(gym.Env):
    def __init__(self):
        super(EconomySimEnv, self).__init__()

        # Define action space
        self.action_space = spaces.Discrete(7)

        # Observation space: [player_money, food_price, food_inventory,
        # fuel_price, fuel_inventory, clothes_price, clothes_inventory]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Initialize the game
        self.game = Game()

        # Initialize previous net worth
        self.previous_net_worth = None

        # Initialize step count
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        observation = self.game.get_observation()

        # Extract prices and inventories from observation
        player_money = observation[0]
        food_price = observation[1]
        food_inventory = observation[2]
        fuel_price = observation[3]
        fuel_inventory = observation[4]
        clothes_price = observation[5]
        clothes_inventory = observation[6]

        # Calculate initial net worth
        self.previous_net_worth = player_money + (
            food_price * food_inventory
            + fuel_price * fuel_inventory
            + clothes_price * clothes_inventory
        )

        # Reset step count
        self.step_count = 0

        return observation, {}

    def step(self, action):
        self.step_count += 1

        # Map the action to the game
        self.game.process_action(action)

        # Advance the game by one day
        self.game.advance_day()

        # Get the observation
        observation = self.game.get_observation()

        # Extract prices and inventories from observation
        player_money = observation[0]
        food_price = observation[1]
        food_inventory = observation[2]
        fuel_price = observation[3]
        fuel_inventory = observation[4]
        clothes_price = observation[5]
        clothes_inventory = observation[6]

        # Calculate the current net worth
        current_net_worth = player_money + (
                food_price * food_inventory
                + fuel_price * fuel_inventory
                + clothes_price * clothes_inventory
        )

        # Calculate the reward as the percentage change in net worth
        previous_net_worth = max(self.previous_net_worth, 1)  # Prevent division by zero
        reward = (current_net_worth - self.previous_net_worth) / previous_net_worth

        # Adjust reward based on inventory diversity
        diversity_bonus = len([inv for inv in [food_inventory, fuel_inventory, clothes_inventory] if inv > 0])
        reward += 0.1 * diversity_bonus  # Encourage having multiple types of inventory

        # Clip the reward to prevent extreme values
        reward = np.clip(reward, -1, 1)

        # Update previous net worth
        self.previous_net_worth = current_net_worth

        # Check if the game is done
        done = self.game.day >= 365

        info = {
            'step_reward': reward,
            'food_inventory': food_inventory,
            'fuel_inventory': fuel_inventory,
            'clothes_inventory': clothes_inventory,
            'player_money': player_money,
            'net_worth': current_net_worth,
            'step_count': self.step_count
        }

        return observation, reward, done, False, info

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        pygame.quit()

# Custom callback to log per-term (episode) information
class CustomLoggerCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.term_rewards = []
        self.term_info = {}
        self.term_count = 0

    def _on_training_start(self):
        # Open the log file and write the header
        with open(self.log_file, 'w') as f:
            f.write('term_count,average_reward,food_inventory,fuel_inventory,clothes_inventory,player_money,net_worth\n')

    def _on_step(self) -> bool:
        # Get the info dict from the environment
        info = self.locals['infos'][0]

        # Accumulate reward
        self.term_rewards.append(info.get('step_reward', 0))

        # Save the last observation data (to log at the end of the term)
        self.term_info = {
            'food_inventory': info.get('food_inventory', 0),
            'fuel_inventory': info.get('fuel_inventory', 0),
            'clothes_inventory': info.get('clothes_inventory', 0),
            'player_money': info.get('player_money', 0),
            'net_worth': info.get('net_worth', 0)
        }

        # If the term is over, log the data
        if self.locals['dones'][0]:
            self.term_count += 1
            average_reward = np.mean(self.term_rewards)

            # Write the data
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{self.term_count},{average_reward},{self.term_info['food_inventory']},{self.term_info['fuel_inventory']},{self.term_info['clothes_inventory']},{self.term_info['player_money']},{self.term_info['net_worth']}\n"
                )

            # Reset the reward list for the next term
            self.term_rewards = []

        return True

    def _on_training_end(self) -> None:
        pass

if __name__ == "__main__":
    # Prompt for model name
    model_name = input("Enter the model name (default: economy_ai_model): ")
    if not model_name:
        model_name = "economy_ai_model"

    # Paths for model and logs
    model_path = f"models/{model_name}"
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create or load environment
    env = Monitor(EconomySimEnv(), filename=log_dir + "train_monitor.csv")
    env = DummyVecEnv([lambda: env])  # Wrap the environment for Stable Baselines3

    # Check if model exists and ask whether to load it
    if os.path.exists(model_path + ".zip"):  # Check if model file exists
        choice = input(
            f"Model '{model_name}' exists. Do you want to load it? (y/n): "
        )
        if choice.lower() == 'y':
            model = PPO.load(model_path, env=env)
        else:
            model = PPO('MlpPolicy', env, verbose=1)
    else:
        model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01)  # Modified entropy coefficient

    total_timesteps = 1000000

    # Create evaluation environment
    eval_env = Monitor(EconomySimEnv(), filename=log_dir + "eval_monitor.csv")
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create callbacks for evaluation and checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix=model_name
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Custom logger callback
    log_file = log_dir + "custom_training_log.csv"
    custom_logger = CustomLoggerCallback(log_file=log_file)

    # Try-except block to handle KeyboardInterrupt and save the model
    try:
        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, custom_logger]
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
        model.save(model_path)
        env.close()
        print("Model saved. Exiting.")
    else:
        # Save the final model
        model.save(model_path)
        env.close()

    # After training or interruption, load the custom training logs and plot the graphs
    # Read the custom log CSV file
    if os.path.exists(log_file):
        # Read the data into a DataFrame
        data = pd.read_csv(log_file)

        # Calculate average reward per term (episode)
        term_groups = data.groupby('term_count').mean()  # This will compute the average for each group

        # Ensure the index is reset to make it plottable
        term_groups = term_groups.reset_index()

        # Plot term vs average reward
        plt.figure()
        plt.plot(term_groups['term_count'], term_groups['average_reward'], label='Average Reward per Term')
        plt.xlabel('Term (Episode)')
        plt.ylabel('Average Reward')
        plt.title('Term vs Average Reward')
        plt.legend()
        plt.savefig(log_dir + 'term_vs_average_reward.png')
        plt.show()

        # Plot net worth over terms
        plt.figure()
        plt.plot(term_groups['term_count'], term_groups['net_worth'], label='Net Worth')
        plt.xlabel('Term (Episode)')
        plt.ylabel('Net Worth')
        plt.title('Term vs Net Worth')
        plt.legend()
        plt.savefig(log_dir + 'term_vs_net_worth.png')
        plt.show()
    else:
        print("No custom training log file found.")
