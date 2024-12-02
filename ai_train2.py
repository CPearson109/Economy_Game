# economy_sim_training.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import csv
import os
from economyGame import Game, RESOURCES
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Initialize Pygame
pygame.init()


class EconomySimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EconomySimEnv, self).__init__()
        self.game = Game()
        self.num_resources = len(RESOURCES)

        # Action space: For each resource, decide the percentage of cash to spend on buying and percentage of inventory to sell.
        # Additionally, decide whether to advance the day.
        self.action_space = spaces.Box(
            low=np.array([0.0] * (2 * self.num_resources) + [0.0]),
            high=np.array([1.0] * (2 * self.num_resources) + [1.0]),
            dtype=np.float32
        )

        # Observation space: Include player money, inventory levels, current prices, previous prices, and other market info.
        obs_low = np.array([0.0] * (1 + self.num_resources * 6), dtype=np.float32)
        obs_high = np.array([np.inf] * (1 + self.num_resources * 6), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize reward tracking variables
        self.previous_net_worth = None
        self.initial_net_worth = None
        self.total_days = 0
        self.total_reward = 0.0
        self.reward_count = 0

        # Variables to store episode data
        self.episode_data = []
        self.current_day_actions = []
        self.current_day = 1

        # Reward strategy parameters
        self.max_inventory_threshold = 100  # Example threshold for penalties
        self.diversity_bonus_per_resource = 0.2  # Bonus per unique resource held
        self.final_reward_weight = 0.5  # Weight for final net worth reward
        self.roi_bonus_weight = 1.0  # Weight for ROI in reward
        self.roi_penalty_weight = 0.5  # Weight for negative ROI penalties

        # Tracking variables for ROI
        self.daily_investment = 0.0
        self.daily_net_profit = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.total_days = 0
        self.total_reward = 0.0
        self.reward_count = 0
        self.previous_net_worth = self.calculate_net_worth()
        self.initial_net_worth = self.previous_net_worth  # Store initial net worth for normalization
        self.current_day_actions = []
        self.current_day = 1
        self.episode_data = []
        self.daily_investment = 0.0
        self.daily_net_profit = 0.0
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        obs = [self.game.player.money]
        for resource in RESOURCES:
            obs.extend([
                self.game.player.inventory.get(resource, 0),
                self.game.market.prices.get(resource, 0.0),
                self.game.market.previous_prices.get(resource, 0.0),
                self.game.market.supply.get(resource, 0.0),
                self.game.market.demand.get(resource, 0.0),
                np.mean(self.game.market.price_history[resource][-5:]) if len(self.game.market.price_history[resource]) >= 5 else self.game.market.prices.get(resource, 0.0)
            ])
        return np.array(obs, dtype=np.float32)

    def calculate_net_worth(self):
        net_worth = self.game.player.money + sum(
            self.game.player.inventory.get(resource, 0) * self.game.market.prices.get(resource, 0.0)
            for resource in RESOURCES
        )
        return net_worth

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        resources = RESOURCES

        previous_observation = self.get_observation().copy()

        # Track transaction costs and investments
        total_buy_cost = 0.0
        total_sell_revenue = 0.0
        daily_investment = 0.0  # Reset daily investment
        daily_net_profit = 0.0    # Reset daily net profit

        # Process buy and sell actions
        for i, resource in enumerate(resources):
            buy_fraction = action[i]
            sell_fraction = action[i + self.num_resources]

            # Buy
            available_cash = self.game.player.money
            buy_amount = (available_cash * buy_fraction) / self.game.market.prices.get(resource, 1.0)
            buy_quantity = int(buy_amount)
            if buy_quantity > 0:
                success = self.game.player.buy(self.game.market, resource, buy_quantity)
                if success:
                    cost = buy_quantity * self.game.market.prices.get(resource, 0.0)
                    total_buy_cost += cost
                    daily_investment += cost
                    self.current_day_actions.append(
                        f"Purchased {buy_quantity} {resource} at ${self.game.market.prices.get(resource, 0.0):.2f} each"
                    )

            # Sell
            inventory = self.game.player.inventory.get(resource, 0)
            sell_quantity = int(inventory * sell_fraction)
            if sell_quantity > 0:
                success = self.game.player.sell(self.game.market, resource, sell_quantity)
                if success:
                    revenue = sell_quantity * self.game.market.prices.get(resource, 0.0)
                    total_sell_revenue += revenue
                    daily_net_profit += revenue
                    self.current_day_actions.append(
                        f"Sold {sell_quantity} {resource} at ${self.game.market.prices.get(resource, 0.0):.2f} each"
                    )

        # Advance day if action[-1] > 0.5
        if action[-1] > 0.5:
            self.current_day_actions.append("Advance day")

            # Record totals, including Water
            totals = {
                'total_cash': round(self.game.player.money, 2),
                'total_net_worth': round(self.calculate_net_worth(), 2),
                'total_food': int(round(self.game.player.inventory.get('Food', 0))),
                'total_fuel': int(round(self.game.player.inventory.get('Fuel', 0))),
                'total_clothes': int(round(self.game.player.inventory.get('Clothes', 0))),
                'total_water': int(round(self.game.player.inventory.get('Water', 0)))  # **Added Water**
            }

            # Record day's data
            self.episode_data.append({
                'day': self.current_day,
                'actions': self.current_day_actions.copy(),
                'totals': totals
            })

            # Reset current day actions and increment day
            self.current_day_actions = []
            self.current_day += 1

            self.game.advance_day()
            self.total_days += 1

            # Calculate daily net profit (including changes in inventory value)
            new_net_worth = self.calculate_net_worth()
            net_worth_change = new_net_worth - self.previous_net_worth
            daily_net_profit += net_worth_change - daily_investment  # Net profit after investment

            # Reset previous net worth for next day
            self.previous_net_worth = new_net_worth
        else:
            # If day is not advanced, net profit is zero
            net_worth_change = 0.0

        # Calculate ROI
        roi = 0.0
        if daily_investment > 0:
            roi = daily_net_profit / daily_investment
        elif daily_net_profit > 0:
            # If no investment but profit was made (e.g., from selling existing inventory)
            roi = daily_net_profit / self.initial_net_worth  # Relative to initial net worth
        else:
            roi = 0.0  # No investment and no profit

        # Initialize reward with ROI
        reward = roi * self.roi_bonus_weight

        # Add penalties for negative ROI
        if roi < 0:
            reward += roi * self.roi_penalty_weight  # Negative reward for negative ROI

        # Penalty for excessive inventory
        excessive_inventory_penalty = 0.0
        for resource in RESOURCES:
            inventory = self.game.player.inventory.get(resource, 0)
            if inventory > self.max_inventory_threshold:
                excessive_inventory = inventory - self.max_inventory_threshold
                excessive_inventory_penalty += excessive_inventory * 0.05  # Penalty rate per excessive unit
        reward -= excessive_inventory_penalty

        # Bonus for portfolio diversity
        active_resources = len([qty for qty in self.game.player.inventory.values() if qty > 0])
        diversity_bonus = active_resources * self.diversity_bonus_per_resource
        reward += diversity_bonus

        # Final episode reward
        done = self.total_days >= 365
        if done:
            final_reward = self.calculate_net_worth() * self.final_reward_weight
            reward += final_reward

            # Record any remaining actions for the last day
            if self.current_day_actions:
                # Record totals, including Water
                totals = {
                    'total_cash': round(self.game.player.money, 2),
                    'total_net_worth': round(self.calculate_net_worth(), 2),
                    'total_food': int(round(self.game.player.inventory.get('Food', 0))),
                    'total_fuel': int(round(self.game.player.inventory.get('Fuel', 0))),
                    'total_clothes': int(round(self.game.player.inventory.get('Clothes', 0))),
                    'total_water': int(round(self.game.player.inventory.get('Water', 0)))  # **Added Water**
                }
                # Record day's data
                self.episode_data.append({
                    'day': self.current_day,
                    'actions': self.current_day_actions.copy(),
                    'totals': totals
                })
                # Reset current day actions
                self.current_day_actions = []
                self.current_day += 1

        # Normalize reward
        reward /= self.initial_net_worth if self.initial_net_worth != 0 else 1.0

        self.total_reward += reward
        self.reward_count += 1

        # Update previous net worth if day advanced
        if action[-1] > 0.5:
            pass  # Already updated above
        else:
            # If day not advanced, keep previous net worth
            pass

        # Prepare observation
        observation = self.get_observation()
        current_net_worth = self.calculate_net_worth()

        # If done, include episode data in info
        info = {
            'net_worth': round(current_net_worth, 2),
            'player_money': round(self.game.player.money, 2),
            'player_inventory': self.game.player.inventory.copy(),
            'days': self.total_days,
            'total_reward': round(self.total_reward, 4),
            'average_reward': round(self.total_reward / self.reward_count, 4) if self.reward_count > 0 else 0,
            'roi': round(roi, 4)
        }

        if done:
            info['episode_data'] = self.episode_data.copy()

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
        self.term_count = 0

        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    'Term', 'Final Net Worth', 'Final Cash',
                    'Final Reward', 'Average Reward',
                    'Food Inventory', 'Fuel Inventory', 'Clothes Inventory', 'Water Inventory', 'ROI'  # **Added ROI**
                ])

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')

        if dones is not None and any(dones):
            done_indices = [i for i, done in enumerate(dones) if done]
            for idx in done_indices:
                self.term_count += 1
                info = infos[idx]

                net_worth = info['net_worth']
                player_money = info['player_money']
                final_reward = info['total_reward']
                average_reward = info['average_reward']
                roi = info.get('roi', 0.0)
                inventories = info['player_inventory']

                food_inventory = int(round(inventories.get('Food', 0)))
                fuel_inventory = int(round(inventories.get('Fuel', 0)))
                clothes_inventory = int(round(inventories.get('Clothes', 0)))
                water_inventory = int(round(inventories.get('Water', 0)))  # **Added Water Inventory**

                with open(self.csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.term_count, net_worth, player_money,
                        final_reward, average_reward,
                        food_inventory, fuel_inventory, clothes_inventory, water_inventory,
                        roi  # **Added ROI**
                    ])
        return True


class BestRunLoggerCallback(BaseCallback):
    def __init__(self, best_run_path, verbose=0):
        super(BestRunLoggerCallback, self).__init__(verbose)
        self.best_run_path = best_run_path
        self.best_net_worth = -np.inf  # Initialize with negative infinity

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')

        if dones is not None and any(dones):
            done_indices = [i for i, done in enumerate(dones) if done]
            for idx in done_indices:
                info = infos[idx]
                net_worth = info['net_worth']
                # Check if this is the best net worth so far
                if net_worth > self.best_net_worth:
                    self.best_net_worth = net_worth
                    # Get the episode data from the info
                    episode_data = info.get('episode_data', [])

                    # Save episode_data to text file in specified format
                    with open(self.best_run_path, mode='w') as file:
                        for day_data in episode_data:
                            day_number = day_data['day']
                            actions = day_data['actions']
                            totals = day_data['totals']
                            file.write(f"Day {day_number}\n")
                            for action_text in actions:
                                file.write(f"{action_text}\n")
                            file.write(f"Total Cash: ${totals['total_cash']:.2f}\n")
                            file.write(f"Total Net Worth: ${totals['total_net_worth']:.2f}\n")
                            file.write(f"Total Food: {totals['total_food']}\n")
                            file.write(f"Total Fuel: {totals['total_fuel']}\n")
                            file.write(f"Total Clothes: {totals['total_clothes']}\n")
                            file.write(f"Total Water: {totals['total_water']}\n")  # **Added Total Water**
                            file.write("\n")  # Add a blank line between days
        return True


if __name__ == "__main__":
    import os

    model_name = input("Enter model name: ")
    model_dir = model_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Paths for model, CSV, best run
    model_path = os.path.join(model_dir, "economy_sim_ppo_model")
    csv_path = os.path.join(model_dir, "economy_sim_results.csv")
    best_run_path = os.path.join(model_dir, "best_run.txt")  # Changed extension to .txt

    env = EconomySimEnv()

    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("Creating new model.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            clip_range=0.2
        )

    # Callback for CSV logging
    csv_logger_callback = CSVLoggerCallback(csv_path=csv_path)

    # Callback for best run
    best_run_callback = BestRunLoggerCallback(best_run_path=best_run_path)

    total_timesteps = 5000000000

    try:
        model.learn(total_timesteps=total_timesteps, callback=[csv_logger_callback, best_run_callback])
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(model_path)
        env.close()
        pygame.quit()
    else:
        model.save(model_path)
        env.close()
        pygame.quit()
