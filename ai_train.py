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

    def __init__(self, game, player):
        super(EconomySimEnv, self).__init__()
        self.game = game  # Shared game instance
        self.player = player  # Individual player instance
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
        self.current_day = self.game.day  # Start from the current game day

        # Reward strategy parameters
        self.max_inventory_threshold = 100  # Threshold for penalties
        self.transaction_cost_rate = 0.01  # 1% transaction cost
        self.diversity_bonus_per_resource = 0.2  # Bonus per unique resource held
        self.final_reward_weight = 0.5  # Weight for final net worth reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Do not reset the game, only reset the player's state if necessary
        self.total_days = 0
        self.total_reward = 0.0
        self.reward_count = 0
        self.previous_net_worth = self.calculate_net_worth()
        self.initial_net_worth = self.previous_net_worth  # Store initial net worth for normalization
        self.current_day_actions = []
        self.current_day = self.game.day
        self.episode_data = []
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        obs = [self.player.money]
        for resource in RESOURCES:
            obs.extend([
                self.player.inventory[resource],
                self.game.market.prices[resource],
                self.game.market.previous_prices[resource],
                self.game.market.supply[resource],
                self.game.market.demand[resource],
                np.mean(self.game.market.price_history[resource][-5:])
                if len(self.game.market.price_history[resource]) >= 5
                else self.game.market.prices[resource]
            ])
        return np.array(obs, dtype=np.float32)

    def calculate_net_worth(self):
        net_worth = self.player.money + sum(
            self.player.inventory[resource] * self.game.market.prices[resource]
            for resource in RESOURCES
        )
        return net_worth

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        resources = RESOURCES

        # Track transaction costs
        total_buy_cost = 0.0
        total_sell_revenue = 0.0

        # Process buy and sell actions
        for i, resource in enumerate(resources):
            buy_fraction = action[i]
            sell_fraction = action[i + self.num_resources]

            # Buy
            available_cash = self.player.money
            buy_amount = (available_cash * buy_fraction) / self.game.market.prices[resource]
            buy_quantity = int(buy_amount)
            if buy_quantity > 0:
                success = self.player.buy(self.game.market, resource, buy_quantity)
                if success:
                    total_buy_cost += buy_quantity * self.game.market.prices[resource]
                    self.current_day_actions.append(
                        f"Purchased {buy_quantity} {resource} at ${self.game.market.prices[resource]:.2f} each"
                    )

            # Sell
            inventory = self.player.inventory[resource]
            sell_quantity = int(inventory * sell_fraction)
            if sell_quantity > 0:
                success = self.player.sell(self.game.market, resource, sell_quantity)
                if success:
                    total_sell_revenue += sell_quantity * self.game.market.prices[resource]
                    self.current_day_actions.append(
                        f"Sold {sell_quantity} {resource} at ${self.game.market.prices[resource]:.2f} each"
                    )

        # Advance turn if action[-1] > 0.5
        done = False
        if action[-1] > 0.5:
            self.current_day_actions.append("End of Turn")

            # Record totals, including Water
            totals = {
                'total_cash': round(self.player.money, 2),
                'total_net_worth': round(self.calculate_net_worth(), 2),
                'total_food': int(round(self.player.inventory.get('Food', 0))),
                'total_fuel': int(round(self.player.inventory.get('Fuel', 0))),
                'total_clothes': int(round(self.player.inventory.get('Clothes', 0))),
                'total_water': int(round(self.player.inventory.get('Water', 0)))  # Added Water
            }

            # Record day's data
            self.episode_data.append({
                'day': self.current_day,
                'actions': self.current_day_actions.copy(),
                'totals': totals
            })

            # Reset current day actions
            self.current_day_actions = []

            # Set done to True to indicate end of agent's turn
            done = True

        observation = self.get_observation()
        current_net_worth = self.calculate_net_worth()

        # Calculate basic reward based on net worth change
        net_worth_change = current_net_worth - self.previous_net_worth

        # Initialize reward with net worth change
        reward = net_worth_change

        # Subtract transaction costs
        transaction_costs = (total_buy_cost + total_sell_revenue) * self.transaction_cost_rate
        reward -= transaction_costs

        # Penalty for excessive inventory
        excessive_inventory_penalty = 0.0
        for resource in RESOURCES:
            if self.player.inventory.get(resource, 0) > self.max_inventory_threshold:
                excessive_inventory = self.player.inventory.get(resource, 0) - self.max_inventory_threshold
                excessive_inventory_penalty += excessive_inventory * 0.05  # Penalty per excessive unit
        reward -= excessive_inventory_penalty

        # Bonus for portfolio diversity
        active_resources = len([qty for qty in self.player.inventory.values() if qty > 0])
        diversity_bonus = active_resources * self.diversity_bonus_per_resource
        reward += diversity_bonus

        # Normalize reward
        reward /= self.initial_net_worth if self.initial_net_worth != 0 else 1.0

        self.total_reward += reward
        self.reward_count += 1
        self.previous_net_worth = current_net_worth

        # Prepare info dictionary
        info = {
            'net_worth': round(current_net_worth, 2),
            'player_money': round(self.player.money, 2),
            'player_inventory': self.player.inventory.copy(),
            'days': self.total_days,
            'total_reward': round(self.total_reward, 4),
            'average_reward': round(self.total_reward / self.reward_count, 4) if self.reward_count > 0 else 0
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
                    'Food Inventory', 'Fuel Inventory', 'Clothes Inventory', 'Water Inventory'  # Added Water Inventory
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
                inventories = info['player_inventory']

                food_inventory = int(round(inventories.get('Food', 0)))
                fuel_inventory = int(round(inventories.get('Fuel', 0)))
                clothes_inventory = int(round(inventories.get('Clothes', 0)))
                water_inventory = int(round(inventories.get('Water', 0)))  # Added Water Inventory

                with open(self.csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.term_count, net_worth, player_money,
                        final_reward, average_reward,
                        food_inventory, fuel_inventory, clothes_inventory, water_inventory  # Added Water Inventory
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
                            file.write(f"Total Water: {totals['total_water']}\n")  # Added Total Water
                            file.write("\n")  # Add a blank line between days
        return True


class TurnEndCallback(BaseCallback):
    def __init__(self):
        super(TurnEndCallback, self).__init__()
        self.turn_end = False

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        if dones is not None and any(dones):
            self.turn_end = True
            return False  # Return False to stop training
        return True


if __name__ == "__main__":
    import os

    model_name = input("Enter model name: ")
    model_dir = model_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    num_agents = 2  # Number of AI agents
    total_days = 365  # Total days to simulate

    # Paths for models
    model_paths = [os.path.join(model_dir, f"economy_sim_ppo_model_agent_{i}") for i in range(num_agents)]
    csv_paths = [os.path.join(model_dir, f"economy_sim_results_agent_{i}.csv") for i in range(num_agents)]
    best_run_paths = [os.path.join(model_dir, f"best_run_agent_{i}.txt") for i in range(num_agents)]

    game = Game()  # Shared game instance

    players = [game.create_player() for _ in range(num_agents)]  # Create individual players
    envs = [EconomySimEnv(game, players[i]) for i in range(num_agents)]
    models = []

    # Initialize models and callbacks
    for i in range(num_agents):
        env = envs[i]
        model_path = model_paths[i]
        csv_path = csv_paths[i]
        best_run_path = best_run_paths[i]
        if os.path.exists(model_path + ".zip"):
            print(f"Loading model for Agent {i} from {model_path}")
            model = PPO.load(model_path, env=env, verbose=1)
        else:
            print(f"Creating new model for Agent {i}.")
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
        models.append(model)

    try:
        for day in range(total_days):
            print(f"Day {day + 1}")
            for agent_idx in range(num_agents):
                env = envs[agent_idx]
                model = models[agent_idx]
                model.set_env(env)
                # Create callbacks for logging
                csv_logger_callback = CSVLoggerCallback(csv_path=csv_paths[agent_idx])
                best_run_callback = BestRunLoggerCallback(best_run_path=best_run_paths[agent_idx])
                turn_end_callback = TurnEndCallback()
                # Train the model until the agent ends its turn
                model.learn(
                    total_timesteps=1000000,
                    callback=[csv_logger_callback, best_run_callback, turn_end_callback],
                    reset_num_timesteps=False
                )
                # Save the model
                model.save(model_paths[agent_idx])
            # After all agents have taken their turns, advance the day
            game.advance_day()
    except KeyboardInterrupt:
        print("Training interrupted. Saving models...")
        for i in range(num_agents):
            models[i].save(model_paths[i])
        pygame.quit()
    else:
        for i in range(num_agents):
            models[i].save(model_paths[i])
        pygame.quit()
