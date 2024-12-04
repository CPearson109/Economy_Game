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

    def __init__(self, game, player, risk_taking, portfolio_diversity_preference, investment_horizon):
        super(EconomySimEnv, self).__init__()
        self.game = game  # Shared game instance
        self.player = player  # Individual player instance
        self.num_resources = len(RESOURCES)

        # Store the custom parameters
        self.risk_taking = risk_taking  # 1 to 5
        self.portfolio_diversity_preference = portfolio_diversity_preference  # 1 to 5
        self.investment_horizon = investment_horizon  # 1 to 5

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

        # Adjust diversity bonus per resource based on preference
        self.diversity_bonus_per_resource = 0.2 * (self.portfolio_diversity_preference / 5)  # Scale from 0.04 to 0.2

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

        # Calculate net worth change
        net_worth_change = current_net_worth - self.previous_net_worth

        # Adjust risk multiplier based on risk_taking parameter
        risk_multiplier = 1 + ((self.risk_taking - 3) / 5)  # risk_taking from 1 to 5, multiplier from 0.6 to 1.4

        # Adjust investment horizon weight
        investment_horizon_weight = (6 - self.investment_horizon) / 5  # investment_horizon from 1 to 5, weight from 1.0 to 0.2

        # Initialize reward with adjusted net worth change
        reward = net_worth_change * risk_multiplier * investment_horizon_weight

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

    # Ask the user for the number of AI agents (up to 4)
    while True:
        try:
            num_agents = int(input("Enter the number of AI agents (1-4): "))
            if 1 <= num_agents <= 4:
                break
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a valid integer between 1 and 4.")

    # Collect customization parameters for each agent
    agent_parameters = []

    for i in range(num_agents):
        print(f"\nConfigure Agent {i}")
        while True:
            try:
                risk_taking = int(input(f"Enter Risk Taking level for Agent {i} (1-5): "))
                if 1 <= risk_taking <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a valid integer between 1 and 5.")

        while True:
            try:
                portfolio_diversity_preference = int(input(f"Enter Portfolio Diversity Preference for Agent {i} (1-5): "))
                if 1 <= portfolio_diversity_preference <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a valid integer between 1 and 5.")

        while True:
            try:
                investment_horizon = int(input(f"Enter Investment Horizon for Agent {i} (1-5): "))
                if 1 <= investment_horizon <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a valid integer between 1 and 5.")

        # Store the parameters
        agent_parameters.append({
            'risk_taking': risk_taking,
            'portfolio_diversity_preference': portfolio_diversity_preference,
            'investment_horizon': investment_horizon
        })

        # Save the parameters to a text file
        params_file_path = os.path.join(model_dir, f"agent_{i}_parameters.txt")
        with open(params_file_path, 'w') as f:
            f.write(f"Agent {i} Parameters:\n")
            f.write(f"Risk Taking: {risk_taking}\n")
            f.write(f"Portfolio Diversity Preference: {portfolio_diversity_preference}\n")
            f.write(f"Investment Horizon: {investment_horizon}\n")

    # Paths for models
    model_paths = [os.path.join(model_dir, f"economy_sim_ppo_model_agent_{i}") for i in range(num_agents)]
    csv_paths = [os.path.join(model_dir, f"economy_sim_results_agent_{i}.csv") for i in range(num_agents)]
    best_run_paths = [os.path.join(model_dir, f"best_run_agent_{i}.txt") for i in range(num_agents)]

    # Initialize term counter
    term_number = 1

    try:
        while True:
            print(f"\nStarting Term {term_number}")
            # Initialize game and players for the new term
            game = Game()  # Reset the game
            players = [game.create_player() for _ in range(num_agents)]  # Create new players

            # Initialize environments with custom parameters
            envs = []
            for i in range(num_agents):
                env = EconomySimEnv(
                    game,
                    players[i],
                    risk_taking=agent_parameters[i]['risk_taking'],
                    portfolio_diversity_preference=agent_parameters[i]['portfolio_diversity_preference'],
                    investment_horizon=agent_parameters[i]['investment_horizon']
                )
                envs.append(env)

            # Load or initialize models
            models = []
            for i in range(num_agents):
                env = envs[i]
                model_path = model_paths[i]
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

            # Run the game for 365 days
            for day in range(365):
                print(f"Day {game.day + 1}")
                for agent_idx in range(num_agents):
                    env = envs[agent_idx]
                    model = models[agent_idx]
                    model.set_env(env)
                    # Create callback for turn end
                    turn_end_callback = TurnEndCallback()
                    # Train the model until the agent ends its turn
                    model.learn(
                        total_timesteps=1000000,
                        callback=[turn_end_callback],
                        reset_num_timesteps=False
                    )
                    # Save the model
                    model.save(model_paths[agent_idx])
                # After all agents have taken their turns, advance the day
                game.advance_day()

            # Collect final results and determine winner
            net_worths = []
            for i in range(num_agents):
                env = envs[i]
                net_worth = env.calculate_net_worth()
                net_worths.append(net_worth)

            # Determine winner(s)
            max_net_worth = max(net_worths)
            winner_indices = [i for i, net_worth in enumerate(net_worths) if net_worth == max_net_worth]

            # Write results to CSV files
            for i in range(num_agents):
                env = envs[i]
                net_worth = net_worths[i]
                player_money = env.player.money
                inventories = env.player.inventory

                food_inventory = int(round(inventories.get('Food', 0)))
                fuel_inventory = int(round(inventories.get('Fuel', 0)))
                clothes_inventory = int(round(inventories.get('Clothes', 0)))
                water_inventory = int(round(inventories.get('Water', 0)))  # Added Water Inventory

                winner = 1 if i in winner_indices else 0

                csv_path = csv_paths[i]
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if os.stat(csv_path).st_size == 0:
                        writer.writerow([
                            'Term', 'Final Net Worth', 'Final Cash',
                            'Food Inventory', 'Fuel Inventory', 'Clothes Inventory', 'Water Inventory',
                            'Winner'
                        ])
                    writer.writerow([
                        term_number, round(net_worth, 2), round(player_money, 2),
                        food_inventory, fuel_inventory, clothes_inventory, water_inventory,
                        winner
                    ])

            # Increment term number
            term_number += 1

    except KeyboardInterrupt:
        print("Simulation interrupted. Saving models...")
        for i in range(num_agents):
            models[i].save(model_paths[i])
        pygame.quit()
