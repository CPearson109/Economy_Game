# ai_learn.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import csv
from economyGame import Game, RESOURCES, BASE_PRICES
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Initialize Pygame
pygame.init()


class EconomySimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EconomySimEnv, self).__init__()
        self.game = Game()
        max_amount = 100
        num_resources = len(RESOURCES)
        self.action_space = spaces.Box(
            low=np.array([0] * (2 * num_resources) + [0]),
            high=np.array([max_amount] * (2 * num_resources) + [1]),
            dtype=np.float32
        )

        obs_low = np.array([0.0] * (1 + num_resources * 4), dtype=np.float32)
        obs_high = np.array([np.inf] * (1 + num_resources * 4), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.previous_net_worth = None
        self.day_counter = 0
        self.total_reward = 0
        self.reward_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.day_counter = 0
        self.total_reward = 0
        self.reward_count = 0
        observation = self.get_observation()
        self.previous_net_worth = self.calculate_net_worth(observation)
        return observation, {}

    def get_observation(self):
        obs = [self.game.player.money]
        for resource in RESOURCES:
            obs.append(self.game.player.inventory[resource])
            obs.append(self.game.market.prices[resource])
            obs.append(self.game.market.supply[resource])
            obs.append(self.game.market.demand[resource])
        return np.array(obs, dtype=np.float32)

    def calculate_net_worth(self, observation):
        player_money = observation[0]
        net_worth = player_money + sum(
            observation[1 + i * 4] * observation[2 + i * 4]
            for i in range(len(RESOURCES))
        )
        return net_worth

    def calculate_reward(self, observation):
        current_net_worth = self.calculate_net_worth(observation)
        net_worth_gain = current_net_worth - self.previous_net_worth
        target_inventory = {"Food": 10, "Fuel": 5, "Clothes": 3}
        inventory_balance_reward = sum(
            -abs(observation[1 + i * 4] - target_inventory[RESOURCES[i]]) * 0.1
            for i in range(len(RESOURCES))
        )

        unused_resource_penalty = sum(
            -observation[1 + i * 4] * 0.05
            for i in range(len(RESOURCES))
        )

        trade_efficiency_reward = 0
        for i, resource in enumerate(RESOURCES):
            inventory_level = observation[1 + i * 4]
            market_price = observation[2 + i * 4]
            if inventory_level > target_inventory[resource] and market_price > BASE_PRICES[resource] * 1.2:
                trade_efficiency_reward += inventory_level * 0.1
            elif inventory_level < target_inventory[resource] and market_price < BASE_PRICES[resource] * 0.8:
                trade_efficiency_reward += (target_inventory[resource] - inventory_level) * 0.1

        bankruptcy_penalty = -100 if current_net_worth <= 0 else 0

        total_reward = (
                net_worth_gain +
                inventory_balance_reward +
                unused_resource_penalty +
                trade_efficiency_reward +
                bankruptcy_penalty
        )

        self.previous_net_worth = current_net_worth

        return total_reward

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        resources = RESOURCES
        num_resources = len(resources)

        for i, resource in enumerate(resources):
            buy_amount = action[i]
            sell_amount = action[i + num_resources]
            buy_quantity = int(np.round(buy_amount))
            sell_quantity = int(np.round(sell_amount))
            if buy_quantity > 0:
                self.game.player.buy(self.game.market, resource, buy_quantity)
            if sell_quantity > 0:
                self.game.player.sell(self.game.market, resource, sell_quantity)

        if action[-1] > 0.5:
            self.game.advance_day()
            self.day_counter += 1

        observation = self.get_observation()
        reward = self.calculate_reward(observation)

        self.total_reward += reward
        self.reward_count += 1

        done = self.day_counter >= 365 or self.calculate_net_worth(observation) <= 0

        info = {
            'reward': reward,
            'player_money': self.game.player.money,
            'net_worth': self.calculate_net_worth(observation),
            'days': self.day_counter,
            'player_inventory': self.game.player.inventory.copy(),
            'average_reward': self.total_reward / self.reward_count if self.reward_count > 0 else 0
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
        self.term_count = 0

        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Term', 'Final Reward', 'Average Reward', 'Final Net Worth', 'Final Cash',
                'Food Inventory', 'Food Value', 'Fuel Inventory', 'Fuel Value',
                'Clothes Inventory', 'Clothes Value'
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
                final_reward = info['reward']
                average_reward = info['average_reward']
                player_money = info['player_money']
                inventories = info['player_inventory']

                food_inventory = inventories['Food']
                fuel_inventory = inventories['Fuel']
                clothes_inventory = inventories['Clothes']

                # Fetch final market prices at the end of the term (day 365)
                food_price = self.locals['env'].envs[0].game.market.prices['Food']
                fuel_price = self.locals['env'].envs[0].game.market.prices['Fuel']
                clothes_price = self.locals['env'].envs[0].game.market.prices['Clothes']

                # Calculate the final value of each resource
                food_value = food_inventory * food_price
                fuel_value = fuel_inventory * fuel_price
                clothes_value = clothes_inventory * clothes_price

                with open(self.csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.term_count, final_reward, average_reward, net_worth, player_money,
                        food_inventory, food_value, fuel_inventory, fuel_value,
                        clothes_inventory, clothes_value
                    ])
        return True


if __name__ == "__main__":
    env = DummyVecEnv([lambda: EconomySimEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    csv_path = "economy_sim_results.csv"
    csv_logger_callback = CSVLoggerCallback(csv_path=csv_path)
    total_timesteps = 10000000
    model.learn(total_timesteps=total_timesteps, callback=csv_logger_callback)
    model.save("economy_sim_ppo_model")
    env.close()
    pygame.quit()
