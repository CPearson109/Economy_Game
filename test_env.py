import gymnasium as gym
import numpy as np
from gymnasium import spaces
from economyGame import Game  # Import your game class


class EconomyGameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # Initialize the economy game
        self.game = Game()

        # Define action space: 0 = no action, 1 = buy resource, 2 = sell resource
        self.action_space = spaces.Discrete(3)  # 3 discrete actions: No action, Buy, Sell

        # Define observation space: player money, resource prices, and inventory
        # Observation contains: player money, prices of 3 resources, and player inventory of 3 resources
        self.observation_space = spaces.Box(
            low=0, high=np.inf,
            shape=(7,),
            # Example: [money, food_price, fuel_price, clothes_price, food_inventory, fuel_inventory, clothes_inventory]
            dtype=np.float32
        )

        self.render_mode = render_mode

    def step(self, action):
        """Executes one step in the game and returns observation, reward, done, info."""
        # Process the action and update the game
        observation, reward, done, info = self.game.step(action)

        # You can also implement `truncated` if you need, but for simplicity, let's just return `done`
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state and returns the initial observation."""
        super().reset(seed=seed)
        self.game.reset()  # Reset the game to the initial state
        return self.game.get_observation(), {}  # Return the initial observation and an empty info dict

