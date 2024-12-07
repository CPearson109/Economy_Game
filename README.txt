README

This project simulates a simplified economy with multiple agents (AI or player-controlled) interacting through buying,
selling, and working. Each agent has configurable parameters (risk taking, portfolio diversity, and investment horizon) influencing their trading strategies.

Key Features:

Resources: Food, Fuel, Clothes, Water.
Agents can buy, sell, and work to earn money.
Prices adjust dynamically based on supply and demand.
Player or AI agents (using the PPO algorithm) try to maximize net worth.
Option to advance one day at a time or fast-forward to day 365.
Controls:

Press I/O/P/W to select a resource (Food/Fuel/Clothes/Water).
Press K to buy or L to sell one unit of the selected resource.
Press SPACE to advance one day.
Press N to run the simulation until day 365.
Setup:

Run economyGame.py to start the simulation.
The AI agents are configured and trained via PPO.
Results and model files are stored in a specified directory.
Additional Files:

economyGame.py: Main simulation.
EconomySimEnv: Gym environment for AI training.
stable_baselines3 for AI model training.
Note:

Requires Python, pygame, gymnasium, stable-baselines3, and matplotlib.
Press CTRL+C to interrupt training. Models and progress will be saved.