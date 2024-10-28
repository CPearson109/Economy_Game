#economyGame.py
import pygame
import random
import sys
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np
=======
import matplotlib.pyplot as pltssss
>>>>>>> 59ed3ff3538a1880d1e3892dc4147ab792ad41b4
=======
import matplotlib.pyplot as pltssss
>>>>>>> 59ed3ff3538a1880d1e3892dc4147ab792ad41b4
=======
import matplotlib.pyplot as pltssss
>>>>>>> 59ed3ff3538a1880d1e3892dc4147ab792ad41b4

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600  # Adjusted height
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simplified Economy Simulation with Player Interaction")

# Fonts
FONT = pygame.font.SysFont("arial", 24, bold=True)
SMALL_FONT = pygame.font.SysFont("arial", 18)

# Colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (230, 230, 230)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (34, 177, 76)
BLUE = (0, 162, 232)
ORANGE = (255, 127, 39)
YELLOW = (255, 242, 0)

RESOURCES = ["Food", "Fuel", "Clothes"]

# Simulation settings (Editable variables for tuning)
NUM_PEOPLE = 100  # Number of people in the simulation
MAX_RESOURCE = {"Food": 30, "Fuel": 20, "Clothes": 10}  # Maximum resources each person can store
CONSUMPTION_RATES = {"Food": 3, "Fuel": 2, "Clothes": 1}  # Daily consumption rate of each resource
BASE_PRICES = {"Food": 3, "Fuel": 6, "Clothes": 9}  # Base market prices for resources

# Production per worker and wage multipliers per resource
PRODUCTION_PER_WORKER = {"Food": 11, "Fuel": 7, "Clothes": 4}  # Production output per worker per day
WAGE_MULTIPLIERS = {"Food": 1.5, "Fuel": 1.5, "Clothes": 1.5}  # Multiplier for wages based on production value

# Wage settings
MINIMUM_WAGE = 10.0  # Minimum wage for all factories
MAX_WAGE_CHANGE = 0.1  # Maximum wage decrease per day (percentage, e.g., 10%)

# Market settings
MOVING_AVERAGE = 10  # Moving average window size for smoothing prices
WEIGHT = 0.15  # Weight for exponential moving average price smoothing

# Reproduction chance
REPRODUCTION_CHANCE = 0.007  # Chance for people to reproduce each day (0.01 = 1%)

# Load images
# Note: You should have the images in the 'images' directory.
# For example, you can use any images you like for the background and resource icons.

try:
    # Background image (Replace 'background.png' with your background image)
    BACKGROUND_IMAGE = pygame.image.load('images/background.png')
    BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (WIDTH, HEIGHT))
except:
    BACKGROUND_IMAGE = None

try:
    # Resource icons (Replace with your own icons)
    FOOD_ICON = pygame.image.load('images/food_icon.png')
    FUEL_ICON = pygame.image.load('images/fuel_icon.png')
    CLOTHES_ICON = pygame.image.load('images/clothes_icon.png')
except:
    FOOD_ICON = None
    FUEL_ICON = None
    CLOTHES_ICON = None

RESOURCE_ICONS = {
    'Food': FOOD_ICON,
    'Fuel': FUEL_ICON,
    'Clothes': CLOTHES_ICON
}

class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.people = [Person() for _ in range(NUM_PEOPLE)]
        self.factories = [Factory(resource) for resource in RESOURCES]
        self.market = Market()
        self.player = Player()
        self.day = 0
        self.random_event = RandomEvent()
        self.selected_resource = "Food"

        # Tracking variables for plotting (if needed)
        self.days = []
        self.prices_history = {resource: [] for resource in RESOURCES}
        self.population_history = []
        self.avg_money_history = []
        self.avg_resources_history = {resource: [] for resource in RESOURCES}
        self.daily_spending_history = {resource: [] for resource in RESOURCES}

    def reset(self):
        self.__init__()  # Re-initialize the game

    def step(self, action):
        # Process the AI's action
        self.process_action(action)

        # Advance the day
        self.advance_day()

        # Get the new observation
        observation = self.get_observation()

        # Calculate the reward
        reward = self.calculate_reward()

        # Check if the game is done
        done = self.day >= 365  # For example, end after 365 days

        # Additional info (can be empty)
        info = {}

        return observation, reward, done, info

    def process_action(self, action):
        # Define how the action affects the game
        if action == 1:  # Buy selected resource
            amount = 1
            self.player.buy(self.market, self.selected_resource, amount)
        elif action == 2:  # Sell selected resource
            amount = 1
            self.player.sell(self.market, self.selected_resource, amount)

    def advance_day(self):
        self.day += 1
        self.random_event.trigger_random_event(self.market, self.factories)
        random.shuffle(self.people)
        for person in self.people:
            person.buy_resources(self.market)
        for person in self.people:
            if person.is_alive:
                person.consume_resources()
        self.people = [person for person in self.people if person.is_alive]
        for factory in self.factories:
            factory.calculate_wage(self.market)
        random.shuffle(self.people)
        for person in self.people:
            if person.decide_to_work(self.market) and not person.working:
                self.assign_worker_to_factory(person)
        for factory in self.factories:
            factory.pay_workers()
        for factory in self.factories:
            factory.produce(self.market)
        for resource in RESOURCES:
            self.market.demand[resource] = len(self.people) * CONSUMPTION_RATES[resource]
        self.market.adjust_prices(self.day)
        new_people = []
        for person in self.people:
            child = person.reproduce()
            if child:
                new_people.append(child)
        self.people.extend(new_people)

    def assign_worker_to_factory(self, person):
        factory_wages = {factory: factory.current_wage for factory in self.factories}
        highest_wage_factory = max(factory_wages, key=lambda f: factory_wages[f])
        highest_wage_factory.accept_worker(person)

    def get_observation(self):
        observation = np.array([
            self.player.money,
            self.market.prices['Food'],
            self.player.inventory['Food'],
            self.market.prices['Fuel'],
            self.player.inventory['Fuel'],
            self.market.prices['Clothes'],
            self.player.inventory['Clothes']
        ], dtype=np.float32)
        return observation

    def calculate_reward(self):
        reward = self.player.money
        return reward

    # The render function for displaying the game state
    def render(self):
        # This is where we will use the draw_window function to display the current game state
        draw_window(WIN, self.people, self.factories, self.market, self.player, self.day, None, self.selected_resource)
        pygame.display.update()  # Update the display after drawing


# Button to end the game
class Button:
    def __init__(self, text, x, y, width, height, color, hover_color, action=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.action = action

    def draw(self, win):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            pygame.draw.rect(win, self.hover_color, self.rect, border_radius=10)
        else:
            pygame.draw.rect(win, self.color, self.rect, border_radius=10)
        pygame.draw.rect(win, BLACK, self.rect, 2, border_radius=10)  # Add border
        text_surface = SMALL_FONT.render(self.text, True, BLACK)
        win.blit(text_surface, (self.rect.x + (self.rect.width - text_surface.get_width()) // 2,
                                self.rect.y + (self.rect.height - text_surface.get_height()) // 2))

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0]:  # Left mouse button is pressed
                if self.action:
                    self.action()

class RandomEvent:
    def __init__(self):
        self.events = [
            {"name": "Drought", "resource": "Food", "impact": self.drought_event},
            {"name": "Fuel Shortage", "resource": "Fuel", "impact": self.fuel_shortage_event},
            {"name": "Clothing Shortage", "resource": "Clothes", "impact": self.clothes_shortage_event},
            {"name": "Economic Recession", "resource": "All", "impact": self.recession_event},
            {"name": "Technological Advancement", "resource": "Random", "impact": self.tech_advancement_event},
            {"name": "Fertile Season", "resource": "Food", "impact": self.fertile_season_event},
            {"name": "Fuel Abundance", "resource": "Fuel", "impact": self.fuel_abundance_event},
            {"name": "Clothing Boom", "resource": "Clothes", "impact": self.clothing_boom_event},
            {"name": "Economic Boom", "resource": "All", "impact": self.economic_boom_event},
            {"name": "Subsidy", "resource": "Random", "impact": self.subsidy_event},
        ]
        self.current_event = None

    def trigger_random_event(self, market, factories):
        if random.random() < 0.05:  # 5% chance of a random event happening each day
            self.current_event = random.choice(self.events)
            self.current_event["impact"](market, factories)

    def drought_event(self, market, factories):
        #print("Event: Drought! Reducing food supply.")
        market.supply["Food"] *= 0.7  # Reduce food supply by 30%

    def fuel_shortage_event(self, market, factories):
        #print("Event: Fuel Shortage! Reducing fuel supply.")
        market.supply["Fuel"] *= 0.6  # Reduce fuel supply by 40%

    def clothes_shortage_event(self, market, factories):
        #print("Event: Clothes Shortage! Reducing Clothes supply.")
        market.supply["Clothes"] *= 0.5  # Reduce Clothes supply by 50%

    def recession_event(self, market, factories):
        #print("Event: Economic Recession! Reducing wages.")
        for factory in factories:
            factory.current_wage *= 0.8  # Reduce wages by 20%

    # Positive Events

    def tech_advancement_event(self, market, factories):
        #print("Event: Technological Advancement! Increasing production.")
        random_resource = random.choice(RESOURCES)
        for factory in factories:
            if factory.resource_type == random_resource:
                factory.production_per_worker *= 1.2  # Increase production by 20%

    def fertile_season_event(self, market, factories):
        #print("Event: Fertile Season! Increasing food supply.")
        market.supply["Food"] *= 1.6  # Increase food supply by 60%

    def fuel_abundance_event(self, market, factories):
        #print("Event: Fuel Abundance! Increasing fuel supply.")
        market.supply["Fuel"] *= 1.5  # Increase fuel supply by 50%

    def clothing_boom_event(self, market, factories):
        #print("Event: Clothing Boom! Increasing clothes production.")
        market.supply["Clothes"] *= 1.4  # Increase clothes supply by 40%

    def economic_boom_event(self, market, factories):
        #print("Event: Economic Boom! Increasing wages and resource supply.")
        for factory in factories:
            factory.current_wage *= 1.2  # Increase wages by 20%
        for resource in RESOURCES:
            market.supply[resource] *= 1.5  # Increase resource supply by 50%

    def subsidy_event(self, market, factories):
        #print("Event: Government Subsidy! Reducing resource prices.")
        random_resource = random.choice(RESOURCES)
        market.prices[random_resource] *= 0.8  # Reduce price by 20%

class Market:
    def __init__(self):
        self.supply = {resource: (NUM_PEOPLE * CONSUMPTION_RATES[resource]) * 3 for resource in RESOURCES}
        self.demand = {resource: NUM_PEOPLE * CONSUMPTION_RATES[resource] for resource in RESOURCES}
        self.prices = {resource: BASE_PRICES[resource] * (self.demand[resource] / self.supply[resource]) for resource in RESOURCES}
        self.price_history = {resource: [self.prices[resource]] for resource in RESOURCES}
        self.inflation_rate = 0.003  # Define an inflation rate of 0.3% per day

    def adjust_prices(self, day):
        for resource in RESOURCES:
            # Prevent division by zero
            if self.supply[resource] == 0:
                self.supply[resource] = 1

            # Apply inflation: Base prices increase by inflation rate each day
            inflated_base_price = BASE_PRICES[resource] * ((1 + self.inflation_rate) ** day)

            # Calculate price based on individual supply and demand
            price = inflated_base_price * (self.demand[resource] / self.supply[resource])

            # Cap the price to prevent it from becoming too high
            max_price = inflated_base_price * 10  # Cap price at 10 times base price
            price = min(price, max_price)

            # Ensure price doesn't go below 10% of base price
            price = max(price, inflated_base_price * 0.1)

            # Check if history is empty and initialize if needed
            if len(self.price_history[resource]) == 0:
                smoothed_price = price  # If no history, start with the current price
            else:
                # EWMA calculation: current_price * weight + previous_average * (1 - weight)
                smoothed_price = (price * WEIGHT) + (self.price_history[resource][-1] * (1 - WEIGHT))

            # Append the smoothed price to price history
            self.price_history[resource].append(smoothed_price)

            # Update the resource price with the latest smoothed price
            self.prices[resource] = smoothed_price

class Person:
    def __init__(self):
        self.money = 100
        self.resources = {
            "Food": random.uniform(10, 20),
            "Fuel": random.uniform(5, 15),
            "Clothes": random.uniform(2, 8)
        }
        self.age = 0
        self.is_alive = True
        self.max_resource = MAX_RESOURCE
        self.working = False
        self.factory = None
        self.daily_spending = {resource: 0 for resource in RESOURCES}  # Track spending per resource
        # Randomized consumption rates (slight variation)
        self.consumption_rates = {
            "Food": CONSUMPTION_RATES["Food"] * random.uniform(0.8, 1.2),
            "Fuel": CONSUMPTION_RATES["Fuel"] * random.uniform(0.8, 1.2),
            "Clothes": CONSUMPTION_RATES["Clothes"] * random.uniform(0.8, 1.2)
        }
        self.laziness_factor = random.uniform(0.0, 0.2)

    def consume_resources(self):
        for resource in RESOURCES:
            self.resources[resource] -= CONSUMPTION_RATES[resource]
            if self.resources[resource] < 0:
                self.is_alive = False

    def decide_to_work(self, market):
        money_threshold = 50  # Lower money threshold to make them work less often
        resource_threshold = {resource: self.consumption_rates[resource] * 2 for resource in RESOURCES}

        # Determine if the person needs money or resources
        needs_money_or_resources = self.money < money_threshold or any(
            self.resources[resource] < resource_threshold[resource] for resource in RESOURCES
        )

        # Check if any resource is in low supply
        resource_low_supply = any(
            market.supply[resource] < market.demand[resource] / 2 for resource in RESOURCES
        )

        # Base decision: Needs money/resources or there's low supply
        wants_to_work = needs_money_or_resources or resource_low_supply

        # Random laziness factor: Even if they want to work, introduce a random chance they don't
        random_factor = random.random()  # Generates a random number between 0 and 1

        # If random_factor is below the laziness_factor, they decide not to work today
        if random_factor < self.laziness_factor:
            return False  # The person decides not to work

        return wants_to_work

    def buy_resources(self, market):
        # Critical threshold to prioritize resource purchasing for reproduction
        critical_replenishment_level = {
            resource: self.consumption_rates[resource] * 4 for resource in RESOURCES  # Adjust this multiplier as needed
        }

        # If reproduction is possible, prioritize reaching a target for reproduction
        reproduction_target_level = {
            resource: self.consumption_rates[resource] * 6 for resource in RESOURCES
        }

        # Randomized target replenishment levels (slight variation)
        target_resource_level = {
            resource: reproduction_target_level[resource] if self.resources[resource] < critical_replenishment_level[resource]
            else self.consumption_rates[resource] * random.uniform(3, 7) for resource in RESOURCES
        }

        # Allocate budget based on importance
        budget_allocation = {'Food': 0.6, 'Fuel': 0.4, 'Clothes': 0.2}
        total_budget = self.money * 0.8  # Spend up to 80% of current money

        for resource in RESOURCES:
            amount_needed = target_resource_level[resource] - self.resources[resource]
            if amount_needed <= 0:
                continue  # No need to buy more of this resource

            amount_to_spend = total_budget * budget_allocation[resource]
            price_per_unit = market.prices[resource]

            # Calculate how many units can be bought
            affordable_amount = amount_to_spend / price_per_unit
            amount_can_buy = min(affordable_amount, market.supply[resource], amount_needed)

            if amount_can_buy > 0:
                self.resources[resource] += amount_can_buy
                self.money -= amount_can_buy * price_per_unit
                self.daily_spending[resource] += amount_can_buy * price_per_unit
                market.supply[resource] -= amount_can_buy

    def reproduce(self):
        # Check if the person has enough resources to support a child
        can_afford_child = all(self.resources[resource] >= CONSUMPTION_RATES[resource] * 3 for resource in RESOURCES)

        # Proceed with reproduction only if the reproduction chance is met and resources are sufficient
        if random.random() < REPRODUCTION_CHANCE and can_afford_child:
            # Halve the parent's resources to share with the child
            child = Person()
            for resource in RESOURCES:
                self.resources[resource] /= 2  # Parent keeps half
                child.resources[resource] = self.resources[resource]  # Child gets half

            return child

        return None

class Player:
    def __init__(self):
        self.money = 100
        self.inventory = {resource: 0 for resource in RESOURCES}

    def buy(self, market, resource, amount):
        price = market.prices[resource]
        total_cost = price * amount
        if total_cost <= self.money and market.supply[resource] >= amount:
            self.money -= total_cost
            self.inventory[resource] += amount
            market.supply[resource] -= amount
            #print(f"Bought {amount} units of {resource} for ${total_cost:.2f}")
        #else:
            #("Cannot complete the purchase.")

    def sell(self, market, resource, amount):
        if self.inventory[resource] >= amount:
            price = market.prices[resource]
            total_earnings = price * amount
            self.money += total_earnings
            self.inventory[resource] -= amount
            market.supply[resource] += amount
            #print(f"Sold {amount} units of {resource} for ${total_earnings:.2f}")
        #else:
            #print("Cannot complete the sale.")

def draw_progress_bar(win, x, y, width, height, progress, bg_color, fg_color):
    # Draw background
    pygame.draw.rect(win, bg_color, (x, y, width, height))
    # Draw border
    pygame.draw.rect(win, BLACK, (x, y, width, height), 1)
    # Draw progress
    inner_width = int(width * progress)
    pygame.draw.rect(win, fg_color, (x, y, inner_width, height))

def draw_panel(win, x, y, width, height, title, content_items, bg_color, text_color):
    # Draw panel background with rounded corners
    pygame.draw.rect(win, bg_color, (x, y, width, height), border_radius=15)
    # Draw panel border
    pygame.draw.rect(win, BLACK, (x, y, width, height), 2, border_radius=15)
    # Render title
    title_surface = FONT.render(title, True, text_color)
    win.blit(title_surface, (x + 20, y + 20))
    # Render content lines
    offset = 60  # Increase offset for better spacing
    for item in content_items:
        if isinstance(item, tuple):
            image, text = item
            if image:
                # Draw the image
                win.blit(image, (x + 20, y + offset))
                # Draw the text next to the image
                line_surface = SMALL_FONT.render(text, True, text_color)
                win.blit(line_surface, (x + 70, y + offset + 10))  # Adjust position
            else:
                # No image, just text
                line_surface = SMALL_FONT.render(text, True, text_color)
                win.blit(line_surface, (x + 20, y + offset))
        else:
            # item is just text
            line_surface = SMALL_FONT.render(item, True, text_color)
            win.blit(line_surface, (x + 20, y + offset))
        offset += 40  # Increase offset for next line

class Factory:
    def __init__(self, resource_type):
        self.resource_type = resource_type
        self.workers = []
        self.production_per_worker = PRODUCTION_PER_WORKER[resource_type]
        self.wage_multiplier = WAGE_MULTIPLIERS[resource_type]
        self.initial_wage = MINIMUM_WAGE
        self.current_wage = MINIMUM_WAGE
        self.previous_wage = MINIMUM_WAGE  # Initialize with minimum wage
        self.max_wage_change = MAX_WAGE_CHANGE  # Maximum wage decrease per day (e.g., 10%)
        self.worker_count = 0  # Track number of workers accepted

    def calculate_wage(self, market):
        # Use the smoothed market price
        market_price = market.prices[self.resource_type]

        # Calculate the desired wage based on production value and wage multiplier
        total_production_value = self.production_per_worker * market_price
        desired_wage = total_production_value * self.wage_multiplier

        # Avoid division by zero in the supply/demand ratio
        if market.supply[self.resource_type] == 0:
            supply_demand_ratio = float('inf')  # If supply is zero, assume demand is much higher
        else:
            supply_demand_ratio = market.demand[self.resource_type] / market.supply[self.resource_type]

        # Adjust wages based on resource supply/demand: high supply = lower wage, low supply = higher wage
        if supply_demand_ratio < 1:  # If supply exceeds demand, reduce wage
            desired_wage *= 0.8  # Reduce wages by 20% when there's excess supply
        elif supply_demand_ratio > 1.5:  # If demand greatly exceeds supply, increase wage
            desired_wage *= 1.2  # Increase wages by 20% when demand is much higher

        # Cap wage growth to slow down excessive wage increase
        if desired_wage > self.previous_wage:
            # Introduce diminishing returns on wage increases
            wage_growth = (desired_wage - self.previous_wage) * 0.3  # Limit wage increase to 30% of the difference
            desired_wage = self.previous_wage + wage_growth

        # Ensure wage is capped at a reasonable maximum and doesn't drop below the minimum wage
        desired_wage = max(MINIMUM_WAGE, desired_wage)

        # Update wages for the factory
        self.initial_wage = desired_wage
        self.previous_wage = desired_wage
        self.current_wage = self.initial_wage  # Reset current wage to initial wage for the day

    def accept_worker(self, person):
        # Offer the current wage
        self.workers.append((person, self.current_wage))
        person.working = True
        person.factory = self

        # Decrease the wage slightly for the next worker
        self.current_wage *= 0.99  # Decrease wage by 1%

        # Ensure wage does not go below MINIMUM_WAGE
        self.current_wage = max(self.current_wage, MINIMUM_WAGE)

        # Increment worker count
        self.worker_count += 1

    def pay_workers(self):
        for person, wage in self.workers:
            person.money += wage

            # Progressive tax or reduce savings rate
            if person.money > 200:  # If money exceeds 200, tax them or reduce income
                tax_rate = 0.1  # Tax 10% of income
                person.money -= wage * tax_rate

            person.working = False
            person.factory = None

    def produce(self, market):
        total_workers = len(self.workers)
        production_amount = self.production_per_worker * total_workers

        # Add randomness to production output (e.g., +/- 10%)
        production_variation = random.uniform(0.9, 1.1)  # 10% random variation
        production_amount *= production_variation
        production_amount = int(production_amount)  # Ensure integer production

        if production_amount > 0:
            market.supply[self.resource_type] += production_amount
        self.workers = []

    def reset_worker_count(self):
        # Reset the worker count to zero at the start of each day
        self.worker_count = 0

def draw_player_inventory(win, x, y, width, height, player):
    # Draw panel background with rounded corners
    pygame.draw.rect(win, GRAY, (x, y, width, height), border_radius=15)
    # Draw panel border
    pygame.draw.rect(win, BLACK, (x, y, width, height), 2, border_radius=15)
    # Render title
    title_surface = FONT.render("Player Inventory", True, BLACK)
    win.blit(title_surface, (x + 20, y + 20))
    # Render money
    money_surface = SMALL_FONT.render(f"Money: ${player.money:.2f}", True, BLACK)
    win.blit(money_surface, (x + 20, y + 60))
    # Render resources as simple counts
    offset = 100
    for resource in RESOURCES:
        # Render resource count next to resource name
        inventory_surface = SMALL_FONT.render(f"{resource}: {player.inventory[resource]:.2f}", True, BLACK)
        win.blit(inventory_surface, (x + 20, y + offset))
        offset += 40


def draw_window(win, people, factories, market, player, day, end_button, selected_resource):
    if BACKGROUND_IMAGE:
        win.blit(BACKGROUND_IMAGE, (0, 0))
    else:
        win.fill(WHITE)  # Fallback to white background if no background image

    # Panel dimensions
    panel_width = WIDTH // 2 - 30
    panel_height = HEIGHT // 2 - 40

    # Top Left Panel - General Info
    general_info = [
        f"Day: {day}",
        f"Population: {len(people)}",
    ]
    if people:
        avg_money = sum(person.money for person in people) / len(people)
        general_info.append(f"Avg Money: ${avg_money:.2f}")
    else:
        general_info.append("Avg Money: N/A")
    draw_panel(win, 20, 20, panel_width, panel_height, "General Info", general_info, GRAY, BLACK)

    # Top Right Panel - Resource Prices
    resource_prices = []
    for resource in RESOURCES:
        icon = pygame.transform.scale(RESOURCE_ICONS[resource], (40, 40)) if RESOURCE_ICONS[resource] else None
        text = f"${market.prices[resource]:.2f}"
        resource_prices.append((icon, text))
    draw_panel(win, WIDTH // 2 + 10, 20, panel_width, panel_height, "Market Prices", resource_prices, GRAY, BLACK)

    # Bottom Left Panel - Player Info
    draw_player_inventory(win, 20, HEIGHT // 2 + 10, panel_width, panel_height, player)

    # Bottom Right Panel - Instructions
    instructions = [
        "Press 'I' for Food",
        "Press 'O' for Fuel",
        "Press 'P' for Clothes",
        "Press K to Buy or L to Sell resource.",
        f"Selected resource: {selected_resource}",
        "Press SPACE to Advance Day"
    ]
    draw_panel(win, WIDTH // 2 + 10, HEIGHT // 2 + 10, panel_width, panel_height, "Instructions", instructions, GRAY, BLACK)

    # Only draw the end game button if it's provided
    if end_button:
        end_button.draw(win)

    pygame.display.update()


def end_game(prices_history, population_history, avg_money_history, avg_resources_history, daily_spending_history, days, player):

    print("Game Over!!!")
    print("Total Money: ", player.money)

    plt.figure(figsize=(10, 5))
    for resource in RESOURCES:
        plt.plot(days, prices_history[resource], label=f"{resource} Price")
    plt.legend()
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.title("Resource Prices Over Time")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(days, population_history, label="Population")
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title("Population Over Time")
    plt.show()

    # Plot daily spending per resource over time
    plt.figure(figsize=(10, 5))
    spending_arrays = [daily_spending_history[resource] for resource in RESOURCES]
    plt.stackplot(days, *spending_arrays, labels=RESOURCES)
    plt.xlabel("Day")
    plt.ylabel("Total Spending")
    plt.title("Total Spending per Resource Over Time")
    plt.legend()
    plt.show()

    pygame.quit()
    sys.exit()

def main():
    clock = pygame.time.Clock()
    people = [Person() for _ in range(NUM_PEOPLE)]
    factories = [Factory(resource) for resource in RESOURCES]
    market = Market()
    player = Player()
    day = 0
    random_event = RandomEvent()  # Create random event system

    selected_resource = None  # Variable to store the selected resource
    action = None  # Variable to store whether user wants to buy or sell

    # For plotting trends
    days = []
    prices_history = {resource: [] for resource in RESOURCES}
    population_history = []
    avg_money_history = []
    avg_resources_history = {resource: [] for resource in RESOURCES}
    daily_spending_history = {resource: [] for resource in RESOURCES}

    # Create the End Game button
    end_button = Button("End Game", WIDTH - 150, HEIGHT - 70, 130, 40, ORANGE, RED, lambda: end_game(
        prices_history, population_history, avg_money_history, avg_resources_history, daily_spending_history, days, player))

    running = True
    game_over = False  # Added this flag to trigger the end of the game

    selected_resource = "Food"

    while running:
        clock.tick(60)

        # Drawing
        draw_window(WIN, people, factories, market, player, day, end_button, selected_resource)
        end_button.check_click()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    selected_resource = "Food"
                elif event.key == pygame.K_o:
                    selected_resource = "Fuel"
                elif event.key == pygame.K_p:
                    selected_resource = "Clothes"

                if event.key == pygame.K_k:  # 'K' for buying 1 unit
                    amount = 1  # Set the amount to 1 unit to buy
                    player.buy(market, selected_resource, amount)
                elif event.key == pygame.K_l:  # 'L' for selling 1 unit
                    amount = 1  # Set the amount to 1 unit to sell
                    player.sell(market, selected_resource, amount)

                #elif event.key == pygame.K_SPACE:
                while day < 365:
                    # Advance day
                    day += 1
                    # Trigger random event with a 5% chance
                    random_event.trigger_random_event(market, factories)
                    # Randomize purchase order
                    random.shuffle(people)
                    # People buy resources
                    for person in people:
                        person.buy_resources(market)
                    # People consume resources
                    for person in people:
                        if person.is_alive:
                            person.consume_resources()
                    # Remove dead people
                    people = [person for person in people if person.is_alive]
                    # Factories calculate wages
                    for factory in factories:
                        factory.calculate_wage(market)
                    # Workers decide where to work
                    random.shuffle(people)

                    # Initialize potential supplies as current supplies
                    potential_supply = {resource: market.supply[resource] for resource in RESOURCES}
                    # Initialize worker counts at factories
                    factory_worker_counts = {factory: 0 for factory in factories}

                    for person in people:
                        if person.decide_to_work(market) and not person.working:
                            # Get the current wages offered by factories
                            factory_wages = {factory: factory.current_wage for factory in factories}

                            # Check if all factories are offering minimum wage
                            all_min_wage = all(wage == MINIMUM_WAGE for wage in factory_wages.values())

                            if all_min_wage:
                                # If all wages are at minimum, assign workers based on potential supply

                                # For each factory, calculate potential supply if the worker joins
                                potential_supplies_after_joining = {}
                                for factory in factories:
                                    resource = factory.resource_type
                                    # Potential supply if the worker joins this factory
                                    potential_supply_if_join = potential_supply[resource] + factory.production_per_worker * (
                                                factory_worker_counts[factory] + 1)
                                    potential_supplies_after_joining[factory] = potential_supply_if_join

                                # The worker chooses the factory where the potential supply is lowest after joining
                                selected_factory = min(potential_supplies_after_joining,
                                                       key=lambda f: potential_supplies_after_joining[f])

                                # Assign the worker to the selected factory
                                selected_factory.accept_worker(person)

                                # Update worker counts
                                factory_worker_counts[selected_factory] += 1

                                # Update potential supply for the resource
                                resource = selected_factory.resource_type
                                potential_supply[resource] += selected_factory.production_per_worker
                            else:
                                # Select the factory offering the highest wage
                                highest_wage_factory = max(factory_wages, key=lambda f: factory_wages[f])
                                # Factory accepts worker
                                highest_wage_factory.accept_worker(person)

                    # Track daily spending per resource
                    daily_spending_total = {resource: sum(person.daily_spending[resource] for person in people) for resource in RESOURCES}

                    # Reset daily spending for each person
                    for person in people:
                        person.daily_spending = {resource: 0 for resource in RESOURCES}

                    # Record daily spending for plotting later
                    for resource in RESOURCES:
                        daily_spending_history[resource].append(daily_spending_total[resource])

                    # Factories pay workers
                    for factory in factories:
                        factory.pay_workers()
                    # Factories produce goods
                    for factory in factories:
                        factory.produce(market)
                    # Calculate market demand based on consumption rates
                    for resource in RESOURCES:
                        market.demand[resource] = len(people) * CONSUMPTION_RATES[resource]
                    # Market adjusts prices
                    market.adjust_prices(day)
                    # People reproduce
                    new_people = []
                    for person in people:
                        child = person.reproduce()
                        if child:
                            new_people.append(child)
                    people.extend(new_people)

                    # Record data for plotting
                    days.append(day)
                    for resource in RESOURCES:
                        prices_history[resource].append(market.prices[resource])
                    population_history.append(len(people))
                    avg_money_history.append(sum(person.money for person in people) / len(people) if people else 0)
                    for resource in RESOURCES:
                        avg_resources_history[resource].append(
                            sum(person.resources[resource] for person in people) / len(people) if people else 0
                        )

                    pygame.event.pump()

                    # Check for simulation end after 1 year
                    if day >= 365:
                        game_over = True  # Set flag to stop the simulation
                        end_game(prices_history, population_history, avg_money_history, avg_resources_history, daily_spending_history, days, player)

if __name__ == "__main__":
    main()
