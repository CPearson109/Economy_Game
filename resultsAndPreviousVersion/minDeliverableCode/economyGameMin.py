# economyGame.py
import pygame
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import false

clock = pygame.time.Clock()
FPS = 5

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600  # Adjusted height
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simplified Economy Simulation")

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

# Simulation settings
NUM_PEOPLE = 100  # Number of people in the simulation
MAX_RESOURCE = {"Food": 30, "Fuel": 20, "Clothes": 10}  # Max resources per person
CONSUMPTION_RATES = {"Food": 3, "Fuel": 2, "Clothes": 1}  # Daily consumption
BASE_PRICES = {"Food": 3, "Fuel": 6, "Clothes": 9}  # Base market prices

# Production per worker and wage multipliers per resource
PRODUCTION_PER_WORKER = {"Food": 11, "Fuel": 7, "Clothes": 4}
WAGE_MULTIPLIERS = {"Food": 1.5, "Fuel": 1.5, "Clothes": 1.5}

# Wage settings
MINIMUM_WAGE = 10.0  # Minimum wage for all factories
MAX_WAGE_CHANGE = 0.1  # Max wage decrease per day (percentage)

# Market settings
MOVING_AVERAGE = 10  # Moving average window size for smoothing prices
WEIGHT = 0.15  # Weight for exponential moving average price smoothing

# Reproduction chance
REPRODUCTION_CHANCE = 0.007  # Chance for people to reproduce each day

# Load images
try:
    # Background image (Replace 'background.png' with your background image)
    BACKGROUND_IMAGE = pygame.image.load('images/background.png')
    BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (WIDTH, HEIGHT))
except:
    BACKGROUND_IMAGE = None

try:
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

def round_currency(value):
    return round(value + 1e-8, 2)  # Round to two decimal places

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
        done = self.day >= 365  # end after 365 days

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

        # Update people in batches
        alive_people = []
        for person in self.people:
            person.buy_resources(self.market)
            person.consume_resources()
            if person.is_alive:
                alive_people.append(person)
        self.people = alive_people

        # Assign workers in batches
        potential_supply = {resource: self.market.supply[resource] for resource in RESOURCES}
        for factory in self.factories:
            factory.reset_worker_count()
        for person in self.people:
            if person.decide_to_work(self.market) and not person.working:
                self.assign_worker_to_factory(person)

        # Factory operations
        for factory in self.factories:
            factory.pay_workers()
            factory.produce(self.market)

        # Market adjustments
        for resource in RESOURCES:
            self.market.demand[resource] = len(self.people) * CONSUMPTION_RATES[resource]
        self.market.adjust_prices(self.day)

        # Reproduction
        self.people.extend(person.reproduce() for person in self.people if person.reproduce())

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

class Market:
    def __init__(self):
        self.supply = {resource: (NUM_PEOPLE * CONSUMPTION_RATES[resource]) * 3 for resource in RESOURCES}
        self.demand = {resource: NUM_PEOPLE * CONSUMPTION_RATES[resource] for resource in RESOURCES}
        self.prices = {resource: BASE_PRICES[resource] * (self.demand[resource] / self.supply[resource]) for resource in RESOURCES}
        self.price_history = {resource: [self.prices[resource]] for resource in RESOURCES}
        self.inflation_rate = 0.003  # Define an inflation rate of 0.3% per day
        self.previous_prices = {resource: BASE_PRICES[resource] for resource in RESOURCES}
        self.price_history = {resource: np.array([self.prices[resource]]) for resource in RESOURCES}

    def adjust_prices(self, day):
        for resource in RESOURCES:
            if self.supply[resource] == 0:
                self.supply[resource] = 1

            inflated_base_price = BASE_PRICES[resource] * ((1 + self.inflation_rate) ** day)
            price = inflated_base_price * (self.demand[resource] / self.supply[resource])
            max_price = inflated_base_price * 10
            price = min(price, max_price)
            price = max(price, inflated_base_price * 0.1)
            if len(self.price_history[resource]) == 0:
                smoothed_price = price
            else:
                smoothed_price = (price * WEIGHT) + (self.price_history[resource][-1] * (1 - WEIGHT))
            self.price_history[resource] = np.append(self.price_history[resource], smoothed_price)
            self.previous_prices[resource] = self.prices[resource]
            self.prices[resource] = round_currency(smoothed_price)

class Person:
    def __init__(self):
        self.money = 100.00
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
        money_threshold = 50
        resource_threshold = {resource: self.consumption_rates[resource] * 2 for resource in RESOURCES}
        needs_money_or_resources = self.money < money_threshold or any(
            self.resources[resource] < resource_threshold[resource] for resource in RESOURCES
        )
        resource_low_supply = any(
            market.supply[resource] < market.demand[resource] / 2 for resource in RESOURCES
        )
        wants_to_work = needs_money_or_resources or resource_low_supply
        random_factor = random.random()
        if random_factor < self.laziness_factor:
            return False
        return wants_to_work

    def buy_resources(self, market):
        critical_replenishment_level = {
            resource: self.consumption_rates[resource] * 4 for resource in RESOURCES
        }
        reproduction_target_level = {
            resource: self.consumption_rates[resource] * 6 for resource in RESOURCES
        }
        target_resource_level = {
            resource: reproduction_target_level[resource] if self.resources[resource] < critical_replenishment_level[resource]
            else self.consumption_rates[resource] * random.uniform(3, 7) for resource in RESOURCES
        }
        budget_allocation = {'Food': 0.6, 'Fuel': 0.4, 'Clothes': 0.2}
        total_budget = self.money * 0.8  # Spend up to 80% of current money

        for resource in RESOURCES:
            amount_needed = target_resource_level[resource] - self.resources[resource]
            if amount_needed <= 0:
                continue
            amount_to_spend = total_budget * budget_allocation[resource]
            price_per_unit = market.prices[resource]
            affordable_amount = amount_to_spend / price_per_unit
            amount_can_buy = min(affordable_amount, market.supply[resource], amount_needed)
            if amount_can_buy > 0:
                self.resources[resource] += amount_can_buy
                self.money -= amount_can_buy * price_per_unit
                self.money = round_currency(self.money)
                self.daily_spending[resource] += amount_can_buy * price_per_unit
                market.supply[resource] -= amount_can_buy

    def reproduce(self):
        can_afford_child = all(self.resources[resource] >= CONSUMPTION_RATES[resource] * 3 for resource in RESOURCES)
        if random.random() < REPRODUCTION_CHANCE and can_afford_child:
            child = Person()
            for resource in RESOURCES:
                self.resources[resource] /= 2
                child.resources[resource] = self.resources[resource]
            return child
        return None

class Player:
    def __init__(self):
        self.money = 100.00
        self.inventory = {resource: 0 for resource in RESOURCES}

    def buy(self, market, resource, amount):
        price = market.prices[resource]
        total_cost = price * amount
        total_cost = round_currency(total_cost)
        if total_cost <= self.money and market.supply[resource] >= amount:
            self.money -= total_cost
            self.money = round_currency(self.money)
            self.inventory[resource] += amount
            market.supply[resource] -= amount
            return True
        else:
            return False

    def sell(self, market, resource, amount):
        if self.inventory[resource] >= amount:
            price = market.prices[resource]
            total_earnings = price * amount
            total_earnings = round_currency(total_earnings)
            self.money += total_earnings
            self.money = round_currency(self.money)
            self.inventory[resource] -= amount
            market.supply[resource] += amount
            return True
        else:
            return False

def draw_progress_bar(win, x, y, width, height, progress, bg_color, fg_color):
    pygame.draw.rect(win, bg_color, (x, y, width, height))
    pygame.draw.rect(win, BLACK, (x, y, width, height), 1)
    inner_width = int(width * progress)
    pygame.draw.rect(win, fg_color, (x, y, inner_width, height))

def draw_panel(win, x, y, width, height, title, content_items, bg_color, text_color):
    pygame.draw.rect(win, bg_color, (x, y, width, height), border_radius=15)
    pygame.draw.rect(win, BLACK, (x, y, width, height), 2, border_radius=15)
    title_surface = FONT.render(title, True, text_color)
    win.blit(title_surface, (x + 20, y + 20))
    offset = 60
    for item in content_items:
        if isinstance(item, tuple):
            image, text = item
            if image:
                win.blit(image, (x + 20, y + offset))
                line_surface = SMALL_FONT.render(text, True, text_color)
                win.blit(line_surface, (x + 70, y + offset + 10))
            else:
                line_surface = SMALL_FONT.render(text, True, text_color)
                win.blit(line_surface, (x + 20, y + offset))
        else:
            line_surface = SMALL_FONT.render(item, True, text_color)
            win.blit(line_surface, (x + 20, y + offset))
        offset += 40

class Factory:
    def __init__(self, resource_type):
        self.resource_type = resource_type
        self.workers = []
        self.production_per_worker = PRODUCTION_PER_WORKER[resource_type]
        self.wage_multiplier = WAGE_MULTIPLIERS[resource_type]
        self.initial_wage = MINIMUM_WAGE
        self.current_wage = MINIMUM_WAGE
        self.previous_wage = MINIMUM_WAGE
        self.max_wage_change = MAX_WAGE_CHANGE
        self.worker_count = 0

    def calculate_wage(self, market):
        market_price = market.prices[self.resource_type]
        total_production_value = self.production_per_worker * market_price
        desired_wage = total_production_value * self.wage_multiplier
        if market.supply[self.resource_type] == 0:
            supply_demand_ratio = float('inf')
        else:
            supply_demand_ratio = market.demand[self.resource_type] / market.supply[self.resource_type]
        if supply_demand_ratio < 1:
            desired_wage *= 0.8
        elif supply_demand_ratio > 1.5:
            desired_wage *= 1.2
        if desired_wage > self.previous_wage:
            wage_growth = (desired_wage - self.previous_wage) * 0.3
            desired_wage = self.previous_wage + wage_growth
        desired_wage = max(MINIMUM_WAGE, desired_wage)
        self.initial_wage = round_currency(desired_wage)
        self.previous_wage = self.initial_wage
        self.current_wage = self.initial_wage

    def accept_worker(self, person):
        self.workers.append((person, self.current_wage))
        person.working = True
        person.factory = self
        self.current_wage *= 0.99  # Decrease wage by 1%
        self.current_wage = max(self.current_wage, MINIMUM_WAGE)
        self.current_wage = round_currency(self.current_wage)
        self.worker_count += 1

    def pay_workers(self):
        for person, wage in self.workers:
            person.money += wage
            person.money = round_currency(person.money)
            if person.money > 200:
                tax_rate = 0.1
                person.money -= wage * tax_rate
                person.money = round_currency(person.money)
            person.working = False
            person.factory = None

    def produce(self, market):
        total_workers = len(self.workers)
        production_amount = self.production_per_worker * total_workers
        production_variation = random.uniform(0.9, 1.1)
        production_amount *= production_variation
        production_amount = int(production_amount)
        if production_amount > 0:
            market.supply[self.resource_type] += production_amount
        self.workers = []

    def reset_worker_count(self):
        self.worker_count = 0

def draw_player_inventory(win, x, y, width, height, player):
    pygame.draw.rect(win, GRAY, (x, y, width, height), border_radius=15)
    pygame.draw.rect(win, BLACK, (x, y, width, height), 2, border_radius=15)
    title_surface = FONT.render("Player Inventory", True, BLACK)
    win.blit(title_surface, (x + 20, y + 20))
    money_surface = SMALL_FONT.render(f"Money: ${player.money:.2f}", True, BLACK)
    win.blit(money_surface, (x + 20, y + 60))
    offset = 100
    for resource in RESOURCES:
        inventory_surface = SMALL_FONT.render(f"{resource}: {player.inventory[resource]:.2f}", True, BLACK)
        win.blit(inventory_surface, (x + 20, y + offset))
        offset += 40

def draw_window(win, people, factories, market, player, day, end_button, selected_resource):
    if BACKGROUND_IMAGE:
        win.blit(BACKGROUND_IMAGE, (0, 0))
    else:
        win.fill(WHITE)
    panel_width = WIDTH // 2 - 30
    panel_height = HEIGHT // 2 - 40
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
    resource_prices = []
    for resource in RESOURCES:
        icon = pygame.transform.scale(RESOURCE_ICONS[resource], (40, 40)) if RESOURCE_ICONS[resource] else None
        text = f"${market.prices[resource]:.2f}"
        resource_prices.append((icon, text))
    draw_panel(win, WIDTH // 2 + 10, 20, panel_width, panel_height, "Market Prices", resource_prices, GRAY, BLACK)
    draw_player_inventory(win, 20, HEIGHT // 2 + 10, panel_width, panel_height, player)
    instructions = [
        "Press 'I' for Food",
        "Press 'O' for Fuel",
        "Press 'P' for Clothes",
        "Press K to Buy or L to Sell resource.",
        f"Selected resource: {selected_resource}",
        "Press SPACE to Advance Day"
    ]
    draw_panel(win, WIDTH // 2 + 10, HEIGHT // 2 + 10, panel_width, panel_height, "Instructions", instructions, GRAY, BLACK)
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

    pygame.quit()
    sys.exit()

def main():
    clock = pygame.time.Clock()
    people = [Person() for _ in range(NUM_PEOPLE)]
    factories = [Factory(resource) for resource in RESOURCES]
    market = Market()
    player = Player()
    day = 0
    selected_resource = None
    action = None
    days = []
    prices_history = {resource: [] for resource in RESOURCES}
    population_history = []
    avg_money_history = []
    avg_resources_history = {resource: [] for resource in RESOURCES}
    daily_spending_history = {resource: [] for resource in RESOURCES}
    end_button = Button("End Game", WIDTH - 150, HEIGHT - 70, 130, 40, ORANGE, RED, lambda: end_game(
        prices_history, population_history, avg_money_history, avg_resources_history, daily_spending_history, days, player))
    running = True
    game_over = False
    selected_resource = "Food"
    while running:
        clock.tick(FPS)
        draw_window(WIN, people, factories, market, player, day, end_button, selected_resource)
        end_button.check_click()
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
                if event.key == pygame.K_k:
                    amount = 1
                    player.buy(market, selected_resource, amount)
                elif event.key == pygame.K_l:
                    amount = 1
                    player.sell(market, selected_resource, amount)

                elif event.key == pygame.K_SPACE:
                    while game_over == false:
                        day += 1
                        #random_event.trigger_random_event(market, factories)
                        random.shuffle(people)
                        for person in people:
                            person.buy_resources(market)
                        for person in people:
                            if person.is_alive:
                                person.consume_resources()
                        people = [person for person in people if person.is_alive]
                        for factory in factories:
                            factory.calculate_wage(market)
                        random.shuffle(people)
                        potential_supply = {resource: market.supply[resource] for resource in RESOURCES}
                        factory_worker_counts = {factory: 0 for factory in factories}
                        for person in people:
                            if person.decide_to_work(market) and not person.working:
                                factory_wages = {factory: factory.current_wage for factory in factories}
                                all_min_wage = all(wage == MINIMUM_WAGE for wage in factory_wages.values())
                                if all_min_wage:
                                    potential_supplies_after_joining = {}
                                    for factory in factories:
                                        resource = factory.resource_type
                                        potential_supply_if_join = potential_supply[resource] + factory.production_per_worker * (
                                                    factory_worker_counts[factory] + 1)
                                        potential_supplies_after_joining[factory] = potential_supply_if_join
                                    selected_factory = min(potential_supplies_after_joining,
                                                           key=lambda f: potential_supplies_after_joining[f])
                                    selected_factory.accept_worker(person)
                                    factory_worker_counts[selected_factory] += 1
                                    resource = selected_factory.resource_type
                                    potential_supply[resource] += selected_factory.production_per_worker
                                else:
                                    highest_wage_factory = max(factory_wages, key=lambda f: factory_wages[f])
                                    highest_wage_factory.accept_worker(person)
                        daily_spending_total = {resource: sum(person.daily_spending[resource] for person in people) for resource in RESOURCES}
                        for person in people:
                            person.daily_spending = {resource: 0 for resource in RESOURCES}
                        for resource in RESOURCES:
                            daily_spending_history[resource].append(daily_spending_total[resource])
                        for factory in factories:
                            factory.pay_workers()
                        for factory in factories:
                            factory.produce(market)
                        for resource in RESOURCES:
                            market.demand[resource] = len(people) * CONSUMPTION_RATES[resource]
                        market.adjust_prices(day)
                        new_people = []
                        for person in people:
                            child = person.reproduce()
                            if child:
                                new_people.append(child)
                        people.extend(new_people)
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
                        if day >= 365:
                            game_over = True
                            end_game(prices_history, population_history, avg_money_history, avg_resources_history, daily_spending_history, days, player)

if __name__ == "__main__":
    main()