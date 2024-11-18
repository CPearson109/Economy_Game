# economyGame_simplified.py

import pygame
import sys
import matplotlib.pyplot as plt
import numpy as np

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simplified Economy Simulation")

# Fonts
FONT = pygame.font.SysFont("arial", 24, bold=True)
SMALL_FONT = pygame.font.SysFont("arial", 18)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (230, 230, 230)
ORANGE = (255, 127, 39)
RED = (255, 0, 0)

# Simulation settings
RESOURCES = ["Food", "Fuel", "Clothes", "Water"]
NUM_PEOPLE = 100
MAX_RESOURCE = {"Food": 30, "Fuel": 20, "Clothes": 10, "Water": 40}
CONSUMPTION_RATES = {"Food": 3, "Fuel": 2, "Clothes": 1, "Water": 4}
BASE_PRICES = {"Food": 3, "Fuel": 6, "Clothes": 9, "Water": 1}
PRODUCTION_PER_WORKER = {"Food": 15, "Fuel": 10, "Clothes": 5, "Water": 20}
WAGE_MULTIPLIERS = {"Food": 3, "Fuel": 3, "Clothes": 3, "Water": 3}
MINIMUM_WAGE = 10.0
MAX_WAGE_CHANGE = 0.1
WEIGHT = 0.3
REPRODUCTION_INTERVAL = 100
INFLATION_RATE = 0.003
PRICE_CHANGE_LIMIT = 1.5  # Allow max 50% daily change

def round_currency(value):
    return round(value + 1e-8, 2)  # Round to two decimal places

class Market:
    def __init__(self):
        initial_supply_multiplier = 5
        self.supply = {resource: (NUM_PEOPLE * CONSUMPTION_RATES[resource]) * initial_supply_multiplier for resource in RESOURCES}
        self.demand = {resource: NUM_PEOPLE * CONSUMPTION_RATES[resource] for resource in RESOURCES}
        self.prices = {
            resource: BASE_PRICES[resource] * (self.demand[resource] / self.supply[resource])
            for resource in RESOURCES
        }
        self.price_history = {resource: [self.prices[resource]] for resource in RESOURCES}
        self.previous_prices = self.prices.copy()

    def adjust_prices(self, day):
        for resource in RESOURCES:
            if self.supply[resource] == 0:
                self.supply[resource] = 1  # Prevent division by zero

            inflated_base_price = BASE_PRICES[resource] * ((1 + INFLATION_RATE) ** day)
            price = inflated_base_price * (self.demand[resource] / self.supply[resource])

            # Limit the price change
            price = min(price, self.prices[resource] * PRICE_CHANGE_LIMIT)
            price = max(price, self.prices[resource] / PRICE_CHANGE_LIMIT)

            # Smooth the price changes
            smoothed_price = (price * WEIGHT) + (self.prices[resource] * (1 - WEIGHT))

            # Update price history and current price
            self.price_history[resource].append(smoothed_price)
            self.previous_prices[resource] = self.prices[resource]
            self.prices[resource] = round_currency(smoothed_price)

class Person:
    def __init__(self):
        self.money = 100.00
        self.resources = {
            "Food": 15,
            "Fuel": 10,
            "Clothes": 5,
            "Water": 25
        }
        self.age = 0
        self.is_alive = True
        self.working = False
        self.factory = None
        self.consumption_rates = CONSUMPTION_RATES.copy()

    def consume_resources(self):
        for resource in RESOURCES:
            self.resources[resource] -= self.consumption_rates[resource]
            if self.resources[resource] < 0:
                self.is_alive = False
        self.age += 1

    def decide_to_work(self, market):
        money_threshold = 150
        resource_threshold = {resource: self.consumption_rates[resource] * 5 for resource in RESOURCES}
        needs_money_or_resources = self.money < money_threshold or any(
            self.resources[resource] < resource_threshold[resource] for resource in RESOURCES
        )

    def buy_resources(self, market):
        target_resource_level = {resource: self.consumption_rates[resource] * 5 for resource in RESOURCES}
        budget_allocation = {resource: 1 / len(RESOURCES) for resource in RESOURCES}
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
                market.supply[resource] -= amount_can_buy

    def reproduce(self):
        can_afford_child = all(
            self.resources[resource] >= self.consumption_rates[resource] * 3
            for resource in RESOURCES
        )
        if self.age > 18 and self.age % REPRODUCTION_INTERVAL == 0 and can_afford_child:
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

class Factory:
    def __init__(self, resource_type):
        self.resource_type = resource_type
        self.workers = []
        self.production_per_worker = PRODUCTION_PER_WORKER[resource_type]
        self.current_wage = MINIMUM_WAGE

    def calculate_wage(self, market):
        market_price = market.prices[self.resource_type]
        total_production_value = self.production_per_worker * market_price
        desired_wage = total_production_value * WAGE_MULTIPLIERS[self.resource_type]
        desired_wage = max(MINIMUM_WAGE, desired_wage)
        self.current_wage = round_currency(desired_wage)

    def accept_worker(self, person):
        self.workers.append(person)
        person.working = True
        person.factory = self

    def pay_workers(self):
        for person in self.workers:
            person.money += self.current_wage
            person.money = round_currency(person.money)
            person.working = False
            person.factory = None

    def produce(self, market):
        total_workers = len(self.workers)
        production_amount = self.production_per_worker * total_workers
        production_amount = int(production_amount)
        market.supply[self.resource_type] += production_amount
        self.workers = []

def draw_panel(win, x, y, width, height, title, content_items, bg_color, text_color):
    pygame.draw.rect(win, bg_color, (x, y, width, height), border_radius=15)
    pygame.draw.rect(win, BLACK, (x, y, width, height), 2, border_radius=15)
    title_surface = FONT.render(title, True, text_color)
    win.blit(title_surface, (x + 20, y + 20))
    offset = 60
    for item in content_items:
        line_surface = SMALL_FONT.render(item, True, text_color)
        win.blit(line_surface, (x + 20, y + offset))
        offset += 30

def draw_window(win, people, market, player, day, selected_resource):
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

    resource_prices = [f"{resource}: ${market.prices[resource]:.2f}" for resource in RESOURCES]
    draw_panel(win, WIDTH // 2 + 10, 20, panel_width, panel_height, "Market Prices", resource_prices, GRAY, BLACK)

    player_info = [
        f"Money: ${player.money:.2f}",
        *[f"{resource}: {player.inventory[resource]:.2f}" for resource in RESOURCES]
    ]
    draw_panel(win, 20, HEIGHT // 2 + 10, panel_width, panel_height, "Player Inventory", player_info, GRAY, BLACK)

    instructions = [
        "Press 'I' for Food",
        "Press 'O' for Fuel",
        "Press 'P' for Clothes",
        "Press 'U' for Water",
        "Press 'K' to Buy or 'L' to Sell resource.",
        f"Selected resource: {selected_resource}",
        "Press SPACE to Advance Day"
    ]
    draw_panel(win, WIDTH // 2 + 10, HEIGHT // 2 + 10, panel_width, panel_height, "Instructions", instructions, GRAY, BLACK)

    pygame.display.update()

def end_game(prices_history, days, player):
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
    people = [Person() for _ in range(NUM_PEOPLE)]
    factories = [Factory(resource) for resource in RESOURCES]
    market = Market()
    player = Player()
    day = 1
    selected_resource = "Food"
    days = [0]
    prices_history = {resource: [market.prices[resource]] for resource in RESOURCES}
    running = True
    game_over = False

    while running:
        draw_window(WIN, people, market, player, day, selected_resource)
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
                elif event.key == pygame.K_u:
                    selected_resource = "Water"
                if event.key == pygame.K_k:
                    amount = 1
                    player.buy(market, selected_resource, amount)
                elif event.key == pygame.K_l:
                    amount = 1
                    player.sell(market, selected_resource, amount)
                elif event.key == pygame.K_SPACE:
                    day += 1

                    # Corrected order of operations
                    # 1. Calculate wages for factories
                    for factory in factories:
                        factory.calculate_wage(market)

                    # 2. Assign workers to factories
                    for person in people:
                        if person.decide_to_work(market) and not person.working:
                            highest_wage_factory = max(factories, key=lambda f: f.current_wage)
                            highest_wage_factory.accept_worker(person)

                    # 3. Factories produce goods
                    for factory in factories:
                        factory.produce(market)

                    # 4. Pay workers
                    for factory in factories:
                        factory.pay_workers()

                    # 5. People buy resources
                    for person in people:
                        person.buy_resources(market)

                    # 6. People consume resources
                    for person in people:
                        if person.is_alive:
                            person.consume_resources()

                    # Remove dead people
                    people = [person for person in people if person.is_alive]

                    # Update market demand
                    for resource in RESOURCES:
                        market.demand[resource] = len(people) * CONSUMPTION_RATES[resource]

                    # Adjust market prices
                    market.adjust_prices(day)

                    # Reproduction
                    new_people = []
                    for person in people:
                        child = person.reproduce()
                        if child:
                            new_people.append(child)
                    people.extend(new_people)

                    days.append(day)
                    for resource in RESOURCES:
                        prices_history[resource].append(market.prices[resource])
                    if day >= 365:
                        game_over = True
                        end_game(prices_history, days, player)


if __name__ == "__main__":
    main()
