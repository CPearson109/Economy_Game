import pygame
import random
import sys
import math
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 800  # Increased height to accommodate more text
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Economy Simulation")

# Fonts
FONT = pygame.font.SysFont("arial", 20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Simulation settings
NUM_PEOPLE = 50
MAX_RESOURCE = 20

# Resources and Consumption Rates
RESOURCES = ["Food", "Fuel", "Clothes"]
CONSUMPTION_RATES = {"Food": 3, "Fuel": 2, "Clothes": 1}

# Production per worker and wage multipliers per resource
PRODUCTION_PER_WORKER = {"Food": 12, "Fuel": 10, "Clothes": 8}
WAGE_MULTIPLIERS = {"Food": 1.5, "Fuel": 2.0, "Clothes": 2.5}

# Minimum wage for factories
MINIMUM_WAGE = 5.0  # Set to a very low value

class Market:
    def __init__(self):
        self.prices = {resource: 10.0 for resource in RESOURCES}
        self.supply = {resource: 500.0 for resource in RESOURCES}  # Start with 500 units of each resource
        self.demand = {resource: 0.0 for resource in RESOURCES}
        self.daily_supply = {resource: 0.0 for resource in RESOURCES}  # Supply added each day

    def adjust_prices(self):
        for resource in RESOURCES:
            if self.demand[resource] > self.supply[resource]:
                # Increase price
                self.prices[resource] *= 1.05  # Increase price by 5%
            elif self.demand[resource] < self.supply[resource]:
                # Decrease price
                self.prices[resource] *= 0.95  # Decrease price by 5%
            # Ensure price doesn't go negative
            self.prices[resource] = max(self.prices[resource], 0.01)
            # Reset daily supply and demand for the next day
            self.daily_supply[resource] = 0.0
            self.demand[resource] = 0.0

class Person:
    def __init__(self):
        self.money = random.uniform(30, 100)
        self.resources = {resource: random.uniform(5, 10) for resource in RESOURCES}
        self.is_alive = True
        self.max_resource = MAX_RESOURCE
        self.working = False
        self.factory = None

    def consume_resources(self):
        for resource in RESOURCES:
            self.resources[resource] -= CONSUMPTION_RATES[resource]
            if self.resources[resource] < 0:
                self.is_alive = False

    def decide_to_work(self, market):
        # Decide to work if money is less than a certain threshold
        if self.money < 200:
            return True
        else:
            return False

    def buy_resources(self, market):
        initial_budget = self.money
        budget_allocation = {'Food': 0.4, 'Fuel': 0.3, 'Clothes': 0.3}
        for resource in RESOURCES:
            amount_to_spend = initial_budget * budget_allocation[resource]
            affordable_amount = amount_to_spend / market.prices[resource]
            amount_can_buy = min(affordable_amount, market.supply[resource], self.max_resource - self.resources[resource])
            if amount_can_buy > 0:
                self.resources[resource] += amount_can_buy
                self.money -= amount_can_buy * market.prices[resource]
                market.demand[resource] += amount_can_buy
                market.supply[resource] -= amount_can_buy

    def reproduce(self):
        reproduction_chance = 0.01  # 1% chance to reproduce each day
        if random.random() < reproduction_chance and self.money > 50:
            self.money /= 2
            child = Person()
            child.money = self.money
            for resource in RESOURCES:
                self.resources[resource] /= 2
                child.resources[resource] = self.resources[resource]
            return child
        return None

class Factory:
    # Optional: Set a maximum number of workers per factory
    MAX_WORKERS = 20  # You can adjust or remove this limit

    def __init__(self, resource_type):
        self.resource_type = resource_type
        self.workers = []
        self.production_per_worker = PRODUCTION_PER_WORKER[resource_type]
        self.wage_multiplier = WAGE_MULTIPLIERS[resource_type]
        self.wage = 0.0
        self.days_unprofitable = 0
        self.money = 2000.0  # Increased starting capital

    def calculate_wage(self, market):
        # Expected production per worker
        expected_production_per_worker = self.production_per_worker
        expected_revenue_per_worker = expected_production_per_worker * market.prices[self.resource_type]
        # Adjusted desired profit margin
        desired_profit_margin = 0.3
        # Set wage so that wage = (revenue per worker) * (1 - profit margin)
        self.wage = max((1 - desired_profit_margin) * expected_revenue_per_worker, MINIMUM_WAGE)

    def accept_workers(self, applicants):
        # Calculate maximum workers factory can afford
        max_affordable_workers = int(self.money / self.wage)
        if self.MAX_WORKERS:
            max_affordable_workers = min(max_affordable_workers, self.MAX_WORKERS)
        if max_affordable_workers <= 0:
            return  # Can't afford any workers
        random.shuffle(applicants)
        self.workers = []
        for person in applicants:
            if len(self.workers) >= max_affordable_workers:
                break
            self.workers.append((person, self.wage))
            person.working = True
            person.factory = self

    def produce(self, market):
        total_workers = len(self.workers)
        # Linear production
        production_amount = self.production_per_worker * total_workers
        if production_amount > 0:
            market.supply[self.resource_type] += production_amount
            market.daily_supply[self.resource_type] = production_amount
        # Pay wages and calculate profit
        total_wages = self.wage * total_workers
        revenue = production_amount * market.prices[self.resource_type]
        profit = revenue - total_wages
        self.money += profit
        # Check profitability
        if profit <= 0:
            self.days_unprofitable += 1
        else:
            self.days_unprofitable = 0
        # Pay workers
        for person, wage in self.workers:
            person.money += wage
            person.working = False
            person.factory = None
        self.workers = []

    def should_close(self):
        return self.days_unprofitable >= 10  # Increased threshold

class CentralBank:
    def __init__(self):
        self.total_money = 0

    def inject_money(self, people):
        for person in people:
            person.money += 10
            self.total_money += 10 * len(people)

def draw_window(win, people, factories, market, day):
    win.fill(WHITE)
    y_offset = 10
    # Display day
    day_text = FONT.render(f"Day: {day}", True, BLACK)
    win.blit(day_text, (10, y_offset))
    y_offset += 30
    # Display population count
    population_text = FONT.render(f"Population: {len(people)}", True, BLACK)
    win.blit(population_text, (10, y_offset))
    y_offset += 30
    if len(people) > 0:
        # Display average money
        avg_money = sum(person.money for person in people) / len(people)
        avg_money_text = FONT.render(f"Average Money: ${avg_money:.2f}", True, BLACK)
        win.blit(avg_money_text, (10, y_offset))
        y_offset += 25
        # Display average resources
        avg_resources = {resource: sum(person.resources[resource] for person in people) / len(people) for resource in RESOURCES}
        for resource in RESOURCES:
            avg_resource_text = FONT.render(f"Avg {resource}: {avg_resources[resource]:.2f}", True, BLACK)
            win.blit(avg_resource_text, (10, y_offset))
            y_offset += 25
    else:
        avg_money_text = FONT.render("Average Money: N/A", True, BLACK)
        win.blit(avg_money_text, (10, y_offset))
        y_offset += 25
    # Display resource prices
    for resource in RESOURCES:
        price_text = FONT.render(f"{resource} Price: ${market.prices[resource]:.2f}", True, BLACK)
        win.blit(price_text, (10, y_offset))
        y_offset += 25
    # Display market supply
    for resource in RESOURCES:
        supply_text = FONT.render(f"{resource} Supply: {market.supply[resource]:.2f}", True, BLACK)
        win.blit(supply_text, (10, y_offset))
        y_offset += 25
    # Display total demand
    for resource in RESOURCES:
        total_demand = len(people) * CONSUMPTION_RATES[resource]
        demand_text = FONT.render(f"{resource} Total Demand: {total_demand:.2f}", True, BLACK)
        win.blit(demand_text, (10, y_offset))
        y_offset += 25
    # Display factory counts
    factory_counts = {resource: 0 for resource in RESOURCES}
    for factory in factories:
        factory_counts[factory.resource_type] += 1
    for resource in RESOURCES:
        factory_text = FONT.render(f"{resource} Factories: {factory_counts[resource]}", True, BLACK)
        win.blit(factory_text, (10, y_offset))
        y_offset += 25
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    people = [Person() for _ in range(NUM_PEOPLE)]
    factories = [Factory(resource) for resource in RESOURCES for _ in range(5)]
    market = Market()
    central_bank = CentralBank()
    day = 0

    # For plotting trends
    days = []
    prices_history = {resource: [] for resource in RESOURCES}

    MAX_FACTORIES_PER_RESOURCE = 10  # Maximum factories per resource

    running = True
    while running:
        clock.tick(60)
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Advance one day
                    day += 1
                    # Central bank injects money
                    central_bank.inject_money(people)
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
                    factory_applications = {factory: [] for factory in factories}
                    for person in people:
                        if person.decide_to_work(market) and not person.working:
                            # Determine the resource the person needs the most
                            needed_resources = {resource: person.max_resource - person.resources[resource] for resource in RESOURCES}
                            most_needed_resource = max(needed_resources, key=needed_resources.get)
                            # Find factories producing that resource
                            preferred_factories = [factory for factory in factories if factory.resource_type == most_needed_resource]
                            if preferred_factories:
                                # Choose the factory offering the highest wage
                                preferred_factory = max(preferred_factories, key=lambda f: f.wage)
                                if preferred_factory.wage >= MINIMUM_WAGE:
                                    factory_applications[preferred_factory].append(person)
                    # Factories accept workers
                    for factory in factories:
                        factory.accept_workers(factory_applications[factory])
                    # Factories produce goods (update market supply)
                    for factory in factories:
                        factory.produce(market)
                    # Remove unprofitable factories
                    factories = [factory for factory in factories if not factory.should_close()]
                    # Randomize purchase order
                    random.shuffle(people)
                    # People buy resources (update market demand)
                    for person in people:
                        person.buy_resources(market)
                    # Market adjusts prices based on updated supply and demand
                    market.adjust_prices()

                    # Demand-based factory creation
                    for resource in RESOURCES:
                        total_factories_of_resource = [factory for factory in factories if factory.resource_type == resource]
                        if market.demand[resource] > market.supply[resource] * 1.2:
                            if len(total_factories_of_resource) < MAX_FACTORIES_PER_RESOURCE:
                                new_factory = Factory(resource)
                                new_factory.money = 1000  # Initial capital
                                factories.append(new_factory)

                    # People reproduce
                    new_people = []
                    for person in people:
                        child = person.reproduce()
                        if child:
                            new_people.append(child)
                    people.extend(new_people)
                    # Check for simulation end
                    if len(people) == 0:
                        print("All people have died.")
                        running = False
                    # Record prices for plotting
                    days.append(day)
                    for resource in RESOURCES:
                        prices_history[resource].append(market.prices[resource])
        # Drawing
        draw_window(WIN, people, factories, market, day)
    # Plotting resource prices over time
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

if __name__ == "__main__":
    main()
