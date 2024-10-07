import pygame
import random
import sys
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 700  # Increased height to fit additional information
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simplified Economy Simulation")

# Fonts
FONT = pygame.font.SysFont("arial", 18)

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
MINIMUM_WAGE = 10.0

class Market:
    def __init__(self):
        self.prices = {resource: 10.0 for resource in RESOURCES}
        self.supply = {resource: 500.0 for resource in RESOURCES}
        self.demand = {resource: 0.0 for resource in RESOURCES}

    def adjust_prices(self):
        for resource in RESOURCES:
            if self.demand[resource] > self.supply[resource]:
                # Increase price
                self.prices[resource] *= 1.05
            elif self.demand[resource] < self.supply[resource]:
                # Decrease price
                self.prices[resource] *= 0.95
            # Ensure price doesn't go negative
            self.prices[resource] = max(self.prices[resource], 0.01)
            # Reset daily demand
            self.demand[resource] = 0.0

class Person:
    def __init__(self):
        self.money = random.uniform(30, 100)
        self.resources = {resource: random.uniform(5, 20) for resource in RESOURCES}
        self.is_alive = True
        self.max_resource = MAX_RESOURCE
        self.working = False
        self.factory = None

    def consume_resources(self):
        for resource in RESOURCES:
            self.resources[resource] -= CONSUMPTION_RATES[resource]
            if self.resources[resource] < 0:
                self.is_alive = False

    def decide_to_work(self):
        return self.money < 200

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
        reproduction_chance = 0.01
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
    def __init__(self, resource_type):
        self.resource_type = resource_type
        self.workers = []
        self.production_per_worker = PRODUCTION_PER_WORKER[resource_type]
        self.wage_multiplier = WAGE_MULTIPLIERS[resource_type]
        self.initial_wage = MINIMUM_WAGE
        self.current_wage = MINIMUM_WAGE
        self.worker_count = 0  # Track number of workers accepted

    def calculate_wage(self, market):
        expected_revenue_per_worker = self.production_per_worker * market.prices[self.resource_type]
        desired_profit_margin = 0.3
        # Set initial wage based on the value of the material produced
        self.initial_wage = max((1 - desired_profit_margin) * expected_revenue_per_worker, MINIMUM_WAGE)
        # Reset current wage to initial wage at the start of the day
        self.current_wage = self.initial_wage

    def accept_workers(self, applicants):
        for person in applicants:
            # Offer the current wage
            self.workers.append((person, self.current_wage))
            person.working = True
            person.factory = self
            # Decrease the wage slightly for the next worker
            self.current_wage *= 0.99  # Decrease wage by 1%
            # Ensure wage does not go below MINIMUM_WAGE
            if self.current_wage < MINIMUM_WAGE:
                self.current_wage = MINIMUM_WAGE
            # Increment worker count
            self.worker_count += 1

    def produce(self, market):
        total_workers = len(self.workers)
        production_amount = self.production_per_worker * total_workers
        if production_amount > 0:
            market.supply[self.resource_type] += production_amount
        # Pay wages and calculate profit
        total_wages = sum(wage for _, wage in self.workers)
        revenue = production_amount * market.prices[self.resource_type]
        profit = revenue - total_wages
        # Pay workers and clear them for the next day
        for person, wage in self.workers:
            person.money += wage
            person.working = False
            person.factory = None
        self.workers = []

    def reset_worker_count(self):
        # Reset the worker count to zero at the start of each day
        self.worker_count = 0


def draw_window(win, people, factories, market, day):
    win.fill(WHITE)
    y_offset = 10
    # Display day
    day_text = FONT.render(f"Day: {day}", True, BLACK)
    win.blit(day_text, (10, y_offset))
    y_offset += 25
    # Display population count
    population_text = FONT.render(f"Population: {len(people)}", True, BLACK)
    win.blit(population_text, (10, y_offset))
    y_offset += 25
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
    # Display supply and demand for each resource
    for resource in RESOURCES:
        # Calculate total demand
        total_demand = CONSUMPTION_RATES[resource] * len(people)
        supply_text = FONT.render(f"{resource} Supply: {market.supply[resource]:.2f}", True, BLACK)
        win.blit(supply_text, (10, y_offset))
        y_offset += 25

    # Display factory worker counts and wages
    y_offset += 10
    factory_header = FONT.render("Factories and Worker Counts:", True, BLACK)
    win.blit(factory_header, (10, y_offset))
    y_offset += 25
    for factory in factories:
        workers_text = FONT.render(f"{factory.resource_type} Factory: {len(factory.workers)} workers (Total accepted: {factory.worker_count})", True, BLACK)
        win.blit(workers_text, (10, y_offset))
        y_offset += 25
        wage_text = FONT.render(f"Starting Wage: ${factory.initial_wage:.2f}", True, BLACK)
        win.blit(wage_text, (10, y_offset))
        y_offset += 25
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    people = [Person() for _ in range(NUM_PEOPLE)]
    # Initialize one factory per resource
    factories = [Factory(resource) for resource in RESOURCES]
    market = Market()
    day = 0


    # For plotting trends
    days = []
    prices_history = {resource: [] for resource in RESOURCES}

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
                    for factory in factories:
                        factory.reset_worker_count()
                    # Advance one day
                    day += 1
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
                        if person.decide_to_work() and not person.working:
                            #compare wages across all factories
                            highest_wage_factory = max(factories, key=lambda factory: factory.current_wage)
                            #apply to highest paying factory
                            factory_applications[highest_wage_factory].append(person)

                    # Factories accept workers
                    for factory in factories:
                        applicants = factory_applications.get(factory, [])
                        factory.accept_workers(applicants)
                    # Factories produce goods
                    for factory in factories:
                        factory.produce(market)
                    # Randomize purchase order
                    random.shuffle(people)
                    # People buy resources
                    for person in people:
                        person.buy_resources(market)
                    # Market adjusts prices
                    market.adjust_prices()
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
        # Reset worker counts for each factory at the end of the day
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
