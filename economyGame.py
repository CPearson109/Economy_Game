import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 800  # Increased height to accommodate more text
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Basic Economy Simulation")

# Fonts
FONT = pygame.font.SysFont("arial", 20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Simulation settings
NUM_PEOPLE = 50
NUM_FACTORIES = 3
MAX_RESOURCE = 20

# Resources and Consumption Rates
RESOURCES = ["Food", "Fuel", "Clothes"]
CONSUMPTION_RATES = {"Food": 3, "Fuel": 2 , "Clothes": 1}

# Production per worker and wage multipliers per resource
PRODUCTION_PER_WORKER = {"Food": 12, "Fuel": 8, "Clothes": 4}
WAGE_MULTIPLIERS = {"Food": 6.0, "Fuel": 8.0, "Clothes": 8.0}

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
            total_supply = self.supply[resource]
            total_demand = self.demand[resource]

            # Avoid division by zero
            if total_supply + total_demand == 0:
                price_change = 0
            else:
                # Calculate the excess demand as a fraction
                excess_demand_fraction = (total_demand - total_supply) / (total_supply + total_demand + 1e-5)

                # Limit the excess demand fraction to be between -1 and +1
                excess_demand_fraction = max(min(excess_demand_fraction, 1), -1)

                # Adjust price by up to +/-10% per day
                price_change = excess_demand_fraction * 0.1

            # Update the price
            self.prices[resource] *= (1 + price_change)

            # Ensure price doesn't go negative
            self.prices[resource] = max(self.prices[resource], 0.01)

            # Reset daily supply and demand for the next day
            self.daily_supply[resource] = 0.0
            self.demand[resource] = 0.0

class Person:
    def __init__(self):
        self.money = random.uniform(80, 120)  # Increased initial money
        self.resources = {resource: random.uniform(5, 10) for resource in RESOURCES}
        self.is_alive = True
        self.max_resource = MAX_RESOURCE
        self.working = False
        self.factory = None
        self.savings_goal = random.uniform(300, 500)  # New attribute: savings goal

    def consume_resources(self):
        for resource in RESOURCES:
            self.resources[resource] -= CONSUMPTION_RATES[resource]
            if self.resources[resource] < 0:
                self.is_alive = False

    def expected_daily_expense(self, market):
        expense = sum(CONSUMPTION_RATES[resource] * market.prices[resource] for resource in RESOURCES)
        return expense

    def decide_to_work(self, market):
        expected_expense = self.expected_daily_expense(market)
        # New logic: decide to work if money is less than savings goal
        if self.money < self.savings_goal:
            return True
        else:
            return False

    def buy_resources(self, market):
        for resource in RESOURCES:
            if self.money > 100:
                # Cap extra items to 5 units each
                needed = min(self.max_resource - self.resources[resource], 5)
            else:
                needed = CONSUMPTION_RATES[resource]
            if needed > 0 and market.supply[resource] > 0:
                affordable_amount = self.money / market.prices[resource]
                amount_can_buy = min(needed, affordable_amount, market.supply[resource])
                if amount_can_buy > 0:
                    self.resources[resource] += amount_can_buy
                    self.money -= amount_can_buy * market.prices[resource]
                    market.demand[resource] += amount_can_buy
                    market.supply[resource] -= amount_can_buy

    def reproduce(self):
        for resource in RESOURCES:
            if self.resources[resource] < self.max_resource:
                return None
        if self.money < 200:  # Increased money threshold for reproduction
            return None
        self.money /= 2
        child = Person()
        child.money = self.money
        child.savings_goal = random.uniform(300, 500)  # Assign a savings goal to the child
        for resource in RESOURCES:
            self.resources[resource] /= 2
            child.resources[resource] = self.resources[resource]
        return child

class Factory:
    def __init__(self, resource_type):
        self.resource_type = resource_type
        self.workers = []
        self.production_per_worker = PRODUCTION_PER_WORKER[resource_type]
        self.wage_multiplier = WAGE_MULTIPLIERS[resource_type]
        self.wage = 0.0

    def calculate_wage(self, market):
        self.wage = max(market.prices[self.resource_type] * self.wage_multiplier, MINIMUM_WAGE)

    def accept_workers(self, applicants):
        max_workers = 50
        random.shuffle(applicants)
        self.workers = []
        for person in applicants[:max_workers]:
            self.workers.append((person, self.wage))
            person.working = True
            person.factory = self

    def produce(self, market):
        production_amount = self.production_per_worker * len(self.workers)
        if production_amount > 0:
            market.supply[self.resource_type] += production_amount
            market.daily_supply[self.resource_type] = production_amount  # Record daily production
        for person, wage in self.workers:
            person.money += wage
            person.working = False
            person.factory = None

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
    # Display factory worker counts
    for factory in factories:
        worker_count = len(factory.workers)
        factory_text = FONT.render(f"{factory.resource_type} Factory Workers: {worker_count}", True, BLACK)
        win.blit(factory_text, (10, y_offset))
        y_offset += 25
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    people = [Person() for _ in range(NUM_PEOPLE)]
    factories = [Factory(resource) for resource in RESOURCES]
    market = Market()
    day = 0

    running = True
    while running:
        clock.tick(60)
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
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
                        if person.decide_to_work(market) and not person.working:
                            # Person prefers factory producing scarce resources
                            sorted_factories = sorted(factories, key=lambda f: market.supply[f.resource_type])
                            # Apply to the factory with the lowest supply first
                            for fac in sorted_factories:
                                if fac.wage >= MINIMUM_WAGE:
                                    factory_applications[fac].append(person)
                                    break
                    # Factories accept workers
                    for factory in factories:
                        factory.accept_workers(factory_applications[factory])
                    # Factories produce goods (update market supply)
                    for factory in factories:
                        factory.produce(market)
                    # Randomize purchase order
                    random.shuffle(people)
                    # People buy resources (update market demand)
                    for person in people:
                        person.buy_resources(market)
                    # Market adjusts prices based on updated supply and demand
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
        # Drawing
        draw_window(WIN, people, factories, market, day)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
