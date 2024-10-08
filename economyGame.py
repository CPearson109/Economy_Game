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
GRAY = (200, 200, 200)
RED = (255, 0, 0)

# Simulation settings
NUM_PEOPLE = 100
MAX_RESOURCE = {
    "Food": 30,    # Max amount for Food
    "Fuel": 20,    # Max amount for Fuel
    "Clothes": 10  # Max amount for Clothes
}

# Define maximum wages per resource type
MAX_WAGE_CAPS = {
    "Food": 200.0,    # Max wage for Food factory
    "Fuel": 200.0,    # Max wage for Fuel factory
    "Clothes": 200.0  # Max wage for Clothes factory
}

# Resources and Consumption Rates
RESOURCES = ["Food", "Fuel", "Clothes"]
CONSUMPTION_RATES = {"Food": 3, "Fuel": 2, "Clothes": 2}
BASE_PRICES = {"Food": 3, "Fuel": 6, "Clothes": 9}

# Production per worker and wage multipliers per resource
PRODUCTION_PER_WORKER = {"Food": 12, "Fuel": 10, "Clothes": 8}
WAGE_MULTIPLIERS = {"Food": 1.0, "Fuel": 1.0, "Clothes": 1.0}

# Minimum wage for factories
MINIMUM_WAGE = 10.0


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
            pygame.draw.rect(win, self.hover_color, self.rect)
        else:
            pygame.draw.rect(win, self.color, self.rect)

        text_surface = FONT.render(self.text, True, BLACK)
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
        self.supply = {
            "Food": 1000.0,
            "Fuel": 750.0,
            "Clothes": 500.0
        }
        self.demand = {resource: NUM_PEOPLE * CONSUMPTION_RATES[resource] for resource in RESOURCES}
        self.prices = {resource: BASE_PRICES[resource] * (self.demand[resource] / self.supply[resource]) for resource in RESOURCES}
        self.price_history = {resource: [self.prices[resource]] for resource in RESOURCES}

    def adjust_prices(self):
        for resource in RESOURCES:
            # Prevent division by zero
            if self.supply[resource] == 0:
                self.supply[resource] = 1

            # Calculate price based on individual supply and demand
            price = BASE_PRICES[resource] * (self.demand[resource] / self.supply[resource])

            # Cap the price to prevent it from becoming too high
            max_price = BASE_PRICES[resource] * 10  # Cap price at 10 times base price
            price = min(price, max_price)

            # Ensure price doesn't go below 10% of base price
            price = max(price, BASE_PRICES[resource] * 0.1)

            # Append the new price to the history
            self.price_history[resource].append(price)

            # Keep only the last N prices for moving average
            N = 3  # Moving average over last 3 days
            if len(self.price_history[resource]) > N:
                self.price_history[resource].pop(0)

            # Calculate the smoothed price using moving average
            smoothed_price = sum(self.price_history[resource]) / len(self.price_history[resource])

            self.prices[resource] = smoothed_price


class Person:
    def __init__(self):
        self.money = 100
        self.resources = {
            "Food": 15.0,
            "Fuel": 10.0,
            "Clothes": 5.0
        }
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
        # Increased thresholds to reduce eagerness to work
        money_threshold = 100  # Lower money threshold to make them work less often
        resource_threshold = {resource: CONSUMPTION_RATES[resource] * 3 for resource in RESOURCES}

        needs_money_or_resources = self.money < money_threshold or any(
            self.resources[resource] < resource_threshold[resource] for resource in RESOURCES
        )

        # Check if any resource is in low supply
        resource_low_supply = any(
            market.supply[resource] < market.demand[resource] for resource in RESOURCES
        )

        return needs_money_or_resources or resource_low_supply

    def buy_resources(self, market):
        # Increased target resource level to increase buying behavior
        target_resource_level = {resource: CONSUMPTION_RATES[resource] * 10 for resource in RESOURCES}
        budget_allocation = {'Food': 0.5, 'Fuel': 0.3, 'Clothes': 0.2}
        total_budget = self.money * 0.8  # Spend up to 80% of current money (increased from 50%)
        for resource in RESOURCES:
            # Determine how much more resource is needed to reach target level
            amount_needed = target_resource_level[resource] - self.resources[resource]
            if amount_needed <= 0:
                continue  # No need to buy more of this resource
            # Allocate budget for this resource
            amount_to_spend = total_budget * budget_allocation[resource]
            # Determine how much can be bought
            affordable_amount = amount_to_spend / market.prices[resource]
            amount_can_buy = min(
                affordable_amount, market.supply[resource], amount_needed
            )
            if amount_can_buy > 0:
                self.resources[resource] += amount_can_buy
                self.money -= amount_can_buy * market.prices[resource]
                market.supply[resource] -= amount_can_buy

    def reproduce(self):
        reproduction_chance = 0.01  # Reduced from 0.02
        if random.random() < reproduction_chance and self.money > 100:
            self.money /= 2
            child = Person()
            child.money = self.money / 2
            for resource in RESOURCES:
                self.resources[resource] /= 2
                child.resources[resource] = self.resources[resource] / 2
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
        self.previous_wage = MINIMUM_WAGE  # Initialize with minimum wage
        self.max_wage_change = 0.1  # Maximum wage decrease per day (e.g., 10%)
        self.worker_count = 0  # Track number of workers accepted
        self.max_wage_cap = MAX_WAGE_CAPS[resource_type]  # Get the max wage cap for the resource type

    def calculate_wage(self, market, num_people):
        # Use the smoothed market price
        market_price = market.prices[self.resource_type]

        # Calculate the desired wage based on production value and wage multiplier
        total_production_value = self.production_per_worker * market_price
        desired_wage = total_production_value * self.wage_multiplier

        # Cap the desired wage between MINIMUM_WAGE and max wage cap
        desired_wage = max(MINIMUM_WAGE, min(desired_wage, self.max_wage_cap))

        # Allow wage to increase to desired wage immediately
        if desired_wage >= self.previous_wage:
            self.initial_wage = desired_wage
        else:
            # Limit wage decrease to max_wage_change percentage
            wage_change = desired_wage - self.previous_wage
            max_decrease = self.previous_wage * self.max_wage_change
            if abs(wage_change) > max_decrease:
                wage_change = -max_decrease
            self.initial_wage = self.previous_wage + wage_change

        # Update previous wage for next day's calculation
        self.previous_wage = self.initial_wage

        # Reset current wage to initial wage at the start of the day
        self.current_wage = self.initial_wage

    def accept_worker(self, person):
        # Offer the current wage
        self.workers.append((person, self.current_wage))
        person.working = True
        person.factory = self

        # Decrease the wage slightly for the next worker
        self.current_wage *= 0.99  # Decrease wage by 0.1%

        # Ensure wage does not go below MINIMUM_WAGE
        self.current_wage = max(self.current_wage, MINIMUM_WAGE)

        # Ensure wage does not exceed the max wage cap for the resource type
        self.current_wage = min(self.current_wage, self.max_wage_cap)

        # Increment worker count
        self.worker_count += 1

    def pay_workers(self):
        for person, wage in self.workers:
            person.money += wage
            person.working = False
            person.factory = None

    def produce(self, market):
        total_workers = len(self.workers)
        production_amount = self.production_per_worker * total_workers
        if production_amount > 0:
            market.supply[self.resource_type] += production_amount
        # Clear workers for the next day
        self.workers = []

    def reset_worker_count(self):
        # Reset the worker count to zero at the start of each day
        self.worker_count = 0


def draw_window(win, people, factories, market, day, end_button):
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
        avg_resources = {
            resource: sum(person.resources[resource] for person in people) / len(people) for resource in RESOURCES
        }
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
        supply_text = FONT.render(f"{resource} Supply: {market.supply[resource]:.2f}", True, BLACK)
        win.blit(supply_text, (10, y_offset))
        y_offset += 25

    # Display factory worker counts and wages
    y_offset += 10
    factory_header = FONT.render("Factories and Worker Counts:", True, BLACK)
    win.blit(factory_header, (10, y_offset))
    y_offset += 25
    for factory in factories:
        workers_text = FONT.render(
            f"{factory.resource_type} Factory: {factory.worker_count} workers", True, BLACK
        )
        win.blit(workers_text, (10, y_offset))
        y_offset += 25
        wage_text = FONT.render(f"Starting Wage: ${factory.initial_wage:.2f}", True, BLACK)
        win.blit(wage_text, (10, y_offset))
        y_offset += 25

    # Draw the End Game button
    end_button.draw(win)

    pygame.display.update()


def end_game(prices_history, population_history, avg_money_history, avg_resources_history, days):
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

    plt.figure(figsize=(10, 5))
    plt.plot(days, avg_money_history, label="Average Money")
    plt.xlabel("Day")
    plt.ylabel("Average Money")
    plt.title("Average Money Over Time")
    plt.show()

    for resource in RESOURCES:
        plt.figure(figsize=(10, 5))
        plt.plot(days, avg_resources_history[resource], label=f"Average {resource}")
        plt.xlabel("Day")
        plt.ylabel(f"Average {resource}")
        plt.title(f"Average {resource} Over Time")
        plt.show()

    pygame.quit()
    sys.exit()


def main():
    clock = pygame.time.Clock()
    people = [Person() for _ in range(NUM_PEOPLE)]
    factories = [Factory(resource) for resource in RESOURCES]
    market = Market()
    day = 0

    # For plotting trends
    days = []
    prices_history = {resource: [] for resource in RESOURCES}
    population_history = []
    avg_money_history = []
    avg_resources_history = {resource: [] for resource in RESOURCES}

    # Create the End Game button
    end_button = Button("End Game", WIDTH - 150, HEIGHT - 50, 130, 40, GRAY, RED, lambda: end_game(
        prices_history, population_history, avg_money_history, avg_resources_history, days))

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
                        factory.calculate_wage(market, len(people))
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

                            # Find the maximum wage offered
                            max_wage = max(factory_wages.values())

                            # Check if all factories are offering minimum wage
                            all_min_wage = all(wage == MINIMUM_WAGE for wage in factory_wages.values())

                            if all_min_wage:
                                # If all wages are at minimum, assign workers based on potential supply

                                # For each factory, calculate potential supply if the worker joins
                                potential_supplies_after_joining = {}
                                for factory in factories:
                                    resource = factory.resource_type
                                    # Potential supply if the worker joins this factory
                                    potential_supply_if_join = potential_supply[resource] + factory.production_per_worker * (factory_worker_counts[factory] + 1)
                                    potential_supplies_after_joining[factory] = potential_supply_if_join

                                # The worker chooses the factory where the potential supply is lowest after joining
                                selected_factory = min(potential_supplies_after_joining, key=lambda f: potential_supplies_after_joining[f])

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

        # Drawing
        draw_window(WIN, people, factories, market, day, end_button)
        end_button.check_click()


if __name__ == "__main__":
    main()
