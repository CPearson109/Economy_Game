import pygame
import random

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3-Player Turn-Based Market Game with Supply-Demand Pricing")

# Fonts and colors
font = pygame.font.SysFont("Arial", 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize player states (3 players)
players = [
    {"money": 1000, "resources": {"Wood": 0, "Iron": 0, "Tools": 0}},
    {"money": 1000, "resources": {"Wood": 0, "Iron": 0, "Tools": 0}},
    {"money": 1000, "resources": {"Wood": 0, "Iron": 0, "Tools": 0}}
]

# Market state
market = {
    "Wood": {
        "supply": 10,
        "max_supply": 20,
        "min_price": 20,
        "max_price": 100
    },
    "Iron": {
        "supply": 10,
        "max_supply": 20,
        "min_price": 50,
        "max_price": 200
    },
    "Tools": {
        "supply": 0,
        "max_supply": 20,
        "min_price": 100,
        "max_price": 400
    }
}

# Function to calculate price based on supply
def calculate_price(resource):
    data = market[resource]
    supply = data["supply"]
    max_supply = data["max_supply"]
    min_price = data["min_price"]
    max_price = data["max_price"]

    # Prevent division by zero
    if max_supply == 0:
        return max_price

    # Calculate price inversely proportional to supply
    price = max_price - ((supply / max_supply) * (max_price - min_price))
    price = max(min_price, min(max_price, price))  # Ensure price stays within bounds
    return round(price)

# Function to display text on the screen
def draw_text(text, x, y, color=BLACK):
    screen.blit(font.render(text, True, color), (x, y))

# Function to handle resource buying (only count action if successful)
def buy_resource(player, resource):
    global market
    price = calculate_price(resource)
    if player["money"] >= price and market[resource]["supply"] > 0:
        player["money"] -= price
        player["resources"][resource] += 1
        market[resource]["supply"] -= 1
        return True  # Action successful
    return False  # Action failed (no money or no supply)

# Function to handle resource selling (only count action if successful)
def sell_resource(player, resource):
    global market
    if player["resources"][resource] > 0:
        price = calculate_price(resource)
        player["money"] += price
        player["resources"][resource] -= 1
        market[resource]["supply"] += 1
        return True  # Action successful
    return False  # Action failed (no resources to sell)

# Function to handle crafting (Wood + Iron = Tools, only count action if successful)
def craft_tools(player):
    if player["resources"]["Wood"] > 0 and player["resources"]["Iron"] > 0:
        player["resources"]["Wood"] -= 1
        player["resources"]["Iron"] -= 1
        player["resources"]["Tools"] += 1
        market["Tools"]["supply"] += 1
        return True  # Action successful
    return False  # Action failed (not enough resources)

# Function to restock the market each turn
def restock_market():
    for resource in market:
        if resource != "Tools":  # Tools are only crafted by the player
            restock_amount = random.randint(1, 3)  # Add a small amount of raw materials
            market[resource]["supply"] = min(market[resource]["max_supply"], market[resource]["supply"] + restock_amount)

    # Remove a percentage of Tools to simulate consumption
    tools_consumed = random.randint(0, market["Tools"]["supply"])
    market["Tools"]["supply"] = max(0, market["Tools"]["supply"] - tools_consumed)

# Main game loop
running = True
clock = pygame.time.Clock()

# Track turns and current player
turn_count = 1
current_player = 0
actions_taken = 0  # Track number of actions taken per player turn
max_actions_per_turn = 5  # Limit of 5 actions per turn

while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and actions_taken < max_actions_per_turn:  # Check if action limit is reached
            if event.key == pygame.K_w:  # Buy Wood
                if buy_resource(players[current_player], "Wood"):
                    actions_taken += 1
            elif event.key == pygame.K_i:  # Buy Iron
                if buy_resource(players[current_player], "Iron"):
                    actions_taken += 1
            elif event.key == pygame.K_s:  # Sell Tools
                if sell_resource(players[current_player], "Tools"):
                    actions_taken += 1
            elif event.key == pygame.K_c:  # Craft Tools
                if craft_tools(players[current_player]):
                    actions_taken += 1
            elif event.key == pygame.K_SPACE:  # End turn, restock market, and move to next player
                restock_market()
                actions_taken = max_actions_per_turn  # Ensure the player finishes their turn

    # When the player has taken 5 actions, move to the next player
    if actions_taken >= max_actions_per_turn:
        current_player = (current_player + 1) % 3  # Move to the next player
        actions_taken = 0  # Reset action count
        turn_count += 1 if current_player == 0 else 0  # Increment turn count after all players finish their turn

    # Update prices based on supply
    wood_price = calculate_price("Wood")
    iron_price = calculate_price("Iron")
    tools_price = calculate_price("Tools")

    # Draw the game interface
    player = players[current_player]
    draw_text(f"Turn: {turn_count} - Player {current_player + 1}'s Turn (Actions: {actions_taken}/{max_actions_per_turn})", 10, 10)
    draw_text(f"Money: ${player['money']}", 10, 40)
    draw_text("Your Resources:", 10, 70)
    draw_text(f"Wood: {player['resources']['Wood']}", 10, 100)
    draw_text(f"Iron: {player['resources']['Iron']}", 10, 130)
    draw_text(f"Tools: {player['resources']['Tools']}", 10, 160)

    draw_text("Market:", 300, 70)
    draw_text(f"Wood - Price: ${wood_price} Supply: {market['Wood']['supply']}", 300, 100)
    draw_text(f"Iron - Price: ${iron_price} Supply: {market['Iron']['supply']}", 300, 130)
    draw_text(f"Tools - Price: ${tools_price} Supply: {market['Tools']['supply']}", 300, 160)

    # Instructions
    draw_text("Press W to buy Wood, I to buy Iron", 10, 250)
    draw_text("Press C to craft Tools (1 Wood + 1 Iron)", 10, 280)
    draw_text("Press S to sell Tools, SPACE to end turn", 10, 310)

    # Update the display
    pygame.display.flip()
    clock.tick(30)

# Quit pygame
pygame.quit()
