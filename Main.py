import pygame
from economyGame import market, restock_market, calculate_price, buy_resource, sell_resource, craft_tools  # Import from economyGame
from AI import AIPlayer  # Import AI player class

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI-Driven Turn-Based Market Game")

# Fonts and colors
font = pygame.font.SysFont("Arial", 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize AI players
ai_players = [AIPlayer(), AIPlayer(), AIPlayer()]  # Three AI players

# Track turns and current player
turn_count = 1
current_player = 0
actions_taken = 0  # Track number of actions taken per AI turn
max_actions_per_turn = 5  # Limit of 5 actions per turn

def draw_text(text, x, y, color=BLACK):
    """Utility function to draw text on screen."""
    screen.blit(font.render(text, True, color), (x, y))

def draw_game_interface():
    """Draw the game interface for the current AI player."""
    player = ai_players[current_player]  # Use AI player

    # Display AI player's resources and market status
    draw_text(f"Turn: {turn_count} - AI Player {current_player + 1}'s Turn (Actions: {actions_taken}/{max_actions_per_turn})", 10, 10)
    draw_text(f"Money: ${player.money}", 10, 40)
    draw_text("AI Player Resources:", 10, 70)
    draw_text(f"Wood: {player.resources['Wood']}", 10, 100)
    draw_text(f"Iron: {player.resources['Iron']}", 10, 130)
    draw_text(f"Tools: {player.resources['Tools']}", 10, 160)

    # Display market state
    wood_price = calculate_price("Wood")
    iron_price = calculate_price("Iron")
    tools_price = calculate_price("Tools")

    draw_text("Market:", 300, 70)
    draw_text(f"Wood - Price: ${wood_price} Supply: {market['Wood']['supply']}", 300, 100)
    draw_text(f"Iron - Price: ${iron_price} Supply: {market['Iron']['supply']}", 300, 130)
    draw_text(f"Tools - Price: ${tools_price} Supply: {market['Tools']['supply']}", 300, 160)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    # All players are AI; simulate their turn
    current_ai = ai_players[current_player]
    current_ai.take_turn(market)  # AI makes decisions automatically

    # After AI finishes turn, restock market and move to next AI player
    restock_market()
    current_player = (current_player + 1) % 3  # Move to next AI player
    actions_taken = 0  # Reset actions for the next player

    # Increment turn count after all AI players finish their turn
    if current_player == 0:
        turn_count += 1

    # Draw the game interface to visualize the current state
    draw_game_interface()

    # Update the display
    pygame.display.flip()
    clock.tick(30)  # Set frame rate to 30 FPS

    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit pygame
pygame.quit()
