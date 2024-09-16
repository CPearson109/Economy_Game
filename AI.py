import random


class AIPlayer:
    def __init__(self):
        # AI starts with the same amount of money and resources as the human players
        self.money = 1000
        self.resources = {"Wood": 0, "Iron": 0, "Tools": 0}
        self.max_actions_per_turn = 5

    def take_turn(self, market):
        """
        This function simulates the AI's turn. It will make up to `max_actions_per_turn`
        decisions based on its current state and the market state.
        """
        actions_taken = 0
        while actions_taken < self.max_actions_per_turn:
            # Decide what to do (buy, sell, or craft)
            action = self.decide_action(market)

            if action == "buy_wood" and self.buy_resource("Wood", market):
                actions_taken += 1
            elif action == "buy_iron" and self.buy_resource("Iron", market):
                actions_taken += 1
            elif action == "sell_tools" and self.sell_resource("Tools", market):
                actions_taken += 1
            elif action == "craft_tools" and self.craft_tools(market):
                actions_taken += 1
            else:
                # No valid action to take, end the turn early
                break

    def decide_action(self, market):
        """
        AI decision-making logic.
        It can decide to buy, sell, or craft based on the current market conditions and its own state.
        The AI prioritizes crafting tools and selling them for profit, while buying resources when they're cheap.
        """
        # Craft tools if possible (highest priority)
        if self.resources["Wood"] > 0 and self.resources["Iron"] > 0:
            return "craft_tools"

        # Sell tools if we have any
        if self.resources["Tools"] > 0:
            return "sell_tools"

        # If wood is cheap, buy it
        if self.money > market["Wood"]["min_price"] and market["Wood"]["supply"] > 0:
            return "buy_wood"

        # If iron is cheap, buy it
        if self.money > market["Iron"]["min_price"] and market["Iron"]["supply"] > 0:
            return "buy_iron"

        # End turn if no actions can be taken
        return "end_turn"

    def buy_resource(self, resource, market):
        """
        AI attempts to buy a resource if it has enough money and there is enough supply.
        """
        price = self.calculate_price(resource, market)
        if self.money >= price and market[resource]["supply"] > 0:
            self.money -= price
            self.resources[resource] += 1
            market[resource]["supply"] -= 1
            return True
        return False

    def sell_resource(self, resource, market):
        """
        AI attempts to sell a resource if it has any of that resource.
        """
        price = self.calculate_price(resource, market)
        if self.resources[resource] > 0:
            self.money += price
            self.resources[resource] -= 1
            market[resource]["supply"] += 1
            return True
        return False

    def craft_tools(self, market):
        """
        AI attempts to craft tools if it has both wood and iron.
        """
        if self.resources["Wood"] > 0 and self.resources["Iron"] > 0:
            self.resources["Wood"] -= 1
            self.resources["Iron"] -= 1
            self.resources["Tools"] += 1
            market["Tools"]["supply"] += 1
            return True
        return False

    def calculate_price(self, resource, market):
        """
        Calculate the price of a resource based on market conditions.
        """
        supply = market[resource]["supply"]
        max_supply = market[resource]["max_supply"]
        min_price = market[resource]["min_price"]
        max_price = market[resource]["max_price"]

        price = max_price - ((supply / max_supply) * (max_price - min_price))
        return max(min_price, min(max_price, price))
