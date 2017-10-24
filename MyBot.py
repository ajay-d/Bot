"""
Welcome to your first Halite-II bot!

This bot's name is Settler. It's purpose is simple (don't expect it to win complex games :) ):
1. Initialize game
2. If a ship is not docked and there are unowned planets
2.a. Try to Dock in the planet if close enough
2.b If not, go towards the planet

Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need
to log anything use the logging module.
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import pandas as pd
import numpy as np

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("Settler")
# Then we print our start message to the logs
logging.info("Starting my Settler bot!")

FEATURE_NAMES = [
    "health",
    "available_docking_spots",
    "remaining_production",
    "signed_current_production",
    "gravity",
    "closest_friendly_ship_distance",
    "closest_enemy_ship_distance",
    "ownership",
    "distance_from_center",
    "weighted_average_distance_from_friendly_ships",
    "is_active"]

def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def distance(x1, y1, x2, y2):
    return math.sqrt(distance2(x1, y1, x2, y2))

class Agent:
    def __init__(self, state_size, action_size):
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            return 0

turn = 0
while True:
    # TURN START
    # Update the map for the new turn and get the latest version
    game_map = game.update_map()

    logging.info('N Players: {}'.format(len(game_map.all_players())))
    N_ships = len(game_map.get_me().all_ships())
    logging.info('My ships: {}'.format(N_ships))
    logging.info('My ID: {}'.format(game_map.my_id))

    N_planets = len(game_map.all_planets())
    logging.info('N planets: {}'.format(N_planets))

    for p in game_map.all_players():
        logging.info('Player: {} Ships: {}'.format(p.id, len(p.all_ships())))

    #Initial our state matrix
    if turn == 0:
        ship_constant = N_ships*100
        df = pd.DataFrame(np.zeros((N_planets, 10)),
            columns=['planet', 'radius', 'spots', 'health', 'current_production', 'remaining_production', 'ownership',
                     'n_docked_ships', 'n_my_ships', 'is_full'])
        df['planet'] = np.arange(0, N_planets, 1)
        for planet in game_map.all_planets():
            #logging.info(df.loc[planet.id]['radius'])
            df.radius.iloc[planet.id] = planet.radius
            df.spots.iloc[planet.id] = planet.num_docking_spots
            df.health.iloc[planet.id] = planet.health
            df.current_production.iloc[planet.id] = planet.current_production
            df.remaining_production.iloc[planet.id] = planet.remaining_resources

    logging.info(df)
    #logging.info(ship_constant)
    my_agent = Agent(10, 2)

    # Here we define the set of commands to be sent to the Halite engine at the end of the turn
    command_queue = []
    # For every ship that I control
    for ship in game_map.get_me().all_ships():
        # If the ship is docked
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            # Skip this ship
            continue

        # For each planet in the game (only non-destroyed planets are included)
        for planet in game_map.all_planets():
            # If the planet is owned
            if planet.is_owned():
                # Skip this planet
                continue

            # If we can dock, let's (try to) dock. If two ships try to dock at once, neither will be able to.
            if ship.can_dock(planet):
                # We add the command by appending it to the command_queue
                command_queue.append(ship.dock(planet))
            else:
                # If we can't dock, we move towards the closest empty point near this planet (by using closest_point_to)
                # with constant speed. Don't worry about pathfinding for now, as the command will do it for you.
                # We run this navigate command each turn until we arrive to get the latest move.
                # Here we move at half our maximum speed to better control the ships
                # In order to execute faster we also choose to ignore ship collision calculations during navigation.
                # This will mean that you have a higher probability of crashing into ships, but it also means you will
                # make move decisions much quicker. As your skill progresses and your moves turn more optimal you may
                # wish to turn that option off.
                navigate_command = ship.navigate(ship.closest_point_to(planet), game_map, speed=hlt.constants.MAX_SPEED/2, ignore_ships=True)
                # If the move is possible, add it to the command_queue (if there are too many obstacles on the way
                # or we are trapped (or we reached our destination!), navigate_command will return null;
                # don't fret though, we can run the command again the next turn)
                if navigate_command:
                    command_queue.append(navigate_command)
            break
    turn += 1
    logging.info(turn)
    # Send our set of commands to the Halite engine for this turn
    game.send_command_queue(command_queue)
    # TURN END
# GAME END
