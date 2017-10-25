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

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import pandas as pd
import numpy as np

from sklearn import preprocessing
from collections import deque

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

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def get_action(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        policy = self.actor.predict(state, batch_size=1)
        return np.random.choice(self.action_size, 1, p=policy[0])[0]

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(50, input_dim=self.state_size, batch_size=1, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(50, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.compile(loss='categorical_crossentropy', optimizer='adam')
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(50, input_dim=self.state_size, batch_size=1, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(50, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.compile(loss='mse', optimizer='adam')
        return critic

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

turn = 0
value = deque(maxlen=2000)
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

    N_enemey_ships = 0
    for p in game_map.all_players():
        logging.info('Player: {} Ships: {}'.format(p.id, len(p.all_ships())))
        if p.id != game_map.my_id:
            N_enemey_ships += len(p.all_ships())


    #Initial our state matrix
    if turn == 0:
        value.append(N_ships / (N_ships + N_enemey_ships) * 100)
        reward = 0
        df = pd.DataFrame(np.zeros((N_planets, 14)),
            columns=['planet', 'radius', 'spots', 
                     'health', 'current_production', 'remaining_production', 'ownership', 'n_docked_ships', 'n_my_ships', 'n_enemy_ships', 'is_full', 
                     'distance', 'nearest_friend', 'nearest_enemy'])
        df['planet'] = np.arange(0, N_planets, 1)
        
        radius_array = np.zeros(N_planets)
        spots_array = np.zeros(N_planets)
        for planet in game_map.all_planets():
            radius_array[planet.id] = planet.radius
            spots_array[planet.id] = planet.num_docking_spots
        df.radius = radius_array
        df.spots = spots_array

    else:
        cur_value = N_ships / (N_ships + N_enemey_ships) * 100
        reward = cur_value - value[turn-1]
        value.append(cur_value)

    logging.info(reward)

    # Here we define the set of commands to be sent to the Halite engine at the end of the turn
    command_queue = []

    #Planet / turn specific factors
    health_array = np.zeros(N_planets)
    cur_prod_array = np.zeros(N_planets)
    rem_prod_array = np.zeros(N_planets)
    own_array = np.zeros(N_planets)
    dock_array = np.zeros(N_planets)
    my_ships_array = np.zeros(N_planets)
    enemy_ships_array = np.zeros(N_planets)
    full_array = np.zeros(N_planets)
    for planet in game_map.all_planets():
        health_array[planet.id] = planet.health
        if planet.owner == game_map.get_me():
            ownership = 1
        elif planet.owner is None:
            ownership = 0
        else:  # owned by enemy
            ownership = -1
        own_array[planet.id] = ownership
        cur_prod_array[planet.id] = planet.current_production
        rem_prod_array[planet.id] = planet.remaining_resources
        dock = len(planet.all_docked_ships())
        dock_array[planet.id] = dock
        
        n_my = 0
        n_enemy = 0
        for s in planet.all_docked_ships():
            if s.owner == game_map.get_me():
                n_my += 1
            else:
                n_enemy += 1
        my_ships_array[planet.id] = n_my
        enemy_ships_array[planet.id] = n_enemy
        if planet.is_full():
            full_array[planet.id] = 1
        else:
            full_array[planet.id] = 0

    df.health = health_array
    df.current_production = cur_prod_array
    df.remaining_production = rem_prod_array
    df.ownership = own_array
    df.n_docked_ships = dock_array
    df.n_my_ships = my_ships_array
    df.n_enemy_ships = enemy_ships_array
    df.is_full = full_array

    # For every ship that I control
    for ship in game_map.get_me().all_ships():

        #Calculate matrix of values for every ship that are ship specific
        dist_array = np.zeros(N_planets)
        my_dist_array = np.zeros(N_planets)
        enemy_dist_array = np.zeros(N_planets)
        for planet in game_map.all_planets():
            logging.info('planet')
            logging.info(planet)
            d = ship.calculate_distance_between(planet)
            dist_array[planet.id] = d

            enemy_best_distance = 10000
            my_best_distance = 10000
            for p in game_map.all_players():
                if p.id != game_map.my_id:
                    for s in p.all_ships():
                        d = s.calculate_distance_between(planet)
                        enemy_best_distance = min(d, enemy_best_distance)
                else:
                    for s in p.all_ships():
                        d = s.calculate_distance_between(planet)
                        my_best_distance = min(d, my_best_distance)
            my_dist_array[planet.id] = my_best_distance
            enemy_dist_array[planet.id] = enemy_best_distance
            
        
        df.distance = dist_array
        df.nearest_friend = my_dist_array
        df.nearest_enemy = enemy_dist_array
        
        train_data = preprocessing.scale(df.drop('planet', axis=1))
        td = train_data.reshape((1,train_data.size))
        #logging.info(df)

        logging.info(td.size)
        #training dim is 1xtotal features / state size
        #plus one more for closest ship
        #output is soft_max length N_planets
        if turn == 0:
            MyAgent = Agent(td.size, N_planets)
            policy = MyAgent.get_action(td)
            

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

