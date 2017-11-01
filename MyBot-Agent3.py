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
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from keras.layers.merge import Add

# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import pandas as pd
import numpy as np
import math

import keras.backend as K
import tensorflow as tf

from sklearn import preprocessing
from collections import deque

sess = tf.Session()
K.set_session(sess)

class Agent:
    def __init__(self, state_size, action_size, sess):
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.actor_state_input, self.actor = self.build_actor()
        self.critic_state_input, self.critic_action_input, self.critic = self.build_critic()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size])
        actor_weights = self.actor.trainable_weights

        self.actor_grads = tf.gradients(self.actor.outputs, actor_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_weights)
        self.optimize = tf.train.AdamOptimizer().apply_gradients(grads)

        self.critic_grads = tf.gradients(self.critic.outputs, self.critic_action_input)

        self.sess.run(tf.global_variables_initializer())
        
        if self.load_model:
            self.actor.load_weights("./model/actor.h5")
            self.critic.load_weights("./model/critic.h5")

    def get_action(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        policy = self.actor.predict(state, batch_size=1)
        return policy

    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        h1 = Dense(250, activation='relu', kernel_initializer='he_uniform')(state_input)
        h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(h1)
        output = Dense(self.action_size, activation='sigmoid', kernel_initializer='he_uniform')(h2)
        
        actor = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.01)
        actor.compile(loss="binary_crossentropy", optimizer=adam)
        
        return state_input, actor

    def build_critic(self):

        state_input = Input(shape=(self.state_size,))
        state_h1 = Dense(200, activation='relu', kernel_initializer='he_uniform')(state_input)
        state_h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(state_h1)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(50, activation='relu', kernel_initializer='he_uniform')(action_input)
        action_h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(action_h1)

        mergedLayer = Add()([state_h2, action_h2])
        mergedLayer_h1 = Dense(100, activation='relu', kernel_initializer='he_uniform')(mergedLayer)
        outputLayer = Dense(1, activation='relu')(mergedLayer_h1)

        critic = Model(inputs=[state_input, action_input], outputs=outputLayer)
        adam = Adam(lr=0.01)
        critic.compile(loss='mse', optimizer=adam)
        
        return state_input, action_input, critic

    def train_model(self, prior_state, state, action, reward):
              
        target_action = self.actor.predict(state, batch_size=1)
        future_reward = self.critic.predict([state, target_action], batch_size=1)[0][0]
        
        reward += self.gamma * future_reward
        target = np.array([reward])

        self.critic.fit([prior_state, action], target, batch_size=1, epochs=1, verbose=0)
        
        grads = self.sess.run(self.critic_grads, 
                              feed_dict={
                                  self.critic_state_input:  prior_state,
                                  self.critic_action_input: action})[0]

        self.sess.run(self.optimize, 
                      feed_dict={
                          self.actor_state_input: prior_state,
                          self.actor_critic_grad: grads})

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("Agent")
# Then we print our start message to the logs
logging.info("Starting my Agent bot!")

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

def hyp(x1, x2):
    return math.sqrt(x1**2 + x2**2)

#Constants to adjust
#Friendly and enemy ships to encode by distance
SHIPS = 5
#Nearest planets to encode
PLANETS = 10
MAX_RADIUS = 16
MAX_SPOTS = 6
MAX_PROD = 2500
MAX_HEALTH = 5000

turn = 0
pct_owned = deque(maxlen=2000)
#A list to hold a dictionary of ship,point tuples
#length is number of turns.  Each turn holds N-dicts for N-ships in that round
ship_states = deque(maxlen=2000)
while True:
    # TURN START
    # Update the map for the new turn and get the latest version
    game_map = game.update_map()

    logging.info('Map Dim X: {}'.format(game_map.width))
    logging.info('Map Dim Y: {}'.format(game_map.height))

    DIST_NORM = hyp(game_map.width, game_map.height)

    logging.info('N Players: {}'.format(len(game_map.all_players())))
    N_ships = len(game_map.get_me().all_ships())
    logging.info('My ships: {}'.format(N_ships))
    logging.info('My ID: {}'.format(game_map.my_id))

    N_planets = len(game_map.all_planets())
    logging.info('N planets: {}'.format(N_planets))

    N_enemy_ships = 0
    for p in game_map.all_players():
        logging.info('Player: {} Ships: {}'.format(p.id, len(p.all_ships())))
        if p.id != game_map.my_id:
            N_enemy_ships += len(p.all_ships())


    #Initial our state matrix
    if turn == 0:
        df = pd.DataFrame(np.zeros((N_planets, 18)),
            columns=['planet', 'x', 'y', 'radius', 'spots',
                     'health', 'current_production', 'remaining_production', 'ownership', 'n_docked_ships', 'n_my_ships', 'n_enemy_ships', 'is_full',
                     'distance', 'angle', 'nearest_friend', 'nearest_enemy', 'n_obstacles'])
        df['planet'] = np.arange(0, N_planets, 1)

        radius_array = np.zeros(N_planets)
        spots_array = np.zeros(N_planets)
        x_array = np.zeros(N_planets)
        y_array = np.zeros(N_planets)
        for planet in game_map.all_planets():
            radius_array[planet.id] = planet.radius / game_map.height
            spots_array[planet.id] = planet.num_docking_spots / MAX_SPOTS
            x_array[planet.id] = planet.x / game_map.width
            y_array[planet.id] = planet.y / game_map.height
        df.radius = radius_array
        df.spots = spots_array
        df.x = x_array
        df.y = y_array


    pct_owned.append(N_ships / (N_ships + N_enemy_ships) * 100)
    logging.info('Pct Ships Owned: {}'.format(pct_owned[turn]))

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

    df.health = health_array / MAX_HEALTH
    df.current_production = cur_prod_array / MAX_PROD
    df.remaining_production = rem_prod_array / MAX_PROD
    df.ownership = own_array
    df.n_docked_ships = dock_array / MAX_SPOTS
    df.n_my_ships = my_ships_array / MAX_SPOTS
    df.n_enemy_ships = enemy_ships_array / MAX_SPOTS
    df.is_full = full_array

    points = (N_ships - N_enemy_ships) * 100
    cur_ship_ids = []
    for s in game_map.get_me().all_ships():
        cur_ship_ids.append(s.id)

    if turn > 0:
        for s in ship_states[turn-1]:
            if s not in cur_ship_ids:
                #if we lost a ship from the last state
                points -= 100
        for ship in game_map.get_me().all_ships():
            if ship.id not in ship_states[turn-1]:
                #if we gained a new ship from the last state
                points += 100

    logging.info('Points: {}'.format(points))

    #Dict to hold ship states and actions
    ship_dict = {}
    # For every ship that I control
    for ship in game_map.get_me().all_ships():

        #Calculate matrix of values for every ship that are ship specific
        dist_array = np.zeros(N_planets)
        angle_array = np.zeros(N_planets)
        my_dist_array = np.zeros(N_planets)
        enemy_dist_array = np.zeros(N_planets)
        obstacle_array = np.zeros(N_planets)
        for planet in game_map.all_planets():
            d = ship.calculate_distance_between(planet)
            dist_array[planet.id] = d
            a = ship.calculate_angle_between(planet)
            angle_array[planet.id] = a

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
            obstacle_array[planet.id] = len(game_map.obstacles_between(ship, planet))


        df.distance = dist_array / game_map.height
        df.angle = angle_array / 360
        df.nearest_friend = my_dist_array / game_map.height
        df.nearest_enemy = enemy_dist_array / game_map.height
        df.obstacle_array = obstacle_array / 10

        #train_data = preprocessing.scale(df.drop('planet', axis=1))
        train_data = df.drop('planet', axis=1)

        #training dim is 1 x total features / state size
        td = train_data.values.reshape((1, train_data.size))
        logging.info('Planet Matrix')
        logging.info(df)
        #logging.info(df.shape)

        #logging.info(td.size)

        ##Add in Friendly and Enemy ship matrix
        all_ships = game_map._all_ships()
        df_ships = pd.DataFrame(np.zeros((len(all_ships), 6)), columns=['ship_id', 'x', 'y', 'player', 'distance', 'status'])

        ship_id_array = []
        ship_x_array = []
        ship_y_array = []
        ship_player_array = []
        ship_dist_array = []
        ship_status_array = []

        ship_min_enemy = [(1000, -1)] * SHIPS
        ship_min_friendly = [(1000, -1)] * SHIPS

        for s in all_ships:
            #ship_id_array = np.append(ship_id_array, s.id)
            ship_id_array.append(s.id)
            ship_x_array.append(s.x)
            ship_y_array.append(s.y)
            ship_player_array.append(s.owner.id)
            ship_status_array.append(s.docking_status)

            dist = ship.calculate_distance_between(s)
            ship_dist_array.append(dist)
            running_max_e, _ = max(ship_min_enemy)
            running_max_f, _ = max(ship_min_friendly)
            if (s.owner.id != game_map.my_id) & (dist < running_max_e):
                ship_min_enemy.pop()
                ship_min_enemy.append((dist, s.id))
                ship_min_enemy = sorted(ship_min_enemy)
            if (s.owner.id == game_map.my_id) & (dist < running_max_f) & (ship.id != s.id):
                ship_min_friendly.pop()
                ship_min_friendly.append((dist, s.id))
                ship_min_friendly = sorted(ship_min_friendly)

        #logging.info(ship_min_enemy)
        #logging.info(ship_min_friendly)

        df_ships.ship_id = ship_id_array
        df_ships.x = ship_x_array
        df_ships.y = ship_y_array
        df_ships.player = ship_player_array
        df_ships.distance = ship_dist_array
        df_ships.status = ship_status_array

        ids_e = []
        ids_f = []
        for _, id, in ship_min_enemy:
            ids_e.append(id)
        for _, id, in ship_min_friendly:
            ids_f.append(id)
        #logging.info(all_ships)
        #logging.info('SHIP')
        #logging.info(df_ships)
        #logging.info(df_ships[df_ships['ship_id'].isin(ids_e)])

        #output is sigmoid(0,1) length 3: percent of x, percent of y, percent of max velocity
        if 'MyAgent' not in locals():
            MyAgent = Agent(td.size, 3, sess)

        policy = MyAgent.get_action(td)

        ship_dict[ship.id] = (td, policy)

        logging.info('POLICY')
        logging.info(policy)

        #logging.info('Points: {}'.format(points))
        #logging.info(ship_dict)

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

    #Train based on previous actions
    if turn > 0:
        prior_turn_dict = ship_states[turn-1]
        for k in prior_turn_dict:
            prior_state, prior_action = prior_turn_dict[k]
            #ship had to be in the last turn and current turn to train
            if k in ship_dict:
                cur_state, cur_action = ship_dict[k]
                MyAgent.train_model(prior_state, cur_state, prior_action, points)
                logging.info('Training worked')
    
    if turn % 50 == 0:
        MyAgent.actor.save_weights("./model/actor.h5")
        MyAgent.critic.save_weights("./model/critic.h5")

    turn += 1
    logging.info(turn)

    #Store ships, actions and values in dict
    ship_states.append(ship_dict)
    # Send our set of commands to the Halite engine for this turn
    game.send_command_queue(command_queue)
    # TURN END
    logging.info('At turn end')
# GAME END

#Rethink rewards:
#+1 for every ship alive from last round
#+100 for new ship
#-100 for dead ship
#maybe +percent of board -50
