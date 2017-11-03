"""
Welcome to your first Halite-II bot!
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
        self.load_model = True

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

        _ , self.actor_target = self.build_actor()
        _ , _ , self.critic_target = self.build_critic()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size])
        actor_weights = self.actor.trainable_weights

        self.actor_grads = tf.gradients(self.actor.outputs, actor_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_weights)
        self.optimize = tf.train.AdamOptimizer().apply_gradients(grads)
        #self.optimize = tf.train.AdadeltaOptimizer().apply_gradients(grads)

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
        h1 = Dense(500, activation='relu', kernel_initializer='he_uniform')(state_input)
        h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(h1)
        h3 = Dense(100, activation='relu', kernel_initializer='he_uniform')(h2)
        output = Dense(self.action_size, activation='sigmoid', kernel_initializer='he_uniform')(h3)

        actor = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.01)
        #'adam', 'adamax', 'adadelta', 'nadam'
        actor.compile(loss="binary_crossentropy", optimizer='adam')

        return state_input, actor

    def build_critic(self):

        state_input = Input(shape=(self.state_size,))
        state_h1 = Dense(500, activation='relu', kernel_initializer='he_uniform')(state_input)
        state_h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(state_h1)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(500, activation='relu', kernel_initializer='he_uniform')(action_input)
        action_h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(action_h1)

        mergedLayer = Add()([state_h2, action_h2])
        mergedLayer_h1 = Dense(500, activation='relu', kernel_initializer='he_uniform')(mergedLayer)
        mergedLayer_h2 = Dense(250, activation='relu', kernel_initializer='he_uniform')(mergedLayer_h1)
        outputLayer = Dense(1, activation='relu')(mergedLayer_h2)

        critic = Model(inputs=[state_input, action_input], outputs=outputLayer)
        adam = Adam(lr=0.01)
        critic.compile(loss='mse', optimizer='adam')

        return state_input, action_input, critic

    def train_model(self, prior_state, state, action, reward):

        target_action = self.actor_target.predict(state, batch_size=1)
        future_reward = self.critic_target.predict([state, target_action], batch_size=1)[0][0]

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

    def update_targets(self):
        actor_weights  = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        self.actor_target.set_weights(actor_weights)
        self.critic_target.set_weights(critic_weights)


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

def angle_between(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360

#Constants to adjust
#Friendly and enemy ships to encode by distance
MAX_SHIPS = 10

MAX_RADIUS = 16
MAX_SPOTS = 6
MAX_PROD = 2500
MAX_HEALTH = 5000
PLANET_MAX_NUM = 28

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
                     'health', 'current_production', 'remaining_production', 'ownership', 'pct_docked_ships', 'pct_my_ships', 'pct_enemy_ships', 'is_full',
                     'distance', 'angle', 'nearest_friend', 'nearest_enemy', 'n_obstacles'])
        df['planet'] = np.arange(0, N_planets, 1)

        radius_array = np.zeros(N_planets)
        spots_array = np.zeros(N_planets)
        x_array = np.zeros(N_planets)
        y_array = np.zeros(N_planets)
        for planet in game_map.all_planets():
            radius_array[planet.id] = planet.radius / game_map.height
            spots_array[planet.id] = planet.num_docking_spots
            x_array[planet.id] = planet.x / game_map.width
            y_array[planet.id] = planet.y / game_map.height
        df.radius = radius_array
        df.spots = spots_array
        df.x = x_array
        df.y = y_array

        ##I need to save the starting number of planets, since they might get destroyed
        N_starting_planets = N_planets


    pct_owned.append(N_ships / (N_ships + N_enemy_ships) * 100)
    logging.info('Pct Ships Owned: {}'.format(pct_owned[turn]))

    # Here we define the set of commands to be sent to the Halite engine at the end of the turn
    command_queue = []

    #Planet / turn specific factors
    health_array = np.zeros(N_starting_planets)
    cur_prod_array = np.zeros(N_starting_planets)
    rem_prod_array = np.zeros(N_starting_planets)
    own_array = np.zeros(N_starting_planets)
    dock_array = np.zeros(N_starting_planets)
    my_ships_array = np.zeros(N_starting_planets)
    enemy_ships_array = np.zeros(N_starting_planets)
    full_array = np.zeros(N_starting_planets)
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
    #These are N counts for the moment
    df.pct_docked_ships = dock_array
    df.pct_my_ships = my_ships_array
    df.pct_enemy_ships = enemy_ships_array
    df.is_full = full_array

    #Convert docking spots to percent and drop total
    df.pct_docked_ships = df.pct_docked_ships / df.spots
    df.pct_my_ships = df.pct_my_ships / df.spots
    df.pct_enemy_ships = df.pct_enemy_ships / df.spots

    ##Calculate points for training
    points = (N_ships - N_enemy_ships) * 100
    cur_ship_ids = []
    for s in game_map.get_me().all_ships():
        cur_ship_ids.append(s.id)

    NEW_SHIP = 0
    if turn > 0:
        for s in ship_states[turn-1]:
            if s not in cur_ship_ids:
                #if we lost a ship from the last state
                points -= 100
        for ship in game_map.get_me().all_ships():
            if ship.id not in ship_states[turn-1]:
                #if we gained a new ship from the last state
                points += 100
                NEW_SHIP = 1

    logging.info('Points: {}'.format(points))
    logging.info('New Ship: {}'.format(NEW_SHIP))

    #Dict to hold ship states and actions
    ship_dict = {}
    # For every ship that I control
    for ship in game_map.get_me().all_ships():

        #Calculate matrix of values for every ship that are ship specific
        dist_array = np.zeros(N_starting_planets)
        angle_array = np.zeros(N_starting_planets)
        my_dist_array = np.zeros(N_starting_planets)
        enemy_dist_array = np.zeros(N_starting_planets)
        obstacle_array = np.zeros(N_starting_planets)
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
        train_data = df.drop(['planet', 'spots'], axis=1)

        N_FEATURES = train_data.shape[1]
        rows_to_add = PLANET_MAX_NUM-train_data.shape[0]

        #training dim is 1 x total features / state size
        td = np.vstack((train_data.values, np.zeros((rows_to_add, N_FEATURES))))
        td = td.reshape((1, td.size))
        logging.info('Planet Matrix')
        logging.info(df)
        logging.info(df.shape)

        logging.info(td.size)
        logging.info(td.shape)

        logging.info('Individual Ship Metrics')
        #Game level metrics too
        ship_metrics = [ship.x / game_map.width, ship.y / game_map.height, ship.health / MAX_HEALTH, pct_owned[turn]/100, N_planets/PLANET_MAX_NUM, N_planets/N_starting_planets]

        logging.info(ship_metrics)
        sm = np.array(ship_metrics).reshape(1, len(ship_metrics))

        td = np.concatenate((td, sm), axis=1)

        ##Add in Friendly and Enemy ship matrix
        all_ships = game_map._all_ships()
        df_ships = pd.DataFrame(np.zeros((len(all_ships), 7)), columns=['ship_id', 'x', 'y', 'player', 'distance', 'angle', 'status'])

        ship_id_array = []
        ship_x_array = []
        ship_y_array = []
        ship_player_array = []
        ship_dist_array = []
        ship_status_array = []
        ship_angle_array = []

        for s in all_ships:
            #ship_id_array = np.append(ship_id_array, s.id)
            ship_id_array.append(s.id)
            ship_x_array.append(s.x)
            ship_y_array.append(s.y)

            if s.owner.id == game_map.get_me():
                ownership = 1
            else:  # owned by enemy
                ownership = 0
            ship_player_array.append(ownership)

            ship_status_array.append(s.docking_status)
            ship_angle_array.append(ship.calculate_angle_between(s))
            dist = ship.calculate_distance_between(s)
            ship_dist_array.append(dist)

        df_ships.ship_id = ship_id_array
        df_ships.x = np.array(ship_x_array) / game_map.width
        df_ships.y = np.array(ship_y_array) / game_map.height
        df_ships.player = ship_player_array
        df_ships.distance = np.array(ship_dist_array) / game_map.height
        df_ships.angle = np.array(ship_angle_array) / 360
        df_ships.status = ship_status_array

        df_ship_train = df_ships.sort_values('distance')
        #split between friendly and enemy ships
        df_ship_train = df_ship_train.drop(['ship_id', 'status'], axis=1)

        df_ship_friendly = df_ship_train[(df_ship_train.player == 1)]
        df_ship_enemy = df_ship_train[(df_ship_train.player == 0)]

        ship_td_friendly = df_ship_friendly.values
        ship_td_enemy = df_ship_enemy.values

        N_SHIP_FEATURES = df_ship_train.shape[1]

        rows_to_add = MAX_SHIPS - ship_td_friendly.shape[0]
        if rows_to_add > 0:
            ship_td_friendly = np.vstack((ship_td_friendly, np.zeros((rows_to_add, N_SHIP_FEATURES))))
        else:
            ship_td_friendly = ship_td_friendly[0:MAX_SHIPS, :]

        rows_to_add = MAX_SHIPS - ship_td_enemy.shape[0]
        if rows_to_add > 0:
            ship_td_enemy = np.vstack((ship_td_enemy, np.zeros((rows_to_add, N_SHIP_FEATURES))))
        else:
            ship_td_enemy = ship_td_enemy[0:MAX_SHIPS, :]

        ship_td_friendly = ship_td_friendly.reshape((1, ship_td_friendly.size))
        ship_td_enemy = ship_td_enemy.reshape((1, ship_td_enemy.size))

        td = np.concatenate((td, ship_td_friendly, ship_td_enemy), axis=1)
        logging.info('Input Dim: {}'.format(td.shape))

        #output is sigmoid(0,1) length 3: percent of x, percent of y, percent of max velocity
        if 'MyAgent' not in locals():
            MyAgent = Agent(td.size, 3, sess)

        policy = MyAgent.get_action(td)

        ship_dict[ship.id] = (td, policy)

        logging.info('Policy: {}'.format(policy))
        logging.info('Points: {}'.format(points))
        #logging.info(ship_dict)

        # If the ship is docking, docked or undocking
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue

        dock_command = None
        for planet in game_map.all_planets():
            if planet.is_full():
                continue
            elif (planet.owner == game_map.get_me()) & (ship.can_dock(planet)) & (NEW_SHIP == 0):
                dock_command = ship.dock(planet)
            elif (planet.owner is None) & (ship.can_dock(planet)) & (NEW_SHIP == 0):
                dock_command = ship.dock(planet)
            # If we can dock, let's (try to) dock. If two ships try to dock at once, neither will be able to.

        if dock_command is None:
            speed = policy[0][2] * hlt.constants.MAX_SPEED
            a = angle_between(ship.x, ship.y, policy[0][0] * game_map.width, policy[0][1] * game_map.height)
            navigate_command = ship.thrust(speed, a)
        else:
            navigate_command = dock_command

        if navigate_command:
            command_queue.append(navigate_command)

    logging.info('Turn: {}'.format(turn))
    logging.info('Command: {}'.format(command_queue))
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
        MyAgent.update_targets()
        MyAgent.actor_target.save_weights("./model/actor.h5")
        MyAgent.critic_target.save_weights("./model/critic.h5")

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

#Windows
#python hlt_client\client.py gym -r "python MyBot.py" -r "python MyBot-Agent.py" -b "halite" -i 100 -H 160 -W 240
#.\halite -d "240 160" -t "python MyBot.py" "python MyBot-Agent.py"
#.\halite -d "240 160" -t "python MyBot-Agent2.py" "python MyBot.py"

#.\halite -d "240 160" -t "python MyBot.py" "python MyBot-adam.py"
#.\halite -d "240 160" -t "python MyBot-adadelta.py" "python MyBot.py"
#python hlt_client\client.py gym -r "python MyBot.py" -r "python MyBot-adam.py" -b "halite" -i 1000 -H 160 -W 240
#python hlt_client\client.py gym -r "python MyBot-adadelta.py" -r "python MyBot.py" -b "halite" -i 1000 -H 160 -W 240
#python hlt_client\client.py gym -r "python MyBot-adadelta.py" -r "python MyBot-adam.py" -b "halite" -i 100 -H 160 -W 240

#Mac
#./halite -d "240 160" -t "python MyBot-Agent.py" "python MyBot.py"
#./hlt_client/client.py gym -r "python MyBot-adadelta.py" -r "python MyBot-adam.py" -b "./halite" -i 100 -H 160 -W 240
