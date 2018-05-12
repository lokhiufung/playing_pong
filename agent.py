import numpy as np
import pandas as pd
import random
from collections import deque
from keras.layers import Conv2D, Dense, Input, Flatten, Reshape, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras.regularizers import l2

from pygame.locals import *
import pygame.event
import sys
import pong

import warnings
warnings.filterwarnings('ignore')

# hyperparameters
learning_rate = 1e-4
batch_size = 32

h1 = 300  # number of hidden units in hidden layer 1
h2 = 300 # number of hidden units in hidden layer 2

K = 1000  # how many frames to update target_network]
observation_steps = 5000
exploration_decay = 10000
replay_buffer = 100000
gamma = 0.99
tau = 0.3
initial_exploration_rate = 1.0
final_exploration_rate = 0.05

exploration_step = (initial_exploration_rate -
                    final_exploration_rate)/exploration_decay
# maybe try to learn from low dimension space first
# then try directly on the pixel space
input_dims = [5, 1]
action_dims = 3
nb_channels = 1

class Agent():
    def __init__(self, optimizer='rmsprop', mode='low_dims'):
        self.mode = mode
        if self.mode == 'low_dims':
            self.action_network = self.create_brain_low_dims(optimizer)
            self.target_network = self.create_brain_low_dims(optimizer)
        if self.mode == 'high_dims':
            self.action_network = self.create_brain_high_dims()
            self.target_network = self.create_brain_high_dims()

        self.memory = deque(maxlen=replay_buffer)

    def create_brain_high_dims(self):
        input_layer = Input(input_dims)
        x = Conv2D(64, kernel_size=[4, 4], strides=[1, 1],
                   padding='valid', activation='relu')(input_layer)
        x = Conv2D(32, kernal_size=[2, 2], strides=[2, 2],
                   padding='valid', activation='relu')(x)
        x = Conv2D(32, kernel_size=[2, 2], strides=[2, 2],
                   padding='valid')
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(action_dims, activation='linear')(x)

        model = Model(input_layer, x)
        rmsprop = RMSprop(lr=learning_rate)
        model.compile(optimizer=rmsprop, loss='mse')
        return model

    def create_brain_low_dims(self, optimizer='rmsprop'):
        input_layer = Input(input_dims)
        x = Reshape((input_dims[0], ))(input_layer)
        x = Dense(h1, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(rate=0.6)(x)
        x = Dense(h2, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dense(action_dims, activation='linear')(x)

        model = Model(input_layer, x)
        if optimizer == 'rmsprop':
            rmsprop = RMSprop(lr=learning_rate)
            model.compile(optimizer=rmsprop, loss='mse')
        elif optimizer == 'adam':
            adam = Adam(lr=learning_rate)
            model.compile(optimizer=adam, loss='mse')
        else:
            sgd = SGD(lr=learning_rate)
            model.compile(optimizer=sgd, loss='mse')

        return model

    def experience_replay(self):
        mini_batch = random.sample(self.memory, batch_size)
        if self.mode == 'low_dims':
            # shape = (batch_size, 5)
            s1 = np.zeros((batch_size, input_dims[0], 1))
            s2 = np.zeros((batch_size, input_dims[0], 1))
        if self.mode == 'high_dims':
            # shape = (batch_size, 64, 64, 1)
            s1 = np.zeros((batch_size, input_dims[0],
                           input_dims[1]), nb_channels)
            s2 = np.zeros((batch_size, input_dims[0],
                           input_dims[1]), nb_channels)
        a = np.zeros(batch_size)
        r = np.zeros(batch_size)

        done = np.zeros(batch_size)

        for i in range(batch_size):

            s1[i] = mini_batch[i][0]
            a[i] = mini_batch[i][1]
            r[i] = mini_batch[i][2]
            s2[i] = mini_batch[i][3]
            done[i] = mini_batch[i][4]

        q = self.action_network.predict(s2)
        a_greedy = np.argmax(q, axis=1)
        q_target = self.action_network.predict(s1)
        q_target_next = self.target_network.predict(s2)

        for i in range(batch_size):
            if not done[i]:
                # double q learning
                q_target[i, int(a[i])] = r[i] + gamma * q_target_next[i, int(a_greedy[i])]
                # dqn
                # q_target[i, int(a[i])] = r[i] + gamma * np.max(q_target_next[i])
            elif done[i]:
                q_target[i, int(a[i])] = r[i]

        # loss on batch
        history = self.action_network.train_on_batch(s1, q_target)
        return history

    def remember(self, state1, action, reward, state2, done):
        self.memory.append((state1, action, reward, state2, done))

    def train_target(self, tau):
        weights = self.action_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = (1 - tau) * target_weights[i] + tau * weights[i]
        self.target_network.set_weights(target_weights)

    def save_model(self):
        # serialize model to JSON
        action_model_json = self.action_network.to_json()
        target_model_json = self.target_network.to_json()
        with open("action.json", "w") as json_file:
            json_file.write(action_model_json)
        with open("target.json", "w") as json_file:
            json_file.write(target_model_json)
        # serialize weights to HDF5
        self.action_network.save_weights("action.h5")
        self.target_network.save_weights("target.h5")


def main():

    exploration_rate = initial_exploration_rate
    game = pong.Pong(mode='low_dims')
    agent = Agent(optimizer='rmsprop', mode='low_dims')
    print(agent.action_network.summary())
    done = False

    training_history = []
    state_1 = np.expand_dims(game.GetPresentFrame(normalize=True), axis=0)

    # loss = []
    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or game.total_score >= 20:
                sys.exit()

        Q = agent.action_network.predict(state_1)
        if np.random.random_sample() > exploration_rate:
            action = np.argmax(Q)
            # print(Q)
        else:
            action = np.random.randint(3)

        reward, state_2 = game.GetNextFrame(action, normalize=True)

        agent.remember(state_1, action, reward,
                       np.expand_dims(state_2, axis=0), done)

        if counter > observation_steps:
            if counter == observation_steps + 1:
                print('learning start!')
            history = agent.experience_replay()

            if counter % 100 == 0:
                training_history.append([history, np.max(Q)])
                pd.DataFrame(training_history, columns=['loss', 'Q']).to_csv('training_history.csv')
                print('{}th frame: loss {}, Q {}'.format(counter - observation_steps, history*1000, Q))

            if counter % K == 0:
                agent.train_target(tau)

            if counter % 1000 == 0:
                agent.save_model()
                print('Made a copy of model')

        # decay exploration rate
        if exploration_rate >= final_exploration_rate:
            exploration_rate -= exploration_step

        # update state and counter
        counter += 1
        state_1 = np.expand_dims(state_2, axis=0)

if __name__ == '__main__':
    main()
