import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Conv2D, Dense, Input, Flatten, Reshape, Dropout
from keras.optimizers import RMSprop
from keras.models import Model

from pygame.locals import *
import pygame.event
import sys
import pong

# hyperparameters
learning_rate = 1e-3
batch_size = 32

K = 1000  # how many frames to update target_network
replay_buffer = 10000
gamma = 0.99
tau = 0.5
initial_exploration_rate = 1.0
final_exploration_rate = 0.1

exploration_step = (initial_exploration_rate -
                    final_exploration_rate)/replay_buffer
# maybe try to learn from low dimension space first
# then try directly on the pixel space
input_dims = [5, 1]
action_dims = 3
nb_channels = 1

class Agent():
    def __init__(self, mode='low_dims'):
        self.mode = mode
        if self.mode == 'low_dims':
            self.action_network = self.create_brain_low_dims()
            self.target_network = self.create_brain_low_dims()
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

    def create_brain_low_dims(self):
        input_layer = Input(input_dims)
        x = Reshape((input_dims[0], ))(input_layer)
        x = Dense(8, activation='relu')(x)
        x = Dropout(rate=0.6)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(rate=0.6)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(rate=0.6)(x)
        x = Dense(action_dims, activation='relu')(x)

        model = Model(input_layer, x)
        rmsprop = RMSprop(lr=learning_rate)
        model.compile(optimizer=rmsprop, loss='mse')
        return model

    def experience_replay(self):
        mini_batch = random.sample(self.memory, batch_size)
        if self.mode == 'low_dims':
            # shape = (batch_size, 6)
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
        q_target = self.target_network.predict(s1)
        q_target_next = self.target_network.predict(s2)

        for i in range(batch_size):
            if not done[i]:
                q_target[i, int(a[i])] = r[i] + gamma * q_target_next[i, int(a_greedy[i])]
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


def main():

    exploration_rate = initial_exploration_rate
    game = pong.Pong(mode='low_dims')
    agent = Agent(mode='low_dims')
    done = False

    loss = []
    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        state_1 = np.expand_dims(game.GetPresentFrame(normalize=True), axis=0)
        if np.random.random_sample() > max(exploration_rate,
                                           final_exploration_rate):
            action = np.argmax(agent.action_network.predict(state_1))
        else:
            action = np.random.randint(3)
        reward, state_2 = game.GetNextFrame(action, normalize=True)
        agent.remember(state_1, action, reward,
                       np.expand_dims(state_2, axis=0), done)

        if len(agent.memory) == replay_buffer:
            history = agent.experience_replay()

            if counter % 100 == 0:
                loss.append(history)
                print('loss at {}th frame is {}'.format(counter - replay_buffer, history*1000))

            if counter % K == 0:
                agent.train_target(tau)
        # decay exploration rate
        if exploration_rate >= final_exploration_rate:
            exploration_rate -= exploration_step
        counter += 1

if __name__ == '__main__':
    main()
