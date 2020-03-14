import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import softmax
from keras.optimizers import Adam
import os
import time
import numpy as np
import math
def onpol_softmax(x, temperature=0.025): 
    """Compute softmax values for each sets of scores in x."""
    x = (x - np.expand_dims(np.max(x, 1), 1))
    x = x/temperature    
    e_x = np.exp(x)
    return e_x / (np.expand_dims(e_x.sum(1), -1) + 1e-5)

def softmax(x, temperature=0.025): 
    #Compute softmax values for each sets of scores in x.
    x1 = x[0][0]/temperature
    x2 = x[0][1]/temperature
    sum = np.exp(x1) + np.exp(x2)    
    return [np.exp(x1) / sum , np.exp(x2) / sum];

class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=20000)
    self.gamma = 0.95    # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.model = self._build_model()

    
  def _build_model(self):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):# We implement the epsilon-greedy policy
    act_values = self.model.predict(state) #oftmax(act_values[0])
    if np.random.rand() <= self.epsilon:
        i = np.random.choice(np.arange(act_values[0].size), p = softmax(act_values, temperature = 0.8))
        check = math.isnan(i)
        if check == True:
          return [np.argmax(act_values[0]), np.amax(act_values[0])]
        else:
          return [i, act_values[0][i]];#random.randrange(self.action_size)
    return [np.argmax(act_values[0]), np.amax(act_values[0])]; # returns action
  def act2(self, state):# We implement the epsilon-greedy policy
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0]) # returns action
  def on_pol2()
  def on_pol(self, state):
    act_values = self.model.predict(state)
    q_table = np.zeros([batch_size])

    if np.random.rand() <= self.epsilon:
      pred_num = onpol_softmax(act_values)
      for i in range(batch_size):
        index = np.random.choice([0,1], p = pred_num[i])
        if math.isnan(index)==True:
          q_table[i] = np.amax(act_values[i]);
        else:
          q_table[i] = act_values[i][index];
    for i in range(batch_size):
      q_table[i] = np.amax(act_values[i])
    return q_table;
                        
    
  def exploit(self, state): # When we test the agent we dont want it to explore anymore, but to exploit what it has learnt
    act_values = self.model.predict(state)
    return np.argmax(act_values[0]) 

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    
    state_b = np.squeeze(np.array(list(map(lambda x: x[0], minibatch))))
    action_b = np.squeeze(np.array(list(map(lambda x: x[1], minibatch))))
    reward_b = np.squeeze(np.array(list(map(lambda x: x[2], minibatch))))
    next_state_b = np.squeeze(np.array(list(map(lambda x: x[3], minibatch))))
    done_b = np.squeeze(np.array(list(map(lambda x: x[4], minibatch))))

    ### Q-learning

    target = (reward_b + self.gamma * self.on_pol(next_state_b))#np.amax(self.model.predict(next_state_b), 1))
    target[done_b==1] = reward_b[done_b==1]
    target_f = self.model.predict(state_b)

    for k in range(target_f.shape[0]):
      target_f[k][action_b[k]] = target[k]
    self.model.train_on_batch(state_b, target_f)
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)
  def save(self, name):
    self.model.save_weights(name)
