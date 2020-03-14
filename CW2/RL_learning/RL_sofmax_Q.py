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

def softmax(x, temperature=0.025): 
    """Compute softmax values for each sets of scores in x."""
    x = (x - np.expand_dims(np.max(x, 1), 1))
    x = x/temperature    
    e_x = np.exp(x)
    return e_x / (np.expand_dims(e_x.sum(1), -1) + 1e-5)

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
    """
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)
    """
    act_values = self.model.predict(state)
    return np.argmax(softmax(act_values)[0]) # returns action

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
    target = (reward_b + self.gamma * np.amax(self.model.predict(next_state_b), 1))
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

    
  EPISODES = 100
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  batch_size = 32
  episode_reward_list = deque(maxlen=100)
  episode_reward_avg = np.zeros([100])
  for e in range(EPISODES):
      state = env.reset()
      state = np.reshape(state, [1, state_size])
      total_reward = 0
      for time in range(1000):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if len(agent.memory) > batcQh_size:
            agent.replay(batch_size)
      episode_reward_list.append(total_reward)
      episode_reward_avg[e] = np.array(episode_reward_list).mean()
      print("episode: {}/{}, score: {}, e: {:.2}, last 100 episodes rew: {:.2f}"
                  .format(e, EPISODES, total_reward, agent.epsilon, episode_reward_avg[e]))
from google.colab  import files

agent.save('weights.h5')
# Download the weights in your PC
files.download('weights.h5') 

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.load('weights.h5')

!apt-get install -y xvfb python-opengl > /dev/null 2>&1
!pip install gym pyvirtualdisplay > /dev/null 2>&1

import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
import numpy as np
import gym
import random

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

env = wrap_env(gym.make('CartPole-v1'))
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
done = False
state = env.reset()
state = np.reshape(state, [1, state_size])
for time in range(1000000000):
    screen = env.render()
    action = agent.exploit(state)
    state, reward, done, _ = env.step(action)
    state = np.reshape(state, [1, state_size])
    if done:
      break
env.close()
show_video()
env.reset()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(0)
plt.grid(axis='both', which = 'both', linewidth = 0.4)
plt.plot(np.linspace(1,100,100),episode_reward_avg, linewidth=0.5, alpha =0.9, label='Reward')
plt.fill_between(np.linspace(1,100,100), np.zeros([len(episode_reward_avg)]), episode_reward_avg, alpha =0.2)
plt.xlabel('epoch')
plt.legend()
plt.show()
