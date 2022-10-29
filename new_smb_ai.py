# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:49:08 2020

@author: Batman
"""
import time
import os
import numpy as np
import collections
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from gym.core import ObservationWrapper
from gym.spaces.box import Box
import cv2
from utils import plotLearning

#PARTE 1 - PREPROCESSAMENTO E FUNCIONALIDADES   
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.float64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = bool)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_ctr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, dones


class PreprocessFrame(ObservationWrapper):
    def __init__(self, shape, env = None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = Box(low = 0.0, high = 1.0, shape = self.shape)
        
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation = cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype = np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs
    
class StackFrames(ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = Box(env.observation_space.low.repeat(repeat, axis = 0), env.observation_space.high.repeat(repeat, axis = 0))
                                     
        self.stack = collections.deque(maxlen = repeat)
    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
           
        return (np.array(self.stack).reshape(self.observation_space.low.shape))
    
    def observation(self, observation):
        self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)
  
#class RepeatAction():
        

def make_env(env_name, shape = (84, 84, 1), repeat = 4):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    
    return env

#------------------------------------------------------------------------------------------------


#PARTE 2 - CONSTRUINDO O AGENTE


#Construindo arquitetura da CNN
class Network(nn.Module):
    def __init__(self, lr, input_dims, n_actions, checkpoint_dir, name):
        super(Network, self).__init__()
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.conv1 = nn.Conv2d(in_channels = input_dims[0], out_channels = 32, kernel_size = 8)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        
        self.fc1 = nn.Linear(in_features = self.count_neurons((input_dims[0], 84, 84)), out_features = 256)
        self.fc2 = nn.Linear(256, self.n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
    def count_neurons(self, image_dim):
        x = Variable(T.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x.data.view(1, -1).size(1)
        
        
    def forward(self, state):
        state = state.float()
        x = F.relu(F.max_pool2d(self.conv1(state), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        actions = self.fc2(x)
        return actions
    
    def save_checkpoint(self):
        print('... salvando checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

#Montando o agente    
class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size,
                 eps_min = 0.01, eps_decay = 1e-5, replace = 1000, env_name = None,
                 algo = None, checkpoint_dir = 'C:/Users/8Bit/Desktop/Machine Learning A-Z/0_SUPER MARIO BROS PROJECT/checkpoints/'):
        
        self.env_name = env_name
        self.algo = algo
        self.replace_target_ctr = replace
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_ctr = 0
        
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay= eps_decay
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.Q_eval = Network(self.lr, self.input_dims, self.n_actions, self.checkpoint_dir,
                              self.env_name + '_' + self.algo + 'q_eval')
        
        self.Q_next = Network(self.lr, self.input_dims, self.n_actions, self.checkpoint_dir,
                              self.env_name + '_' + self.algo + 'q_next')
        
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation])
        
            #tensor sai com formato [1, 4, 84, 84], converter para [4, 1, 84, 84]
            #state = state.numpy()
            #state = T.from_numpy(np.moveaxis(state, 1, 0))
            
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
            
            if action >= 7 and action < 14:
                action -= 7
            elif action >= 14 and action < 21:
                action -= 14
            elif action >= 21:
                action -= 21
            else:
                action = action
            
        else:
            action = np.random.choice(self.action_space)
        
        #print(SIMPLE_MOVEMENT[action])
        return action
    
    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_decay
        else:
            self.epsilon = self.eps_min
            
            
            
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(state)
        actions = T.tensor(action)
        rewards = T.tensor(reward)
        new_states = T.tensor(new_state)
        dones = T.tensor(done)
        
        return states, actions, rewards, new_states, dones
    
    def replace_target_network(self):
        if self.learn_step_ctr % self.replace_target_ctr == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            
    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()
        
    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()
        
        
    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, new_states, dones = self.sample_memory()
        
        q_pred = self.Q_eval.forward(states)
        q_next = self.Q_next.forward(new_states).max(dim = 1)[0]
        q_next[dones] = 0.0
        q_target = reward + self.gamma * q_next
        
        #converte formato [32] pra [32, 1]
        q_target = q_target.unsqueeze(1)
        
        loss = self.Q_eval.loss(q_target, q_pred)
        loss = Variable(loss, requires_grad = True)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_ctr += 1
        self.decrement_epsilon()
     
#------------------------------------------------------------------------------------------------
        
#PARTE 3 - MONTANDO O AMBIENTE

#Importando o ambiente
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

#Iniciando o ambiente
#env = gym_super_mario_bros.make("SuperMarioBros-v0")
#env = PreprocessFrame((84, 84, 1), env)
#env = image_preprocessing.PreprocessImage(gym_super_mario_bros.make("SuperMarioBros-v0"), width = 84, height = 84, grayscale = True)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = make_env("SuperMarioBros-v0")
load_checkpoint = False
n_games = 400
scores = []
eps_history = []
steps_array = []
n_steps = 0
agent = Agent(lr = 0.01, input_dims = env.observation_space.shape, n_actions = env.action_space.n,
              gamma = 0.99, epsilon = 1.0, mem_size = 10000, batch_size = 32,
              env_name = "SuperMarioBros-v0", algo = 'DQNAgent')

if load_checkpoint:
    agent.load_models()

for episode in range(n_games):
    done = False
    start = time.time()
    score = 0
    best_score = -np.inf
    obs = env.reset()
    
    
    while not done:
        action = agent.choose_action(obs)
        new_obs, reward, done, info = env.step(action)
        score += reward
        
        if not load_checkpoint:
            agent.store_transition(obs, action, reward, new_obs, int(done))
            agent.learn()
        obs = new_obs
        n_steps +=1
        #env.render()
        
    scores.append(score)
    steps_array.append(n_steps)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    end = time.time()
    print(f'episode: {episode} score: {score} average score: {avg_score} epsilon: {agent.epsilon}) elapsed time: {end - start} seconds')
    
    if avg_score > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = avg_score
    
    #x = np.array([i+1 for i in range(n_games)])
    #filename = 'smb-preformance.png'
    #plotLearning(x, scores, eps_history, filename)
# Close device
env.close()