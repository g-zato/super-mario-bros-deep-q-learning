"""
Created on Thu Jun 25 01:08:55 2020

@author: Batman
"""
#APIs and libraries
import time
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import optim as optim
from torch.autograd import Variable

#Importing other files
import experience_replay, image_preprocessing

#Importing the environment
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY


#Part 1 - Building the AI


#Building the brain

class CNN(nn.Module):
    
    def __init__(self, number_actions):        #cria as variaveis para as demais funçoes
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 8)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = self.num_neurons_calculator((1, 84, 84)), out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = number_actions)
        
    def num_neurons_calculator(self, image_dim):  #cria uma imagem aleatoria para passar pelas camadas e descobrir quantos neuronios devem ter na primeira camada escondida
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x.data.view(1, -1).size(1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = x.view(x.size(0), -1)       #flattening
        x = F.relu(self.fc1(x))     #conexao entre flattening layer e hidden layer
        x = F.relu(self.fc2(x))     #conexao entre hidden layer e output layer
        return x    #output com q valores


#Building the body
        
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
        
    def forward(self, outputs)   :
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions

#Assembling the AI

class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()


#Part 2 - Implementing Deep Convolutional Q-Learning
        
    

# Make gym environment
env = image_preprocessing.PreprocessImage(gym_super_mario_bros.make('SuperMarioBros-v0'), width = 84, height = 84, grayscale = True)
env = gym.wrappers.Monitor(env, "videos", force = True)
env = JoypadSpace(env, RIGHT_ONLY)
number_actions = env.action_space.n
print(env.action_space.n)

#Building the AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 2.0)
ai = AI(brain = cnn, body = softmax_body)   

#Implementing Experience Replay
n_steps = experience_replay.NStepProgress(env = env, ai= ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)


#Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs= []
    targets = []
    
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumulative_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumulative_reward = step.reward + gamma * cumulative_reward
            print(str(step.reward))     #TESTEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumulative_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)
 
#Calculating the average movement in the last 100 steps
    
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
        
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
            
    def average(self):
        print(self.list_of_rewards)  #TESTEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        return np.mean(self.list_of_rewards)
    
ma = MA(100)    
   
#Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.002)  
nb_epochs = 100
#step = 0


for epoch in range(1, nb_epochs+1):
    #memory.run_steps(200)
    #while True:
        #env.render()
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()       #initializes the optimizer
        loss_error.backward()       #Apply backward propagation
        optimizer.step()            #Apply stochastic gradient descent
    
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), (avg_reward))) 
    # If done break loop
    '''if done:
        break'''
    
#Close the environment
env.close()