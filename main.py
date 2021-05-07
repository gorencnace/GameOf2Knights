import gym
from tqdm import tqdm
from time import time
from gym_knights.envs.knights_env import KnightsEnv
import gym_knights
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(64, 10)
        self.hidden2 = nn.Linear(10, 10)
        #self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 8)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        #x = self.hidden3(x)
        #x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

env = gym.make('knights-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

num_episodes = 10000000
show = 1000

BATCH_SIZE = 128
GAMMA = 0.7
'''
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 2000'''
TARGET_UPDATE = 50
epsilon = 0.9999995
time1 = 0
time2 = 0

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net1 = Network().to(device)
target_net1 = Network().to(device)
target_net1.load_state_dict(policy_net1.state_dict())
target_net1.eval()

optimizer1 = optim.RMSprop(policy_net1.parameters())
memory1 = ReplayMemory(10000)

policy_net2 = Network().to(device)
target_net2 = Network().to(device)
target_net2.load_state_dict(policy_net2.state_dict())
target_net2.eval()

optimizer2 = optim.RMSprop(policy_net2.parameters())
memory2 = ReplayMemory(10000)



def select_action(state, player, step):
    global steps_done
    sample = random.random()
    eps_threshold = epsilon**step
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if player == 1:
                a = policy_net1(state)
            else:
                a = policy_net2(state)
            b = a.max(1)
            c = b[1]
            d = c.view(1,1)
            return d#policy_net1(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []



def optimize_model(player):
    if player == 1:
        memory = memory1
        policy_net = policy_net1
        target_net = target_net1
        optimizer = optimizer1
    else:
        memory = memory2
        policy_net = policy_net2
        target_net = target_net2
        optimizer = optimizer2

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net1
    a = policy_net(state_batch)
    b = a.gather(1, action_batch)
    state_action_values = b

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net1; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net1.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    if i_episode % show == 0:
        print(f'Episode {i_episode}:')
        env.render()
    current_player = 1
    state = torch.from_numpy(np.reshape(state.flatten(), (1, 64))).float().to(device)
    #with tqdm(total=num_episodes, desc=str(i_episode) + "/" + str(num_episodes), unit='episodes') as prog_bar:
    for t in count():
        if current_player == 1:
            memory = memory1
            time1 += 1
            time = time1
        else:
            memory = memory2
            time2 += 1
            time = time2
        # Select and perform an action
        action = select_action(state, current_player, time)
        next_state, reward, done, _ = env.step((current_player, action.item()))
        reward = torch.tensor([reward], device=device)

        next_player = (t + 1) % 2 + 1
        next_state = torch.from_numpy(np.reshape(next_state.flatten(), (1, 64))).float().to(device)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Perform one step of the optimization (on the policy network)
        optimize_model(current_player)
        if i_episode % show == 0:
            env.render()
            print(f'Player {current_player}: {time}\n')
        current_player = next_player
        if done:
            episode_durations.append(t + 1)
            break
        #prog_bar.update(1)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net1.load_state_dict(policy_net1.state_dict())
        target_net2.load_state_dict(policy_net2.state_dict())

print('Complete')
env.close()
