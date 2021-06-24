import pickle

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
import torch.nn.functional as f
import torchvision.transforms as T


class DQNCNNPolicy(nn.Module):
    def __init__(self):
        super(DQNCNNPolicy, self).__init__()
        self.inp = nn.Conv2d(1, 16, (2, 2))
        self.conv_1 = nn.Conv2d(16, 32, (2, 2))
        self.linear_1 = nn.Linear(32 * 3 * 3, 64)
        self.linear_2 = nn.Linear(64, 8)

    def forward(self, x):
        x = f.relu(self.inp(x))
        x = f.relu(self.conv_1(x))
        x = f.relu(x.flatten(1))
        x = f.relu(self.linear_1(x))
        return f.relu(self.linear_2(x))

def learnn():
    transition = namedtuple('transitions', ('state', 'action', 'reward', 'next_state'))


    class ReplayMemory:
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []

        def push(self, state, action, reward, next_state):
            if len(self.memory) > self.capacity:
                self.memory.pop()
            self.memory.append(transition(state, action, reward, next_state))

        def sample_batch(self, batch_size):
            if len(self) > batch_size:
                return random.sample(self.memory, batch_size)
            return []

        def __len__(self):
            return len(self.memory)


    def get_greedy_action(policy, state):
        assert state.shape == (1, 5, 5)
        state = state.unsqueeze(0)
        return policy(state)


    def get_epsilon_greedy_action(policy, state, env, eps, player):
        prob = np.random.uniform()

        if prob > eps:
            with torch.no_grad():
                return get_greedy_action(policy, state).argmax().unsqueeze(0)
        return torch.from_numpy(np.array([env.random_action(player)]))


    BATCH_SIZE = 32
    GAMMA = 0.9


    def optimize_model(policy, target, criterion, optimizer, memory, device):
        sample = memory.sample_batch(BATCH_SIZE)

        non_final_states_mask = [True if item.next_state is not None else False for item in sample]

        states = torch.stack([item.state for item in sample])
        actions = torch.stack([item.action for item in sample])
        rewards = torch.stack([item.reward for item in sample])

        non_final_states = torch.stack([item.next_state for item in sample if item.next_state is not None]).to(device)
        next_states = torch.zeros(states.shape).to(device)
        next_states[non_final_states_mask] = non_final_states

        aaaa = policy(states).reshape(32, 8, 1)
        state_action = aaaa.gather(1, actions)
        next_state_action = target(next_states).max(1)[0].unsqueeze(0).T.detach()

        expected = rewards + GAMMA * next_state_action
        optimizer.zero_grad()
        loss = criterion(state_action, expected)
        loss.backward()
        optimizer.step()
        return loss.item()


    ALPHA = 0.001
    n_episodes = 100000
    E = 0.999995
    POLICY_UPDATE = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DQNCNNPolicy().to(device)
    target_policy = DQNCNNPolicy().to(device)
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval()

    env = gym.make('knights-v0').unwrapped
    replay_memory = ReplayMemory(5000)
    loss_criterion = nn.MSELoss()
    optimzier_func = optim.SGD(policy.parameters(), lr=ALPHA)

    with tqdm(range(n_episodes), unit='episode') as tq_ep:
        for ep in tq_ep:
            EPSILON = E**(ep + 1)
            tq_ep.set_description(f"Episode: {ep + 1}")
            done = False
            state = env.reset()
            state = torch.from_numpy(np.reshape(state.flatten(), (1, 5, 5))).float().to(device)
            ep_loss, ep_reward = 0, 0
            current_player = 1
            while not done:
                if current_player == 2:
                    action = env.random_action(current_player)
                    next_state, _, done, _ = env.step((current_player, action))
                else:
                    action = get_epsilon_greedy_action(policy, state, env, EPSILON, current_player)
                    next_state, reward, done, _ = env.step((current_player, action))
                    reward = torch.from_numpy(np.array([[reward]])).float().to(device)
                    action = torch.reshape(action, (1, 1)).long().to(device)
                    next_state = torch.from_numpy(np.reshape(next_state.flatten(), (1, 5, 5))).float().to(device)
                    ep_reward += reward[0].item()
                if done:
                    next_state = None
                if current_player == 1:
                    replay_memory.push(state, action, reward, next_state)

                if len(replay_memory) > BATCH_SIZE:
                    ep_loss += optimize_model(policy, target_policy, loss_criterion,
                                              optimzier_func, replay_memory, device)
                state = next_state
            tq_ep.set_postfix(ep_reward=ep_reward)

            if ep % POLICY_UPDATE == 0:
                target_policy.load_state_dict(policy.state_dict())

    torch.save({'player1': policy.state_dict(),
                'optimizer1_dict': optimzier_func.state_dict()},
               'models/network_agent1_test')
    torch.load('models/network_agent1_test')

def get_win_percentages(agent, n_rounds=100):
    env = gym.make('knights-v0')
    w1 = 0
    w2 = 0
    i1 = 0
    i2 = 0
    for i in range(n_rounds):
        done = False
        state = env.reset()
        state = torch.from_numpy(np.reshape(state.flatten(), (1, 5, 5))).float()
        current_player = 1
        env.render()
        while not done:
            if current_player == 2:
                action = env.random_action(current_player)
                next_state, reward, done, _ = env.step((current_player, action))
            else:
                a = agent(state, env.get_legal_moves(current_player))
                action = a  # .max(1)[1].view(1, 1).item()
                next_state, reward, done, _ = env.step((current_player, action))
            state = torch.from_numpy(np.reshape(next_state.flatten(), (1, 5 * 5))).float()
            # state = next_state
            if done:
                if reward == env.invalid_move:
                    if current_player == 1:
                        i1 += 1
                    else:
                        i2 += 1
                if current_player == 1:
                    w2 += 1
                else:
                    w1 += 1
            current_player = current_player % 2 + 1
            env.render()

    print("Agent 1 Win Percentage:", np.round(w1 / n_rounds, 2))
    print("Agent 2 Win Percentage:", np.round(w2 / n_rounds, 2))
    print("Number of Invalid Plays by Agent 1:", i1)
    print("Number of Invalid Plays by Agent 2:", i2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNCNNPolicy().to(device)
checkpoint = torch.load('models/network_agent1_test')
agent.load_state_dict(checkpoint['player1'])
agent.eval()

def agent1(obs, valid_moves):
    with torch.no_grad():
        state = torch.tensor(obs, dtype=torch.float).reshape(1, 1, 5, 5).to(device)
        c = agent(state)
        col = c.argmax().item()
    if len(valid_moves) == 0 or int(col) in valid_moves:
        return int(col)
    else:
        probs = c.cpu().detach().numpy()[0]
        max_prob = min(probs)
        max_move = np.where(probs == max_prob)[0]
        for move in valid_moves:
            if probs[move] >= max_prob:
                max_move = move
                max_prob = probs[move]
        return max_move
'''
        return random.choice(
            valid_moves
        )'''

get_win_percentages(agent1)






class Network(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.hidden1 = nn.Linear(n * n, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.hidden3 = nn.Linear(50, 50)
        self.hidden4 = nn.Linear(50, 50)
        self.hidden5 = nn.Linear(50, 50)
        self.hidden6 = nn.Linear(50, 50)
        self.hidden7 = nn.Linear(50, 50)
        self.hidden8 = nn.Linear(50, 50)
        self.hidden9 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 8)

        self.relu = nn.Sigmoid()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)

        x = self.hidden2(x)
        x = self.relu(x)

        x = self.hidden3(x)
        x = self.relu(x)

        x = self.hidden4(x)
        x = self.relu(x)

        x = self.hidden5(x)
        x = self.relu(x)

        x = self.hidden6(x)
        x = self.relu(x)

        x = self.hidden7(x)
        x = self.relu(x)

        x = self.hidden8(x)
        x = self.relu(x)

        x = self.hidden9(x)
        x = self.relu(x)

        x = self.output(x)
        x = self.softmax(x)

        return x


class Memory():
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def get_tensors(self, device):
        obs = torch.cat([torch.unsqueeze(x, 0) for x in self.observations])
        try:
            act = torch.cat(self.actions).view(len(self.actions), 1)
        except:
            x = 1
        rew = torch.cat(self.rewards).view(len(self.rewards), 1)
        return obs, act, rew


def learn():
    env = gym.make('knights-v0').unwrapped

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    START = 0
    END = 40000
    RENDER = 5000

    EPSILON = 0.99985**START

    N = 5

    player1 = Network(N).to(device)
    #player2 = Network(N).to(device)

    if START != 0:
        checkpoint = torch.load('models/network_minus')
        player1.load_state_dict(checkpoint['player1'])
        player1.train()
        #player2.load_state_dict(checkpoint['player2'])
        #player2.train()
        optimizer1 = optim.Adam(player1.parameters(), lr=0.001)
        optimizer1.load_state_dict(checkpoint['optimizer1_dict'])
        #optimizer2 = optim.Adam(player2.parameters(), lr=0.001)
        #optimizer2.load_state_dict(checkpoint['optimizer2_dict'])
    else:
        optimizer1 = optim.Adam(player1.parameters(), lr=0.001)
        #optimizer2 = optim.Adam(player2.parameters(), lr=0.001)

    def loss_function(x, y, z):
        # a = nn.LogSoftmax(dim=1)
        b = nn.NLLLoss()
        # m = a(x)
        n = b(x.transpose(1, 2), y)
        k = torch.mean(n * z)
        return k

    memory1 = Memory()
    #memory2 = Memory()

    def get_action(model, state, valid):
        act = np.random.choice(['model', 'random'], 1, p=[1 - EPSILON, EPSILON])[0]
        probabilities = model(state)
        if act == 'model':
            action = probabilities.max(1)[1].view(1)
        else:
            if not valid:
                valid = [0, 1, 2, 3, 4, 5, 6, 7]
            action = torch.tensor([np.random.choice(valid)], device=device, dtype=torch.long)
        return action, probabilities.cpu().detach().numpy()[0]

    with tqdm(total=END - START, unit='episode') as prog_bar:
        for i_episode in range(START, END):
            memory1.clear()
            #memory2.clear()
            state = env.reset()
            if (i_episode + 1) % RENDER == 0:
                print(f'Episode {i_episode + 1}:')
                env.render()
            current_player = 1
            observation = torch.from_numpy(np.reshape(state.flatten(), (1, N * N))).float().to(device)
            done = False
            EPSILON = EPSILON * 0.99985
            while not done:
                if current_player == 1:
                    player_net = player1
                    memory = memory1
                #else:
                    #player_net = player2
                    #memory = memory2

                valid_actions = env.get_legal_moves(current_player)
                if current_player == 1:
                    action, probabilities = get_action(player_net, observation, valid_actions)
                    temp_action = action.item()
                    if len(valid_actions) > 0 and temp_action not in valid_actions:
                        _, fake_reward, _, _ = env.fake_step(current_player, action.item())
                        fake_reward = torch.tensor([fake_reward], device=device)
                        memory.add_to_memory(observation, action, fake_reward)
                        temp_prob = min(probabilities)
                        for i_act in valid_actions:
                            if temp_prob <= probabilities[i_act]:
                                temp_prob = probabilities[i_act]
                                temp_action = i_act
                        action = torch.from_numpy(np.array([temp_action])).long().to(device)

                    next_observation, reward, done, _ = env.step((current_player, action.item()))
                    reward = torch.tensor([reward], device=device)
                    next_observation = torch.from_numpy(np.reshape(next_observation.flatten(), (1, N * N))).float().to(device)
                    memory.add_to_memory(observation, action, reward)
                else:
                    action = env.random_action(current_player)
                    next_observation, _, done, _ = env.step((current_player, action.item()))
                    next_observation = torch.from_numpy(np.reshape(next_observation.flatten(), (1, N * N))).float().to(
                        device)
                observation = next_observation
                current_player = current_player % 2 + 1

                if (i_episode + 1) % RENDER == 0:
                    env.render()

                if done:
                    observations, actions, rewards = memory1.get_tensors(device)
                    optimizer1.zero_grad()
                    outputs = player1(observations)
                    loss = loss_function(outputs, actions, rewards)
                    loss.backward()
                    optimizer1.step()

                    #observations, actions, rewards = memory2.get_tensors(device)
                    #optimizer2.zero_grad()
                    #outputs = player2(observations)
                    #loss = loss_function(outputs, actions, rewards)
                    #loss.backward()
                    #optimizer2.step()
            prog_bar.update(1)

    torch.save({'player1': player1.state_dict(),
                #'player2': player2.state_dict(),
                'optimizer1_dict': optimizer1.state_dict()},
                #'optimizer2_dict': optimizer2.state_dict()},
               'models/network_agent1')


def get_win_percentages(n_rounds=100):
    env = gym.make('knights-v0')
    agent = Network(5)
    checkpoint = torch.load('models/network_minus')
    agent.load_state_dict(checkpoint['player1'])
    agent.eval()
    w1 = 0
    w2 = 0
    i1 = 0
    i2 = 0
    for i in range(n_rounds):
        done = False
        state = torch.from_numpy(np.reshape(env.reset().flatten(), (1, 5 * 5))).float()
        current_player = 1
        while not done:
            if current_player == 2:
                action = env.random_action(current_player)
                next_state, reward, done, _ = env.step((current_player, action))
            else:
                a = agent(state)
                action = a.max(1)[1].view(1, 1).item()
                next_state, reward, done, _ = env.step((current_player, action))
            state = torch.from_numpy(np.reshape(next_state.flatten(), (1, 5 * 5))).float()
            #state = next_state
            if done:
                if reward == env.invalid_move:
                    if current_player == 1:
                        i1 += 1
                    else:
                        i2 += 1
                if current_player == 1:
                    w2 += 1
                else:
                    w1 += 1

    print("Agent 1 Win Percentage:", np.round(w1/n_rounds, 2))
    print("Agent 2 Win Percentage:", np.round(w2/n_rounds, 2))
    print("Number of Invalid Plays by Agent 1:", i1)
    print("Number of Invalid Plays by Agent 2:", i2)

def play(p):
    def load_player(p):
        player = Network(5)
        checkpoint = torch.load('models/network_minus')
        player.load_state_dict(checkpoint['player' + str(p % 2 + 1) + ''])
        player.eval()
        return player

    if p == 1:
        computer = load_player(2)
    else:
        computer = load_player(1)

    env = gym.make('knights-v0')
    state = torch.from_numpy(np.reshape(env.reset().flatten(), (1, 5 * 5))).float()
    current_player = 1
    done = False
    env.render()
    while not done:
        if current_player == p:
            legal = env.get_legal_moves(current_player)
            print('Select a move from: ' + str(legal))
            action = int(input())
        else:
            a = computer(state)
            action = a.max(1)[1].view(1, 1).item()
        next_state, _, done, _ = env.step((current_player, action))
        state = torch.from_numpy(np.reshape(next_state.flatten(), (1, 5 * 5))).float()  # .to(device)
        env.render()
        current_player = current_player % 2 + 1
    print('Player ' + str(current_player) + ' won!')
    env.close()

'''
if __name__ == '__main__':
    learn()
    #play(2)
    get_win_percentages(n_rounds=100)
'''







'''
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def pop(self):
        self.memory.pop()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train():
    env = gym.make('knights-v0').unwrapped

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_episodes = 0
    end_episodes = 10000
    show = 1000

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 2000
    TARGET_UPDATE = 100
    epsilon = 0.9995

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    N = 5

    policy_net1 = Network(N).to(device)
    target_net1 = Network(N).to(device)
    policy_net2 = Network(N).to(device)
    target_net2 = Network(N).to(device)

    if start_episodes != 0:
        checkpoint = torch.load('models/network_reward')
        policy_net1.load_state_dict(checkpoint['policy_net1_dict'])
        policy_net1.train()
        target_net1.load_state_dict(checkpoint['target_net1_dict'])
        target_net2.train()
        policy_net2.load_state_dict(checkpoint['policy_net2_dict'])
        policy_net2.state_dict()
        target_net2.load_state_dict(checkpoint['target_net2_dict'])
        target_net2.state_dict()
        optimizer1 = optim.RMSprop(policy_net1.parameters())
        optimizer1.load_state_dict(checkpoint['optimizer1_dict'])
        optimizer2 = optim.RMSprop(policy_net2.parameters())
        optimizer2.load_state_dict(checkpoint['optimizer2_dict'])
        with open('models/memory1', 'rb') as handle:
            memory1 = pickle.load(handle)
        with open('models/memory2', 'rb') as handle:
            memory2 = pickle.load(handle)

    else:
        target_net1.load_state_dict(policy_net1.state_dict())
        target_net1.eval()
        target_net2.load_state_dict(policy_net2.state_dict())
        target_net2.eval()
        optimizer1 = optim.RMSprop(policy_net1.parameters())
        optimizer2 = optim.RMSprop(policy_net2.parameters())
        memory1 = ReplayMemory(500)
        memory2 = ReplayMemory(500)

    def select_action(state, player, step):
        global steps_done
        sample = random.random()
        eps_threshold = epsilon ** step
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
                d = c.view(1, 1)
                return d  # policy_net1(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
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
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    for i_episode in range(start_episodes, end_episodes):
        # Initialize the environment and state
        state = env.reset()
        if i_episode % show == 0:
            print(f'Episode {i_episode}:')
            env.render()
        current_player = 1
        state = torch.from_numpy(np.reshape(state.flatten(), (1, N * N))).float().to(device)
        done = False
        while not done:
            if current_player == 1:
                memory = memory1
                target_net = target_net1
                policy_net = policy_net1
            else:
                memory = memory2
                target_net = target_net2
                policy_net = policy_net2

            # Select and perform an action
            action = select_action(state, current_player, i_episode)
            legal = env.get_legal_moves(current_player)

            while len(legal) > 0 and action not in legal:
                fake_next_state, fake_reward, _, _ = env.fake_step((current_player, action.item()))
                fake_reward = torch.tensor([fake_reward], device=device)
                fake_next_state = torch.from_numpy(np.reshape(fake_next_state.flatten(), (1, N * N))).float().to(device)
                memory.push(state, action, fake_next_state, fake_reward)
                optimize_model(current_player)
                
                if time % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                time += 1
                action = select_action(state, current_player, i_episode)

            next_state, reward, done, _ = env.step((current_player, action.item()))
            reward = torch.tensor([reward], device=device)

            next_player = current_player % 2 + 1
            next_state = torch.from_numpy(np.reshape(next_state.flatten(), (1, N * N))).float().to(device)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Perform one step of the optimization (on the policy network)
            optimize_model(current_player)

            if i_episode % show == 0:
                env.render()
                # print(f'Player {current_player}: {time}\n')

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if current_player == 1:
                time1 = time
            else:
                time2 = time
            current_player = next_player

        if i_episode % 5000 == 0:
            print('SAVE ' + str(i_episode))
            torch.save({'target_net1_dict': target_net1.state_dict(),
                        'target_net2_dict': target_net2.state_dict(),
                        'policy_net1_dict': policy_net1.state_dict(),
                        'policy_net2_dict': policy_net2.state_dict(),
                        'optimizer1_dict': optimizer1.state_dict(),
                        'optimizer2_dict': optimizer2.state_dict()},
                       'models/network_reward')

            with open('models/memory1', 'wb') as handle:
                pickle.dump(memory1, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('models/memory2', 'wb') as handle:
                pickle.dump(memory2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    target_net1.load_state_dict(policy_net1.state_dict())
    target_net2.load_state_dict(policy_net2.state_dict())
    print('Complete')
    env.close()

    torch.save({'target_net1_dict': target_net1.state_dict(),
                'target_net2_dict': target_net2.state_dict(),
                'policy_net1_dict': policy_net1.state_dict(),
                'policy_net2_dict': policy_net2.state_dict(),
                'optimizer1_dict': optimizer1.state_dict(),
                'optimizer2_dict': optimizer2.state_dict()},
               'models/network_reward')

    with open('models/memory1', 'wb') as handle:
        pickle.dump(memory1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('models/memory2', 'wb') as handle:
        pickle.dump(memory2, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''