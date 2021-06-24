import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from IPython.display import clear_output

class KnightsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5):
        super(KnightsEnv, self).__init__()
        self.n = n
        # Define action and observation space
        # They must be gym.spaces objects
        # Action - contains all of the actions possible for an agent to take in the environment
        self.action_space = spaces.Discrete(8)
        # Observation - contains all of the environment's data to be observed by the agent
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n, self.n), dtype=np.uint8)
        #self.punishment = 150
        #self.step_punishment = -16
        #self.reward = -10
        self.won = 1
        self.lost = -1
        self.step_reward = 1/25
        self.invalid_move = -10

    def _next_observation(self):
        obs = self.board
        #obs = np.append(obs, [[self.current_player, 0, 0, 0, 0, 0, 0, 0]], axis=0)
        return obs

    def get_legal_moves(self, player):
        current_pos = np.where(self.board == player)
        actions = []
        next_pos = current_pos[0] + 2, current_pos[1] + 1
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(0)

        next_pos = current_pos[0] + 1, current_pos[1] + 2
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(1)

        next_pos = current_pos[0] - 1, current_pos[1] + 2
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(2)

        next_pos = current_pos[0] - 2, current_pos[1] + 1
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(3)

        next_pos = current_pos[0] - 2, current_pos[1] - 1
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(4)

        next_pos = current_pos[0] - 1, current_pos[1] - 2
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(5)

        next_pos = current_pos[0] + 1, current_pos[1] - 2
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(6)

        next_pos = current_pos[0] + 2, current_pos[1] - 1
        if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
            actions.append(7)

        return actions

    def _take_action(self, player, action):
        current_pos = np.where(self.board == player)
        self.board[current_pos] = -1

        if action == 0:
            next_pos = current_pos[0] + 2, current_pos[1] + 1
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 1:
            next_pos = current_pos[0] + 1, current_pos[1] + 2
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 2:
            next_pos = current_pos[0] - 1, current_pos[1] + 2
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 3:
            next_pos = current_pos[0] - 2, current_pos[1] + 1
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 4:
            next_pos = current_pos[0] - 2, current_pos[1] - 1
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 5:
            next_pos = current_pos[0] - 1, current_pos[1] - 2
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 6:
            next_pos = current_pos[0] + 1, current_pos[1] - 2
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        elif action == 7:
            next_pos = current_pos[0] + 2, current_pos[1] - 1
            if 0 <= next_pos[0] < self.n and 0 <= next_pos[1] < self.n and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = player
            else:
                self.state = 'L'

        if next_pos[0] < 0 or next_pos[0] >= self.n or next_pos[1] < 0 or next_pos[1] >= self.n:
            self.state = 'I'

    def fake_step(self, player, action):
        current_pos = np.where(self.board == player)
        fake_board = self.board.copy()
        fake_board[current_pos] = -1
        return fake_board, self.lost, True, {}


    def step(self, action):
        # Execute one time step within the environment
        lm = self.get_legal_moves(action[0])
        self._take_action(action[0], action[1])
        #print(self.board)

        if self.state == 'L':
            #print(f'Player {action[0] % 2 + 1} won.')
            reward = self.lost
            done = True
        elif self.state == 'I':
            reward = self.invalid_move
            done = True
        elif not self.get_legal_moves(action[0] % 2 + 1):
            reward = self.won
            done = False
        else:
            reward = self.step_reward
            done = False

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.board = np.zeros((self.n, self.n))
        self.board[0][0] = 1
        self.board[-1][-1] = 2
        self.state = 'P'
        return self._next_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        # Print, render 3d objects, etc.
        str = ''
        for r in self.board:
            for c in r:
                if c == 1:
                    str += '♘'
                elif c == 2:
                    str += '♞'
                elif c == 0:
                    str += ' ⃞ '
                else:
                    str += ' ⃠ '
            str += '\n'
        print(str)

    def random_action(self, player):
        moves = self.get_legal_moves(player)
        if len(moves) == 0:
            moves = [0, 1, 2, 3, 4, 5, 6, 7]
        return np.random.choice(moves)

    def close(self):
        pass