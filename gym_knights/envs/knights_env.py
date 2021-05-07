import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from IPython.display import clear_output

class KnightsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(KnightsEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Action - contains all of the actions possible for an agent to take in the environment
        self.action_space = spaces.Discrete(8)
        # Observation - contains all of the environment's data to be observed by the agent
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.uint8)

    def _next_observation(self):
        obs = self.board
        #obs = np.append(obs, [[self.current_player, 0, 0, 0, 0, 0, 0, 0]], axis=0)
        return obs

    def _take_action(self, action):
        current_pos = np.where(self.board == action[0]) #self.current_player)

        if action[1] == 0:
            next_pos = current_pos[0] + 2, current_pos[1] + 1
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 1:
            next_pos = current_pos[0] + 1, current_pos[1] + 2
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 2:
            next_pos = current_pos[0] - 1, current_pos[1] + 2
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 3:
            next_pos = current_pos[0] - 2, current_pos[1] + 1
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 4:
            next_pos = current_pos[0] - 2, current_pos[1] - 1
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 5:
            next_pos = current_pos[0] - 1, current_pos[1] - 2
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 6:
            next_pos = current_pos[0] + 1, current_pos[1] - 2
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

        elif action[1] == 7:
            next_pos = current_pos[0] + 2, current_pos[1] - 1
            self.board[current_pos] = -1
            if 0 <= next_pos[0] < 8 and 0 <= next_pos[1] < 8 and int(self.board[next_pos]) not in (-1, 1, 2):
                self.board[next_pos] = action[0]
            else:
                self.state = 'W'

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        #print(self.board)

        if self.state == 'W':
            #print(f'Player {action[0] % 2 + 1} won.')
            reward = -500
            done = True
        else:
            reward = 10
            done = False

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.board = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 2]])
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


    def close(self):
        pass