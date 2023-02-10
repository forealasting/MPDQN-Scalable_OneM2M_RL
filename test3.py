import numpy as np


class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, gamma=0.9, max_epsilon=1, min_epsilon=0.1, epsilon_decay=1 / 300):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.full((3, 11, 10, 5), -np.iinfo(np.int32).max)  # -2147483647
        # Kmax = 3  #  max number of replicas
        # u = 11    #  divide cpu utilization into 11 degrees
        # c = 10    #  divide cpus into 10 degrees


    def choose_action(self, state):
        available_actions = self.get_available_actions(state)
        s = list(state)
        s[2] = int(s[2] * 10 - 1)
        s[1] = int(s[1])
        # action selection
        if self.epsilon > np.random.uniform():
            # choose random action
            action = np.random.choice(available_actions)
        else:
            # print(state[0], state[1], state[2])
            # choose greedy action
            q_values = self.q_table[s[0], s[1], s[2], :]
            q_values[np.isin(range(5), available_actions, invert=True)] = -np.iinfo(np.int32).max
            action = np.argmax(q_values)

        return action

    def learn(self, state, a, r, next_state, done):
        s = list(state)
        s_ = list(next_state)
        # state  = [1, 0.0, 0.5]
        s[2] = int(s[2] * 10 - 1)
        s[1] = int(s[1])

        s_[2] = int(s_[2] * 10 - 1)
        s_[1] = int(s_[1])

        q_predict = self.q_table[s[0], s[1], s[2], a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[s_[0], s_[1], s_[2], :])
        self.q_table[s[0], s[1], s[2], a] = q_predict + self.lr * (q_target - q_predict)
        print(self.q_table[s[0], s[1], s[2], a])
        # linearly decrease epsilon
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )

    def get_available_actions(self, state):
        # S ={𝑘, 𝑢, 𝑐}
        # 𝑘 (replica): 1 ~ 3                          actual value : same
        # 𝑢 (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
        # 𝑐 (used cpus) : 0.1 0.2 ... 1               actual value : same
        # action_space = ['-r', -1, 0, 1, 'r']
        actions = [0, 1, 2, 3, 4]  # action index
        if state[0] == 1:
            actions.remove(1)
        if state[0] == 3:
            actions.remove(3)
        if state[2] == 1:
            actions.remove(4)
        if state[2] == 0.5:
            actions.remove(0)

        return actions