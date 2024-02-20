import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table

    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        player_info = state['observation']['players'][0]
        position = player_info['position']
        curr_cart = player_info['curr_cart']
        return json.dumps({'position': position, 'curr_cart': curr_cart}, sort_keys=True)

    def check_add(self, state):
        if self.trans(state) not in self.qtable.index:
            self.qtable.loc[self.trans(state)] = pd.Series(np.zeros(self.action_space),
                                                           index=[i for i in range(self.action_space)])

    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        self.check_add(state)
        self.check_add(next_state)
        q_sa = self.qtable.loc[self.trans(state), action]
        max_next_q_sa = self.qtable.loc[self.trans(next_state), :].max()
        new_q_sa = q_sa + self.alpha * (rwd + self.gamma * max_next_q_sa - q_sa)
        self.qtable.loc[self.trans(state), action] = new_q_sa

    def action_prob(self, state):
        self.check_add(state)
        p = np.random.uniform(0, 1)
        self.epsilon *= 0.99
        if p <= self.epsilon:
            return np.array([1/self.action_space for i in range(self.action_space)])
        else:
            prob = F.softmax(torch.tensor(self.qtable.loc[self.trans(state)].to_list()), dim=0).detach().numpy()
            return prob

    def choose_action(self, state):
        # implement the action selection for the fully trained agent
        self.check_add(state)
        p = np.random.uniform(0, 1)
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])
        else:
            prob = F.softmax(torch.tensor(self.qtable.loc[self.trans(state)].to_list()), dim=0).detach().numpy()
            return np.random.choice(np.flatnonzero(prob == prob.max()))

