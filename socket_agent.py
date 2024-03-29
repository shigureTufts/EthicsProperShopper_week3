# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import json
import random
import socket
import pandas as pd
import numpy as np

from env import SupermarketEnv
from utils import recv_socket_data
from Q_Learning_agent import QLAgent


# garlic [7.5, 17.5]
# carrot [13.5, 17.5]
target_pos = [1, 17.5]
def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def calculate_reward(previous_state, current_state):
    # design your own reward function here
    # You should design a function to calculate the reward for the agent to guide the agent to do the desired task

    reward = 0
    step_penalty = -0.2 # Small penalty for each step
    reward += step_penalty

    # handle the reward of approaching the target (negative reward while leaving to target)
    if current_state["violations"] == '':
        target_dis_now = euclidean_distance(
            [round(i, 1) for i in current_state['observation']['players'][0]['position']],
            target_pos)
        target_dis_pre = euclidean_distance(
            [round(i, 1) for i in previous_state['observation']['players'][0]['position']],
            target_pos)
        reward += reward + (target_dis_pre - target_dis_now) * 10
    else:  # violations
        reward = -20
    return reward


if __name__ == "__main__":

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1  # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)  # give it more aggressive decisions , alpha=0.3, gamma=0.8, epsilon=0.9, decay=0.5

    ####################
    # Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    agent.qtable = pd.read_json('qtable.json')
    ####################

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 100
    episode_length = 200
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        # slowly increase the episode length to make the agent learn more but explore too much at the beginning
        # episode_length += i
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state

            # one hack of making the learning easier is to reduce the action space
            # if the agent has a cart, exclude the interaction from the action space
            #
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]
            print("Sending action: ", action)

            # send several identical commands
            index = np.ones(2)
            for i in index:
                sock_game.send(str.encode(action))  # send action to env
                next_state = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(next_state)

            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state)  # You need to define this function
            print("------------------------------------------")
            print(reward, action_commands[action_index])
            print("------------------------------------------")
            # Update Q-table
            agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state
            agent.qtable.to_json('qtable.json')

            if euclidean_distance(state['observation']['players'][0]['position'], target_pos) <= 1.0 or cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()
