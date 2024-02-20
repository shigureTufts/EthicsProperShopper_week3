#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd


cart = False
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

def distance_to_cart(state):
    agent_position = state['observation']['players'][0]['position']
    if agent_position[0] > 1.5:
        cart_distances = [euclidean_distance(agent_position, cart_pos_right)]
    else:
        cart_distances = [euclidean_distance(agent_position, cart_pos_left)]
    return min(cart_distances)

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def calculate_reward(previous_state, current_state):
    # design your own reward function here
    # You should design a function to calculate the reward for the agent to guide the agent to do the desired task
    global cart
    global exit_pos

    reward = 0
    step_penalty = -0.1  # Small penalty for each step
    reward += step_penalty
    # handle the reward of getting a cart

    if current_state['observation']['players'][0]['curr_cart'] != -1 and previous_state['observation']['players'][0][
        'curr_cart'] == -1:
        # only give reward for once to avoid random reward from picking up a cart from the ground
        if not cart:
            reward += 100
            cart = True
    if current_state['observation']['players'][0]['curr_cart'] == -1 and previous_state['observation']['players'][0][
        'curr_cart'] != -1:
        reward -= 100

    # handle the reward of approaching the cart
    if current_state['observation']['players'][0]['curr_cart'] == -1:
        cart_dis_now = min(euclidean_distance(current_state['observation']['players'][0]['position'], cart_pos_left),
                           euclidean_distance(current_state['observation']['players'][0]['position'], cart_pos_right))
        cart_dis_pre = min(euclidean_distance(previous_state['observation']['players'][0]['position'], cart_pos_left),
                           euclidean_distance(previous_state['observation']['players'][0]['position'], cart_pos_right))
        reward = reward + (cart_dis_pre - cart_dis_now) * 10
    print(current_state['observation']['players'][0]['curr_cart'])

    # handle the reward of reaching the exit
    # Theortically we need this, but since the env auto shuts down, we might not get the reward
    if abs(current_state['observation']['players'][0]['position'][0]) < 0.4 and abs(
            current_state['observation']['players'][0]['position'][1] - exit_pos[1]) < 0.1 and \
            current_state['observation']['players'][0]['curr_cart'] != -1:
        reward += 100

    # handle the reward of approaching the exit
    if current_state['observation']['players'][0]['curr_cart'] != -1:
        exit_dis_now = euclidean_distance(current_state['observation']['players'][0]['position'], exit_pos)
        exit_dis_pre = euclidean_distance(previous_state['observation']['players'][0]['position'], exit_pos)
        reward = reward + (exit_dis_pre - exit_dis_now) * 20

    return reward


if __name__ == "__main__":

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1  # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)

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
    episode_length = 10
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        # slowly increase the episode length to make the agent learn more but explore too much at the beginning
        episode_length += i
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state

            # one hack of making the learning easier is to reduce the action space
            # if the agent has a cart, exclude the interaction from the action space
            #
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]

            print("Sending action: ", action)
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

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

