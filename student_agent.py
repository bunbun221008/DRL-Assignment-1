# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import json


def get_q_state(obs ):

        # TODO: Represent the state using agent position, direction, key possession, door status, and etc.
        #create a list of 4 stations position where positions are unknown
        stations_pos = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        taxi_row, taxi_col, stations_pos[0][0], stations_pos[0][1], stations_pos[1][0], stations_pos[1][1], stations_pos[2][0], stations_pos[2][1], stations_pos[3][0], stations_pos[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs


        return ( obstacle_north, obstacle_south, obstacle_east, obstacle_west)





def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    # Load the pre-trained Q-table
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)


    
    
    
    state = get_q_state(obs)
    

    # Initialize the state in the Q-table if not already present.
    if state not in Q_table:
        Q_table[state] = np.zeros(6)

    # Implement ε-greedy policy for action selection.
    epsilon = 0
    if np.random.rand() < epsilon:
        action = np.random.choice(6)  # Explore.
    else:
        action = np.argmax(Q_table[state])  # Exploit.

    


    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

