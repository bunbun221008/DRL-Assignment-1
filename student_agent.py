# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import json


def get_q_state(obs, last_action, Passenger_Pos, Has_Passenger, Checked_Stations, Checked_Destinations ):

        # TODO: Represent the state using agent position, direction, key possession, door status, and etc.
        #create a list of 4 stations position where positions are unknown
        sta_row = [0, 0, 0, 0]
        sta_column = [0, 0, 0, 0]
        Near_sta = [0,0,0,0]
        taxi_row, taxi_col, sta_row[0],sta_column[0],sta_row[1],sta_column[1],sta_row[2],sta_column[2],sta_row[3],sta_column[3],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

        rel_sta_pos = [[0,0], [0,0], [0,0], [0,0]]
        for i in range(4):
            rel_sta_pos[i] = [sta_row[i] - taxi_row  , sta_column[i] - taxi_col]
            if np.abs(rel_sta_pos[i][0]) + np.abs(rel_sta_pos[i][1]) <= 1:
                Near_sta[i] = 1
            
            if rel_sta_pos[i][0] >0:
                rel_sta_pos[i][0] = 1 
            elif rel_sta_pos[i][0] <0:  
                rel_sta_pos[i][0] = -1  

            if rel_sta_pos[i][1] >0:
                rel_sta_pos[i][1] = 1
            elif rel_sta_pos[i][1] <0:
                rel_sta_pos[i][1] = -1      
        if Passenger_Pos[0] == -1:
            rel_passenger_pos = [2,2]
        else:
            rel_passenger_pos = [Passenger_Pos[0] - taxi_row, Passenger_Pos[1] - taxi_col]
            if rel_passenger_pos[0] >0:
                rel_passenger_pos[0] = 1
            elif rel_passenger_pos[0] <0:
                rel_passenger_pos[0] = -1
            if rel_passenger_pos[1] >0:
                rel_passenger_pos[1] = 1
            elif rel_passenger_pos[1] <0:
                rel_passenger_pos[1] = -1
        

        

        return (tuple(rel_sta_pos[0]),tuple(rel_sta_pos[1]),tuple(rel_sta_pos[2]),tuple(rel_sta_pos[3]), obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, last_action, tuple(rel_passenger_pos), Has_Passenger, tuple(Checked_Stations), Checked_Destinations)


Passenger_Pos = [-1,-1]
Has_Passenger = False
Has_Picked_Up = False
last_action = -1
Checked_Stations = [0,0,0,0]
Checked_Destinations = -1

def get_action(obs):
    global last_action
    global Passenger_Pos
    global Has_Passenger
    global Has_Picked_Up
    global Checked_Stations
    global Checked_Destinations
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    # Load the pre-trained Q-table
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)


    sta_row = [0, 0, 0, 0]
    sta_column = [0, 0, 0, 0]
    
    taxi_row, taxi_col, sta_row[0],sta_column[0],sta_row[1],sta_column[1],sta_row[2],sta_column[2],sta_row[3],sta_column[3],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs


    if not Has_Picked_Up:
        if last_action == 4 and passenger_look and ((taxi_col == sta_column[0] and taxi_row == sta_row[0]) or (taxi_col == sta_column[1] and taxi_row == sta_row[1]) or (taxi_col == sta_column[2] and taxi_row == sta_row[2]) or (taxi_col == sta_column[3] and taxi_row == sta_row[3])):
            Has_Picked_Up = True
            Has_Passenger = True
            Passenger_Pos[0] = taxi_row
            Passenger_Pos[1] = taxi_col
    else:
        if (not Has_Passenger) and last_action == 4 and Passenger_Pos[0] == taxi_row and Passenger_Pos[1] == taxi_col:
            Has_Passenger = True

    if Has_Passenger:
        Passenger_Pos[0] = taxi_row
        Passenger_Pos[1] = taxi_col
        if last_action == 5:
            Has_Passenger = False
            Passenger_Pos[0] = taxi_row
            Passenger_Pos[1] = taxi_col
    
    Near_Stations = [0,0,0,0]
            
    if np.abs(taxi_row - sta_row[0]) + np.abs(taxi_col - sta_column[0]) <= 0:
        Near_Stations[0] = 1
    if np.abs(taxi_row - sta_row[1]) + np.abs(taxi_col - sta_column[1]) <= 0:
        Near_Stations[1] = 1 
    if np.abs(taxi_row - sta_row[2]) + np.abs(taxi_col - sta_column[2]) <= 0:
        Near_Stations[2] = 1
    if np.abs(taxi_row - sta_row[3]) + np.abs(taxi_col - sta_column[3]) <= 0:
        Near_Stations[3] = 1
    
    if Near_Stations[0] == 1:
        Checked_Stations[0] = 1
        if destination_look:
            Checked_Destinations = 0
    if Near_Stations[1] == 1:
        Checked_Stations[1] = 1
        if destination_look:
            Checked_Destinations = 1
    if Near_Stations[2] == 1:
        Checked_Stations[2] = 1
        if destination_look:
            Checked_Destinations = 2
    if Near_Stations[3] == 1:
        Checked_Stations[3] = 1
        if destination_look:
            Checked_Destinations = 3
    
    
    state = get_q_state(obs, last_action, Passenger_Pos, Has_Passenger,Checked_Stations, Checked_Destinations)
    

    # Initialize the state in the Q-table if not already present.
    if state not in Q_table:
        Q_table[state] = np.zeros(6)

    # Implement ε-greedy policy for action selection.
    epsilon = 0
    if np.random.rand() < epsilon:
        action = np.random.choice(6)  # Explore.
    else:
        action = np.argmax(Q_table[state])  # Exploit.

    last_action = action

    print("Passenger_Pos: ", Passenger_Pos)
    print("Has_Passenger: ", Has_Passenger)
    print("Has_Picked_Up: ", Has_Picked_Up)

    

    


    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

