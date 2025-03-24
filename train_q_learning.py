import pickle
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from simple_custom_taxi_env import SimpleTaxiEnv
import json
import matplotlib.pyplot as plt



def tabular_q_learning_adjust(episodes=100000, alpha=0.01, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99995, reward_shaping=True,
                              q_table=None, debug=False):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    """
    ✅ Implement Tabular Q-learning with Reward Shaping
    - Modify reward shaping to accelerate learning.
    - Adjust epsilon decay to ensure sufficient exploration.
    - Ensure the agent learns the full sequence: "pick up key → open door → reach goal".
    """
    action_n = 6
    Near_Passenger = + 5
    Near_Destination = + 5
    Pick_Up_Passenger = 10
    Repeat_Pick_Up_Passenger = - 10
    Check_station = +5
    Turn_Back = -2

    MAX_STEPS = 100
    TOO_LONG = -5

    env = SimpleTaxiEnv()
    env.reset()

    if q_table is None:
        q_table = {}

    rewards_per_episode = []
    epsilon = epsilon_start


    def get_q_state(obs, last_action, Passenger_Pos, Has_Passenger, Checked_Stations, Checked_Destinations ):

        # TODO: Represent the state using agent position, direction, key possession, door status, and etc.
        #create a list of 4 stations position where positions are unknown
        stations_pos = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        taxi_row, taxi_col, stations_pos[0][0], stations_pos[0][1], stations_pos[1][0], stations_pos[1][1], stations_pos[2][0], stations_pos[2][1], stations_pos[3][0], stations_pos[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs


        # Sort stations_pos based on the first column, then the second column
        sorted_indices = np.lexsort((stations_pos[:, 1], stations_pos[:, 0]))
        stations_pos = stations_pos[sorted_indices]

        rel_sta_pos = [[0,0], [0,0], [0,0], [0,0]]
        for i in range(4):
            rel_sta_pos[i] = [stations_pos[i][0] - taxi_row  , stations_pos[i][1] - taxi_col]
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
        

        

        return (tuple(rel_sta_pos[0]),tuple(rel_sta_pos[1]),tuple(rel_sta_pos[2]),tuple(rel_sta_pos[3]), obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, last_action, tuple(rel_passenger_pos), Has_Passenger, Checked_Stations, Checked_Destinations)


    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        last_action = -1
        total_reward = 0
        episode_step = 0
        stop = 0
        Passenger_Pos = [-1,-1]
        Has_Passenger = False
        Has_Picked_Up = False
        Checked_Stations = -1
        Checked_Destinations = -1
        state = get_q_state(obs, last_action, Passenger_Pos, Has_Passenger, Checked_Stations, Checked_Destinations)


        while not (done or stop):
            # TODO: Initialize the state in the Q-table if not already present.
            if state not in q_table:
                q_table[state] = np.zeros(action_n)

            
            # TODO: Implement ε-greedy policy for action selection.
            if np.random.rand() < epsilon:
                action = np.random.choice(action_n)  # Explore.
            else:
                action = np.argmax(q_table[state])  # Exploit.


            


            # Execute the selected action.
            obs, reward, done,  _ = env.step(action)

            stations_pos = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
            taxi_row, taxi_col, stations_pos[0][0], stations_pos[0][1], stations_pos[1][0], stations_pos[1][1], stations_pos[2][0], stations_pos[2][1], stations_pos[3][0], stations_pos[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs


            # Sort stations_pos based on the first column, then the second column
            sorted_indices = np.lexsort((stations_pos[:, 1], stations_pos[:, 0]))
            stations_pos = stations_pos[sorted_indices]
        
            if not Has_Picked_Up:
                if action == 4 and passenger_look and ((taxi_col == stations_pos[0][0] and taxi_row == stations_pos[0][1]) or (taxi_col == stations_pos[1][0] and taxi_row == stations_pos[1][1]) or (taxi_col == stations_pos[2][0] and taxi_row == stations_pos[2][1]) or (taxi_col == stations_pos[3][0] and taxi_row == stations_pos[3][1])):
                    Has_Picked_Up = True
                    Has_Passenger = True
                    Passenger_Pos[0] = taxi_row
                    Passenger_Pos[1] = taxi_col
            else:
                if (not Has_Passenger) and action == 4 and Passenger_Pos[0] == taxi_row and Passenger_Pos[1] == taxi_col:
                    Has_Passenger = True
        
            if Has_Passenger:
                Passenger_Pos[0] = taxi_row
                Passenger_Pos[1] = taxi_col
                if action == 5:
                    Has_Passenger = False
                    Passenger_Pos[0] = taxi_row
                    Passenger_Pos[1] = taxi_col
            
            Near_Stations = [0,0,0,0]
            
            if np.abs(taxi_row - stations_pos[0][0]) + np.abs(taxi_col - stations_pos[0][1]) <= 0:
                Near_Stations[0] = 1
            if np.abs(taxi_row - stations_pos[1][0]) + np.abs(taxi_col - stations_pos[1][1]) <= 0:
                Near_Stations[1] = 1
            if np.abs(taxi_row - stations_pos[2][0]) + np.abs(taxi_col - stations_pos[2][1]) <= 0:
                Near_Stations[2] = 1
            if np.abs(taxi_row - stations_pos[3][0]) + np.abs(taxi_col - stations_pos[3][1]) <= 0:
                Near_Stations[3] = 1
            
            if Checked_Stations == -1:
                if Near_Stations[0] == 1:
                    Checked_Stations = 0
                    if destination_look:
                        Checked_Destinations = 0
            elif Checked_Stations == 0:
                if Near_Stations[1] == 1:
                    Checked_Stations = 1
                    if destination_look:
                        Checked_Destinations = 1
            elif Checked_Stations == 1:
                if Near_Stations[2] == 1:
                    Checked_Stations = 2
                    if destination_look:
                        Checked_Destinations = 2
            elif Checked_Stations == 2:
                if Near_Stations[3] == 1:
                    Checked_Stations = 3
                    if destination_look:
                        Checked_Destinations = 3

            next_state = get_q_state(obs, action, Passenger_Pos, Has_Passenger, Checked_Stations, Checked_Destinations)
            episode_step += 1

            

            # ✅ TODO: Implement reward shaping.

            shaped_reward = 0
            if episode_step >= MAX_STEPS:
                stop = True
                shaped_reward += TOO_LONG

            if state[12] == 0 and next_state[12] == 1:
                shaped_reward += Pick_Up_Passenger

            if next_state[8] == 1 and state[8] == 0:
                shaped_reward += Near_Passenger
            if next_state[8] == 0 and state[8] == 1:
                shaped_reward -= Near_Passenger

            if Checked_Stations > state[13] :
                shaped_reward += Check_station
            
            

            if (next_state[10],state[10]) == (0,1) or (next_state[10],state[10]) == (1,0) or (next_state[10],state[10]) == (2,3) or (next_state[10],state[10]) == (3,2):
                shaped_reward += Turn_Back


            # Update total reward.
            reward += shaped_reward

            total_reward += reward

            # TODO: Initialize the next state in the Q-table if not already present.
            if next_state not in q_table:
                q_table[next_state] = np.zeros(action_n)

            # TODO: Apply Q-learning update rule (Bellman equation).
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            # Move to the next state.
            last_action = action
            state = next_state


        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)


        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    return q_table, rewards_per_episode


if __name__ == "__main__":
    q_table, rewards = tabular_q_learning_adjust(episodes=20000, alpha=0.01, gamma=0.99,
                                                 epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99996, reward_shaping=True,
                                                 q_table=None, debug=True)
    print("Training Complete")
    print("Q-table size:", len(q_table))
    print("Last 100 episode average reward:", np.mean(rewards[-100:]))


    # Save the Q-table for testing
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training Progress")
    plt.show()

