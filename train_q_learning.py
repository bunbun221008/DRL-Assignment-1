import pickle
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from simple_custom_taxi_env import SimpleTaxiEnv

def tabular_q_learning_adjust(episodes=10000, alpha=0.1, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9998, reward_shaping=True,
                              q_table=None, debug=False):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    """
    ✅ Implement Tabular Q-learning with Reward Shaping
    - Modify reward shaping to accelerate learning.
    - Adjust epsilon decay to ensure sufficient exploration.
    - Ensure the agent learns the full sequence: "pick up key → open door → reach goal".
    """
    action_n = 6
    DROP_KEY = -0.5
    TAKE_KEY = 0.5
    CLOSE_DOOR = -0.5
    OPEN_DOOR = 0.5
    TOO_LONG = -0.5

    env = SimpleTaxiEnv()
    env.reset()

    if q_table is None:
        q_table = {}

    rewards_per_episode = []
    epsilon = epsilon_start


    def get_q_state(obs):

        # TODO: Represent the state using agent position, direction, key possession, door status, and etc.
        #create a list of 4 stations position where positions are unknown
        sta_row = [0, 0, 0, 0]
        sta_column = [0, 0, 0, 0]
        taxi_row, taxi_col, sta_row[0],sta_column[0],sta_row[1],sta_column[1],sta_row[2],sta_column[2],sta_row[3],sta_column[3],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

        rel_sta_pos = [[0,0], [0,0], [0,0], [0,0]]
        for i in range(4):
            rel_sta_pos[i] = [sta_row[i] - taxi_row  , sta_column[i] - taxi_col]
            if rel_sta_pos[i][0] >0:
                rel_sta_pos[i][0] = 1 
            elif rel_sta_pos[i][0] <0:  
                rel_sta_pos[i][0] = -1        

        

        return (obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])


    for episode in range(episodes):
        obs, _ = env.reset()
        state = get_q_state(obs)  
        done = False
        total_reward = 0
        episode_step = 0
        stop = 0

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
            next_state = get_q_state(obs)
            episode_step += 1

            if episode_step >= 200:
                stop = True

            # ✅ TODO: Implement reward shaping.

            shaped_reward = 0.2

            


            # Update total reward.
            reward += shaped_reward
            total_reward += reward

            # TODO: Initialize the next state in the Q-table if not already present.
            if next_state not in q_table:
                q_table[next_state] = np.zeros(action_n)

            # TODO: Apply Q-learning update rule (Bellman equation).
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            # Move to the next state.
            state = next_state


        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)


        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    return q_table, rewards_per_episode


if __name__ == "__main__":
    q_table, rewards = tabular_q_learning_adjust(episodes=10000, alpha=0.1, gamma=0.99,
                                                 epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9998, reward_shaping=True,
                                                 q_table=None, debug=True)
    print("Training Complete")
    print("Q-table size:", len(q_table))
    print("Last 100 episode average reward:", np.mean(rewards[-100:]))


    # Save the Q-table for testing
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f, protocol=4)

