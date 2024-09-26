import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, action_space, state_space, num_autobots, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space = action_space
        self.state_space = state_space
        self.num_autobots = num_autobots
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_tables = [{} for _ in range(num_autobots)]  # Q-tables for each autobot

    def get_state_key(self, state, autobot_id):
        # Create a unique key for each state
        return (tuple(state['positions'][autobot_id]) + 
                tuple([state['directions'][autobot_id]]) + 
                tuple(map(tuple, state['destinations'])) + 
                tuple(state['obstacles'].flatten()))

    def get_q_value(self, autobot_id, state_key, action):
        return self.q_tables[autobot_id].get((state_key, action), 0.0)

    def choose_actions(self, state):
        actions = []
        for i in range(self.num_autobots):
            state_key = self.get_state_key(state, i)
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.action_space))
            else:
                q_values = [self.get_q_value(i, state_key, a) for a in range(self.action_space)]
                actions.append(np.argmax(q_values))
        return actions

    def learn(self, states, actions, rewards, next_states, done):
        for i in range(self.num_autobots):
            state_key = self.get_state_key(states, i)
            next_state_key = self.get_state_key(next_states, i)
            
            best_next_action = max(range(self.action_space), 
                                   key=lambda a: self.get_q_value(i, next_state_key, a))
            td_target = rewards[i] + self.discount_factor * self.get_q_value(i, next_state_key, best_next_action) * (1 - done)
            current_q = self.get_q_value(i, state_key, actions[i])
            new_q = current_q + self.learning_rate * (td_target - current_q)
            self.q_tables[i][(state_key, actions[i])] = new_q

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_agent(self, filename='agent_state.pkl'):
        agent_state = {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'q_tables': self.q_tables
        }
        with open(filename, 'wb') as f:
            pickle.dump(agent_state, f)
        print(f"Agent state saved to {filename}")

    def load_agent(self, filename='agent_state.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                agent_state = pickle.load(f)
            self.learning_rate = agent_state['learning_rate']
            self.discount_factor = agent_state['discount_factor']
            self.epsilon = agent_state['epsilon']
            self.epsilon_decay = agent_state['epsilon_decay']
            self.min_epsilon = agent_state['min_epsilon']
            self.q_tables = agent_state['q_tables']
            print(f"Agent state loaded from {filename}")
        else:
            print(f"No agent state file found at {filename}. Starting with default values.")

    def get_q_tables_size(self):
        return sum(len(q_table) for q_table in self.q_tables)
