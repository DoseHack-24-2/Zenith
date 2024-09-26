import tkinter as tk
import numpy as np
from autobot_env import AutobotEnv
from gui import AutobotGUI
from agent import QLearningAgent
import time

def parse_input_file(file_path):
    with open(file_path, 'r') as file:
        grid = [list(line.strip()) for line in file]
    
    # Automatically determine the number of bots
    num_bots = sum(1 for row in grid for cell in row if cell.isupper() and cell != 'X')
    
    return grid, num_bots



def run_episode(env, agent, gui, episode_num, max_steps=1000, delay=0.2):
    state = env.reset()
    total_rewards = np.zeros(env.num_autobots)
    done = False
    step = 0

    while not done and step < max_steps:
        actions = agent.choose_actions(state)
        next_state, rewards, done, _ = env.step(actions)
        
        agent.learn(state, actions, rewards, next_state, done)
        state = next_state
        total_rewards += rewards
        
        gui.update_grid(env, total_rewards)
        gui.root.update()
        
        step += 1
        time.sleep(delay)

    rewards_str = ", ".join([f"{reward:.2f}" for reward in total_rewards])
    print(f"Episode {episode_num:3d}: Steps = {step:4d}, Rewards = [{rewards_str}]")

    return total_rewards, step

def main():
    # Parse the input file
    input_file_path = "custom_inputs\i2.txt"  # You can change this to accept command-line arguments
    grid, num_bots = parse_input_file(input_file_path)

    # Set up the environment
    env = AutobotEnv(grid)

    # Calculate the state space size
    state_space_size = (env.grid_height * env.grid_width) * 4 * num_bots
    
    # Set up the agent
    agent = QLearningAgent(
        action_space=env.action_space.n,
        state_space=state_space_size,
        num_autobots=num_bots,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    # Load the agent's state if it exists
    agent.load_agent()
    
    # Set up the GUI
    root = tk.Tk()
    gui = AutobotGUI(root, env)
    
    # Run episodes
    num_episodes = 800  # You can adjust this number
    print("Training started...")
    for episode in range(num_episodes):
        _, _ = run_episode(env, agent, gui, episode + 1)
        agent.update_epsilon()

    agent.save_agent()
    print("Training complete and agent state saved.")
    print(f"Q-tables size: {agent.get_q_tables_size()} state-action pairs")
    
    # Keep the GUI open after training
    root.mainloop()

if __name__ == "__main__":
    main()