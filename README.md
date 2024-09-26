<h1 align="center" id="title">Autobot Grid Navigation with Reinforcement Learning</h1>

<p id="description">This project implements a multi-autobot navigation system using Q-learning a reinforcement learning algorithm. Autobots are tasked with navigating through a grid filled with obstacles from specified start positions to designated destinations. The system visualizes the autobots' progress on a Tkinter-based GUI and logs the time and rewards for each episode.</p>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Reinforcement Learning: The project uses a Q-learning agent to guide autobots.
*   Customizable Grid: Users can define grids with obstacles autobot start positions and destinations.Customizable Grid: Users can define grids with obstacles autobot start positions and destinations.
*   Real-Time GUI: Visualize autobot movements rewards and steps via a Tkinter-based GUI.Real-Time GUI: Visualize autobot movements rewards and steps via a Tkinter-based GUI.
*   Time Tracking: Displays the actual time each episode takes to complete.Time Tracking: Displays the actual time each episode takes to complete.
*   State Persistence: Saves and loads Q-learning agent states between runs.State Persistence: Saves and loads Q-learning agent states between runs.

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone this repository:</p>

```
git clone https://github.com/your-username/autobot-grid-navigation.git cd autobot-grid-navigation
```

<p>2. Install the required dependencies:</p>

```
pip install -r numpy gym
```

<p>3. (Optional) Set up a virtual environment:</p>

```
python -m venv venv source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   OPENAI Gymnasium
*   Python

<p>
  Algorithm Explanation

The project implements Q-learning, a reinforcement learning algorithm where each autobot learns the optimal path to its destination based on rewards. The environment provides rewards based on:

  Movement towards the destination.
  Collisions with obstacles or other autobots (negative reward).
  Reaching the correct destination (positive reward).

The Q-learning agent maintains Q-tables for each autobot, where the state-action pairs are stored and updated as the autobots interact with the environment.
Key Concepts

  Q-learning: An off-policy reinforcement learning algorithm.
  Exploration vs. Exploitation: Controlled by the ε-greedy policy (ε decays over time).
  Rewards: Positive for reaching destinations, negative for collisions and invalid moves.
  State Space: Comprises autobot positions, directions, and obstacle locations.
</p>
