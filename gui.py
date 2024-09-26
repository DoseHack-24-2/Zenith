import tkinter as tk
import numpy as np

class AutobotGUI:
    def __init__(self, root, env):
        self.root = root
        self.env = env
        self.grid_height = env.grid_height
        self.grid_width = env.grid_width
        self.cell_size = 50  # Size of each cell in the grid

        self.root.title("Multi-Autobot Grid Traversal")
        
        # Create canvas to represent the grid
        self.canvas = tk.Canvas(self.root, width=self.grid_width*self.cell_size, height=self.grid_height*self.cell_size)
        self.canvas.pack()

        # Create labels for rewards
        self.reward_labels = []
        for i in range(env.num_autobots):
            label = tk.Label(self.root, text=f"Autobot {i+1} Reward: 0")
            label.pack()
            self.reward_labels.append(label)

        # Draw initial grid
        self.draw_grid()

    def draw_grid(self):
        """Draw the grid with autobot positions, obstacles, and destinations."""
        self.canvas.delete("all")  # Clear the canvas before redrawing
        
        # Draw the grid cells, obstacles, and destinations
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                fill_color = "white"
                
                if (i, j) in self.env.obstacles:
                    fill_color = "gray"
                elif any(np.array_equal(dest, [i, j]) for dest in self.env.destinations):
                    fill_color = "green"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="black")
                
                # Add labels for destinations
                if self.env.grid[i][j].islower():
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=self.env.grid[i][j].upper())
        
        # Draw the autobots (balls) on top of obstacles
        for i, pos in enumerate(self.env.autobot_positions):
            x1 = pos[1] * self.cell_size
            y1 = pos[0] * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            fill_color = f"#{hash(i) % 0xFFFFFF:06x}"  # Generate a unique color for each autobot
            self.canvas.create_oval(x1, y1, x2, y2, fill=fill_color, outline="black")
            self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=self.env.grid[pos[0]][pos[1]])

    def update_grid(self, env, rewards=None):
        """Update the grid based on the current state of the environment."""
        self.env = env
        self.draw_grid()
        if rewards is not None:
            for i, reward in enumerate(rewards):
                self.reward_labels[i].config(text=f"Autobot {i+1} Reward: {reward:.2f}")
        self.root.update()