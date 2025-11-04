import gymnasium as gym
import famnit_gym
import numpy as np
import queue
from famnit_gym.wrappers.sokoban import Keyboard


# ------------ Map creation and setup -------------------
# ------------------- Initialization --------------------
# Integer	Meaning
# 0	Floor
# 1	Wall
# 2	Crate
# 3	Goal
# 4	Crate on a goal
# 5	Player


map = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 3, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 2, 0, 1],
    [1, 0, 0, 0, 0, 0, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])  


options = {
    'map_template': map , 
    'scale': 0.75  
}

env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
env = Keyboard(env)

# Reset the environment.
observation, info = env.reset()

# Run until game over,
done = False
while not done:
    # Use input will be used instead of the given action.
    _, _, terminated, truncated, _ = env.step(0)

    # The episode is truncated if the user quits the game.
    done = terminated or truncated

# Close the environment.
env.close()

# done = False
# while not done:
#     action = env.action_space.sample()
#     _, _, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated






# ------------------- BFS --------------------

#working with (x,y) coordinates as its more intiuitive
initial_state  = (6,3)
print("Initial State:", initial_state)
transitions = {}  
queue = queue.Queue()  

transitions[initial_state] = (None, None)  
queue.put(initial_state)

def valid_actions(state):
    actions = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

    x, y = state
    rows, cols = map.shape

    for dx, dy in directions:
        nx, ny = x + dx, y + dy  # next position

        # Check bounds
        if not (0 <= nx < cols and 0 <= ny < rows):
            continue

        tile = map[ny, nx]  # NumPy uses (row, col) = (y, x)
        if tile == 1:  # Wall
            continue

        # If there’s a crate, check space beyond it
        if tile in (2, 4): 
            bx, by = nx + dx, ny + dy
            if not (0 <= bx < cols and 0 <= by < rows):
                continue
            beyond = map[by, bx]
            if beyond in (1, 2, 4):  # wall or another crate
                continue

        # Valid move -> add action tuple
        actions.append((dx, dy))

    return actions



print("Valid actions from initial state:", valid_actions(initial_state))




def bfs():
    while not queue.empty():
        current_state = queue.get()

        # recreating patjh when goal is found
        if map[current_state[1], current_state[0]] in (3, 4):  
            path = []
            while current_state is not None:
                prev_state, action = transitions[current_state]
                if action is not None:
                    path.append(action)
                current_state = prev_state
            path.reverse()
            return path

        # Explore valid actions
        for action in valid_actions(current_state):
            dx, dy = action
            next_state = (current_state[0] + dx, current_state[1] + dy)

            if next_state not in transitions:
                transitions[next_state] = (current_state, action)
                queue.put(next_state)
    
    return None  # Return None if no path is found.

solution_path = bfs()
print("Solution path (dx, dy):", solution_path) 

# Close the environment.
