import gymnasium as gym
import famnit_gym
import numpy as np

# Create and reset the environment.

map = np.array(  # The array can be of any dimensions greater than or equal to 2 × 2.
    [
        [0,0,0,0,0,0,1,0,0],
        [2,0,1,1,1,3,1,0,0],
        [0,1,2,0,1,0,0,0,0],
        [0,1,0,0,1,0,0,0,0],
        [0,0,2,0,0,0,0,0,0],
        [0,5,0,0,2,0,2,0,0],
        [0,0,0,0,0,0,0,0,0],
    ]
)


options = {
    'map_template': map , # An integer 0 - 999 for a hardoded level, or a numpy array for a custom level.
    'scale': 0.75  # Scale the image when render_mode='human'.
}

env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
observation, info = env.reset()

# Execute random actions.
done = False
while not done:
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Close the environment.
env.close()