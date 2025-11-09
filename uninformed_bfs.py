import gymnasium as gym
import famnit_gym
import numpy as np
import time

# ------------ Map creation and setup -------------------
# 0 = Floor, 1 = Wall, 2 = Crate, 3 = Goal, 4 = Crate on Goal, 5 = Player

map_template = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 3, 0, 1], 
    [1, 0, 2, 0, 0, 0, 0, 1], 
    [1, 0, 0, 0, 0, 0, 5, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1]
])

options = {'map_template': map_template, 'scale': 0.75}

# ------------------- Utility Functions --------------------

def get_ground_state(y, x, template_map):
    """Returns 3 (Goal) or 0 (Floor) for a given coordinate."""
    if 0 <= y < template_map.shape[0] and 0 <= x < template_map.shape[1]:
        # a cell is a goal spot if the original template had a 3  there
        if template_map[y, x] == 3:
            return 3  # Goal
    return 0  # Floor

def is_goal(m):
    # Goal reached if no plain crates (2) exist
    map_array = np.array(m)
    return not np.any(map_array == 2)

def valid_moves(player_pos, m):
    moves = []
    x, y = player_pos
    map_array = np.array(m).reshape(map_template.shape)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        # Check for wall
        if map_array[ny, nx] == 1:
            continue
        
        # Check for crate in the next cell
        if map_array[ny, nx] in (2, 4):
            bx, by = nx + dx, ny + dy
            # Check if crate can be pushed 
            if map_array[by, bx] in (0, 3):
                moves.append((dx, dy, True)) #valid
        
        # Empty or goal
        elif map_array[ny, nx] in (0, 3):
             moves.append((dx, dy, False)) # Valid simple move

    return moves

# Move + new map state
def apply_move(x, y, m, dx, dy, push):
    # Use the global map_template to determine static goal locations
    global map_template 

    # Create a copy of the map to modify
    map_array = np.array(m).reshape(map_template.shape).copy()
    nx, ny = x + dx, y + dy
    
    #  Handle Crate Movement (if pushing) 
    if push:
        bx, by = nx + dx, ny + dy
        
        # Old position to ground
        if map_array[ny, nx] == 2:      # Crate leaves a Floor
            map_array[ny, nx] = get_ground_state(ny, nx, map_template)
        elif map_array[ny, nx] == 4:    # Crate leaves a Goal
            map_array[ny, nx] = get_ground_state(ny, nx, map_template)

        # Update new position
        if map_array[by, bx] == 0:
            map_array[by, bx] = 2       
        elif map_array[by, bx] == 3:
            map_array[by, bx] = 4       
            
    # Player Movement
    
    # TRevert to prev based on og map
    map_array[y, x] = get_ground_state(y, x, map_template)
    
    map_array[ny, nx] = 5
    
    return (nx, ny, tuple(map_array.flatten()))
def is_deadlock(map_array, template_map):
    for y in range(1, map_array.shape[0] - 1):
        for x in range(1, map_array.shape[1] - 1):
            if map_array[y, x] == 2:  # crate not on goal
                # Check corners
                if ((map_array[y-1, x] == 1 or map_array[y+1, x] == 1) and
                    (map_array[y, x-1] == 1 or map_array[y, x+1] == 1)):
                    return True
    return False


# ------------------- BFS Implementation --------------------

def solve_sokoban_bfs(initial_state):
    # Initial state (player x,y + flattened map)
    visited = set()
    visited.add(initial_state)
    bfs_queue = queue.Queue()
    bfs_queue.put(initial_state)
    transitions = {initial_state: (None, None)}  # (parent, move)

    # BFS loop
    solution_state = None
    start_time = time.time()

    while not bfs_queue.empty():
        x, y, m = bfs_queue.get()
        
        if is_goal(m):
            solution_state = (x, y, m)
            break
        
        # Early exit safeguard for large state spaces
        if time.time() - start_time > 10 and bfs_queue.qsize() > 50000:
            print("Search space too large")
            break
            
        for dx, dy, push in valid_moves((x, y), m):
            new_state = apply_move(x, y, m, dx, dy, push)
            map_array = np.array(new_state[2]).reshape(map_template.shape)

            if is_deadlock(map_array, map_template):
                continue  # prune
            
            if new_state not in visited:
                visited.add(new_state)
                # parent + action
                transitions[new_state] = ((x, y, m), (dx, dy)) 
                bfs_queue.put(new_state)

    print(f"BFS finished. States explored: {len(visited)}")

    # Reconstruct path
    solution_path = []
    if solution_state:
        state = solution_state
        # Parent pointers
        while transitions[state][0] is not None:
            parent, move = transitions[state]
            solution_path.append(move)
            state = parent
        return solution_path[::-1] # Reverse the path
    
    return None

# ------------------- Main Execution --------------------

# Initial state (player x,y + flattened map)
initial_state = (6, 3, tuple(map_template.flatten())) 


print("\n--- Running BFS ---")
solution_path = solve_sokoban_bfs(initial_state)
print(f"BFS path length: {len(solution_path) if solution_path else 0}")
print("Solution path (dx, dy):", solution_path)

# ------------------- Redraw path in environment -------------------
if solution_path:
    print("\nExecuting solution in the environment...")
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    time.sleep(1)

    move_to_action = { (0, -1):0,
         (1, 0):1,
         (0, 1):2,
        (-1, 0):3}

    for dx, dy in solution_path:
        action = move_to_action[(dx, dy)]
        _, _, terminated, truncated, _ = env.step(action)
        time.sleep(0.3)
        if terminated or truncated:
            break

    env.close()
else:
    print("No solution found for the given map configuration.")