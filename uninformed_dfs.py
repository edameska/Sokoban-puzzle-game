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
    if template_map[y, x] == 3:
        return 3  # Goal
    return 0  # Floor

def is_goal(m):
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
                moves.append((dx, dy, True)) # valid push
        # Empty or goal
        elif map_array[ny, nx] in (0, 3):
            moves.append((dx, dy, False)) # simple move
    return moves

def is_dead_square(y, x, template_map):
    """Check if a crate pushed to (y,x) is irrecoverably stuck (in a corner not a goal)."""
    if template_map[y, x] == 3:
        return False  # Goal square is fine
    # Check for corner: surrounded by walls or borders in two perpendicular directions
    walls = 0
    if template_map[y-1, x] == 1 or template_map[y+1, x] == 1:
        walls += 1
    if template_map[y, x-1] == 1 or template_map[y, x+1] == 1:
        walls += 1
    return walls >= 2  # Dead corner if surrounded in two directions

def apply_move(x, y, m, dx, dy, push):
    global map_template
    map_array = np.array(m).reshape(map_template.shape).copy()
    nx, ny = x + dx, y + dy

    if push:
        bx, by = nx + dx, ny + dy
        # If pushing into dead square, prune immediately
        if is_dead_square(by, bx, map_template):
            return None

        # Old position (crate leaves)
        map_array[ny, nx] = get_ground_state(ny, nx, map_template)

        # Update new crate position
        if map_array[by, bx] == 0:
            map_array[by, bx] = 2
        elif map_array[by, bx] == 3:
            map_array[by, bx] = 4

    # Move player
    map_array[y, x] = get_ground_state(y, x, map_template)
    map_array[ny, nx] = 5
    return (nx, ny, tuple(map_array.flatten()))


# ------------------- DFS with Iterative Deepening + Pruning --------------------

def dfs_search(state, depth, max_depth, visited, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None  # Timeout
    
    x, y, m = state
    if is_goal(m):
        return []  # Goal reached, return empty move list
    
    if depth >= max_depth:
        return None

    for dx, dy, push in valid_moves((x, y), m):
        new_state = apply_move(x, y, m, dx, dy, push)
        if new_state is None:
            continue  # Pruned dead state
        if new_state in visited:
            continue
        visited.add(new_state)
        result = dfs_search(new_state, depth + 1, max_depth, visited, start_time, time_limit)
        if result is not None:
            return [(dx, dy)] + result
    return None

def solve_sokoban_dfs_iterative(initial_state, max_depth=50, time_limit=20):
    start_time = time.time()
    for depth in range(1, max_depth + 1):
        print(f"Searching with depth limit: {depth}")
        visited = {initial_state}
        path = dfs_search(initial_state, 0, depth, visited, start_time, time_limit)
        if path is not None:
            print(f"Solution found at depth {depth}")
            return path
        if time.time() - start_time > time_limit:
            print("Time limit reached.")
            break
    print("No solution found within limits.")
    return None


# ------------------- Main Execution --------------------

initial_state = (6, 3, tuple(map_template.flatten()))

solution_path = solve_sokoban_dfs_iterative(initial_state)

print("Solution path (dx, dy):", solution_path)
print(f"Path length: {len(solution_path) if solution_path else 0}")

if solution_path:
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    time.sleep(1)

    move_to_action = {
        (0, -1): 0,
        (1, 0): 1,
        (0, 1): 2,
        (-1, 0): 3
    }

    for dx, dy in solution_path:
        action = move_to_action[(dx, dy)]
        _, _, terminated, truncated, _ = env.step(action)
        time.sleep(0.3)
        if terminated or truncated:
            break
    env.close()
else:
    print("No solution found for the given map configuration.")
