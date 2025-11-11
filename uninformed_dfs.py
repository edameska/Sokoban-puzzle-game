import gymnasium as gym
import famnit_gym
import numpy as np
import time

# ------------ Map creation and setup -------------------
# 0 = Floor, 1 = Wall, 2 = Crate, 3 = Goal, 4 = Crate on Goal, 5 = Player

map_template = np.array([
    [1,1,1,1,1,1,1],
    [1,5,0,2,0,3,1],
    [1,0,0,0,0,0,1],
    [1,0,0,2,0,3,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1]
])
options = {'map_template': map_template, 'scale': 0.75}

# ------------------- Utility Functions --------------------

def get_ground_state(y, x, template_map):
    return 3 if template_map[y, x] == 3 else 0

def is_goal(m):
    """Check if all crates are on goals (2 should no longer exist)."""
    map_array = np.array(m)
    return not np.any(map_array == 2)

def valid_moves(player_pos, m):
    """Return all valid moves (dx, dy, push) from current state."""
    moves = []
    x, y = player_pos
    map_array = np.array(m).reshape(map_template.shape)
    rows, cols = map_array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # Check bounds
        if not (0 <= nx < cols and 0 <= ny < rows):
            continue
        # Check wall
        if map_array[ny, nx] == 1:
            continue
        # Check crate
        if map_array[ny, nx] in (2, 4):
            bx, by = nx + dx, ny + dy
            # Check bounds for crate push
            if not (0 <= bx < cols and 0 <= by < rows):
                continue
            if map_array[by, bx] in (0, 3):
                moves.append((dx, dy, True))
        # Empty or goal
        elif map_array[ny, nx] in (0, 3):
            moves.append((dx, dy, False))
    return moves

def is_dead_square(y, x, template_map):
    """Check if a crate pushed to (y,x) is irrecoverably stuck (in a corner not a goal)."""
    if template_map[y, x] == 3:
        return False
    walls = 0
    if template_map[y-1, x] == 1 or template_map[y+1, x] == 1:
        walls += 1
    if template_map[y, x-1] == 1 or template_map[y, x+1] == 1:
        walls += 1
    return walls >= 2

def apply_move(x, y, m, dx, dy, push):
    map_array = np.array(m).reshape(map_template.shape).copy()
    nx, ny = x + dx, y + dy

    if push:
        bx, by = nx + dx, ny + dy
        # Check dead corner
        if is_dead_square(by, bx, map_template):
            return None
        # Restore previous crate position
        if map_array[ny, nx] == 4:
            map_array[ny, nx] = 3
        else:
            map_array[ny, nx] = get_ground_state(ny, nx, map_template)
        # Move crate
        map_array[by, bx] = 4 if map_array[by, bx] == 3 else 2

    # Move player
    map_array[y, x] = get_ground_state(y, x, map_template)
    map_array[ny, nx] = 5
    return (nx, ny, tuple(map_array.flatten()))

# ------------------- DFS with Iterative Deepening --------------------

def dfs_search(state, depth, max_depth, visited, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None
    x, y, m = state
    if is_goal(m):
        return []
    if depth >= max_depth:
        return None
    for dx, dy, push in valid_moves((x, y), m):
        new_state = apply_move(x, y, m, dx, dy, push)
        if new_state is None or new_state in visited:
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

# Find initial player position in map
player_pos = np.argwhere(map_template == 5)[0]
initial_state = (player_pos[1], player_pos[0], tuple(map_template.flatten()))

solution_path = solve_sokoban_dfs_iterative(initial_state)

print("Solution path (dx, dy):", solution_path)
print(f"Path length: {len(solution_path) if solution_path else 0}")

# Execute in famnit_gym if solution exists
if solution_path:
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    time.sleep(1)

    move_to_action = {
        (0, -1): 0,  # up
        (1, 0): 1,   # right
        (0, 1): 2,   # down
        (-1, 0): 3   # left
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
