import gymnasium as gym
import famnit_gym
import numpy as np
import time

# ------------ Map creation and setup -------------------
# 0 = Floor, 1 = Wall, 2 = Crate, 3 = Goal, 4 = Crate on Goal, 5 = Player

#most complex
map1=np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,0,0,0,0,0,3,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,0,1,1,0,0,1,1,0,0,1],
    [1,3,0,0,0,0,0,0,0,3,0,1],
    [1,0,2,0,0,0,0,2,0,0,0,1],
    [1,0,0,0,1,5,1,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

#most crates
map2=np.array([
    [1,1,1,1,1,1,1,1],
    [1,0,0,2,0,3,5,1],
    [1,0,2,0,0,1,3,1],
    [1,3,3,2,0,2,2,1],
    [1,3,3,2,0,0,0,1],
    [1,1,1,1,1,1,1,1]
])

#biggest map
map3=np.array([
    [0,0,0,1,1,1,1,1,1,1,1],
    [0,0,0,1,0,0,0,0,0,0,1],
    [0,0,1,1,0,0,0,0,2,0,1],
    [0,1,0,0,0,1,1,0,1,1,1],
    [1,1,0,0,1,0,0,3,0,0,1],
    [1,0,0,1,3,2,2,0,0,1,1],
    [1,5,1,3,0,0,0,0,1,1,0],
    [1,1,0,0,0,0,0,1,1,0,0],
    [1,1,1,1,1,1,1,1,0,0,0]
])

#least
map4=np.array([
    [1,1,1,1,1,1,1],
    [1,5,0,2,0,3,1],
    [1,1,0,2,0,1,1],
    [1,3,0,0,0,0,1],
    [1,1,1,1,1,1,1]
])

#2nd simplest
map5=np.array([
    [1,1,1,1,1,1],
    [1,5,2,3,1,1],
    [1,2,2,1,1,1],
    [1,0,3,0,0,1],
    [1,0,0,0,0,1],
    [1,3,3,2,0,1],
    [1,1,1,1,1,1]
])
maps = [map1, map2, map3, map4, map5]
map_names = ["map1", "map2", "map3", "map4", "map5"]
map_template=None
options = {'map_template': map_template, 'scale': 0.75}
# ------------------- Utility Functions --------------------

def get_ground_state(y, x, template_map):
    return 3 if template_map[y, x] == 3 else 0

def is_goal(m):
    # Check if all crates are on goals 
    map_array = np.array(m)
    return not np.any(map_array == 2)

def valid_moves(player_pos, m):
    #Return all valid moves (dx, dy, push) from current state
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
    #Check if a crate pushed to (y,x) is in a corner
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
        if is_dead_square(by, bx, map_template):
            return None
        # Restore prev crate position
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
def dfs_search(state, depth, max_depth, visited, start_time, time_limit, states_counter):
    if time.time() - start_time > time_limit:
        return None
    x, y, m = state
    states_counter[0] += 1  # count this state
    if is_goal(m):
        return []
    if depth >= max_depth:
        return None
    for dx, dy, push in valid_moves((x, y), m):
        new_state = apply_move(x, y, m, dx, dy, push)
        if new_state is None or new_state in visited:
            continue
        visited.add(new_state)
        result = dfs_search(new_state, depth + 1, max_depth, visited, start_time, time_limit, states_counter)
        if result is not None:
            return [(dx, dy)] + result
    return None

def solve_sokoban_dfs_iterative(initial_state, max_depth=100, time_limit=600):
    start_time = time.time()
    states_counter = [0]  # mutable counter
    for depth in range(1, max_depth + 1):
        print(f"Searching with depth limit: {depth}")
        visited = {initial_state}
        path = dfs_search(initial_state, 0, depth, visited, start_time, time_limit, states_counter)
        if path is not None:
            elapsed = time.time() - start_time
            return path, states_counter[0], elapsed
        if time.time() - start_time > time_limit:
            print("Time limit reached.")
            break
    elapsed = time.time() - start_time
    return None, states_counter[0], elapsed

# ------------------- Main Execution --------------------
with open("dfs_results.txt", "w") as f:
    for i, m in enumerate(maps):
        map_template = m
        options['map_template'] = map_template

        player_pos_arr = np.argwhere(map_template == 5)
        if player_pos_arr.size == 0:
            print(f"No player found in {map_names[i]}, skipping map.")
            continue
        player_pos = player_pos_arr[0]
        initial_state = (player_pos[1], player_pos[0], tuple(map_template.flatten()))

        print(f"\n--- Running DFS on {map_names[i]} ---")
        solution_path, states_explored, elapsed = solve_sokoban_dfs_iterative(initial_state)

        path_length = len(solution_path) if solution_path else 0
        print(f"{map_names[i]} DFS path length: {path_length}, States explored: {states_explored}, Time: {elapsed:.2f}s")
        f.write(f"{map_names[i]}: Time = {elapsed:.2f}s, Path length = {path_length}, States explored = {states_explored}\n")
