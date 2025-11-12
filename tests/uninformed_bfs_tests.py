import gymnasium as gym
import famnit_gym
import numpy as np
import time
import queue

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
    map_array = np.array(m)
    return not np.any(map_array == 2)

def valid_moves(player_pos, m):
    moves = []
    x, y = player_pos
    arr = np.array(m).reshape(map_template.shape)
    rows, cols = arr.shape
    directions = [(-1,0),(1,0),(0,-1),(0,1)]  # left, right, up, down

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < cols and 0 <= ny < rows):
            continue
        if arr[ny, nx] == 1:
            continue  # wall
        if arr[ny, nx] in (2,4):
            bx, by = nx + dx, ny + dy
            if not (0 <= bx < cols and 0 <= by < rows):
                continue
            if arr[by, bx] in (0,3):
                moves.append((dx, dy, True))
        elif arr[ny, nx] in (0,3):
            moves.append((dx, dy, False))
    return moves

def apply_move(x, y, m, dx, dy, push):
    arr = np.array(m).reshape(map_template.shape).copy()
    nx, ny = x + dx, y + dy

    if push:
        bx, by = nx + dx, ny + dy
        # Restore previous square
        arr[ny, nx] = get_ground_state(ny, nx, map_template)
        # Move crate
        arr[by, bx] = 4 if arr[by, bx] == 3 else 2

    # Move player
    arr[y, x] = get_ground_state(y, x, map_template)
    arr[ny, nx] = 5
    return (nx, ny, tuple(arr.flatten()))

def is_deadlock(arr, template):
    for y in range(1, arr.shape[0]-1):
        for x in range(1, arr.shape[1]-1):
            if arr[y, x] == 2 and template[y, x] != 3:
                if ((arr[y-1, x] == 1 or arr[y+1, x] == 1) and
                    (arr[y, x-1] == 1 or arr[y, x+1] == 1)):
                    return True
    return False

# ------------------- BFS Implementation --------------------
def solve_sokoban_bfs(initial_state):
    visited = set()
    visited.add(initial_state)
    bfs_queue = queue.Queue()
    bfs_queue.put(initial_state)
    transitions = {initial_state: (None, None)}

    solution_state = None
    start_time = time.time()

    while not bfs_queue.empty():
        x, y, m = bfs_queue.get()

        if is_goal(m):
            solution_state = (x, y, m)
            break

        if time.time() - start_time > 600:  # timeout safeguard
            print("Timeout during BFS")
            break

        for dx, dy, push in valid_moves((x, y), m):
            new_state = apply_move(x, y, m, dx, dy, push)
            if new_state is None:
                continue
            arr = np.array(new_state[2]).reshape(map_template.shape)
            if is_deadlock(arr, map_template):
                continue
            if new_state not in visited:
                visited.add(new_state)
                transitions[new_state] = ((x, y, m), (dx, dy))
                bfs_queue.put(new_state)

    elapsed = time.time() - start_time
    states_explored = len(visited)

    # Reconstruct path
    path = []
    if solution_state:
        state = solution_state
        while transitions[state][0] is not None:
            parent, move = transitions[state]
            path.append(move)
            state = parent
        path = path[::-1]

    return path, states_explored, elapsed

# ------------------- Main Execution --------------------
with open("bfs_results.txt", "w") as f:
    for i, m in enumerate(maps):
        map_template = m
        options['map_template'] = map_template

        player_pos_arr = np.argwhere(map_template == 5)
        if player_pos_arr.size == 0:
            print(f"No player found in {map_names[i]}, skipping map.")
            continue
        player_pos = player_pos_arr[0]
        initial_state = (player_pos[1], player_pos[0], tuple(map_template.flatten()))

        print(f"\n--- Running BFS on {map_names[i]} ---")
        solution_path, states_explored, elapsed = solve_sokoban_bfs(initial_state)

        path_length = len(solution_path) if solution_path else 0
        print(f"{map_names[i]} BFS path length: {path_length}, States explored: {states_explored}, Time: {elapsed:.2f}s")
        f.write(f"{map_names[i]}: Time = {elapsed:.2f}s, Path length = {path_length}, States explored = {states_explored}\n")
