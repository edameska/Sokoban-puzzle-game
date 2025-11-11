import gymnasium as gym
import famnit_gym
import numpy as np
import time
import queue

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

        if time.time() - start_time > 15:  # timeout safeguard
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

    print(f"BFS finished. States explored: {len(visited)}")

    # Reconstruct path
    path = []
    if solution_state:
        state = solution_state
        while transitions[state][0] is not None:
            parent, move = transitions[state]
            path.append(move)
            state = parent
        return path[::-1]
    return None

# ------------------- Main Execution --------------------

# Auto-detect initial player position
player_pos = np.argwhere(map_template == 5)[0]
initial_state = (player_pos[1], player_pos[0], tuple(map_template.flatten()))

print("\n--- Running BFS ---")
solution_path = solve_sokoban_bfs(initial_state)
print(f"BFS path length: {len(solution_path) if solution_path else 0}")
print("Solution path (dx, dy):", solution_path)

# ------------------- Execute in Environment -------------------
if solution_path:
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    time.sleep(1)

    move_to_action = {
        (0, -1):0,  # up
        (1, 0):1,   # right
        (0, 1):2,   # down
        (-1, 0):3   # left
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
