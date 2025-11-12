import gymnasium as gym
import famnit_gym
import numpy as np
import heapq, time

# ------------------- Utility Functions --------------------

def get_ground_state(y, x, template):
    # return ground type 
    return 3 if template[y, x] == 3 else 0

def is_goal(m):
    # check if all crates are on goals
    arr = np.array(m).reshape(map_template.shape)
    return not np.any(arr == 2)

def find_player(m):
    #Return (x, y) of player in the map
    arr = np.array(m).reshape(map_template.shape)
    pos = np.argwhere(arr == 5)
    if pos.size == 0:
        raise ValueError("Player not found in the map!")
    y, x = pos[0]
    return x, y

def valid_moves(player_pos, m):
    # return all valid moves (dx, dy, push) from current state
    moves = []
    x, y = player_pos
    arr = np.array(m).reshape(map_template.shape)
    rows, cols = arr.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < cols and 0 <= ny < rows):
            continue
        if arr[ny, nx] == 1:
            continue
        if arr[ny, nx] in (2, 4):  # crate
            bx, by = nx + dx, ny + dy
            if not (0 <= bx < cols and 0 <= by < rows):
                continue
            if arr[by, bx] in (0, 3):
                moves.append((dx, dy, True))
        elif arr[ny, nx] in (0, 3):
            moves.append((dx, dy, False))
    return moves

def apply_move(x, y, m, dx, dy, push):
    # apply move and return new state
    arr = np.array(m).reshape(map_template.shape).copy()
    nx, ny = x + dx, y + dy

    if push:
        bx, by = nx + dx, ny + dy
        arr[ny, nx] = get_ground_state(ny, nx, map_template)
        arr[by, bx] = 4 if map_template[by, bx] == 3 else 2

    arr[y, x] = get_ground_state(y, x, map_template)
    arr[ny, nx] = 5
    return (nx, ny, tuple(arr.flatten()))

def is_deadlock(arr, template):
    # detect crates in corners not on goals
    for y in range(1, arr.shape[0]-1):
        for x in range(1, arr.shape[1]-1):
            if arr[y, x] in (2,):
                if template[y, x] == 3:
                    continue
                if ((arr[y-1, x] == 1 or arr[y+1, x] == 1) and
                    (arr[y, x-1] == 1 or arr[y, x+1] == 1)):
                    return True
    return False

def heuristic_box_to_goal(arr, template):
    #sum of Manhattan distances from each crate to nearest goal
    arr = np.array(arr).reshape(template.shape)
    crates = np.argwhere(arr == 2)
    goals = np.argwhere(template == 3)
    total = 0
    for cy, cx in crates:
        total += min(abs(cy - gy) + abs(cx - gx) for gy, gx in goals)
    return total

# ------------------- A* Implementation --------------------

def reconstruct_path(transitions, end_state):
    path = []
    s = end_state
    while transitions[s][0] is not None:
        parent, move = transitions[s]
        path.append(move)
        s = parent
    return path[::-1]

def solve_sokoban_astar(initial_state, timeout=600):
    visited = set()
    pq = []
    transitions = {initial_state: (None, None)}

    g = 0
    h = heuristic_box_to_goal(initial_state[2], map_template)
    heapq.heappush(pq, (g + h, g, initial_state))
    start_time = time.time()

    while pq:
        f, g, state = heapq.heappop(pq)
        if state in visited:
            continue
        visited.add(state)

        if is_goal(state[2]):
            print(f"A* finished. States explored: {len(visited)}")
            return reconstruct_path(transitions, state)

        x, y = state[0], state[1]
        for dx, dy, push in valid_moves((x, y), state[2]):
            new_state = apply_move(x, y, state[2], dx, dy, push)
            if new_state in visited:
                continue
            arr = np.array(new_state[2]).reshape(map_template.shape)
            if is_deadlock(arr, map_template):
                continue
            new_g = g + 1
            h = heuristic_box_to_goal(arr, map_template)
            heapq.heappush(pq, (new_g + h, new_g, new_state))
            transitions[new_state] = (state, (dx, dy))

        if time.time() - start_time > timeout:
            print("A* timeout.")
            break

    print("A* failed (no solution).")
    return None
# ------------ Map creation -------------------
map1 = np.array([
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

map2 = np.array([
    [1,1,1,1,1,1,1,1],
    [1,0,0,2,0,3,5,1],
    [1,0,2,0,0,1,3,1],
    [1,3,3,2,0,2,2,1],
    [1,3,3,2,0,0,0,1],
    [1,1,1,1,1,1,1,1]
])

map3 = np.array([
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

map4 = np.array([
    [1,1,1,1,1,1,1],
    [1,5,0,2,0,3,1],
    [1,1,0,2,0,1,1],
    [1,3,0,0,0,0,1],
    [1,1,1,1,1,1,1]
])

map5 = np.array([
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
options = {'map_template': None, 'scale': 0.75}

# ------------------- Main Execution -------------------
with open("astar_results.txt", "w") as f:
    for i, m in enumerate(maps):
        map_template = m
        options['map_template'] = map_template

        try:
            player_x, player_y = find_player(map_template)
        except ValueError:
            print(f"No player found in {map_names[i]}, skipping map.")
            continue

        initial_state = (player_x, player_y, tuple(map_template.flatten()))
        print(f"\n--- Running A* on {map_names[i]} ---")
        start_time = time.time()
        astar_path = solve_sokoban_astar(initial_state)
        elapsed = time.time() - start_time

        print(f"{map_names[i]} A* path length: {len(astar_path) if astar_path else 0}")
        f.write(f"{map_names[i]}: Time = {elapsed:.2f}s, Path length = {len(astar_path) if astar_path else 0}\n")

        if astar_path:
            continue
        else:
            print(f"No solution found for {map_names[i]}.")
