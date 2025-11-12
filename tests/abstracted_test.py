import gymnasium as gym
import famnit_gym
import numpy as np
from collections import deque
import heapq
import time

# ---------------- Maps ----------------
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
dirs = [(-1,0),(1,0),(0,-1),(0,1)]
move_to_action = {(0,-1):0, (1,0):1, (0,1):2, (-1,0):3}

# ---------------- Utilities (unchanged) ----------------
def get_ground_state(y, x, template_map):
    return 3 if template_map[y, x] == 3 else 0

def is_goal(boxes, goals):
    return all(b in goals for b in boxes)

def player_path(arr, start, goal, box_positions):
    visited = set()
    queue = deque([(start, [])])
    while queue:
        (y, x), path = queue.popleft()
        if (y, x) == goal:
            return path
        if (y, x) in visited:
            continue
        visited.add((y, x))
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= ny < arr.shape[0] and 0 <= nx < arr.shape[1]:
                if arr[ny, nx] in (0,3) and (ny, nx) not in box_positions:
                    queue.append(((ny,nx), path + [(dx, dy)]))
    return None

def apply_move(player_pos, boxes, arr, move, push_idx=None):
    py, px = player_pos
    dx, dy = move
    nx, ny = px + dx, py + dy
    new_boxes = list(boxes)
    new_arr = np.array(arr).reshape(arr.shape).copy()

    if push_idx is not None:
        by, bx = boxes[push_idx]
        by_new, bx_new = by + dy, bx + dx
        if not (0 <= by_new < arr.shape[0] and 0 <= bx_new < arr.shape[1]):
            return player_pos, boxes, tuple(arr.flatten())
        new_boxes[push_idx] = (by_new, bx_new)
        new_arr[by_new, bx_new] = 2
        new_arr[by, bx] = get_ground_state(by, bx, arr)

    new_arr[ny, nx] = 5
    new_arr[py, px] = get_ground_state(py, px, arr)
    return (ny, nx), tuple(new_boxes), tuple(new_arr.flatten())

# ---------------- Abstracted Box-Centric A* ----------------
def solve_sokoban_astar(initial_player_pos, boxes, goals, arr):
    start_state = (initial_player_pos, tuple(boxes), tuple(arr.flatten()))
    pq = []
    heapq.heappush(pq, (0, start_state))
    visited = set()
    transitions = {start_state:(None, None)}
    start_time = time.time()
    
    while pq:
        g, state = heapq.heappop(pq)
        player_pos, box_positions, m = state
        arr_state = np.array(m).reshape(arr.shape)

        if is_goal(box_positions, goals):
            # reconstruct path
            path = []
            s = state
            while transitions[s][0] is not None:
                parent, moves = transitions[s]
                path = moves + path
                s = parent
            return path, time.time()-start_time, len(visited)

        if state in visited:
            continue
        visited.add(state)

        for idx, (by, bx) in enumerate(box_positions):
            for dx, dy in dirs:
                target_box_pos = (by + dy, bx + dx)
                push_pos = (by - dy, bx - dx)

                if not (0 <= target_box_pos[0] < arr.shape[0] and 0 <= target_box_pos[1] < arr.shape[1]):
                    continue
                if not (0 <= push_pos[0] < arr.shape[0] and 0 <= push_pos[1] < arr.shape[1]):
                    continue
                if arr_state[target_box_pos[0], target_box_pos[1]] not in (0,3):
                    continue

                other_boxes = tuple(b for i,b in enumerate(box_positions) if i != idx)
                p_path = player_path(arr_state, player_pos, push_pos, other_boxes)
                if p_path is None:
                    continue

                new_player_pos = player_pos
                temp_arr = np.array(arr_state).reshape(arr.shape)
                moves = []
                for move_p in p_path:
                    new_player_pos, _, temp_arr_flat = apply_move(new_player_pos, box_positions, temp_arr, move_p)
                    temp_arr = np.array(temp_arr_flat).reshape(arr.shape)
                    moves.append(move_p)

                new_player_pos, new_box_positions, temp_arr_flat = apply_move(new_player_pos, box_positions, temp_arr, (dx, dy), push_idx=idx)
                moves.append((dx, dy))
                new_state = (new_player_pos, new_box_positions, temp_arr_flat)

                if new_state not in visited:
                    heapq.heappush(pq, (g+len(moves), new_state))
                    transitions[new_state] = (state, moves)

        if time.time()-start_time > 300:
            print("Timeout")
            break
    return None, time.time()-start_time, len(visited)

# ---------------- Main Execution ----------------
with open("abstracted_astar_results.txt", "w") as f:
    for i, m in enumerate(maps):
        map_template = m
        options['map_template'] = map_template
        initial_player_pos = tuple(np.argwhere(map_template==5)[0])
        initial_box_pos = [tuple(g) for g in np.argwhere(map_template==2)]
        goals = [tuple(g) for g in np.argwhere(map_template==3)]

        print(f"\n--- Abstracted A* on {map_names[i]} ---")
        full_path, elapsed, states_explored = solve_sokoban_astar(initial_player_pos, initial_box_pos, goals, map_template)

        path_len = len(full_path) if full_path else 0
        f.write(f"{map_names[i]}: Time={elapsed:.2f}s, Path length={path_len}, States explored={states_explored}\n")
        print(f"{map_names[i]} done: Time={elapsed:.2f}s, Path length={path_len}, States explored={states_explored}")
