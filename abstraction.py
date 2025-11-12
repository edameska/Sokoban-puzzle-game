import gymnasium as gym
import famnit_gym
import numpy as np
from collections import deque
import heapq
import time

# ---------------- Map Setup ----------------
map_template =  np.array([
    [1,1,1,1,1,1,1],
    [1,5,0,2,0,3,1],
    [1,0,0,0,0,0,1],
    [1,0,0,2,0,3,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1]
])
options = {'map_template': map_template, 'scale': 0.75}

# Directions: left, right, up, down (dx, dy)
dirs = [(-1,0),(1,0),(0,-1),(0,1)]
move_to_action = {(0,-1):0, (1,0):1, (0,1):2, (-1,0):3}

# ---------------- Utilities ----------------
def get_ground_state(y, x, template_map):
    return 3 if template_map[y, x] == 3 else 0

def is_goal(box_positions, goals):
    return all(pos in goals for pos in box_positions)

def player_path(arr, start, goal, box_positions):
    # compute path from player to position behind box where we should push
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

def apply_move(player_pos, box_positions, arr, move, push_box_idx=None):
    py, px = player_pos
    dx, dy = move
    nx, ny = px + dx, py + dy
    new_box_positions = list(box_positions)
    new_arr = np.array(arr).reshape(map_template.shape).copy()

    if push_box_idx is not None: #handle box push if there is one
        bx, by = box_positions[push_box_idx][1], box_positions[push_box_idx][0]
        bx_new, by_new = bx + dx, by + dy
        if not (0 <= by_new < arr.shape[0] and 0 <= bx_new < arr.shape[1]):
            return player_pos, box_positions, tuple(arr.flatten())
        new_box_positions[push_box_idx] = (by_new, bx_new)
        new_arr[by_new, bx_new] = 2
        new_arr[by, bx] = get_ground_state(by, bx, map_template)

    # Move player
    new_arr[ny, nx] = 5
    new_arr[py, px] = get_ground_state(py, px, map_template)

    return (ny, nx), tuple(new_box_positions), tuple(new_arr.flatten())

# ---------------- A* with box abstraction ----------------
def heuristic(box_positions, goals):
    # sum of Manhattan distances from boxes to nearest goals
    return sum(min(abs(b[0]-g[0]) + abs(b[1]-g[1]) for g in goals) for b in box_positions)

def solve_sokoban_astar(initial_player_pos, initial_box_positions, goals, arr):
    # Only considers box-pushing actions instead of all player movements
    start_state = (initial_player_pos, tuple(initial_box_positions), tuple(arr.flatten()))
    pq = []
    heapq.heappush(pq, (heuristic(initial_box_positions, goals), 0, start_state))
    visited = set()
    transitions = {start_state:(None, None)}
    start_time = time.time()

    while pq:
        f, g, state = heapq.heappop(pq)
        player_pos, box_positions, m = state
        arr_state = np.array(m).reshape(arr.shape)

        if is_goal(box_positions, goals):
            path = []
            s = state
            while transitions[s][0] is not None:
                parent, moves = transitions[s]
                path = moves + path
                s = parent
            return path

        if state in visited:
            continue
        visited.add(state)

        #look at possible box pushes
        for idx, box in enumerate(box_positions):
            by, bx = box
            for dx, dy in dirs:
                target_box_pos = (by+dy, bx+dx)
                push_pos = (by-dy, bx-dx)

                # Bounds and obstacles
                if not (0 <= target_box_pos[0] < arr.shape[0] and 0 <= target_box_pos[1] < arr.shape[1]):
                    continue
                if not (0 <= push_pos[0] < arr.shape[0] and 0 <= push_pos[1] < arr.shape[1]):
                    continue
                if arr_state[target_box_pos[0], target_box_pos[1]] not in (0,3):
                    continue

                # check if player can reach the pushing position
                p_path = player_path(arr_state, player_pos, push_pos, box_positions)
                if p_path is None:
                    continue

                #move behind box
                new_player_pos = player_pos
                temp_arr = np.array(arr_state)
                moves = []
                for move_p in p_path:
                    new_player_pos, _, temp_arr_flat = apply_move(new_player_pos, box_positions, temp_arr, move_p)
                    temp_arr = np.array(temp_arr_flat).reshape(arr.shape)
                    moves.append(move_p)

                #push box
                new_player_pos, new_box_positions, temp_arr_flat = apply_move(new_player_pos, box_positions, temp_arr, (dx, dy), push_box_idx=idx)
                moves.append((dx, dy))
                new_state = (new_player_pos, new_box_positions, temp_arr_flat)

                if new_state not in visited:
                    g_new = g + len(moves)
                    f_new = g_new + heuristic(new_box_positions, goals)
                    heapq.heappush(pq, (f_new, g_new, new_state))
                    transitions[new_state] = (state, moves)

        if time.time() - start_time > 30:
            print("Timeout")
            break

    print("No solution")
    return None

# ---------------- Main ----------------
initial_player_pos = tuple(np.argwhere(map_template==5)[0])
initial_box_positions = [tuple(b) for b in np.argwhere(map_template==2)]
goals = [tuple(g) for g in np.argwhere(map_template==3)]

full_path = solve_sokoban_astar(initial_player_pos, initial_box_positions, goals, map_template)

print("\n---  A* with box abstraction---")
if full_path:
    print(f"Full path length: {len(full_path)}")
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    for dx, dy in full_path:
        action = move_to_action.get((dx, dy))
        if action is None:
            continue
        _, _, terminated, truncated, _ = env.step(action)
        time.sleep(0.2)
        if terminated or truncated:
            break
    env.close()
else:
    print("No solution found")
