import gymnasium as gym
import famnit_gym
import numpy as np
from collections import deque
import heapq
import time

# ---------------- Map Setup ----------------
map_template = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 5, 0, 2, 0, 3, 1],
    [1, 0, 0, 0, 2, 0, 1],
    [1, 0, 2, 0, 0, 3, 1],
    [1, 0, 0, 0, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1]
])
options = {'map_template': map_template, 'scale': 0.75}

# Directions: left, right, up, down (dx, dy)
dirs = [(-1,0),(1,0),(0,-1),(0,1)]
move_to_action = {(0,-1):0, (1,0):1, (0,1):2, (-1,0):3}

# ---------------- Utilities ----------------
def get_ground_state(y, x, template_map):
    return 3 if template_map[y, x] == 3 else 0

def is_goal(boxes, goals):
    return all(b in goals for b in boxes)

# BFS to see if player can reach a target position without crossing crates
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

# Apply a move; push=True if pushing a specific crate
def apply_move(player_pos, boxes, arr, move, push_idx=None):
    py, px = player_pos
    dx, dy = move
    nx, ny = px + dx, py + dy
    new_boxes = list(boxes)
    new_arr = np.array(arr).reshape(map_template.shape).copy()
    
    if push_idx is not None:
        by, bx = boxes[push_idx]
        by_new, bx_new = by + dy, bx + dx
        # Bounds check
        if not (0 <= by_new < arr.shape[0] and 0 <= bx_new < arr.shape[1]):
            return player_pos, boxes, tuple(arr.flatten())
        new_boxes[push_idx] = (by_new, bx_new)
        new_arr[by_new, bx_new] = 2
        new_arr[by, bx] = get_ground_state(by, bx, map_template)
    
    # Move player
    new_arr[ny, nx] = 5
    new_arr[py, px] = get_ground_state(py, px, map_template)
    
    return (ny, nx), tuple(new_boxes), tuple(new_arr.flatten())

# ---------------- Player-aware Box-Centric A* for multiple crates ----------------
def solve_sokoban_astar(initial_player_pos, boxes, goals, arr):
    start_state = (initial_player_pos, tuple(boxes), tuple(arr.flatten()))
    pq = []
    heapq.heappush(pq, (0, start_state))
    visited = set()
    transitions = {start_state:(None, None)}  # state -> (parent_state, moves)
    start_time = time.time()
    
    while pq:
        g, state = heapq.heappop(pq)
        player_pos, box_positions, m = state
        arr_state = np.array(m).reshape(map_template.shape)
        
        if is_goal(box_positions, goals):
            # reconstruct full path
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

                # Can player reach push position without crossing other boxes?
                other_boxes = tuple(b for i,b in enumerate(box_positions) if i != idx)
                p_path = player_path(arr_state, player_pos, push_pos, other_boxes)
                if p_path is None:
                    continue

                # Apply player moves
                new_player_pos = player_pos
                temp_arr = np.array(arr_state).reshape(map_template.shape)
                moves = []
                for move_p in p_path:
                    new_player_pos, _, temp_arr_flat = apply_move(new_player_pos, box_positions, temp_arr, move_p, push_idx=None)
                    temp_arr = np.array(temp_arr_flat).reshape(map_template.shape)
                    moves.append(move_p)
                
                # Apply push
                new_player_pos, new_box_positions, temp_arr_flat = apply_move(new_player_pos, box_positions, temp_arr, (dx, dy), push_idx=idx)
                moves.append((dx, dy))
                new_state = (new_player_pos, new_box_positions, temp_arr_flat)

                if new_state not in visited:
                    heapq.heappush(pq, (g+len(moves), new_state))
                    transitions[new_state] = (state, moves)

        if time.time()-start_time > 30:
            print("Timeout")
            break
    print("No solution")
    return None

# ---------------- Main ----------------
initial_player_pos = tuple(np.argwhere(map_template==5)[0])
initial_box_pos = [tuple(g) for g in np.argwhere(map_template==2)]
goals = [tuple(g) for g in np.argwhere(map_template==3)]
full_path = solve_sokoban_astar(initial_player_pos, initial_box_pos, goals, map_template)

print("\n--- Player-aware Sokoban A* ---")
if full_path:
    print(f"Full path length: {len(full_path)}")
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    import time
    for dx,dy in full_path:
        action = move_to_action.get((dx,dy))
        if action is None: continue
        _,_,terminated,truncated,_ = env.step(action)
        time.sleep(0.2)
        if terminated or truncated: break
    env.close()
else:
    print("No solution found")
