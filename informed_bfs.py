"""
Added/Implemented Features:
    Full A* search algorithm using a priority queue (min-heap) with f = g + h cost function.
    Heuristic function (heuristic_box_to_goal) calculating the sum of Manhattan distances
        from each crate to its nearest goal — ensures admissibility.
    Path reconstruction using a 'transitions' dictionary to recover the full solution path.
    Deadlock detection (is_deadlock) to prune unsolvable states early and improve efficiency.
    Timeout control to prevent infinite search loops on difficult maps. *currently 10s)
    Execution of the final solution path within the famnit_gym Sokoban environment for visualization.
"""
import gymnasium as gym
import famnit_gym
import numpy as np
import heapq, time

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

def get_ground_state(y, x, template):
    """Return the underlying ground type (goal or floor) at a given position."""
    if 0 <= y < template.shape[0] and 0 <= x < template.shape[1]:
        if template[y, x] == 3:
            return 3
    return 0

def is_goal(m):
    """Check if all crates are on goals."""
    return not np.any(np.array(m) == 2)

def valid_moves(player_pos, m):
    """Return all valid moves (dx, dy, push) from current state."""
    moves = []
    x, y = player_pos
    arr = np.array(m).reshape(map_template.shape)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if arr[ny, nx] == 1:
            continue  # wall

        # If there’s a crate, check if it can be pushed
        if arr[ny, nx] in (2, 4):
            bx, by = nx + dx, ny + dy
            if arr[by, bx] in (0, 3):
                moves.append((dx, dy, True))  # push move
        elif arr[ny, nx] in (0, 3):
            moves.append((dx, dy, False))  # normal move
    return moves

def apply_move(x, y, m, dx, dy, push):
    """Apply a move (with or without pushing a crate) and return the new state."""
    arr = np.array(m).reshape(map_template.shape).copy()
    nx, ny = x + dx, y + dy

    if push:
        bx, by = nx + dx, ny + dy
        # Move crate forward
        arr[ny, nx] = get_ground_state(ny, nx, map_template)
        arr[by, bx] = 2 if arr[by, bx] == 0 else 4

    # Move player
    arr[y, x] = get_ground_state(y, x, map_template)
    arr[ny, nx] = 5
    return (nx, ny, tuple(arr.flatten()))

def is_deadlock(arr, template):
    """Detect simple deadlocks: crates in corners not on goals."""
    for y in range(1, arr.shape[0] - 1):
        for x in range(1, arr.shape[1] - 1):
            if arr[y, x] == 2 and template[y, x] != 3:
                # Crate in a corner (not a goal)
                if ((arr[y - 1, x] == 1 or arr[y + 1, x] == 1) and
                    (arr[y, x - 1] == 1 or arr[y, x + 1] == 1)):
                    return True
    return False

def heuristic_box_to_goal(map_array, template):
    """Sum of Manhattan distances from each crate to its nearest goal (admissible)."""
    crates = np.argwhere(map_array == 2)
    goals = np.argwhere(template == 3)
    total = 0

    for cy, cx in crates:
        # Find nearest goal for each crate
        min_dist = min(abs(cy - gy) + abs(cx - gx) for gy, gx in goals)
        total += min_dist
    return total

# ------------------- A* (informed search) --------------------

def solve_sokoban_astar(initial_state):
    """Run A* search to solve the Sokoban puzzle using the box-to-goal heuristic."""
    visited = set()
    pq = []
    transitions = {initial_state: (None, None)}

    x, y, m = initial_state
    g = 0
    h = heuristic_box_to_goal(np.array(m).reshape(map_template.shape), map_template)
    heapq.heappush(pq, (g + h, g, initial_state))  # (f = g + h, g, state)

    start = time.time()
    while pq:
        f, g, state = heapq.heappop(pq)
        x, y, m = state

        if state in visited:
            continue
        visited.add(state)

        # Goal check
        if is_goal(m):
            print(f"A* finished. States explored: {len(visited)}")
            return reconstruct_path(transitions, state)

        # Explore valid moves
        for dx, dy, push in valid_moves((x, y), m):
            new_state = apply_move(x, y, m, dx, dy, push)
            if new_state in visited:
                continue

            arr = np.array(new_state[2]).reshape(map_template.shape)
            if is_deadlock(arr, map_template):
                continue  # skip deadlocked states

            new_g = g + 1
            h = heuristic_box_to_goal(arr, map_template)
            heapq.heappush(pq, (new_g + h, new_g, new_state))
            transitions[new_state] = (state, (dx, dy))

        # Timeout safety
        if time.time() - start > 10:
            print("A* timeout.")
            break

    print("A* failed (no solution found or timed out).")
    return None

# ------------------- Helper: Reconstruct Path --------------------

def reconstruct_path(transitions, end_state):
    """Backtrack from goal to start to reconstruct the path of moves."""
    path = []
    s = end_state
    while transitions[s][0] is not None:
        parent, move = transitions[s]
        path.append(move)
        s = parent
    return path[::-1]

# ------------------- Main Execution --------------------

initial_state = (6, 3, tuple(map_template.flatten()))

print("\n--- Running A* ---")
astar_path = solve_sokoban_astar(initial_state)
print(f"A* path length: {len(astar_path) if astar_path else 0}")

if astar_path:
    print("\nExecuting A* solution in environment...")
    env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
    env.reset()
    time.sleep(1)

    move_to_action = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
    for dx, dy in astar_path:
        action = move_to_action[(dx, dy)]
        _, _, terminated, truncated, _ = env.step(action)
        time.sleep(0.3)
        if terminated or truncated:
            break
    env.close()
else:
    print("No solution found for the given map configuration.")