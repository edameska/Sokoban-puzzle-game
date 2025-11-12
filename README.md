# Sokoban Search Algorithms

This repository contains Python implementations of four Sokoban-solving algorithms using the `famnit_gym` environment:

1. **Breadth-First Search (BFS)**  
2. **Depth-First Search (DFS)**  
3. **A\***   
4. **Abstracted A\*** (Box-Centric Push-Based Search)

Each algorithm is implemented in a separate script.

## Requirements

It’s recommended to use a virtual environment to keep dependencies isolated.

### For Windows
```bash
python -m venv <venv_name>
<venv_name>\Scripts\activate
```
### For Linux/Mac
```bash
python3 -m venv <venv_name>
source <venv_name>/bin/activate
```
Before running any scripts, install the required packages:

```bash
pip install -r requirements.txt
```

## Running the algorithms
Each algorithm is tested on the same map in the examples. You can change the map by changing the `map_template` variable on the top of each script. Feel free to use one of the maps we provide in the maps.txt file or use your own. 
<br>
Run each algorithm separately:
1. Breadth-First Search
```bash
python3 uninformed_bfs.py
```
1. Depth-First Search
```bash
python3 uninformed_dfs.py
```
1. A* Algorithm
```bash
python3 astar.py
```
4. Abstracted A* (Box Abstraction)
```bash
python3 abstraction.py
```
## Test files
Additionally, we also provide the tests we used. Each algorithm was tested on 5 maps with different sizes and complexities. To verify correctness and test performance run:
```bash
python3 tests/uninformed_dfs_tests.py
python3 tests/uninformed_bfs_tests.py
python3 tests/astar.py
python3 tests/abstracted_test.py
```
Each test file outputs a .txt file containing metrics for each different map, such as:
- Path length
- Execution time
- Number of explored states
