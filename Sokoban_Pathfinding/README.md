# Sokoban Pathfinding Prototype

An interactive grid-navigation notebook that compares Random Search, Depth-First Search (DFS), Breadth-First Search (BFS), and A*. It includes handcrafted and randomly generated levels, manual controls, animated search execution, and a reproducible benchmark.

The notebook represents the pathfinding foundation of a Sokoban-style game: the player navigates around static walls from a start cell to a goal. Box-pushing mechanics are not included in this prototype.

## Results

All three systematic search algorithms solve the nine predefined levels. BFS and A* consistently find shortest paths, while DFS sometimes returns longer routes because it explores depth before path length. A* uses Manhattan distance to guide the search toward the goal.

## Run

Open `sokoban_pathfinding_prototype.ipynb` in Google Colab or Jupyter Notebook and run the cells from top to bottom. The manual controls work in a live notebook environment, while GitHub displays the saved algorithm outputs and benchmark results.

Install the local dependencies with:

```bash
pip install -r requirements.txt
```
