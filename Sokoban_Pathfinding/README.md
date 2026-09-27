# Sokoban Pathfinding Prototype

This notebook covers the pathfinding part of a Sokoban-style game. The grid contains a player, walls, a starting position, and a goal. Box-pushing is not implemented, so this should be considered a search and movement prototype rather than a complete Sokoban game.

## What is included

- nine manually defined levels, from 2 × 2 to 10 × 10;
- a random grid generator;
- manual controls with Jupyter widgets;
- Random Search, Depth-First Search (DFS), Breadth-First Search (BFS), and A*;
- a benchmark that runs DFS, BFS, and A* on every predefined level.

## Results

DFS, BFS, and A* solve all nine levels. BFS and A* return the same shortest path length on every level. DFS finds a longer route on six of the nine levels, which is expected because it explores depth before path length.

A* uses Manhattan distance as its heuristic. On this grid, the heuristic is admissible because movement is limited to four directions and every move has the same cost.

## Run

Open `sokoban_pathfinding_prototype.ipynb` in Google Colab or Jupyter Notebook and run the cells from top to bottom. GitHub displays the saved paths and benchmark output, while the manual controls require a live notebook session.

```bash
pip install -r requirements.txt
```
