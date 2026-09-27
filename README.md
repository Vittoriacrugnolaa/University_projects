# University Projects

This repository contains three university projects covering deep learning, classical machine learning, and search algorithms. Each folder includes the notebook or report together with the information needed to understand and reproduce the work.

## Multimodal News Classification

A multi-input TensorFlow model for assigning several subject labels to a news article. It combines raw text, Bag-of-Words features, and publication metadata in three separate branches.

- **Data:** 11,000 articles and 18 labels
- **Model:** Embedding and Bi-LSTM for text, Dense layers for Bag-of-Words and metadata
- **Test results:** micro-F1 0.684 and macro-F1 0.656 with validation-based thresholds

[Project details](Multimodal_News_Classification/) · [Notebook](Multimodal_News_Classification/notebooks/multimodal_news_classification.ipynb)

## Student Persistence and Dropout

A classification study focused on identifying students at risk of dropping out. The work covers data exploration, feature preparation, class imbalance, model comparison, nested cross-validation, and model interpretation.

- **Selected model:** L1-regularized logistic regression with random oversampling
- **Test result:** F1-score 0.807
- **Focus:** an interpretable pipeline for understanding the main factors associated with dropout

[Project details](Student_Dropout/) · [Notebook](Student_Dropout/Student_Persistence_Dropout.ipynb) · [Report](Student_Dropout/Student_Dropout_essay.pdf)

## Sokoban Pathfinding Prototype

A Python notebook that tests movement and search on Sokoban-style grids with static walls. It compares Random Search, DFS, BFS, and A* across nine levels.

- **Result:** BFS and A* find the same shortest path length on every predefined level
- **Scope:** player navigation and pathfinding; box-pushing is not implemented

[Project details](Sokoban_Pathfinding/) · [Notebook](Sokoban_Pathfinding/sokoban_pathfinding_prototype.ipynb)
