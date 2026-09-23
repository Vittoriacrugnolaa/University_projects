# Multimodal Multi-Label News Classification

[![Python](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%20%E2%80%93%202.20-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vittoriacrugnolaa/University_projects/blob/main/Multimodal_News_Classification/notebooks/multimodal_news_classification.ipynb)

A self-contained TensorFlow notebook for multi-label classification of news articles. The original three-input design is retained: raw text, a supplied Bag-of-Words representation, and publication metadata are processed in separate branches and combined before the 18 sigmoid outputs.

The notebook is committed with all cells executed, so its model summary, validation search, training history, test metrics, classification report, confusion matrices, and loss chart can be inspected directly on GitHub.

## Results

All 11,000 available articles are used with a stratified 70/15/15 train, validation, and test split. Preprocessing, early stopping, model selection, and decision thresholds use training or validation data only; the test set is evaluated once at the end.

| Decision rule | Exact-match accuracy | Micro-F1 | Macro-F1 | Weighted-F1 | Hamming accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Threshold 0.50 | **0.507** | 0.682 | 0.644 | 0.677 | **0.926** |
| Validation-tuned thresholds | 0.506 | **0.684** | **0.656** | **0.684** | 0.922 |

Validation-tuned thresholds improve macro-F1 by balancing precision and recall across labels. Both decision rules are reported in the notebook to make the trade-off explicit.

## Model

```text
Text -> tokenization -> embedding -> Bi-LSTM ------------------
                                                               |
BoW -> L1 normalization -> Dense ------------------------------|-> concatenate -> Dense -> 18 sigmoid outputs
                                                               |
Year and month -> scaling and one-hot encoding -> Dense -------
```

The selected network contains approximately 2.67 million trainable parameters. Its configuration is chosen through the compact validation search already included in the notebook, followed by early stopping and learning-rate reduction.

## Data

| Input | Shape | Processing |
| --- | ---: | --- |
| Article text | 11,000 strings | Cleaning, integer encoding, embedding, Bi-LSTM |
| Bag-of-Words | 11,000 × 10,000 | L1 row normalization, Dense layer |
| Year and month | 11,000 × 2 | Standard scaling and one-hot encoding |
| Targets | 11,000 × 18 | Multi-label binary matrix |

The archive is excluded from Git because it contains third-party article text and no redistribution license was supplied. Place the original `input_data.zip` in `data/`, or upload it when the Colab notebook requests it. See [data/README.md](data/README.md) for the expected schema and checksum.

The dataset provides label indices but not human-readable category names, so the evaluation uses `label_0` through `label_17` without inventing a taxonomy.

## Run in Google Colab

Open the notebook with the badge above and run all cells. When requested, upload the original `input_data.zip` archive. A GPU is optional.

## Run locally

```bash
git clone https://github.com/Vittoriacrugnolaa/University_projects.git
cd University_projects/Multimodal_News_Classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place the archive at `data/input_data.zip`, then open [the notebook](notebooks/multimodal_news_classification.ipynb).

## Repository contents

```text
Multimodal_News_Classification/
├── data/README.md
├── notebooks/multimodal_news_classification.ipynb
├── README.md
└── requirements.txt
```

## Author

[Vittoria Crugnola](https://github.com/Vittoriacrugnolaa)
