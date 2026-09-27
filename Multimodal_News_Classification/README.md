# Multimodal Multi-Label News Classification

A TensorFlow notebook for assigning multiple subject labels to news articles.

Each article is represented by three inputs: raw text, a 10,000-dimensional Bag-of-Words vector, and publication metadata. The model processes them in separate branches and combines the three representations before producing 18 independent label probabilities.

[Open the notebook in Google Colab](https://colab.research.google.com/github/Vittoriacrugnolaa/University_projects/blob/main/Multimodal_News_Classification/notebooks/multimodal_news_classification.ipynb)

## Results

The dataset contains 11,000 articles and is divided into stratified training, validation, and test sets (70/15/15). The test set is used only for the final evaluation.

| Decision rule | Exact-match accuracy | Micro-F1 | Macro-F1 | Weighted-F1 | Hamming accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Threshold 0.50 | **0.507** | 0.682 | 0.644 | 0.677 | **0.926** |
| Thresholds selected on validation data | 0.506 | **0.684** | **0.656** | **0.684** | 0.922 |

Test binary cross-entropy: **0.246**.

The per-label thresholds improve recall and macro-F1 with a small reduction in Hamming accuracy. The training history also shows some overfitting after the best validation epoch, which is handled with early stopping and restoration of the best weights.

## Model

```text
Raw text -> tokenization -> embedding -> Bi-LSTM ----------------
                                                                 |
Bag-of-Words -> L1 normalization -> Dense -----------------------|-> Concatenate -> Dense -> 18 sigmoid outputs
                                                                 |
Year and month -> scaling and one-hot encoding -> Dense ----------
```

The selected model has about 2.67 million trainable parameters. A small validation search compares three configurations before the final training run.

## Data

| Input | Shape | Processing |
| --- | ---: | --- |
| Article text | 11,000 strings | Cleaning, tokenization, embedding, Bi-LSTM |
| Bag-of-Words | 11,000 × 10,000 | L1 normalization, Dense layer |
| Year and month | 11,000 × 2 | Standard scaling and one-hot encoding |
| Targets | 11,000 × 18 | Multi-label binary matrix |

The dataset is not included because the archive contains third-party article text and no redistribution license was provided. To reproduce the notebook, place the original `input_data.zip` in `data/` or upload it when prompted in Colab. The expected schema and checksum are listed in [data/README.md](data/README.md).

The dataset contains label indices but no category names, so the notebook reports `label_0` through `label_17`.

## Run locally

```bash
git clone https://github.com/Vittoriacrugnolaa/University_projects.git
cd University_projects/Multimodal_News_Classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place the archive at `data/input_data.zip`, then open [the notebook](notebooks/multimodal_news_classification.ipynb). It is committed with the training output, metrics, classification report, confusion matrices, and loss curve already visible.
