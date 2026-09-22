import numpy as np

from news_classifier.evaluation import (
    decode_to_valid_patterns,
    multilabel_metrics,
    tune_per_label_thresholds,
)


def test_valid_pattern_decoder_never_creates_unknown_combination():
    patterns = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int8)
    probabilities = np.array([[0.9, 0.4, 0.4], [0.2, 0.8, 0.7]])

    decoded = decode_to_valid_patterns(probabilities, patterns)

    assert np.array_equal(decoded, patterns)


def test_threshold_tuning_and_metrics():
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    probabilities = np.array([[0.4, 0.1], [0.45, 0.2], [0.2, 0.6], [0.1, 0.7]])

    thresholds = tune_per_label_thresholds(y_true, probabilities)
    predictions = (probabilities >= thresholds).astype(np.int8)
    metrics = multilabel_metrics(y_true, predictions)

    assert metrics["subset_accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
