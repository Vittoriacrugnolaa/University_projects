"""TensorFlow model definition for the three-branch classifier."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_global_determinism(seed: int = 213) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def build_text_vectorizer(
    training_texts: np.ndarray,
    *,
    max_tokens: int = 20_000,
    sequence_length: int = 80,
) -> keras.layers.TextVectorization:
    """Adapt a serializable text vectorizer using training text only."""

    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
        output_sequence_length=sequence_length,
        name="text_vectorization",
    )
    vectorizer.adapt(np.asarray(training_texts, dtype=object))
    return vectorizer


def build_model(
    *,
    text_vectorizer: keras.layers.TextVectorization,
    bow_dimension: int,
    metadata_dimension: int,
    number_of_labels: int,
    embedding_dimension: int = 64,
    recurrent_units: int = 64,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the multimodal multi-label neural network."""

    text_input = keras.Input(shape=(), dtype=tf.string, name="text")
    token_ids = text_vectorizer(text_input)
    text_features = keras.layers.Embedding(
        input_dim=len(text_vectorizer.get_vocabulary()),
        output_dim=embedding_dimension,
        mask_zero=True,
        name="embedding",
    )(token_ids)
    text_features = keras.layers.Bidirectional(
        keras.layers.LSTM(recurrent_units),
        name="bidirectional_lstm",
    )(text_features)
    text_features = keras.layers.Dropout(dropout_rate, name="text_dropout")(
        text_features
    )

    bow_input = keras.Input(shape=(bow_dimension,), dtype=tf.float32, name="bow")
    bow_features = keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        name="bow_dense",
    )(bow_input)
    bow_features = keras.layers.Dropout(dropout_rate, name="bow_dropout")(
        bow_features
    )

    metadata_input = keras.Input(
        shape=(metadata_dimension,),
        dtype=tf.float32,
        name="metadata",
    )
    metadata_features = keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer="he_normal",
        name="metadata_dense",
    )(metadata_input)

    features = keras.layers.Concatenate(name="feature_fusion")(
        [text_features, bow_features, metadata_features]
    )
    features = keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_dense",
    )(features)
    features = keras.layers.Dropout(dropout_rate, name="shared_dropout")(features)
    output = keras.layers.Dense(
        number_of_labels,
        activation="sigmoid",
        name="label_probabilities",
    )(features)

    model = keras.Model(
        inputs={"text": text_input, "bow": bow_input, "metadata": metadata_input},
        outputs=output,
        name="multimodal_news_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(
                curve="PR",
                multi_label=True,
                num_labels=number_of_labels,
                name="macro_pr_auc",
            ),
        ],
    )
    return model


def training_callbacks(patience: int = 4) -> list[keras.callbacks.Callback]:
    """Callbacks used for stable, validation-driven training."""

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
        ),
    ]
