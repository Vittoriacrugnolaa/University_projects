"""Data loading, validation, splitting, and feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import pickle
import zipfile

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler


EXPECTED_ARCHIVE_SHA256 = (
    "fb6c46eb6658d9c71e56dfd614f04b607c23ed81aad39f5f9047df3e28fde075"
)
EXPECTED_KEYS = frozenset({"texts", "years", "months", "X_bow", "labels"})


@dataclass(frozen=True)
class NewsDataset:
    """Validated inputs for multimodal hierarchical news classification."""

    texts: np.ndarray
    years: np.ndarray
    months: np.ndarray
    bow: sparse.csr_matrix
    labels: np.ndarray

    @property
    def metadata(self) -> np.ndarray:
        return np.column_stack((self.years, self.months))


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class PreparedFeatures:
    bow_train: np.ndarray
    bow_validation: np.ndarray
    bow_test: np.ndarray
    metadata_train: np.ndarray
    metadata_validation: np.ndarray
    metadata_test: np.ndarray
    bow_preprocessor: Pipeline
    metadata_preprocessor: ColumnTransformer


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("The dataset payload must be a dictionary.")

    missing = EXPECTED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"Dataset is missing required keys: {sorted(missing)}")

    n_samples = len(payload["texts"])
    for key in ("years", "months", "X_bow", "labels"):
        if len(payload[key]) != n_samples:
            raise ValueError(
                f"Inconsistent sample count for {key!r}: "
                f"expected {n_samples}, found {len(payload[key])}."
            )

    bow = np.asarray(payload["X_bow"])
    labels = np.asarray(payload["labels"])
    months = np.asarray(payload["months"])

    if bow.ndim != 2 or labels.ndim != 2:
        raise ValueError("X_bow and labels must both be two-dimensional arrays.")
    if not np.issubdtype(bow.dtype, np.number):
        raise TypeError("X_bow must contain numeric values.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a binary matrix.")
    if not np.isin(months, np.arange(1, 13)).all():
        raise ValueError("months must contain integers from 1 through 12.")

    return payload


def load_dataset(
    archive_path: str | Path,
    *,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
) -> NewsDataset:
    """Load the pinned dataset archive and validate its schema.

    The source is a pickle file, so its exact archive hash is checked before
    deserialization. Do not disable this check for data from an untrusted source.
    """

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    actual_hash = file_sha256(archive_path)
    if expected_sha256 and actual_hash != expected_sha256:
        raise ValueError(
            "Dataset checksum mismatch. Refusing to deserialize an unexpected "
            f"pickle archive. Expected {expected_sha256}, found {actual_hash}."
        )

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if members != ["input_data.pkl"]:
            raise ValueError(
                "Expected the archive to contain only input_data.pkl; "
                f"found {members}."
            )
        with archive.open("input_data.pkl") as handle:
            payload = _validate_payload(pickle.load(handle))

    dataset = NewsDataset(
        # Object dtype keeps variable-length Python strings. A fixed-width NumPy
        # Unicode dtype would allocate every row at the length of the longest
        # article and is rejected by Keras 3 model inputs.
        texts=np.asarray(payload["texts"], dtype=object),
        years=np.asarray(payload["years"], dtype=np.int32),
        months=np.asarray(payload["months"], dtype=np.int8),
        bow=sparse.csr_matrix(payload["X_bow"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.float32),
    )
    return dataset


def split_dataset(
    labels: np.ndarray,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 213,
) -> SplitIndices:
    """Create deterministic iterative-stratified train/validation/test splits."""

    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be smaller than 1.")

    all_indices = np.arange(len(labels))
    holdout_size = validation_size + test_size
    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=random_state,
    )
    train_relative, holdout_relative = next(first_split.split(all_indices, labels))
    train_indices = all_indices[train_relative]
    holdout_indices = all_indices[holdout_relative]

    relative_test_size = test_size / holdout_size
    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=random_state + 1,
    )
    validation_relative, test_relative = next(
        second_split.split(holdout_indices, labels[holdout_indices])
    )

    return SplitIndices(
        train=train_indices,
        validation=holdout_indices[validation_relative],
        test=holdout_indices[test_relative],
    )


def prepare_features(
    dataset: NewsDataset,
    splits: SplitIndices,
    *,
    bow_components: int = 256,
    random_state: int = 213,
) -> PreparedFeatures:
    """Fit all numerical preprocessing on training data only."""

    max_components = min(
        dataset.bow.shape[1] - 1,
        len(splits.train) - 1,
    )
    if not 1 <= bow_components <= max_components:
        raise ValueError(
            f"bow_components must be between 1 and {max_components}; "
            f"received {bow_components}."
        )

    bow_preprocessor = Pipeline(
        steps=[
            ("normalize", Normalizer(norm="l1", copy=True)),
            (
                "reduce",
                TruncatedSVD(
                    n_components=bow_components,
                    algorithm="randomized",
                    n_iter=7,
                    random_state=random_state,
                ),
            ),
        ]
    )

    bow_train = bow_preprocessor.fit_transform(dataset.bow[splits.train])
    bow_validation = bow_preprocessor.transform(dataset.bow[splits.validation])
    bow_test = bow_preprocessor.transform(dataset.bow[splits.test])

    month_encoder = OneHotEncoder(
        categories=[np.arange(1, 13)],
        handle_unknown="ignore",
        sparse_output=False,
    )
    metadata_preprocessor = ColumnTransformer(
        transformers=[
            ("year", StandardScaler(), [0]),
            ("month", month_encoder, [1]),
        ],
        verbose_feature_names_out=False,
    )

    metadata = dataset.metadata
    metadata_train = metadata_preprocessor.fit_transform(metadata[splits.train])
    metadata_validation = metadata_preprocessor.transform(
        metadata[splits.validation]
    )
    metadata_test = metadata_preprocessor.transform(metadata[splits.test])

    return PreparedFeatures(
        bow_train=np.asarray(bow_train, dtype=np.float32),
        bow_validation=np.asarray(bow_validation, dtype=np.float32),
        bow_test=np.asarray(bow_test, dtype=np.float32),
        metadata_train=np.asarray(metadata_train, dtype=np.float32),
        metadata_validation=np.asarray(metadata_validation, dtype=np.float32),
        metadata_test=np.asarray(metadata_test, dtype=np.float32),
        bow_preprocessor=bow_preprocessor,
        metadata_preprocessor=metadata_preprocessor,
    )
