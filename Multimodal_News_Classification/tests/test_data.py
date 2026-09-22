import pickle
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from news_classifier.data import file_sha256, load_dataset, split_dataset


def test_load_dataset_and_stratified_split(tmp_path):
    sample_count = 40
    labels = np.tile(
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int64),
        (sample_count // 2, 1),
    )
    payload = {
        "texts": [f"sample {index}" for index in range(sample_count)],
        "years": [2020] * sample_count,
        "months": [1 + index % 12 for index in range(sample_count)],
        "X_bow": np.eye(sample_count, 10, dtype=np.int64),
        "labels": labels,
    }

    pickle_path = tmp_path / "input_data.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle)

    archive_path = tmp_path / "input_data.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(pickle_path, arcname="input_data.pkl")

    dataset = load_dataset(archive_path, expected_sha256=file_sha256(archive_path))
    splits = split_dataset(dataset.labels, random_state=7)

    assert dataset.bow.shape == (sample_count, 10)
    assert len(splits.train) == 28
    assert len(splits.validation) == 6
    assert len(splits.test) == 6
    assert set(splits.train).isdisjoint(splits.validation)
    assert set(splits.train).isdisjoint(splits.test)
    assert set(splits.validation).isdisjoint(splits.test)
