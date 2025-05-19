from unittest.mock import patch

import hypothesis.strategies as st
import pytest
from datasets import Dataset, DatasetDict
from hypothesis import given, settings

from sat.data.splitter import StreamingKFoldSplitter


def make_synthetic_dataset(num_items=100, id_field="id"):
    return Dataset.from_dict(
        {
            id_field: [str(i) for i in range(num_items)],
            "feature": [i * 2 for i in range(num_items)],
        }
    )


def make_existing_split_dict(num_items=100, id_field="id"):
    # 60% train, 20% val, 20% test
    n_train = int(num_items * 0.6)
    n_val = int(num_items * 0.2)

    # Create non-overlapping IDs for each split
    train_data = {
        id_field: [str(i) for i in range(0, n_train)],
        "feature": [i * 2 for i in range(0, n_train)],
    }
    val_data = {
        id_field: [str(i) for i in range(n_train, n_train + n_val)],
        "feature": [i * 2 for i in range(n_train, n_train + n_val)],
    }
    test_data = {
        id_field: [str(i) for i in range(n_train + n_val, num_items)],
        "feature": [i * 2 for i in range(n_train + n_val, num_items)],
    }

    train = Dataset.from_dict(train_data)
    val = Dataset.from_dict(val_data)
    test = Dataset.from_dict(test_data)
    return DatasetDict({"train": train, "val": val, "test": test})


class DummyConfig(dict):
    def get(self, k, default=None):
        return self[k] if k in self else default


def assert_disjoint_and_complete(result, num_items, id_field="id"):
    train_ids = set(result["train"][id_field])
    val_ids = set(result["val"][id_field])
    test_ids = set(result["test"][id_field])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    all_ids = train_ids | val_ids | test_ids
    assert all_ids == set(str(i) for i in range(num_items))


@given(num_items=st.integers(min_value=20, max_value=100))
@settings(max_examples=10, deadline=None)
def test_hash_split_property(num_items):
    cfg = DummyConfig(dataset_name="synthetic", streaming=False)
    with patch("hydra.utils.call", return_value=make_synthetic_dataset(num_items)):
        splitter = StreamingKFoldSplitter(
            "synthetic", id_field="id", k=None, val_ratio=0.2
        )
        result = splitter.load_split(cfg)
        assert_disjoint_and_complete(result, num_items)


@given(
    num_items=st.integers(min_value=20, max_value=100),
    k=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=10, deadline=None)
def test_hash_split_kfold_property(num_items, k):
    if k >= num_items:
        return  # Skip degenerate folds
    cfg = DummyConfig(dataset_name="synthetic", streaming=False)
    with patch("hydra.utils.call", return_value=make_synthetic_dataset(num_items)):
        splitter = StreamingKFoldSplitter(
            "synthetic", id_field="id", k=k, val_ratio=0.2
        )
        for fold in range(k):
            result = splitter.load_split(cfg, fold_index=fold)
            train_ids = set(result["train"]["id"])
            val_ids = set(result["val"]["id"])
            test_ids = set(result["test"]["id"])
            assert train_ids.isdisjoint(val_ids)
            assert train_ids.isdisjoint(test_ids)
            assert val_ids.isdisjoint(test_ids)
            assert len(val_ids) > 0
            assert len(train_ids) > 0
            assert len(test_ids) > 0
        # Determinism: should get same split for same fold
        for fold in range(k):
            result1 = splitter.load_split(cfg, fold_index=fold)
            result2 = splitter.load_split(cfg, fold_index=fold)
            assert set(result1["val"]["id"]) == set(result2["val"]["id"])


@given(num_items=st.integers(min_value=20, max_value=100))
@settings(max_examples=10, deadline=None)
def test_existing_split_property(num_items):
    cfg = DummyConfig(dataset_name="synthetic", streaming=False)
    with patch("hydra.utils.call", return_value=make_existing_split_dict(num_items)):
        splitter = StreamingKFoldSplitter(
            "synthetic",
            id_field="id",
            k=None,
            test_split_strategy="existing",
            split_names=("train", "val", "test"),
        )
        result = splitter.load_split(cfg)
        assert_disjoint_and_complete(result, num_items)


@given(k=st.integers(min_value=2, max_value=10))
@settings(max_examples=10, deadline=None)
def test_existing_split_kfold_property(k):
    cfg = DummyConfig(dataset_name="synthetic", streaming=False)
    with patch("hydra.utils.call", return_value=make_existing_split_dict(100)):
        splitter = StreamingKFoldSplitter(
            "synthetic",
            id_field="id",
            k=k,
            test_split_strategy="existing",
            split_names=("train", "val", "test"),
        )
        for fold in range(k):
            result = splitter.load_split(cfg, fold_index=fold)
            train_ids = set(result["train"]["id"])
            val_ids = set(result["val"]["id"])
            test_ids = set(result["test"]["id"])
            assert train_ids.isdisjoint(val_ids)
            assert train_ids.isdisjoint(test_ids)
            assert val_ids.isdisjoint(test_ids)
            assert len(val_ids) > 0
            assert len(train_ids) > 0
            assert len(test_ids) > 0


def test_missing_id_field():
    cfg = DummyConfig(dataset_name="synthetic", streaming=False)
    # Dataset missing id_field
    ds = Dataset.from_dict({"feature": [1, 2, 3]})
    with patch("hydra.utils.call", return_value=ds):
        splitter = StreamingKFoldSplitter("synthetic", id_field="id", k=2)
        with pytest.raises(ValueError, match="Required id_field 'id' not found"):
            splitter.load_split(cfg, fold_index=0)


@pytest.mark.parametrize("streaming", [False, True])
def test_hash_split_basic(streaming):
    cfg = DummyConfig(dataset_name="synthetic", streaming=streaming)
    # Patch hydra.utils.call to return our synthetic dataset
    with patch("hydra.utils.call", return_value=make_synthetic_dataset(100)):
        splitter = StreamingKFoldSplitter(
            "synthetic", id_field="id", k=None, val_ratio=0.2
        )
        result = splitter.load_split(cfg)
        assert set(result.keys()) == {"train", "val", "test"}
        # Check no overlap between splits
        train_ids = set(result["train"]["id"])
        val_ids = set(result["val"]["id"])
        test_ids = set(result["test"]["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        # Check all ids are present
        all_ids = train_ids | val_ids | test_ids
        assert all_ids == set(str(i) for i in range(100))


@pytest.mark.parametrize("streaming", [False, True])
def test_hash_split_kfold(streaming):
    cfg = DummyConfig(dataset_name="synthetic", streaming=streaming)
    with patch("hydra.utils.call", return_value=make_synthetic_dataset(100)):
        splitter = StreamingKFoldSplitter(
            "synthetic", id_field="id", k=3, val_ratio=0.2
        )
        for fold in range(3):
            result = splitter.load_split(cfg, fold_index=fold)
            if streaming:
                train_ids = set([ex["id"] for ex in result["train"]])
                val_ids = set([ex["id"] for ex in result["val"]])
                test_ids = set([ex["id"] for ex in result["test"]])
            else:
                train_ids = set(result["train"]["id"])
                val_ids = set(result["val"]["id"])
                test_ids = set(result["test"]["id"])
            assert train_ids.isdisjoint(val_ids)
            assert train_ids.isdisjoint(test_ids)
            assert val_ids.isdisjoint(test_ids)
            assert len(val_ids) > 0
            assert len(train_ids) > 0
            assert len(test_ids) > 0
        # Determinism: running again gives same split
        result1 = splitter.load_split(cfg, fold_index=0)
        result2 = splitter.load_split(cfg, fold_index=0)
        if streaming:
            result1_val_ids = set([ex["id"] for ex in result1["val"]])
            result2_val_ids = set([ex["id"] for ex in result2["val"]])
        else:
            result1_val_ids = set(result1["val"]["id"])
            result2_val_ids = set(result2["val"]["id"])
        assert result2_val_ids == result1_val_ids


@pytest.mark.parametrize("streaming", [False, True])
def test_existing_split(streaming):
    cfg = DummyConfig(dataset_name="synthetic", streaming=streaming)
    with patch("hydra.utils.call", return_value=make_existing_split_dict(100)):
        splitter = StreamingKFoldSplitter(
            "synthetic",
            id_field="id",
            k=None,
            test_split_strategy="existing",
            split_names=("train", "val", "test"),
        )
        result = splitter.load_split(cfg)
        assert set(result.keys()) == {"train", "val", "test"}
        # Check no overlap between splits
        if streaming:
            train_ids = set([ex["id"] for ex in result["train"]])
            val_ids = set([ex["id"] for ex in result["val"]])
            test_ids = set([ex["id"] for ex in result["test"]])
        else:
            train_ids = set(result["train"]["id"])
            val_ids = set(result["val"]["id"])
            test_ids = set(result["test"]["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        all_ids = train_ids | val_ids | test_ids
        assert all_ids == set(str(i) for i in range(100))


@pytest.mark.parametrize("streaming", [False, True])
def test_existing_split_kfold(streaming):
    cfg = DummyConfig(dataset_name="synthetic", streaming=streaming)
    with patch("hydra.utils.call", return_value=make_existing_split_dict(100)):
        splitter = StreamingKFoldSplitter(
            "synthetic",
            id_field="id",
            k=2,
            test_split_strategy="existing",
            split_names=("train", "val", "test"),
        )
        for fold in range(2):
            result = splitter.load_split(cfg, fold_index=fold)
            if streaming:
                # For streaming datasets, we need to iterate through them to collect IDs
                train_ids = set([ex["id"] for ex in result["train"]])
                val_ids = set([ex["id"] for ex in result["val"]])
                test_ids = set([ex["id"] for ex in result["test"]])
            else:
                train_ids = set(result["train"]["id"])
                val_ids = set(result["val"]["id"])
                test_ids = set(result["test"]["id"])
            assert train_ids.isdisjoint(val_ids)
            assert train_ids.isdisjoint(test_ids)
            assert val_ids.isdisjoint(test_ids)
            assert len(val_ids) > 0
            assert len(train_ids) > 0
            assert len(test_ids) > 0


@pytest.mark.parametrize("streaming", [False, True])
def test_edge_cases(streaming):
    # Empty dataset
    cfg = DummyConfig(dataset_name="synthetic", streaming=streaming)
    with patch("hydra.utils.call", return_value=make_synthetic_dataset(0)):
        splitter = StreamingKFoldSplitter("synthetic", id_field="id", k=2)
        result = splitter.load_split(cfg, fold_index=0)
        assert len(result["train"]) == 0
        assert len(result["val"]) == 0
        assert len(result["test"]) == 0
    # Single item
    with patch("hydra.utils.call", return_value=make_synthetic_dataset(1)):
        splitter = StreamingKFoldSplitter("synthetic", id_field="id", k=2)
        result = splitter.load_split(cfg, fold_index=0)
        assert len(result["train"]) + len(result["val"]) + len(result["test"]) == 1
