"""An efficient dataset splitter for large data."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hashlib
from typing import Any, Dict, Tuple

import hydra
from datasets import Dataset, DatasetDict, IterableDataset
from omegaconf import DictConfig


class StreamingKFoldSplitter:
    """
    A utility for performing k-fold cross-validation and/or single train/val/test splits
    on Hugging Face datasets in streaming mode. Supports hash-based and pre-defined splits.

    Parameters:
    -----------
    id_field : str
        The name of the field to use as a stable, unique identifier for hashing (e.g., "id", "text").
    k : int or None, default=None
        The number of folds for k-fold cross-validation. Set to None for a single val split.
    val_ratio : float, default=0.2
        Only used if `k` is None. Specifies the percentage of train+val to use for validation.
    test_ratio : float, default=0.1
        Used when test_split_strategy is "hash". Proportion of data to assign to test set.
    test_split_strategy : str, default="hash"
        Either "hash" (generate test set by hashing) or "existing" (use dataset's built-in test split).
    split_names : tuple(str), default=("train", "val", "test")
        Tuple of split names to use when combining for k-fold, if using "existing" test split.

    Methods:
    --------
    load_streaming_split(fold_index=None)
        Returns (train_dataset, val_dataset, test_dataset) for the given fold index.

    Example Usage:
    --------------
    # 1. Hash-based test split and single val fold
    splitter = StreamingKFoldSplitter("ag_news", id_field="text", k=None, val_ratio=0.2)
    dataset = splitter.load_split()

    # 2. k-fold CV using hash-based test split
    splitter = StreamingKFoldSplitter("ag_news", id_field="text", k=5)
    dataset = splitter.load_split(fold_index=0)

    # 3. Reuse built-in test split and do CV on train+val
    splitter = StreamingKFoldSplitter(
        "glue", id_field="sentence1", k=5,
        test_split_strategy="existing",
        split_names=("train", "val", "test")
    )
    dataset = splitter.load_split(fold_index=3)
    """

    def __init__(
        self,
        id_field: str = "id",
        k: int = None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        test_split_strategy: str = "hash",
        split_names: Tuple[str] = ("train", "val", "test"),
    ):
        self.id_field = id_field
        self.k = k
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.test_split_strategy = test_split_strategy
        self.split_names = split_names

    def _normalize_hash(self, s: str) -> float:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h, 16) / 2**128

    def _fold(self, s: str) -> int:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h, 16) % self.k

    def _get_fold(self, example: Dict[str, Any]) -> int:
        """Get the fold assignment for a single example."""
        if self.k is None:
            return (
                0
                if self._normalize_hash(str(example[self.id_field]))
                < (1 - self.val_ratio)
                else 1
            )
        return self._fold(str(example[self.id_field]))

    def _is_in_fold(self, example: Dict[str, Any], fold_index: int) -> bool:
        """Check if an example should be in the specified fold."""
        return self._get_fold(example) == fold_index

    def _split_by_folds(
        self, examples: Dataset, fold_index: int
    ) -> Tuple[Dataset, Dataset]:
        """Split examples into train/val based on fold assignments using Dataset.filter."""
        train_data = examples.filter(lambda ex: not self._is_in_fold(ex, fold_index))
        val_data = examples.filter(lambda ex: self._is_in_fold(ex, fold_index))
        return train_data, val_data

    def _concat_datasets(self, dataset1: Dataset, dataset2: Dataset) -> Dataset:
        """Concatenate two datasets."""
        combined = [ex for ex in dataset1] + [ex for ex in dataset2]
        return Dataset.from_list(combined)

    def load_split(self, cfg: DictConfig = None, fold_index: int = None) -> DatasetDict:
        """
        Returns train, val, test datasets based on configuration, supporting both streaming and in-memory loading.

        Parameters:
        -----------
        cfg: DictConfig or None
            Configuration object containing the dataset loading parameters
        fold_index : int or None
            If k-fold CV is being used, specify which fold to use as validation.
            If None and k is None, performs a one-off train/val split.

        Returns:
        --------
        DatasetDict containing train, val, test datasets
        """
        if cfg is None:
            raise ValueError("cfg must be provided")

        streaming = cfg.get("streaming", False)
        full_dataset = hydra.utils.call(cfg)

        # Schema check: ensure id_field exists in dataset columns
        if isinstance(full_dataset, DatasetDict):
            # Check if id_field exists in all splits
            for split_name, split_data in full_dataset.items():
                if self.id_field not in split_data.column_names:
                    raise ValueError(
                        f"Required id_field '{self.id_field}' not found in {split_name} split columns: {split_data.column_names}"
                    )
        else:
            if self.id_field not in full_dataset.column_names:
                raise ValueError(
                    f"Required id_field '{self.id_field}' not found in dataset columns: {full_dataset.column_names}"
                )

        # 1. Create initial train/val/test split
        if self.test_split_strategy == "existing":
            test_dataset = full_dataset[self.split_names[-1]]
            train_split = full_dataset[self.split_names[0]]
            val_split = full_dataset[self.split_names[1]]
            if streaming:
                # This creates a new IterableDataset that chains the two datasets
                def chain_iterables():
                    for example in train_split:
                        yield example
                    for example in val_split:
                        yield example

                train_val_dataset = IterableDataset.from_generator(chain_iterables)
            else:
                train_val_dataset = self._concat_datasets(train_split, val_split)
        else:

            def is_test(example: Dict[str, Any]) -> bool:
                return (
                    self._normalize_hash(str(example[self.id_field])) < self.test_ratio
                )

            test_dataset = full_dataset["train"].filter(is_test)
            train_val_dataset = full_dataset["train"].filter(lambda x: not is_test(x))

        # 2. Compute folds (independent of streaming/in-memory and split strategy)
        def is_val(example: Dict[str, Any]) -> bool:
            return self._is_in_fold(example, fold_index or 0)

        def is_train(example: Dict[str, Any]) -> bool:
            return not self._is_in_fold(example, fold_index or 0)

        # 3. Apply fold assignment logic
        if self.test_split_strategy == "existing" and self.k is None:
            # Simple case: no k-fold, use existing splits as-is
            train_dataset = train_split
            val_dataset = val_split
        else:
            # K-fold case: split train_val_dataset according to folds
            train_dataset = train_val_dataset.filter(is_train)
            val_dataset = train_val_dataset.filter(is_val)

        result = {"train": train_dataset, "test": test_dataset}
        if val_dataset is not None:
            result["valid"] = val_dataset
        return DatasetDict(result)
