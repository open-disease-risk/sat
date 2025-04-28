"""An efficient dataset splitter for large data."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hashlib
from itertools import chain

from datasets import load_dataset


class StreamingKFoldSplitter:
    """
    A utility for performing k-fold cross-validation and/or single train/val/test splits
    on Hugging Face datasets in streaming mode. Supports hash-based and pre-defined splits.

    Parameters:
    -----------
    dataset_name : str
        The Hugging Face dataset identifier (e.g., "ag_news", "glue", or a local path).
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
    split_names : tuple(str), default=("train", "validation", "test")
        Tuple of split names to use when combining for k-fold, if using "existing" test split.

    Methods:
    --------
    load_streaming_split(fold_index=None)
        Returns (train_dataset, val_dataset, test_dataset) for the given fold index.

    Example Usage:
    --------------
    # 1. Hash-based test split and single val fold
    splitter = StreamingKFoldSplitter("ag_news", id_field="text", k=None, val_ratio=0.2)
    train, val, test = splitter.load_streaming_split()

    # 2. k-fold CV using hash-based test split
    splitter = StreamingKFoldSplitter("ag_news", id_field="text", k=5)
    train, val, test = splitter.load_streaming_split(fold_index=0)

    # 3. Reuse built-in test split and do CV on train+val
    splitter = StreamingKFoldSplitter(
        "glue", id_field="sentence1", k=5,
        test_split_strategy="existing",
        split_names=("train", "validation", "test")
    )
    train, val, test = splitter.load_streaming_split(fold_index=3)
    """

    def __init__(
        self,
        dataset_name,
        id_field="id",
        k=None,
        val_ratio=0.2,
        test_ratio=0.1,
        test_split_strategy="hash",
        split_names=("train", "validation", "test"),
    ):
        self.dataset_name = dataset_name
        self.id_field = id_field
        self.k = k
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.test_split_strategy = test_split_strategy
        self.split_names = split_names

    def _normalize_hash(self, s):
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h, 16) / 2**128

    def _fold(self, s):
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h, 16) % self.k

    def load_streaming_split(self, fold_index=None):
        """
        Returns train, val, test datasets as streaming filters based on configuration.

        Parameters:
        -----------
        fold_index : int or None
            If k-fold CV is being used, specify which fold to use as validation.
            If None and k is None, performs a one-off train/val split.

        Returns:
        --------
        (train_dataset, val_dataset, test_dataset) : tuple of streaming datasets
        """
        # Load dataset splits
        if self.test_split_strategy == "existing":
            test_dataset = load_dataset(
                self.dataset_name, split=self.split_names[-1], streaming=True
            )
            train_val_streams = [
                load_dataset(self.dataset_name, split=split, streaming=True)
                for split in self.split_names[:-1]
            ]
            train_val_dataset = chain(*train_val_streams)
        else:
            full_dataset = load_dataset(
                self.dataset_name, split="train", streaming=True
            )

            def is_test(example):
                return (
                    self._normalize_hash(str(example[self.id_field])) < self.test_ratio
                )

            def is_trainval(example):
                return not is_test(example)

            test_dataset = full_dataset.filter(is_test)
            train_val_dataset = full_dataset.filter(is_trainval)

        # Split train_val into train/val depending on whether we're using k-fold or ratio
        def is_val(example):
            h = self._normalize_hash(str(example[self.id_field]))
            if self.k is None:
                return h >= (1 - self.val_ratio)
            return self._fold(str(example[self.id_field])) == fold_index

        def is_train(example):
            h = self._normalize_hash(str(example[self.id_field]))
            if self.k is None:
                return h < (1 - self.val_ratio)
            return self._fold(str(example[self.id_field])) != fold_index

        train_dataset = train_val_dataset.filter(is_train)
        val_dataset = (
            train_val_dataset.filter(is_val)
            if (fold_index is not None or self.k is None)
            else None
        )

        return train_dataset, val_dataset, test_dataset
