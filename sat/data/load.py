"""PyTorch Dataset and Dataloader classes"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"


from logging import DEBUG, ERROR

import datasets
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.utils import logging

logger = logging.get_default_logger()


@log_on_start(DEBUG, "Split dataset...", logger=logger)
@log_on_error(
    ERROR,
    "Error splitting dataset: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
def split_dataset(dataDict: DictConfig, dataset: datasets.Dataset, ds_key="train"):
    if dataDict.perform_split:
        if isinstance(dataset, datasets.DatasetDict):
            dataset = dataset[ds_key]

        ds_dict = {}
        for split_name in dataDict.splits:
            logger.debug(f"Filtering out the split for {split_name}")
            # Use a partial function to avoid closure issues with loop variables
            ds_dict[split_name] = dataset.filter(
                lambda x, split=split_name: x[dataDict.split_col] == split
            )

        dataset = datasets.DatasetDict(ds_dict)

    return dataset
