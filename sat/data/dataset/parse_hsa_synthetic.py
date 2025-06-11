"""Process the HSA synthetic data.

Simulated dataset from:
@inproceedings{tjandra2021hierarchical,
title={A hierarchical approach to multi-event survival analysis},
author={Tjandra, Donna and He, Yifei and Wiens, Jenna},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={35},
number={1},
pages={591--599},
year={2021}
}

This script processes the HSA synthetic dataset and saves it in the required format.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path

import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sat.utils import logging

from . import utils

logger = logging.get_default_logger()


# Token and numeric processing functions moved to utils.py


@dataclass
class hsa:
    source: str
    processed_dir: str
    name: str
    scale_method: str
    scale_numerics: bool = True
    min_scale_numerics: float = 1.0

    @log_on_start(DEBUG, "Create hsa data representation...")
    @log_on_error(
        ERROR,
        "Error creating hsa data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
        # 1. read data
        logger.debug("Read data source")
        df = pd.read_csv(self.source, index_col="id")

        numeric_features = [
            "x_1",
            "x_2",
            "x_3",
            "x_4",
            "x_5",
            "x_6",
            "x_7",
            "x_8",
            "x_9",
            "x_10",
            "x_11",
            "x_12",
            "x_13",
            "x_14",
            "x_15",
        ]

        df_features = df[numeric_features]
        df_targets = df[["event1", "event2", "duration1", "duration2"]]
        df_targets.loc[:, "event1"] = df_targets.loc[:, "event1"].astype(int)
        df_targets.loc[:, "event2"] = df_targets.loc[:, "event2"].astype(int)
        df_targets["durations"] = df[["duration1", "duration2"]].values.tolist()
        df_targets["events"] = df[["event1", "event2"]].values.tolist()

        logger.debug(f"features: {df_features.head()}")
        logger.debug(f"Targets: {df_targets.head()}")

        # 2. encode the features
        # differentiate modalities, i.e., token = 0, numerics = 1
        modality = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # min/max scaling of the numeric features
        if self.scale_numerics:
            if self.scale_method == "min_max":
                scaler = MinMaxScaler()
                logger.debug("Perform min/max scaling of the numeric features")
                df_features.loc[:, numeric_features] = (
                    scaler.fit_transform(df_features[numeric_features])
                    + self.min_scale_numerics
                )
            elif self.scale_method == "standard":
                scaler = StandardScaler()
                logger.debug("Perform standard scaling of the numeric features")
                df_features.loc[:, numeric_features] = scaler.fit_transform(
                    df_features[numeric_features]
                )
            else:
                raise ValueError(
                    f"scale_method {self.scale_method} not supported. Use 'min_max' or 'standard'"
                )

        df_features.loc[:, "x"] = ""
        df_features.loc[:, "x"] = df_features.loc[:, "x"].astype("object")

        for index, _ in df_features.iterrows():
            df_features.at[index, "x"] = " ".join(
                utils.tokens(df_features.iloc[index], modality)
            )
        df_features.loc[:, "numerics"] = ""
        df_features.loc[:, "numerics"] = df_features.loc[:, "numerics"].astype("object")

        for index, _ in df_features.iterrows():
            df_features.at[index, "numerics"] = utils.numerics(
                df_features.iloc[index], modality
            )

        # Create the modality column - first create an empty column with object type
        df_features.loc[:, "modality"] = ""
        df_features.loc[:, "modality"] = df_features.loc[:, "modality"].astype("object")

        # Then set the modality value for each row
        for index, _ in df_features.iterrows():
            df_features.at[index, "modality"] = modality

        # 4. Create final dataframe with all data
        logger.debug("Create final dataframe")
        data = pd.DataFrame(
            data={
                "x": df_features["x"],
                "modality": df_features["modality"],
                "numerics": df_features["numerics"],
                "events": df_targets["events"],
                "durations": df_targets["durations"],
            },
            index=df_features.index,
        ).reset_index(level=0)

        # 5. Save to file
        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        data.to_json(Path(f"{out_dir}/{self.name}.json"), orient="records", lines=True)
