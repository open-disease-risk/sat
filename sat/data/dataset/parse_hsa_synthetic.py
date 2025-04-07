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

"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path

import numpy as np
import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sat.utils import logging
from sat.utils.data import train_val_test

logger = logging.get_default_logger()


def tokens(row, modalities):
    (idx,) = np.where(np.array(modalities) == 0)
    toks = list(row.index[0 : len(modalities)])
    for i in idx:
        toks[i] = row.iloc[i]
    return list(toks)


def numerics(row, modalities):
    (idx,) = np.where(np.array(modalities) == 1)
    numerics = [1.0] * len(modalities)
    for i in idx:
        numerics[i] = row.iloc[i]
    return numerics


@dataclass
class hsa:
    source: str
    processed_dir: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    name: str
    scale_method: str
    scale_numerics: bool = True
    min_scale_numerics: float = 1.0
    kfold: int = 0

    @log_on_start(DEBUG, "Create hsa data representation...")
    @log_on_error(
        ERROR,
        "Error creating hsa data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def prepare(self) -> None:
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

        for index, row in df_features.iterrows():
            df_features.at[index, "x"] = " ".join(tokens(row, modality))

        df_features.loc[:, "numerics"] = ""
        df_features.loc[:, "numerics"] = df_features.loc[:, "numerics"].astype("object")

        for index, row in df_features.iterrows():
            df_features.at[index, "numerics"] = numerics(row, modality)

        df_features.loc[:, "modality"] = ""
        df_features.loc[:, "modality"] = df_features.loc[:, "modality"].astype("object")

        for index, row in df_features.iterrows():
            df_features.at[index, "modality"] = modality

        # 4. create train/val/test split
        logger.debug("Create train/val/test split")
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(
            X=df_features,
            y=df_targets,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            validation_ratio=self.validation_ratio,
        )

        # 5. save data frames
        logger.debug("Save the dataframes")
        logger.debug("Combine features and targets and save into CSV files")

        train_data = pd.DataFrame(
            data={
                "x": X_train["x"],
                "modality": X_train["modality"],
                "numerics": X_train["numerics"],
                "events": y_train["events"],
                "durations": y_train["durations"],
                "split": "train",
            },
            index=X_train.index,
        )

        val_data = pd.DataFrame(
            data={
                "x": X_val["x"],
                "modality": X_val["modality"],
                "numerics": X_val["numerics"],
                "events": y_val["events"],
                "durations": y_val["durations"],
                "split": "valid",
            },
            index=X_val.index,
        )

        test_data = pd.DataFrame(
            data={
                "x": X_test["x"],
                "modality": X_test["modality"],
                "numerics": X_test["numerics"],
                "events": y_test["events"],
                "durations": y_test["durations"],
                "split": "test",
            },
            index=X_test.index,
        )

        if self.kfold > 1:
            logger.debug("Create kfold splits")
            df_train_data = pd.concat([X_train, X_val])
            df_y_train_data = pd.concat([y_train, y_val])

            kf = KFold(n_splits=self.kfold, shuffle=True)
            for i, (train_index, test_index) in enumerate(kf.split(df_train_data)):
                X_train_kf, X_test_kf = (
                    df_train_data.iloc[train_index],
                    df_train_data.iloc[test_index],
                )
                y_train_kf, y_test_kf = (
                    df_y_train_data.iloc[train_index],
                    df_y_train_data.iloc[test_index],
                )

                train_data_kf = pd.DataFrame(
                    data={
                        "x": X_train_kf["x"],
                        "modality": X_train_kf["modality"],
                        "numerics": X_train_kf["numerics"],
                        "events": y_train_kf["events"],
                        "durations": y_train_kf["durations"],
                        "split": "train",
                    },
                    index=X_train_kf.index,
                )

                test_data_kf = pd.DataFrame(
                    data={
                        "x": X_test_kf["x"],
                        "modality": X_test_kf["modality"],
                        "numerics": X_test_kf["numerics"],
                        "events": y_test_kf["events"],
                        "durations": y_test_kf["durations"],
                        "split": "valid",
                    },
                    index=X_test_kf.index,
                )

                outDir = Path(f"{self.processed_dir}/{self.name}")
                outDir.mkdir(parents=True, exist_ok=True)
                data_kf = pd.concat(
                    [train_data_kf, test_data_kf, test_data]
                ).reset_index(level=0)
                data_kf.to_json(
                    Path(f"{outDir}/{i}_{self.name}.json"),
                    orient="records",
                    lines=True,
                )

        outDir = Path(f"{self.processed_dir}/{self.name}")
        outDir.mkdir(parents=True, exist_ok=True)
        data = pd.concat([train_data, val_data, test_data]).reset_index(level=0)
        data.to_json(Path(f"{outDir}/{self.name}.json"), orient="records", lines=True)
