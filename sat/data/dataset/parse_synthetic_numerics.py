"""Process the synthetic dataset."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import numpy as np
import pandas as pd

from dataclasses import dataclass
from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold

from sat.utils import logging

from sat.utils.data import train_val_test


logger = logging.get_default_logger()


def tokens(row, modalities, offset):
    (idx,) = np.where(np.array(modalities) == 0)
    toks = list(row.index[:-1])
    for i in idx:
        toks[i] = row.iloc[i + offset]
    return list(toks)


def numerics(row, modalities, offset):
    (idx,) = np.where(np.array(modalities) == 1)
    numerics = [1.0] * len(modalities)
    for i in idx:
        numerics[i] = row.iloc[i + offset]
    return numerics


@dataclass
class synthetic:
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

    @log_on_start(DEBUG, "Create metabric data representation...")
    @log_on_error(
        ERROR,
        "Error creating synthetic data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def prepare(self) -> None:
        # 1. combine the test and training sets from H5 sources
        logger.debug("Read original synthetic data")
        df = pd.read_csv(self.source)
        df_features = df.iloc[:, 4:]
        df_targets = df.iloc[:, :2]

        df_features["index"] = range(len(df_features))
        df_targets["index"] = range(len(df_targets))
        df_features.set_index("index", inplace=True)
        df_targets.set_index("index", inplace=True)
        logger.debug(f"features: {df_features.head()}")
        logger.debug(f"Targets: {df_targets.head()}")

        # 2. encode the features
        # differentiate modalities, i.e., token = 0, numerics = 1
        modality = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # create tokens for a transformer model
        logger.debug("Prepend column name to the features turning them into tokens")
        logger.debug(f"Original feature columns: {df_features.columns}")

        # min/max scaling of the numeric features
        if self.scale_numerics:
            if self.scale_method == "min_max":
                scaler = MinMaxScaler()
                logger.debug("Perform min/max scaling of the numeric features")
                df_features[
                    [
                        "feature1",
                        "feature2",
                        "feature3",
                        "feature4",
                        "feature5",
                        "feature6",
                        "feature7",
                        "feature8",
                        "feature9",
                        "feature10",
                        "feature11",
                        "feature12",
                    ]
                ] = (
                    scaler.fit_transform(
                        df_features[
                            [
                                "feature1",
                                "feature2",
                                "feature3",
                                "feature4",
                                "feature5",
                                "feature6",
                                "feature7",
                                "feature8",
                                "feature9",
                                "feature10",
                                "feature11",
                                "feature12",
                            ]
                        ]
                    )
                    + self.min_scale_numerics
                )
            elif self.scale_method == "standard":
                scaler = StandardScaler()
                logger.debug("Perform standard scaling of the numeric features")
                df_features[
                    [
                        "feature1",
                        "feature2",
                        "feature3",
                        "feature4",
                        "feature5",
                        "feature6",
                        "feature7",
                        "feature8",
                        "feature9",
                        "feature10",
                        "feature11",
                        "feature12",
                    ]
                ] = scaler.fit_transform(
                    df_features[
                        [
                            "feature1",
                            "feature2",
                            "feature3",
                            "feature4",
                            "feature5",
                            "feature6",
                            "feature7",
                            "feature8",
                            "feature9",
                            "feature10",
                            "feature11",
                            "feature12",
                        ]
                    ]
                )
            else:
                raise ValueError(
                    f"scale_method {self.scale_method} not supported. Use 'min_max' or 'standard'"
                )

        df_features["x"] = ""
        df_features["x"] = df_features["x"].astype("object")

        for index, row in df_features.iterrows():
            df_features.at[index, "x"] = " ".join(tokens(row, modality, 0))

        df_features["numerics"] = ""
        df_features["numerics"] = df_features["numerics"].astype("object")

        for index, row in df_features.iterrows():
            df_features.at[index, "numerics"] = numerics(row, modality, 0)

        df_features["modality"] = ""
        df_features["modality"] = df_features["modality"].astype("object")

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
                "event": y_train["label"],
                "duration": y_train["time"],
                "split": "train",
            },
            index=X_train.index,
        )

        val_data = pd.DataFrame(
            data={
                "x": X_val["x"],
                "modality": X_val["modality"],
                "numerics": X_val["numerics"],
                "event": y_val["label"],
                "duration": y_val["time"],
                "split": "valid",
            },
            index=X_val.index,
        )

        test_data = pd.DataFrame(
            data={
                "x": X_test["x"],
                "modality": X_test["modality"],
                "numerics": X_test["numerics"],
                "event": y_test["label"],
                "duration": y_test["time"],
                "split": "test",
            },
            index=X_test.index,
        )

        if self.kfold > 0:
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
                        "event": y_train_kf["label"],
                        "duration": y_train_kf["time"],
                        "split": "train",
                    },
                    index=X_train_kf.index,
                )

                test_data_kf = pd.DataFrame(
                    data={
                        "x": X_test_kf["x"],
                        "modality": X_test_kf["modality"],
                        "numerics": X_test_kf["numerics"],
                        "event": y_test_kf["label"],
                        "duration": y_test_kf["time"],
                        "split": "valid",
                    },
                    index=X_test_kf.index,
                )

                outDir = Path(f"{self.processed_dir}/{self.name}_{i}")
                outDir.mkdir(parents=True, exist_ok=True)
                data_kf = pd.concat(
                    [train_data_kf, test_data_kf, test_data]
                ).reset_index(level=0)
                data_kf.to_json(
                    Path(f"{outDir}/{self.name}_{i}.json"),
                    orient="records",
                    lines=True,
                )

        outDir = Path(f"{self.processed_dir}/{self.name}")
        outDir.mkdir(parents=True, exist_ok=True)
        data = pd.concat([train_data, val_data, test_data]).reset_index(level=0)
        data.to_json(Path(f"{outDir}/{self.name}.json"), orient="records", lines=True)
