"""Process the Metabric data.

1. Read the H5 file from: https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data.
2. Combine the train/test sets and split into train/val/test
3. Save as pandas dataframes

"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
class metabric:
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
        "Error creating metabric data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def prepare(self) -> None:
        # 1. combine the test and training sets from H5 sources
        logger.debug("Combine train/test sets from H5 source")
        with h5py.File(self.source) as file:
            keys = file.keys()
            features = []
            targets = []
            for key in keys:
                logger.debug(f"Process key: {key}")

                content = file[key]
                features.append(pd.DataFrame(data=np.asarray(content["x"])))
                targets.append(
                    pd.DataFrame(
                        data=np.hstack(
                            (
                                np.asarray(content["t"]).reshape((-1, 1)),
                                np.asarray(content["e"]).reshape((-1, 1)),
                            )
                        )
                    )
                )
                logger.debug(
                    f"Features with columns {features[-1].columns} added: {features[-1].head()}"
                )
                logger.debug(
                    f"Targets with columns {targets[-1].columns} added: {targets[-1].head()}"
                )
            df_features = pd.concat(features)
            df_targets = pd.concat(targets)

            df_features["index"] = range(len(df_features))
            df_targets["index"] = range(len(df_targets))
            df_features.set_index("index", inplace=True)
            df_targets.set_index("index", inplace=True)
            logger.debug(f"features: {df_features.head()}")
            logger.debug(f"Targets: {df_targets.head()}")

            # 2. encode the features
            # differentiate modalities, i.e., token = 0, numerics = 1
            modality = [1, 1, 1, 1, 0, 0, 0, 0, 1]

            # create tokens for a transformer model
            logger.debug("Prepend column name to the features turning them into tokens")
            logger.debug(f"Original feature columns: {df_features.columns}")
            new_feature_columns = []
            new_feature_columns.extend(list(map(lambda c: "x" + str(c), range(9))))
            new_target_columns = ["t", "e"]
            logger.debug(f"New feature columns: {new_feature_columns}")
            logger.debug(f"New target columns: {new_target_columns}")
            df_features.columns = new_feature_columns
            df_targets.columns = new_target_columns

            # min/max scaling of the numeric features
            if self.scale_numerics:
                if self.scale_method == "min_max":
                    scaler = MinMaxScaler()
                    logger.debug("Perform min/max scaling of the numeric features")
                    df_features[["x0", "x1", "x2", "x3", "x8"]] = (
                        scaler.fit_transform(
                            df_features[["x0", "x1", "x2", "x3", "x8"]]
                        )
                        + self.min_scale_numerics
                    )
                elif self.scale_method == "standard":
                    scaler = StandardScaler()
                    logger.debug("Perform standard scaling of the numeric features")
                    df_features[["x0", "x1", "x2", "x3", "x8"]] = scaler.fit_transform(
                        df_features[["x0", "x1", "x2", "x3", "x8"]]
                    )
                else:
                    raise ValueError(
                        f"scale_method {self.scale_method} not supported. Use 'min_max' or 'standard'"
                    )

            cols_categorical = ["x4", "x5", "x6", "x7"]

            logger.debug("Prepend feature name to categorical values")
            for c in cols_categorical:
                logger.debug(f"Map feature {c}")
                df_features[c] = df_features[c].apply(lambda x: c + "_" + str(x))

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
                    "event": y_train["e"],
                    "duration": y_train["t"],
                    "split": "train",
                },
                index=X_train.index,
            )

            val_data = pd.DataFrame(
                data={
                    "x": X_val["x"],
                    "modality": X_val["modality"],
                    "numerics": X_val["numerics"],
                    "event": y_val["e"],
                    "duration": y_val["t"],
                    "split": "valid",
                },
                index=X_val.index,
            )

            test_data = pd.DataFrame(
                data={
                    "x": X_test["x"],
                    "modality": X_test["modality"],
                    "numerics": X_test["numerics"],
                    "event": y_test["e"],
                    "duration": y_test["t"],
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
                            "event": y_train_kf["e"],
                            "duration": y_train_kf["t"],
                            "split": "train",
                        },
                        index=X_train_kf.index,
                    )

                    test_data_kf = pd.DataFrame(
                        data={
                            "x": X_test_kf["x"],
                            "modality": X_test_kf["modality"],
                            "numerics": X_test_kf["numerics"],
                            "event": y_test_kf["e"],
                            "duration": y_test_kf["t"],
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
            data.to_json(
                Path(f"{outDir}/{self.name}.json"), orient="records", lines=True
            )
