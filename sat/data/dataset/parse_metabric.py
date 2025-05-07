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
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from sat.utils import logging
from sat.utils.data import train_val_test

logger = logging.get_default_logger()


@dataclass
class metabric:
    source: str
    processed_dir: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    n_bins: int
    encode: str
    strategy: str
    name: str
    flip_event: bool = False
    kfold: int = 0

    @log_on_start(DEBUG, "Create metabric data representation...")
    @log_on_error(
        ERROR,
        "Error creating metabric data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
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
            logger.debug("Encode the features only")
            cols_categorical = [4, 5, 6, 7]
            cols_standardize = [0, 1, 2, 3, 8]
            df_feat_standardize = df_features[cols_standardize]
            df_feat_standardize_disc = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.encode,
                strategy=self.strategy,
            ).fit_transform(df_feat_standardize)
            df_feat_standardize_disc = pd.DataFrame(
                df_feat_standardize_disc, columns=cols_standardize
            )

            # must be categorical feature ahead of numerical features!
            df_feat = pd.concat(
                [df_features[cols_categorical], df_feat_standardize_disc], axis=1
            )

            logger.debug("Run the label encoder on each feature")
            vocab_size = 0
            for _, feat in enumerate(cols_categorical):
                df_feat[feat] = (
                    LabelEncoder().fit_transform(df_features[feat]).astype(int)
                    + vocab_size
                )
                vocab_size = df_feat[feat].max() + 1

            # create tokens for a transformer model
            logger.debug("Prepend column name to the features turning them into tokens")

            cols = cols_categorical
            cols.extend(cols_standardize)
            logger.debug(f"Original feature columns: {df_feat.columns}")
            logger.debug(f"Original target columns: {df_targets.columns}")
            new_feature_columns = []
            main_feature_cols = list(map(lambda c: "x" + str(c), cols))
            new_feature_columns.extend(main_feature_cols)
            new_target_columns = ["t", "e"]
            logger.debug(f"New feature columns: {new_feature_columns}")
            logger.debug(f"New target columns: {new_target_columns}")
            df_feat.columns = new_feature_columns
            df_targets.columns = new_target_columns

            # Create a function that takes both column name and value to avoid closure issues
            def prefix_with_colname(colname, value):
                return f"{colname}_{value}"

            for col in df_feat.columns:
                logger.debug(f"Map feature {col}")
                # Using a partial function to avoid closure issues with loop variables
                df_feat[col] = df_feat[col].apply(
                    lambda x, col=col: prefix_with_colname(col, x)
                )

            # 3. create a dummy dataset where the event is flipped to 2 randomly
            if self.flip_event:
                df_candidate = df_targets.loc[df_targets["e"] == 1]
                df_update = df_candidate.sample(frac=0.5, replace=False)
                df_update["e"] = 2
                df_targets.update(df_update)

            # 4. create train/val/test split
            logger.debug("Create train/val/test split")
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(
                X=df_feat,
                y=df_targets,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                validation_ratio=self.validation_ratio,
            )

            # 5. save data frames
            logger.debug("Save the dataframes")
            logger.debug("Combine features and targets and save into CSV files")

            train_data = pd.DataFrame(
                {
                    "x": X_train.agg(" ".join, axis=1),
                    "event": y_train["e"],
                    "duration": y_train["t"],
                    "split": "train",
                },
                index=X_train.index,
            )

            val_data = pd.DataFrame(
                {
                    "x": X_val.agg(" ".join, axis=1),
                    "event": y_val["e"],
                    "duration": y_val["t"],
                    "split": "valid",
                },
                index=X_val.index,
            )

            test_data = pd.DataFrame(
                {
                    "x": X_test.agg(" ".join, axis=1),
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
                            "x": X_train_kf[main_feature_cols].agg(" ".join, axis=1),
                            "event": y_train_kf["e"],
                            "duration": y_train_kf["t"],
                            "split": "train",
                        },
                        index=X_train_kf.index,
                    )

                    test_data_kf = pd.DataFrame(
                        data={
                            "x": X_test_kf[main_feature_cols].agg(" ".join, axis=1),
                            "event": y_test_kf["e"],
                            "duration": y_test_kf["t"],
                            "split": "valid",
                        },
                        index=X_test_kf.index,
                    )

                    outDir = Path(f"{self.processed_dir}/{self.name}")
                    outDir.mkdir(parents=True, exist_ok=True)
                    data_kf = pd.concat(
                        [train_data_kf, test_data_kf, test_data]
                    ).reset_index(level=0)
                    data_kf.to_csv(
                        Path(f"{outDir}/{i}_{self.name}.csv"),
                        index=False,
                    )

            outDir = Path(f"{self.processed_dir}/{self.name}")
            outDir.mkdir(parents=True, exist_ok=True)
            data = pd.concat([train_data, val_data, test_data]).reset_index()
            data.to_csv(
                Path(f"{outDir}/{self.name}.csv"),
                index=False,
            )
