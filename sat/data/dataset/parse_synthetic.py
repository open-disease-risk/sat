"""Process the synthetic dataset

1. Read the H5 file from: https://github.com/chl8856/DeepHit/blob/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv
2. Combine the train/test sets and split into train/val/test
3. Save as pandas dataframes

"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path

import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from sat.utils import logging
from sat.utils.data import train_val_test

logger = logging.get_default_logger()


@dataclass
class synthetic:
    source: str
    processed_dir: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    n_bins: int
    encode: str
    strategy: str
    name: str
    kfold: int = 0

    @log_on_start(DEBUG, "Create synthetic data representation...")
    @log_on_error(
        ERROR,
        "Error creating synthetic data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def prepare(self) -> None:
        # 1. read original data
        logger.debug("Read original synthetic data")
        df = pd.read_csv(self.source)
        cols_standardize = df.columns[4:]
        df_features = df[cols_standardize]
        df_targets = df[df.columns[:2]]

        logger.debug(f"features: {df_features.head()}")
        logger.debug(f"Targets: {df_targets.head()}")

        # 2. encode the features
        logger.debug("Encode the features only")
        feat_standardize = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy,
        ).fit_transform(df_features)

        df_feat = pd.DataFrame(feat_standardize, columns=cols_standardize)

        logger.debug("Run the label encoder on each feature")
        vocab_size = 0
        for _, feat in enumerate(df_feat.columns):
            df_feat[feat] = (
                LabelEncoder().fit_transform(df_feat[feat]).astype(int) + vocab_size
            )
            vocab_size = df_feat[feat].max() + 1

        # create tokens for a transformer model
        logger.debug("Prepend column name to the features turning them into tokens")
        logger.debug(f"Feature types: {df_feat.dtypes}")
        main_feature_cols = df_feat.columns
        for c in df_feat.columns:
            logger.debug(f"Map feature {c}")
            df_feat[c] = df_feat[c].apply(lambda x: c + "_" + str(x))

        # 3. create train/val/test split
        logger.debug("Create train/val/test split")
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(
            X=df_feat,
            y=df_targets,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            validation_ratio=self.validation_ratio,
        )

        # 4. save data frames
        logger.debug("Save the dataframes")
        logger.debug("Combine features and targets and save into CSV files")
        train_data = pd.DataFrame(
            {
                "x": X_train.agg(" ".join, axis=1),
                "event": y_train["label"],
                "duration": y_train["time"],
                "split": "train",
            },
            index=X_train.index,
        )

        val_data = pd.DataFrame(
            {
                "x": X_val.agg(" ".join, axis=1),
                "event": y_val["label"],
                "duration": y_val["time"],
                "split": "valid",
            },
            index=X_val.index,
        )

        test_data = pd.DataFrame(
            {
                "x": X_test.agg(" ".join, axis=1),
                "event": y_test["label"],
                "duration": y_test["time"],
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
                        "event": y_train_kf["label"],
                        "duration": y_train_kf["time"],
                        "split": "train",
                    },
                    index=X_train_kf.index,
                )

                test_data_kf = pd.DataFrame(
                    data={
                        "x": X_test_kf[main_feature_cols].agg(" ".join, axis=1),
                        "event": y_test_kf["label"],
                        "duration": y_test_kf["time"],
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

        data = pd.concat([train_data, val_data, test_data]).reset_index()
        data.to_json(Path(f"{outDir}/{self.name}.json"), orient="records", lines=True)
