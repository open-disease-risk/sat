"""Process the Metabric data.

1. Read the H5 file from: https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data.
2. Combine the train/test sets
3. Process and save the data in the required format

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sat.utils import logging

from . import utils

logger = logging.get_default_logger()

# Token and numeric processing functions moved to utils.py


@dataclass
class metabric:
    source: str
    processed_dir: str
    name: str
    scale_method: str
    scale_numerics: bool = True
    min_scale_numerics: float = 1.0

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

            # Create a function that takes both column name and value to avoid closure issues
            def prefix_with_colname(colname, value):
                return f"{colname}_{value}"

            for col in cols_categorical:
                logger.debug(f"Map feature {col}")
                # Using a partial function to avoid closure issues with loop variables
                df_features[col] = df_features[col].apply(
                    lambda x, col=col: prefix_with_colname(col, x)
                )

            df_features["x"] = ""
            df_features["x"] = df_features["x"].astype("object")

            for index, _ in df_features.iterrows():
                df_features.at[index, "x"] = " ".join(
                    utils.tokens(df_features.iloc[index], modality)
                )

            df_features["numerics"] = ""
            df_features["numerics"] = df_features["numerics"].astype("object")

            for index, _ in df_features.iterrows():
                df_features.at[index, "numerics"] = utils.numerics(
                    df_features.iloc[index], modality
                )

            df_features["modality"] = ""
            df_features["modality"] = df_features["modality"].astype("object")

            for index, _ in df_features.iterrows():
                df_features.at[index, "modality"] = modality

            # 4. Create final dataframe with all data
            logger.debug("Create final dataframe")
            data = pd.DataFrame(
                data={
                    "x": df_features["x"],
                    "modality": df_features["modality"],
                    "numerics": df_features["numerics"],
                    "event": df_targets["e"],
                    "duration": df_targets["t"],
                },
                index=df_features.index,
            ).reset_index(level=0)

            # 5. Save to file
            out_dir = Path(f"{self.processed_dir}/{self.name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            data.to_json(
                Path(f"{out_dir}/{self.name}.json"), orient="records", lines=True
            )
