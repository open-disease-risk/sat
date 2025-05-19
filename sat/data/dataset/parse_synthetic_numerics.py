"""Process the synthetic dataset.

1. Read the synthetic dataset
2. Process and save the data in the required format
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


@dataclass
class synthetic:
    source: str
    processed_dir: str
    name: str
    scale_method: str
    scale_numerics: bool = True
    min_scale_numerics: float = 1.0

    @log_on_start(DEBUG, "Create metabric data representation...")
    @log_on_error(
        ERROR,
        "Error creating synthetic data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
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
        data.to_json(Path(f"{out_dir}/{self.name}.json"), orient="records", lines=True)
