"""Process the SEER breast cancer dataset.

1. Read SEER file from data/seer/
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


# Token and numeric processing functions moved to utils.py


@dataclass
class seer:
    source: str
    processed_dir: str
    name: str
    scale_method: str
    scale_numerics: bool = True
    min_scale_numerics: float = 1.0

    @log_on_start(DEBUG, "Create seer data representation...")
    @log_on_error(
        ERROR,
        "Error creating seer data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
        # 1. combine the test and training sets from H5 sources
        logger.debug("Combine train/test sets from H5 source")

        df = pd.read_csv(self.source, comment="#", index_col=None)
        df.rename(columns={"duration": "duration1"}, inplace=True)
        df["duration2"] = df["duration1"]
        df["event1"] = df["event_breast"]
        df["event2"] = df["event_heart"]

        df_targets = df[["event1", "event2", "duration1", "duration2"]]
        df_targets["durations"] = df[["duration1", "duration2"]].values.tolist()
        df_targets["events"] = df[["event1", "event2"]].values.tolist()
        df_features = df[
            [
                "sex",
                "year_diagnosis",
                "race",
                "histology_type",
                "laterality",
                "seq_number",
                "er_status_breast_cancer",
                "pr_status_breast_cancer",
                "summary_stage_2000",
                "rx_summ",
                "reason_no_surgery",
                "first_malignant_indicator",
                "diagnostic_confirmation",
                "median_household_income",
                "regional_nodes_examined",
                "CS_tumor_size",
                "total_number_benign_tumors",
                "total_number_malignant_tumors",
            ]
        ]

        categorical_features = [
            "sex",
            "year_diagnosis",
            "race",
            "histology_type",
            "laterality",
            "seq_number",
            "er_status_breast_cancer",
            "pr_status_breast_cancer",
            "summary_stage_2000",
            "rx_summ",
            "reason_no_surgery",
            "first_malignant_indicator",
            "diagnostic_confirmation",
            "median_household_income",
        ]

        numeric_features = [
            "regional_nodes_examined",
            "CS_tumor_size",
            "total_number_benign_tumors",
            "total_number_malignant_tumors",
        ]

        logger.debug(f"features: {df_features.head()}")
        logger.debug(f"Targets: {df_targets.head()}")

        # 2. encode the features
        # differentiate modalities, i.e., token = 0, numerics = 1
        modality = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

        # create tokens for a transformer model
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

        logger.debug("Prepend feature name to categorical values")

        # Create a function that takes both column name and value to avoid closure issues
        def prefix_with_colname(colname, value):
            return f"{colname}_{value}"

        for col in categorical_features:
            logger.debug(f"Map feature {col}")
            # Using a partial function to avoid closure issues with loop variables
            df_features.loc[:, col] = df_features[col].apply(
                lambda x, col=col: prefix_with_colname(col, x)
            )

        df_features.loc[:, "x"] = ""
        df_features.loc[:, "x"] = df_features["x"].astype("object")

        for index, _ in df_features.iterrows():
            df_features.at[index, "x"] = " ".join(
                utils.tokens(df_features.iloc[index], modality)
            )

        df_features.loc[:, "numerics"] = ""
        df_features.loc[:, "numerics"] = df_features["numerics"].astype("object")

        for index, _ in df_features.iterrows():
            df_features.at[index, "numerics"] = utils.numerics(
                df_features.iloc[index], modality
            )

        df_features.loc[:, "modality"] = ""
        df_features.loc[:, "modality"] = df_features["modality"].astype("object")

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
