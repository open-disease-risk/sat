"""Feature extractor for a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import pickle
from logging import DEBUG, ERROR
from pathlib import Path
from typing import List, Union

import numpy as np
from logdecorator import log_on_end, log_on_error, log_on_start
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin

from sat.utils import logging

logger = logging.get_default_logger()


class SAFeatureExtractor(FeatureExtractionMixin):
    @log_on_start(
        DEBUG,
        "Instantiate SAFeatureExtractor({label_transform_path}, {do_data_transform})",
        logger=logger,
    )
    @log_on_error(
        ERROR,
        "Error during SAFeatureExtractor.__init__(): {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def __init__(
        self,
        label_transform_path: Path = None,
        do_data_transform: bool = True,
        duration_col: str = "duration",
        event_col: str = "event",
        transformed_duration_cols: List[str] = ["t", "e"],
        **kwargs,
    ):
        self.label_transform_path = label_transform_path
        self.do_data_transform = do_data_transform
        self.duration_col = duration_col
        self.event_col = event_col
        self.transformed_duration_cols = transformed_duration_cols
        self.labtrans = None
        self.is_data_transform_loaded = False

        super().__init__(**kwargs)

    @log_on_start(
        DEBUG,
        "Load label transformer",
        logger=logger,
    )
    @log_on_error(
        ERROR,
        "Error during loading the label transformer: {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    def _load_label_transformer(self):
        if self.do_data_transform and not self.is_data_transform_loaded:
            logger.debug("Load label transformer.")
            pickle_file = self.label_transform_path
            with open(pickle_file, "rb") as pf:
                self.labtrans = pickle.load(pf)
                self.is_data_transform_loaded = True

    @log_on_start(
        DEBUG,
        "Transform features in SAFeatureExtractor.__call__()",
        logger=logger,
    )
    @log_on_error(
        ERROR,
        "Error during SAFeatureExtractor.__call__(): {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def __call__(
        self, data: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
    ) -> BatchFeature:
        self._load_label_transformer()

        duration = np.array(data[self.duration_col])
        event = np.array(data[self.event_col])

        duration = duration[:, np.newaxis] if duration.ndim == 1 else duration
        event = event[:, np.newaxis] if event.ndim == 1 else event

        num_events = event.shape[1]
        if self.do_data_transform:
            y_trans = self.labtrans.transform(
                duration.ravel(),
                event.ravel(),
            )
            t = y_trans[0].reshape(-1, num_events)
            f = y_trans[2].reshape(-1, num_events)
        else:
            t = np.array(data[self.transformed_duration_cols[0]]).reshape(
                -1, num_events
            )
            f = np.array(data[self.transformed_duration_cols[1]]).reshape(
                -1, num_events
            )

        e = event.reshape(-1, num_events)
        d = duration.reshape(-1, num_events)

        logger.debug(
            f"Dimensions for t: {t.shape}, e: {e.shape}, f: {f.shape}, d: {d.shape}"
        )
        data["labels"] = np.concatenate((t, e, f, d), axis=1)

        return data
