"""Evaluation modules for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import evaluate
import torch

import numpy as np
import pandas as pd

from evaluate import EvaluationModule

from sat.utils import logging

logger = logging.get_default_logger()


class SurvivalEvaluationModule(EvaluationModule):
    def survival_predictions(self, predictions):
        """Process prediction data efficiently for survival analysis.
        
        Args:
            predictions: Can be a numpy array, PyTorch tensor, or dictionary
                         containing 'hazard', 'risk', and 'survival' keys.
                         
        Returns:
            Processed prediction tensor with shape (batch_size, 3, event_size, duration_cuts-1)
        """

        try:
            logger.debug(f"type of predictions: {type(predictions)}")
            # predictions can be a numpy array
            if hasattr(predictions, "shape"):
                logger.info(
                    f"Predictions already has shape {predictions.shape}, returning as is"
                )
                return predictions[:, :, :, 1:]

            # or predictions can be a dictionary
            if not isinstance(predictions, dict):
                raise ValueError(
                    f"Expected dictionary or numpy array, got {type(predictions)}"
                )

            # Check if required keys exist
            required_keys = ["hazard", "risk", "survival"]
            for key in required_keys:
                if key not in predictions:
                    raise ValueError(f"Missing required key in predictions: {key}")

            # Ensure predictions have the right format
            hazard_pred = predictions["hazard"]
            risk_pred = predictions["risk"]
            survival_pred = predictions["survival"]

            # check prediction structure
            if not (
                hazard_pred.shape == risk_pred.shape
                and hazard_pred.shape == survival_pred.shape
                and hazard_pred.ndim >= 3
            ):
                raise ValueError(
                    f"Shape mismatch: hazard {hazard_pred.shape}, "
                    f"risk {risk_pred.shape}, survival {survival_pred.shape}"
                )
            
            # Extract dimensions once for reuse
            batch_size = hazard_pred.shape[0]
            event_size = hazard_pred.shape[1]
            duration_cuts = hazard_pred.shape[2]
            
            if duration_cuts <= 1:
                error_msg = f"Duration cuts must be greater than 1, got {duration_cuts}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Use np.empty for faster allocation when values will be completely overwritten
            result = np.empty(
                (batch_size, 3, event_size, duration_cuts - 1), dtype=np.float32
            )
            
            # Direct assignment to avoid stack and transpose operations
            result[:, 0, :, :] = hazard_pred[:, :, 1:]
            result[:, 1, :, :] = risk_pred[:, :, 1:]
            result[:, 2, :, :] = survival_pred[:, :, 1:]
            
            return result
        
        except Exception as e:
            logger.error(f"Error in survival_predictions: {e}")
            logger.debug(f"Got predictions: {predictions}")
            # Return empty array if we encounter any exception
            return np.array([])


class ComputeBrier(SurvivalEvaluationModule):
    def __init__(self, cfg, survival_train_path, duration_cuts, per_horizon=False):
        self.cfg = cfg
        self.survival_train = pd.read_csv(survival_train_path)

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.duration_cuts = df.cuts.values[1:-1]  # we do not need the start points
        self.per_horizon = per_horizon

    def compute_event(self, predictions, references, event):
        logger.debug(f"predictions shape: {predictions.shape}")
        logger.debug(f"references shape: {references.shape}")

        # Check if predictions is valid for computing metrics
        if predictions.size == 0 or predictions.ndim < 4:
            logger.warning(
                f"Invalid predictions for compute_event: shape {predictions.shape}"
            )
            # Return default values for metrics
            return {
                f"brier_{event}th_event": 0.5,  # Default Brier score (0.5 is baseline random)
                f"brier_{event}th_event_n": 0,
            }

        brier_score = evaluate.load("./sat/evaluate/brier_score")

        events_train = self.survival_train.iloc[:, (1 * self.cfg.num_events + event)]
        durations_train = self.survival_train.iloc[:, (3 * self.cfg.num_events + event)]
        events_test = references[:, (1 * self.cfg.num_events + event)].astype(bool)
        durations_test = references[:, (3 * self.cfg.num_events + event)]

        preds = predictions[:, 2, event]

        train_set = np.stack([events_train, durations_train], axis=1)

        metric_dict = {}
        quantile_incr = 1.0 / preds.shape[1]
        horizons = np.arange(1, preds.shape[1]) * quantile_incr

        n = events_test.sum()
        et_test = np.stack([events_test, durations_test], axis=1)

        # Compute metrics
        ibrs, brs = brier_score.compute(
            references=et_test,
            predictions=preds[:, :-1],
            train_set=train_set,
            duration_cuts=self.duration_cuts,
            per_horizon=self.per_horizon,
        )

        # Record results
        for j in range(len(brs)):
            metric_dict[f"brier_{event}th_event_{horizons[j]}"] = brs[j]

        metric_dict[f"brier_{event}th_event"] = ibrs
        metric_dict[f"brier_{event}th_event_n"] = n

        return metric_dict

    def compute(self, predictions, references):
        predictions = self.survival_predictions(predictions)
        logger.debug(f"survival predictions shape {predictions.shape}")

        brier_mean = 0.0
        brier_balanced_mean = 0.0
        brier_n = 0

        metrics_dict = {}
        for i in range(self.cfg.num_events):
            metric_results = self.compute_event(predictions, references, i)
            metrics_dict.update(metric_results)
            brier_mean += (
                metrics_dict[f"brier_{i}th_event"]
                * metrics_dict[f"brier_{i}th_event_n"]
            )
            brier_balanced_mean += metrics_dict[f"brier_{i}th_event"]
            brier_n += metrics_dict[f"brier_{i}th_event_n"]

        # Prevent division by zero
        if brier_n > 0:
            brier_mean /= brier_n
        else:
            brier_mean = 0.5  # Default value when no events

        metrics_dict["brier_weighted_avg"] = brier_mean
        metrics_dict["brier_avg"] = brier_balanced_mean / max(1, self.cfg.num_events)

        return metrics_dict


class ComputeCIndex(SurvivalEvaluationModule):
    def __init__(self, cfg, survival_train_path, duration_cuts):
        self.cfg = cfg
        self.survival_train = pd.read_csv(survival_train_path)
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.duration_cuts = df.cuts.values[1:]  # we do not need the start point

    def compute_event(self, predictions, references, event):
        logger.debug(f"predictions shape: {predictions.shape}")
        logger.debug(f"references shape: {references.shape}")

        # Check if predictions is valid for computing metrics
        if predictions.size == 0 or predictions.ndim < 4:
            logger.warning(
                f"Invalid predictions for compute_event: shape {predictions.shape}"
            )
            # Return default values for metrics
            return {
                f"ipcw_{event}th_event": 0.5,  # Default c-index (0.5 is random)
                f"ipcw_avg_{event}th_event": 0.5,
                f"ipcw_{event}th_event_n": 0,
            }

        c_index = evaluate.load("./sat/evaluate/concordance_index_ipcw")

        events_train = self.survival_train.iloc[:, (1 * self.cfg.num_events + event)]
        durations_train = self.survival_train.iloc[:, (3 * self.cfg.num_events + event)]
        events_test = references[:, (1 * self.cfg.num_events + event)].astype(bool)
        durations_test = references[:, (3 * self.cfg.num_events + event)]

        # Safely get predictions
        preds = predictions[:, 1, event]  # Get risk predictions

        train_set = np.stack([events_train, durations_train], axis=1)

        metric_dict = {}
        quantile_incr = 1.0 / preds.shape[1]
        horizons = np.arange(1, preds.shape[1] + 1) * quantile_incr

        n = events_test.sum()
        et_test = np.stack([events_test, durations_test], axis=1)

        # Compute c-index with error handling
        cindeces = c_index.compute(
            references=et_test,
            predictions=preds,
            train_set=train_set,
            duration_cuts=self.duration_cuts,
        )

        # Record results
        for j in range(len(cindeces)):
            metric_dict[f"ipcw_{event}th_event_{horizons[j]}"] = cindeces[j]

        metric_dict[f"ipcw_avg_{event}th_event"] = np.mean(cindeces)
        metric_dict[f"ipcw_{event}th_event"] = cindeces[-1]

        # Always record sample size
        metric_dict[f"ipcw_{event}th_event_n"] = n

        return metric_dict

    def compute(self, predictions, references):
        predictions = self.survival_predictions(predictions)
        cindex_mean = 0.0
        cindex_weighted_avg_mean = 0.0
        cindex_avg_mean = 0.0
        cindex_n = 0
        metrics_dict = {}

        for i in range(self.cfg.num_events):
            metric_results = self.compute_event(predictions, references, i)
            metrics_dict.update(metric_results)
            cindex_mean += (
                metrics_dict[f"ipcw_{i}th_event"] * metrics_dict[f"ipcw_{i}th_event_n"]
            )
            cindex_weighted_avg_mean += (
                metrics_dict[f"ipcw_avg_{i}th_event"]
                * metrics_dict[f"ipcw_{i}th_event_n"]
            )
            cindex_avg_mean += metrics_dict[f"ipcw_avg_{i}th_event"]
            cindex_n += metrics_dict[f"ipcw_{i}th_event_n"]

        # Prevent division by zero
        if cindex_n > 0:
            cindex_mean /= cindex_n
            weighted_avg = cindex_weighted_avg_mean / cindex_n
        else:
            cindex_mean = 0.5  # Default value (random predictor)
            weighted_avg = 0.5

        metrics_dict["ipcw"] = cindex_mean
        metrics_dict["ipcw_weighted_avg"] = weighted_avg
        metrics_dict["ipcw_avg"] = cindex_avg_mean / max(1, self.cfg.num_events)

        return metrics_dict


class ComputeMismatch(EvaluationModule):
    def __init__(self, duration_cuts, max_time):
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.duration_cuts = torch.tensor(df.cuts.values[1:-1]).to(
            torch.float32
        )  # we do not need the start and end points
        self.max_time = max_time

    def compute(self, predictions, references):
        predictions = self.survival_predictions(predictions)
        logger.debug(f"predictions: {predictions}")
        logger.debug(f"references: {references}")
        mismatch = evaluate.load("./sat/evaluate/mismatch")

        metric_dict = {}

        loss = mismatch.compute(
            predictions=predictions,
            references=references,
            duration_cuts=self.duration_cuts,
            max_time=self.max_time,
        )
        metric_dict["mismatch_loss"] = loss

        return metric_dict


class ComputeNLLPHazardLoss(EvaluationModule):
    def __init__(self, importance_sample_weights, per_event=False):
        self.importance_sample_weights = importance_sample_weights
        self.per_event = per_event

    def compute(self, predictions, references):
        predictions = self.survival_predictions(predictions)
        nllphazard = evaluate.load("./sat/evaluate/nllphazard")

        metric_dict = {}

        loss, event_losses = nllphazard.compute(
            predictions=predictions,
            references=references,
            importance_sampling_weights=self.importance_sample_weights,
            per_event=self.per_event,
        )
        metric_dict["nllph_hazard_loss"] = loss
        for j in range(len(event_losses)):
            metric_dict[f"nllph_hazard_{j}th_event_loss"] = event_losses[j]

        return metric_dict


class RegressionEvaluationModule(EvaluationModule):
    def __init__(self):
        pass


class ComputeL1(RegressionEvaluationModule):
    def __init__(
        self,
        training_set,
        l1_type,
        num_events,
        importance_sample_weights: str = None,
        per_event=False,
    ):
        self.training_set = training_set
        self.l1_type = l1_type
        self.num_events = num_events
        self.per_event = per_event
        self.importance_sample_weights = importance_sample_weights

    def compute(self, predictions, references):
        l1 = evaluate.load("./sat/evaluate/l1")

        metric_dict = {}

        loss, event_losses = l1.compute(
            predictions=predictions["time_to_event"],
            references=references,
            training_set=self.training_set,
            importance_sample_weights=self.importance_sample_weights,
            l1_type=self.l1_type,
            num_events=self.num_events,
            per_event=self.per_event,
        )

        metric_dict["l1_loss"] = loss
        for j in range(len(event_losses)):
            metric_dict[f"l1_{j}th_event_loss"] = event_losses[j]

        return metric_dict


class ComputeMSELoss(RegressionEvaluationModule):
    def __init__(
        self,
        training_set: str,
        l2_type: str,
        num_events: int,
        importance_sample_weights: str = None,
        per_event: bool = False,
    ):
        self.training_set = training_set
        self.l2_type = l2_type
        self.num_events = num_events
        self.importance_sample_weights = importance_sample_weights
        self.per_event = per_event

    def compute(self, predictions, references):
        mse = evaluate.load("./sat/evaluate/mse")

        metric_dict = {}

        loss, event_losses = mse.compute(
            predictions=predictions["time_to_event"],
            references=references,
            training_set=self.training_set,
            importance_sample_weights=self.importance_sample_weights,
            l2_type=self.l2_type,
            num_events=self.num_events,
            per_event=self.per_event,
        )
        metric_dict["mse_loss"] = loss
        for j in range(len(event_losses)):
            metric_dict[f"mse_{j}th_event_loss"] = event_losses[j]

        return metric_dict


class ComputeTTERanking(RegressionEvaluationModule):
    def __init__(self, num_events: int):
        self.num_events = num_events

    def compute(self, predictions, references):
        tte_r = evaluate.load("./sat/evaluate/tte_ranking")

        metric_dict = {}

        loss, event_losses = tte_r.compute(
            predictions=predictions["time_to_event"],
            references=references,
            num_events=self.num_events,
        )
        metric_dict["tte_ranking_count"] = loss
        for j in range(len(event_losses)):
            metric_dict[f"tte_ranking_{j}th_event_count"] = event_losses[j]

        return metric_dict


class ComputeCrossEntropyLoss(EvaluationModule):
    def __init__(
        self,
        event_time_thr: float,
        training_set: str,
        num_events: int,
        l_type: str = "margin",
        importance_sample_weights: str = None,
        per_event: bool = False,
    ):
        self.event_time_thr = event_time_thr
        self.num_events = num_events
        self.per_event = per_event
        self.training_set = training_set
        self.l_type = l_type
        self.importance_sample_weights = importance_sample_weights

    def compute(self, predictions, references):
        ce = evaluate.load("./sat/evaluate/cross-entropy")

        metric_dict = {}

        loss, event_losses = ce.compute(
            predictions=predictions["event"],
            references=references,
            event_time_thr=self.event_time_thr,
            training_set=self.training_set,
            l_type=self.l_type,
            importance_sample_weights=self.importance_sample_weights,
            num_events=self.num_events,
            per_event=self.per_event,
        )
        metric_dict["ce_loss"] = loss
        for j in range(len(event_losses)):
            metric_dict[f"ce_{j}th_event_loss"] = event_losses[j]

        return metric_dict


class ComputeOneCalibration(EvaluationModule):
    def __init__(
        self, bins: int, num_events: int, duration_cuts: str, event_time_thr: float
    ):
        self.bins = bins
        self.num_events = num_events
        self.event_time_thr = event_time_thr
        self.config_name = "survival"

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.duration_cuts = df.cuts.values[
            1:-1
        ]  # we do not need the start and end points

    def reshape(self, predictions):
        if "survival" in predictions:
            self.config_name = "survival"
            # we only care about the survival predictions between the first and last cut
            predictions = predictions["survival"][
                :, :, 1:-1
            ]  # need this to be batch x events x duration cuts
        elif "event" in predictions:
            # dimensions: events  x batch x prediction (1 - to be squeezed)
            self.config_name = "classification"
            predictions = np.squeeze(
                predictions["event"], axis=2
            )  # need this to be batch x events

        return predictions

    def compute(self, predictions, references):
        predictions = self.reshape(predictions)
        oc = evaluate.load("./sat/evaluate/one_calibration", self.config_name)

        metric_dict = oc.compute(
            predictions=predictions,
            references=references,
            bins=self.bins,
            num_events=self.num_events,
            duration_cuts=self.duration_cuts,
            event_time_thr=self.event_time_thr,
        )

        return metric_dict
