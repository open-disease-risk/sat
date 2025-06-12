"""Custom FEMR-compatible labelers for survival analysis.

This module extends FEMR's labeling capabilities with specialized labelers for:
- Competing risks survival analysis
- Single event survival analysis
- Custom event type detection
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import abc
import datetime
from typing import Dict, List, Optional, Union

from femr.labelers import Labeler
from meds import Patient

from sat.utils import logging

from ..utils import ensure_datetime
from .schema import ExtendedLabel, LabelType

logger = logging.get_default_logger()


class CohortLabeler(Labeler, abc.ABC):
    """FEMR-compatible labeler for cohort selection.

    This labeler returns a single event indicator for the cohort.

    Args:
        name: Unique identifier for this labeler
        label_type: Type of event this labeler is associated with
    """

    def __init__(self, name: str, label_type: LabelType):
        super().__init__()
        self.name = name
        self.label_type = label_type


class CustomEventLabeler(CohortLabeler):
    """
    Labeler for survival and competing risks (first-match approach).

    Args:
        name: Unique identifier for this labeler
        event_codes: List[str] of codes that define the event of interest
        competing_event: bool, whether this labeler represents a competing event
        label_type: Type of event this labeler is associated with
        time_field: Name of the time field to use (default: 'time')
        max_time: Maximum time for censoring. Accepts:
            - ISO8601 string (e.g. "2023-12-31T23:59:59")
            - float/int (days since epoch, 1970-01-01)
            - datetime.datetime
            - None (default: datetime.max)

        condition_codes: Optional list of codes that must also be present
        time_window: Optional float specifying the time window for condition codes
        sequence_required: Whether condition must follow primary event
        time_unit: Unit for time differences ('days', 'hours', 'minutes', 'seconds', default: 'days')
        mode: Mode for label generation ('all' or 'first')

    Note: Instantiate one labeler per event type (including competing events).
    When using time_window with datetime data, the time_unit determines how the
    window is interpreted (e.g., time_window=7 with time_unit='days' means 7 days).
    """

    def __init__(
        self,
        name: str,
        event_codes: List[str],
        competing_event: bool = False,
        label_type: LabelType = LabelType.OUTCOME,
        time_field: str = "time",
        max_time: Union[str, float, int, datetime.datetime, None] = None,
        condition_codes: Optional[List[str]] = None,
        time_window: Optional[float] = None,
        sequence_required: bool = False,
        time_unit: str = "days",  # Unit for time differences: 'days', 'hours', 'minutes', 'seconds'
        mode: str = "all",  # New mode parameter
    ):
        super().__init__(name, label_type)
        self.event_codes = set(event_codes)
        self.competing_event = competing_event
        self.label_type = label_type
        self.time_field = time_field

        # Ensure max_time is always a datetime object
        if max_time is None:
            self.max_time = datetime.datetime.max
        elif isinstance(max_time, (float, int)):
            # Interpret as days from epoch
            self.max_time = datetime.datetime(1970, 1, 1) + datetime.timedelta(
                days=max_time
            )
        elif isinstance(max_time, str):
            try:
                # Accept ISO8601 strings
                self.max_time = datetime.datetime.fromisoformat(max_time)
            except Exception as e:
                raise ValueError(
                    f"Could not parse max_time string '{max_time}' as datetime: {e}"
                )
        elif isinstance(max_time, datetime.datetime):
            self.max_time = max_time
        else:
            raise ValueError(
                "max_time must be a string (ISO8601), datetime, or float/int (days since epoch)"
            )

        self.condition_codes = set(condition_codes) if condition_codes else set()
        self.time_window = time_window
        self.sequence_required = sequence_required
        self.time_unit = time_unit
        self.mode = mode  # Store mode as instance attribute

    def get_schema(self) -> Dict:
        """Return the schema for this labeler."""
        return {
            "event": {
                "type": "int",
                "description": "Event indicator (1=event, 0=censored)",
            },
            "duration": {"type": "float", "description": "Time to event or censoring"},
            "primary_code": {
                "type": "str",
                "description": "The specific code that triggered the event",
            },
        }

    def label(self, patient: Patient) -> List[ExtendedLabel]:
        """Label a patient based on the event codes and conditions."""
        labels = []

        # First, find any matching events based on criteria
        for event in patient["events"]:
            event[self.time_field] = ensure_datetime(event[self.time_field])
        for event in patient["events"]:
            event_time = event[self.time_field]
            if event_time is None or event_time > self.max_time:
                continue  # skip events with no valid time or after max_time
            if event["code"] in self.event_codes:
                condition_met = True
                if self.condition_codes:
                    condition_met = False
                    for cond_event in patient["events"]:
                        if cond_event["code"] in self.condition_codes:
                            cond_time = cond_event[self.time_field]
                            if cond_time is None:
                                continue  # skip events with no valid time (e.g., demographic)
                            if self.sequence_required and cond_time <= event_time:
                                continue
                            delta = cond_time - event_time
                            if self.time_unit == "days":
                                time_diff = abs(delta.days)
                            elif self.time_unit == "hours":
                                time_diff = abs(delta.total_seconds() / 3600)
                            elif self.time_unit == "minutes":
                                time_diff = abs(delta.total_seconds() / 60)
                            else:  # seconds
                                time_diff = abs(delta.total_seconds())

                            if (
                                self.time_window is None
                                or time_diff <= self.time_window
                            ):
                                condition_met = True
                                break
                if condition_met:
                    if self.mode == "first" and labels:
                        break
                    labels.append(
                        ExtendedLabel(
                            event_category=self.name,
                            label_type=self.label_type,
                            prediction_time=event_time,
                            boolean_value=condition_met,
                            competing_event=self.competing_event,
                        )
                    )

        # Calculate a reasonable censor time based on patient events
        last_observed_time = self.max_time  # Default to max_time
        if patient["events"]:
            valid_times = [
                event[self.time_field]
                for event in patient["events"]
                if event[self.time_field] is not None
            ]
            if valid_times:
                last_observed_time = max(valid_times)

        # If no event labels were found, add a censoring at last observation time
        if not labels:
            labels.append(
                ExtendedLabel(
                    event_category=self.name,
                    label_type=self.label_type,
                    prediction_time=last_observed_time,
                    boolean_value=False,
                    competing_event=self.competing_event,
                )
            )

        # ALWAYS add a guaranteed max_time label to ensure something survives anchor time filtering
        # Only add if we don't already have a label at max_time
        has_max_time_label = any(
            label["prediction_time"] == self.max_time for label in labels
        )
        if not has_max_time_label:
            # Add a "guaranteed" censoring label at max_time that will survive any anchor time filtering
            labels.append(
                ExtendedLabel(
                    event_category=self.name,
                    label_type=self.label_type,
                    prediction_time=self.max_time,
                    boolean_value=False,  # Always censored
                    competing_event=self.competing_event,
                )
            )

        return labels
