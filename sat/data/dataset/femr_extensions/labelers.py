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
import logging
from typing import Dict, List, Optional

from femr.labelers import Labeler
from meds import Patient

from .schema import ExtendedLabel, LabelType

logger = logging.getLogger(__name__)


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
        max_time: Maximum time for censoring (default: 3650.0)
        condition_codes: Optional list of codes that must also be present
        time_window: Optional float specifying the time window for condition codes
        sequence_required: Whether condition must follow primary event
        time_unit: Unit for time differences ('days', 'hours', 'minutes', 'seconds', default: 'days')

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
        max_time: float = 3650.0,
        condition_codes: Optional[List[str]] = None,
        time_window: Optional[float] = None,
        sequence_required: bool = False,
        time_unit: str = "days",  # Unit for time differences: 'days', 'hours', 'minutes', 'seconds'
    ):
        super().__init__(name, label_type)
        self.event_codes = set(event_codes)
        self.competing_event = competing_event
        self.time_field = time_field
        self.max_time = max_time
        self.condition_codes = set(condition_codes or [])
        self.time_window = time_window
        self.sequence_required = sequence_required
        self.time_unit = time_unit

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
        """
        Create a label for the patient using first-match logic for this event type.
        Sets boolean_value and competing_event flags appropriately.
        If condition_codes are provided, only set boolean_value=True if at least one condition event is present (optionally within time_window and/or after the primary event if sequence_required).
        """
        if patient["events"]:
            last_event_time = patient["events"][-1][self.time_field]
        else:
            last_event_time = self.max_time

        first_event_time = datetime.datetime(9999, 12, 31)
        for event in patient["events"]:
            if event["code"] in self.event_codes:
                event_time = event[self.time_field]
                if event_time < first_event_time:
                    first_event_time = event_time

        # Determine effective time and condition
        if first_event_time != datetime.datetime(9999, 12, 31):
            effective_time = first_event_time
            # Check condition codes if provided
            condition_met = True
            if self.condition_codes:
                condition_met = False
                for event in patient["events"]:
                    if event["code"] not in self.condition_codes:
                        continue
                    cond_time = event[self.time_field]
                    # Sequence requirement: condition after primary event
                    if self.sequence_required and cond_time < first_event_time:
                        continue
                    # Time window requirement
                    if self.time_window is not None:
                        # Calculate time difference based on data type
                        if isinstance(cond_time, (datetime.datetime, datetime.date)):
                            delta = cond_time - first_event_time
                            # Convert to the appropriate unit
                            if self.time_unit == "days":
                                time_diff = abs(delta.days)
                            elif self.time_unit == "hours":
                                time_diff = abs(delta.total_seconds() / 3600)
                            elif self.time_unit == "minutes":
                                time_diff = abs(delta.total_seconds() / 60)
                            else:  # seconds
                                time_diff = abs(delta.total_seconds())
                        else:
                            time_diff = abs(cond_time - first_event_time)

                        if time_diff > self.time_window:
                            continue
                    condition_met = True
                    break
            if condition_met:
                return [
                    ExtendedLabel(
                        event_category=self.name,
                        label_type=self.label_type,
                        prediction_time=effective_time,
                        boolean_value=condition_met,
                        competing_event=self.competing_event,
                    )
                ]

        return [
            ExtendedLabel(
                event_category=self.name,
                label_type=self.label_type,
                prediction_time=last_event_time,
                boolean_value=False,
                competing_event=self.competing_event,
            )
        ]
