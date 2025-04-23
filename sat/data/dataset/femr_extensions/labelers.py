"""Custom FEMR-compatible labelers for survival analysis.

This module extends FEMR's labeling capabilities with specialized labelers for:
- Competing risks survival analysis
- Single event survival analysis
- Custom event type detection
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import abc
import logging
from typing import Dict, List, Optional

from femr.labelers import Labeler
from meds import Patient

from .schema import ExtendedLabel, EventType

logger = logging.getLogger(__name__)

class CohortLabeler(Labeler, abc.ABC):
    """FEMR-compatible labeler for cohort selection.

    This labeler returns a single event indicator for the cohort.

    Args:
        name: Unique identifier for this labeler
        event_type: Type of event this labeler is associated with
    """
    def __init__(self, name: str, event_type: EventType):
        super().__init__()
        self.name = name
        self.event_type = event_type

class CompetingRiskLabeler(CohortLabeler):
    """FEMR-compatible labeler supporting competing risks.

    This labeler supports multiple competing event types, each with its own
    set of defining event codes. It returns separate events, durations, and
    event types for each risk.

    Args:
        name: Unique identifier for this labeler
        event_type: Type of event this labeler is associated with
        event_codes: Dictionary mapping event types to lists of relevant codes
        time_field: Name of the time field to use (default: 'time')
    """

    def __init__(
        self,
        name: str,
        event_codes: Dict[str, List[str]],
        event_type: EventType = EventType.OUTCOME,
        time_field: str = "time",
        max_time: float = 3650.0,
    ):
        super().__init__(name, event_type)
        self.event_codes = event_codes
        self.time_field = time_field
        self.event_categories = list(event_codes.keys())
        self.max_time = max_time

    def get_schema(self) -> Dict:
        """Return the schema for this labeler."""
        return {
            "events": {"type": "List[int]", "description": "Event indicators"},
            "durations": {"type": "List[float]", "description": "Times to event"},
            "event_categories": {"type": "List[str]", "description": "Event category names"},
        }

    def label(self, patient: Patient) -> List[ExtendedLabel]:
        """Create competing risk labels for the patient.

        Args:
            patient: MEDS Patient object with events

        Returns:
            List of ExtendedLabel objects
        """

        # Track events for each risk type
        results: List[ExtendedLabel] = []

        # Process each competing risk type
        # Map event types to their earliest event (if any)
        event_occurrences = {}
        for event_category, codes in self.event_codes.items():
            for event in patient["events"]:
                if event["code"] in codes:
                    event_time = event[self.time_field]
                    if event_category not in event_occurrences or event_time < event_occurrences[event_category][1]:
                        event_occurrences[event_category] = (event, event_time)

        # Find the first event among all event types
        first_event_category = None
        first_event_time = float("inf")
        for event_category, (event, event_time) in event_occurrences.items():
            if event_time < first_event_time:
                first_event_time = event_time
                first_event_category = event_category

        # Determine the maximum observed event time for the patient
        if patient["events"]:
            last_event_time = patient["events"][-1][self.time_field]
        else:
            last_event_time = self.max_time  # fallback if no events

        # Use last_event_time as censoring time if no event observed
        effective_time = first_event_time if first_event_time != float("inf") else last_event_time

        # Use max_time as censoring time if no event observed
        effective_time = min(effective_time, self.max_time)

        # Create labels for each event category
        results: List[ExtendedLabel] = []
        for event_category in self.event_codes.keys():
            label: ExtendedLabel = {
                "event_category": self.name + "_" + event_category,
                "prediction_time": effective_time,
            }
            if event_category == first_event_category:
                # This is the observed event
                label["competing_event"] = True
                label["boolean_value"] = True
            else:
                # Censored at the time of the observed event
                label["competing_event"] = True
                label["boolean_value"] = False
            results.append(label)

        # Log the result for debugging
        logger.debug(
            f"Patient {patient}: labels={results}"
        )

        # Return the combined results
        return results


class SurvivalLabeler(CohortLabeler):
    """FEMR-compatible labeler for single-event survival analysis.

    This labeler identifies a single event type based on provided event codes.
    It returns a binary event indicator and duration for standard survival analysis.

    Args:
        name: Unique identifier for this labeler
        event_type: Type of event this labeler is associated with
        event_codes: List of codes that define the event of interest
        time_field: Name of the time field to use (default: 'time')
    """

    def __init__(
        self,
        name: str,
        event_codes: List[str],
        event_type: EventType = EventType.OUTCOME,
        time_field: str = "time",
        max_time: float = 3650.0,
    ):
        super().__init__(name, event_type)
        self.event_codes = set(event_codes)
        self.time_field = time_field
        self.max_time = max_time

    def get_schema(self) -> Dict:
        """Return the schema for this labeler."""
        return {
            "event": {
                "type": "int",
                "description": "Event indicator (1=event, 0=censored)",
            },
            "duration": {"type": "float", "description": "Time to event or censoring"},
        }

    def label(self, patient: Patient) -> List[ExtendedLabel]:
        """Create survival labels for the patient.

        Args:
            patient: MEDS Patient object with events

        Returns:
            List of ExtendedLabel objects
        """
        # Find the first matching event by code
        for event in patient["events"]:
            if event["code"] in self.event_codes:
                event_time = event[self.time_field]
                label: ExtendedLabel = {
                    "event_category": self.name,
                    "prediction_time": event_time,
                    "boolean_value": True,
                    "competing_event": False,
                }
                return [label]

        # No event found, censored at last event time
        if patient["events"]:
            last_event_time = patient["events"][-1][self.time_field]
        else:
            last_event_time = self.max_time

        label: ExtendedLabel = {
            "event_category": self.name,
            "prediction_time": last_event_time,
            "boolean_value": False,
            "competing_event": False,
        }
        return [label]


class CustomEventLabeler(CohortLabeler):
    """FEMR-compatible labeler for custom event detection with advanced filtering.

    This labeler identifies events based on complex criteria including code patterns,
    event sequences, and temporal relationships.

    Args:
        name: Unique identifier for this labeler
        event_type: Type of event this labeler is associated with
        primary_codes: List of codes that define the primary event
        condition_codes: Optional list of codes that must also be present
        exclusion_codes: Optional list of codes that exclude a patient
        time_window: Optional window (in days) for condition codes
        sequence_required: Whether condition codes must follow primary codes
        time_field: Name of the time field to use (default: 'days')
        max_time: Maximum follow-up time for censored patients (default: 3650.0)
    """

    def __init__(
        self,
        name: str,
        primary_codes: List[str],
        event_type: EventType = EventType.OUTCOME,
        condition_codes: Optional[List[str]] = None,
        exclusion_codes: Optional[List[str]] = None,
        time_window: Optional[float] = None,
        sequence_required: bool = False,
        time_field: str = "time",
        max_time: float = 3650.0,
    ):
        super().__init__(name, event_type)
        self.primary_codes = set(primary_codes)
        self.condition_codes = set(condition_codes or [])
        self.exclusion_codes = set(exclusion_codes or [])
        self.time_window = time_window
        self.sequence_required = sequence_required
        self.time_field = time_field
        self.max_time = max_time

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
            "condition_met": {
                "type": "bool",
                "description": "Whether condition criteria were also met",
            },
        }

    def label(self, patient: Patient) -> List[ExtendedLabel]:
        """Create custom event labels for the patient.

        Args:
            patient: FEMR Patient object with events

        Returns:
            List of ExtendedLabel objects
        """
        # Check exclusion criteria first
        patient_codes = {event["code"] for event in patient["events"]}

        if patient["events"]:
            last_event_time = patient["events"][-1][self.time_field]
        else:
            last_event_time = self.max_time

        # If any exclusion code is present, patient is excluded
        if self.exclusion_codes and self.exclusion_codes.intersection(patient_codes):
            logger.debug(
                f"Patient {patient}: Excluded due to exclusion codes"
            )
            label: ExtendedLabel = {
                "event_category": self.name,
                "prediction_time": last_event_time,
                "boolean_value": False,
                "competing_event": False,
            }
            return [label]

        # Find primary events
        primary_events = []
        for event in patient["events"]:
            if event["code"] in self.primary_codes:
                primary_events.append(event)

        if not primary_events:
            logger.debug(f"Patient {patient}: No primary events found")
            label: ExtendedLabel = {
                "event_category": self.name,
                "prediction_time": last_event_time,
                "boolean_value": False,
                "competing_event": False,
            }
            return [label]

        # Sort by time
        primary_events.sort(key=lambda e: e[self.time_field])
        earliest_primary = primary_events[0]
        primary_time = earliest_primary[self.time_field]
        primary_code = earliest_primary["code"]

        # Check condition codes if needed
        condition_met = True
        if self.condition_codes:
            condition_met = False
            for event in patient["events"]:
                if event["code"] not in self.condition_codes:
                    continue

                event_time = event[self.time_field]

                # Check sequence requirement
                if self.sequence_required and event_time < primary_time:
                    continue

                # Check time window requirement
                if self.time_window is not None:
                    time_diff = abs(event_time - primary_time)
                    if time_diff > self.time_window:
                        continue

                # If we get here, condition is met
                condition_met = True
                break

        # Determine final event status
        if condition_met:
            logger.debug(
                f"Patient {patient}: Event at time {primary_time}, code {primary_code}"
            )
            label: ExtendedLabel = {
                "event_category": self.name,
                "prediction_time": primary_time,
                "boolean_value": True,
                "competing_event": False,
            }
            return [label]
        else:
            logger.debug(
                f"Patient {patient}: Primary event found but condition not met"
            )

            label: ExtendedLabel = {
                "event_category": self.name,
                "prediction_time": last_event_time,
                "boolean_value": False,
                "competing_event": False,
            }
            return [label]
