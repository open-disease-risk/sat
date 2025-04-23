"""
Extended Label schema for FEMR extensions.
Adds a 'competing_event' flag to the base meds.Label schema.
"""
from typing import Optional
from meds.schema import Label
from enum import Enum

class EventType(str, Enum):
    ANCHOR = "anchor"
    OUTCOME = "outcome"
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"

class ExtendedLabel(Label, total=False):
    """
    Extension of the base meds.Label schema to include a competing_event flag.
    Args:
        competing_event (bool): Whether this label represents a competing event.
        event_category (str): The category of event this label represents.
    """
    competing_event: Optional[bool]
    event_category: Optional[str]
    event_type: EventType = EventType.OUTCOME