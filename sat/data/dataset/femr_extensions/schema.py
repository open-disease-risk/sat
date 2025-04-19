"""
Extended Label schema for FEMR extensions.
Adds a 'competing_event' flag to the base meds.Label schema.
"""
from typing import Optional
from meds.schema import Label

class ExtendedLabel(Label, total=False):
    """
    Extension of the base meds.Label schema to include a competing_event flag.
    Args:
        competing_event (bool): Whether this label represents a competing event.
        event_type (str): The type of event this label represents.
    """
    competing_event: Optional[bool]
    event_type: Optional[str]
