from enum import StrEnum

class RecordingMode(StrEnum):
    """Enum for recording modes."""
    TRIGGER = "triggered"
    CONTINUOUS = "continuous"