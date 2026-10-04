from pathlib import Path

import pandas as pd

from analysis.streaming_detector import StreamingAnomalyDetector


class EventCSVLogger:
    """Save and reload one run's detector-event snapshot."""

    TIMESTAMP_COLUMNS = ["start", "triggered_at", "end"]

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def save(self, events: pd.DataFrame, run_id: str) -> Path:
        if not run_id.strip():
            raise ValueError("run_id cannot be empty.")

        required = StreamingAnomalyDetector.EVENT_COLUMNS
        missing = set(required) - set(events.columns)

        if missing:
            raise ValueError(f"Missing event columns: {sorted(missing)}")

        records = events[required].copy()

        if records["event_id"].duplicated().any():
            raise ValueError("Event IDs must be unique within a run.")

        records.insert(0, "run_id", run_id)

        for column in self.TIMESTAMP_COLUMNS:
            records[column] = pd.to_datetime(
                records[column], utc=True, errors="raise"
            )
            if records[column].isna().any():
                raise ValueError(f"Missing event timestamps: {column}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace this run's snapshot without appending duplicate events.
        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )
        records.to_csv(temporary_path, index=False)
        temporary_path.replace(self.output_path)

        return self.output_path

    def load(self) -> pd.DataFrame:
        records = pd.read_csv(
            self.output_path,
            dtype={"run_id": "string"},
            float_precision="round_trip",
        )

        for column in self.TIMESTAMP_COLUMNS:
            records[column] = pd.to_datetime(
                records[column], utc=True, errors="raise"
            )

        return records