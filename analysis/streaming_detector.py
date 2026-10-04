import numpy as np
import pandas as pd


class StreamingAnomalyDetector:
    """Track continuous threshold exceedances one reading at a time."""

    EVENT_COLUMNS = [
        "event_id", "axis", "severity", "threshold",
        "start", "triggered_at", "end",
        "duration_seconds", "peak_residual",
        "status", "close_reason",
    ]

    def __init__(
        self,
        thresholds: pd.DataFrame,
        duration_seconds: float,
        max_gap_seconds: float,
    ):
        if not np.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive.")
        if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive.")
        if thresholds.empty or not thresholds.index.is_unique:
            raise ValueError("Threshold axes must be nonempty and unique.")

        self.axes = list(thresholds.index)
        self.duration_seconds = float(duration_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.limits = {}

        for axis in self.axes:
            min_c = float(thresholds.loc[axis, "candidate_MinC"])
            max_c = float(thresholds.loc[axis, "candidate_MaxC"])

            if not np.isfinite([min_c, max_c]).all():
                raise ValueError("Thresholds must be finite.")
            if not 0 < min_c < max_c:
                raise ValueError("Require 0 < MinC < MaxC.")

            self.limits[(axis, "Alert")] = min_c
            self.limits[(axis, "Error")] = max_c

        self.runs = {key: None for key in self.limits}
        self.events = []
        self.last_timestamp = None
        self.finished = False

    def _close(self, key, reason):
        run = self.runs[key]

        if run is not None and run["event"] is not None:
            run["event"]["status"] = "closed"
            run["event"]["close_reason"] = reason

        self.runs[key] = None

    def update(self, timestamp, residuals) -> dict:
        """Process one reading and return active severity for each axis."""
        if self.finished:
            raise RuntimeError("Create a new detector for a new stream.")

        timestamp = pd.to_datetime(timestamp, utc=True)
        if pd.isna(timestamp):
            raise ValueError("Timestamp is missing.")

        values = {axis: float(residuals[axis]) for axis in self.axes}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Residuals must be finite.")

        if self.last_timestamp is not None:
            gap = (timestamp - self.last_timestamp).total_seconds()

            if gap <= 0:
                raise ValueError("Timestamps must strictly increase.")

            if gap > self.max_gap_seconds:
                for key in self.runs:
                    self._close(key, "recording_gap")

        for key, threshold in self.limits.items():
            axis, severity = key
            value = values[axis]

            if value < threshold:
                self._close(key, "below_threshold")
                continue

            run = self.runs[key]

            if run is None:
                run = {
                    "start": timestamp,
                    "peak": value,
                    "event": None,
                }
                self.runs[key] = run

            run["peak"] = max(run["peak"], value)
            duration = (timestamp - run["start"]).total_seconds()

            if run["event"] is None and duration >= self.duration_seconds:
                event = {
                    "event_id": len(self.events) + 1,
                    "axis": axis,
                    "severity": severity,
                    "threshold": threshold,
                    "start": run["start"],
                    "triggered_at": timestamp,
                    "end": timestamp,
                    "duration_seconds": duration,
                    "peak_residual": run["peak"],
                    "status": "active",
                    "close_reason": None,
                }
                self.events.append(event)
                run["event"] = event

            if run["event"] is not None:
                run["event"].update({
                    "end": timestamp,
                    "duration_seconds": duration,
                    "peak_residual": run["peak"],
                })

        self.last_timestamp = timestamp

        active = {}
        for axis in self.axes:
            active[axis] = "Normal"

            for severity in ("Alert", "Error"):
                run = self.runs[(axis, severity)]
                if run is not None and run["event"] is not None:
                    active[axis] = severity

        return active

    def finish(self):
        """Close remaining records without claiming signal recovery."""
        for key in self.runs:
            self._close(key, "stream_end")
        self.finished = True

    def event_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events, columns=self.EVENT_COLUMNS)