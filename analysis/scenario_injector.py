import numpy as np
import pandas as pd


class ScenarioInjector:
    """Create controlled scenarios for candidate detection-rule checks."""

    def __init__(self, axes: list[str]):
        self.axes = list(axes)

    def inject(
        self,
        baseline: pd.DataFrame,
        predictions: pd.DataFrame,
        thresholds: pd.DataFrame,
    ):
        if not baseline.index.equals(predictions.index):
            raise ValueError("Baseline and prediction indexes must match.")

        if len(baseline) < 412:
            raise ValueError("At least 412 baseline samples are required.")

        frame = baseline.copy().reset_index(drop=True)
        predicted = predictions[self.axes].reset_index(drop=True)

        timestamps = pd.to_datetime(frame["sampled_at"], utc=True)
        intervals = timestamps.diff().dt.total_seconds().dropna()

        if not np.allclose(intervals, 2.0):
            raise ValueError("These scenarios require a two-second cadence.")

        if not np.isfinite(predicted.to_numpy()).all():
            raise ValueError("Predictions must be finite.")

        if predicted.lt(0).any().any():
            raise ValueError("Scenario construction requires nonnegative predictions.")

        specifications = [
            ("brief_high", 100, 102, "high"),
            ("sustained_moderate", 200, 210, "moderate"),
            ("sustained_high", 300, 310, "high"),
            ("gap_interrupted_high", 400, 410, "high"),
        ]

        labels = []

        for axis in self.axes:
            min_c = float(thresholds.loc[axis, "candidate_MinC"])
            max_c = float(thresholds.loc[axis, "candidate_MaxC"])

            if not np.isfinite([min_c, max_c]).all():
                raise ValueError("Thresholds must be finite.")
            if not 0 < min_c < max_c:
                raise ValueError("Require 0 < MinC < MaxC.")

            for name, start, end, magnitude in specifications:
                target_residual = (
                    (min_c + max_c) / 2
                    if magnitude == "moderate"
                    else max_c + 0.1 * max_c
                )

                # Controlled boundary observations isolate each scenario.
                for boundary in (start - 1, end + 1):
                    frame.loc[boundary, axis] = predicted.loc[boundary, axis]

                frame.loc[start:end, axis] = (
                    predicted.loc[start:end, axis] + target_residual
                )

                labels.append({
                    "scenario": name,
                    "axis": axis,
                    "start": timestamps.iloc[start],
                    "end": timestamps.iloc[end],
                    "span_seconds": (
                        timestamps.iloc[end] - timestamps.iloc[start]
                    ).total_seconds(),
                    "target_residual": target_residual,
                    "observed_samples": (
                        8 if name == "gap_interrupted_high"
                        else end - start + 1
                    ),
                    "longest_segment_seconds": (
                        6.0 if name == "gap_interrupted_high"
                        else (end - start) * 2.0
                    ),
                })

        # Retain rows 400–403 and 407–410:
        # each segment lasts 6 seconds; the intervening gap is 8 seconds.
        frame = frame.drop(index=[404, 405, 406]).reset_index(drop=True)

        frame["state"] = np.where(
            frame[self.axes].ne(0).any(axis=1),
            "RUNNING",
            "IDLE",
        )

        return frame, pd.DataFrame(labels)