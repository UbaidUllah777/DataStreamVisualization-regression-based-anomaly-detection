import numpy as np
import pandas as pd

from analysis.streaming_detector import StreamingAnomalyDetector


class SensitivityEvaluator:
    """Evaluate additive shifts defined independently of alert thresholds."""

    def __init__(
        self,
        axes,
        regression,
        training_std,
        thresholds,
        duration_seconds,
        max_gap_seconds,
    ):
        self.axes = list(axes)
        self.regression = regression
        self.training_std = training_std.copy()
        self.thresholds = thresholds.copy()
        self.duration_seconds = duration_seconds
        self.max_gap_seconds = max_gap_seconds

    def evaluate(
        self,
        baseline,
        starts=(1000, 5000, 10000),
        multipliers=(1, 2, 4, 8),
        durations=(4, 12, 30),
    ):
        results = []

        # Each case is a separate experiment with fresh detector state.
        for start_position in starts:
            window = baseline.iloc[
                start_position:start_position + 50
            ].copy().reset_index(drop=True)

            if len(window) != 50:
                raise ValueError("Insufficient baseline samples.")

            intervals = (
                window["sampled_at"].diff().dt.total_seconds().dropna()
            )
            if not np.allclose(intervals, 2.0):
                raise ValueError("Expected a two-second baseline cadence.")

            predictions = self.regression.predict(window)
            baseline_residuals = window[self.axes] - predictions

            for axis in self.axes:
                for multiplier in multipliers:
                    increase = float(
                        multiplier * self.training_std[axis]
                    )

                    for duration in durations:
                        injection_start = window["sampled_at"].iloc[5]
                        injection_end = (
                            injection_start
                            + pd.Timedelta(seconds=duration)
                        )

                        mask = window["sampled_at"].between(
                            injection_start, injection_end
                        )

                        shifted_residuals = baseline_residuals.copy()
                        shifted_residuals.loc[mask, axis] += increase

                        detector = StreamingAnomalyDetector(
                            thresholds=self.thresholds,
                            duration_seconds=self.duration_seconds,
                            max_gap_seconds=self.max_gap_seconds,
                        )

                        for timestamp, values in zip(
                            window["sampled_at"],
                            shifted_residuals[self.axes].to_numpy(),
                        ):
                            detector.update(
                                timestamp,
                                dict(zip(self.axes, values)),
                            )

                        detector.finish()
                        events = detector.event_frame()

                        # A detection must trigger during the injection.
                        matches = events.loc[
                            events["axis"].eq(axis)
                            & events["triggered_at"].between(
                                injection_start, injection_end
                            )
                        ]

                        alerts = matches.loc[
                            matches["severity"].eq("Alert")
                        ]
                        errors = matches.loc[
                            matches["severity"].eq("Error")
                        ]

                        results.append({
                            "baseline_start_row": start_position,
                            "axis": axis,
                            "std_multiplier": multiplier,
                            "added_value": increase,
                            "injection_duration_seconds": duration,
                            "Alert_detected": not alerts.empty,
                            "Error_detected": not errors.empty,
                            "first_detection_delay_seconds": (
                                (
                                    matches["triggered_at"].min()
                                    - injection_start
                                ).total_seconds()
                                if not matches.empty else np.nan
                            ),
                        })

        return pd.DataFrame(results)