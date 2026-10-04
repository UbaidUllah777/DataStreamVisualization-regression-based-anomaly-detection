import numpy as np
import pandas as pd


class TrainingDataInspector:
    """Inspect historical axis measurements without modifying the source."""

    def __init__(self, frame: pd.DataFrame, axes: list[str]):
        self.axes = list(axes)
        self.frame = (
            frame.copy()
            .sort_values("sampled_at")
            .reset_index(drop=True)
        )

    def axis_summary(self) -> pd.DataFrame:
        """Describe each axis, including missing and non-finite values."""
        values = self.frame[self.axes]

        summary = values.describe().T.rename(
            columns={"50%": "median", "25%": "q1", "75%": "q3"}
        )

        summary["missing_count"] = values.isna().sum()
        summary["infinite_count"] = np.isinf(values).sum()
        summary["zero_pct"] = values.eq(0).mean() * 100

        return summary[
            [
                "count", "missing_count", "infinite_count",
                "mean", "std", "min", "q1", "median",
                "q3", "max", "zero_pct"
            ]
        ]

    def activity_summary(self) -> pd.DataFrame:
        """Summarize the recorded operating-state labels."""
        counts = self.frame["state"].value_counts()

        return pd.DataFrame({
            "samples": counts,
            "percentage": counts / len(self.frame) * 100,
        }).rename_axis("state")

    def sampling_summary(self) -> pd.DataFrame:
        """Describe elapsed time between consecutive measurements."""
        intervals = (
            self.frame["sampled_at"]
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        return pd.DataFrame([{
            "interval_count": len(intervals),
            "minimum_seconds": intervals.min(),
            "median_seconds": intervals.median(),
            "mean_seconds": intervals.mean(),
            "maximum_seconds": intervals.max(),
            "gaps_over_10_seconds": int(intervals.gt(10).sum()),
        }])

    def recording_gaps(self, cutoff_seconds: float = 10.0) -> pd.DataFrame:
        """List recording gaps above a descriptive inspection cutoff."""
        timestamps = self.frame["sampled_at"]

        gaps = pd.DataFrame({
            "previous_timestamp": timestamps.shift(),
            "next_timestamp": timestamps,
            "gap_seconds": timestamps.diff().dt.total_seconds(),
        })

        return gaps.loc[
            gaps["gap_seconds"] > cutoff_seconds
        ].reset_index(drop=True)