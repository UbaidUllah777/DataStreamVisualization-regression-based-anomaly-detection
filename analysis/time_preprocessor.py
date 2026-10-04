# This class learns the training start time and reuses it whenever it transforms data. That ensures future testing data uses the same time reference.

import pandas as pd


class TimePreprocessor:
    """Create elapsed seconds using one fixed training-time reference."""

    def __init__(self):
        self.reference_time_ = None

    def _timestamps(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            raise ValueError("The dataset is empty.")

        timestamps = pd.to_datetime(
            frame["sampled_at"],
            utc=True,
            errors="raise",
        )

        if timestamps.isna().any():
            raise ValueError("Timestamps contain missing values.")

        if timestamps.duplicated().any():
            raise ValueError("Timestamps contain duplicates.")

        return timestamps

    def fit(self, training_frame: pd.DataFrame):
        timestamps = self._timestamps(training_frame)
        self.reference_time_ = timestamps.min()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.reference_time_ is None:
            raise RuntimeError("Call fit() on training data first.")

        timestamps = self._timestamps(frame)

        prepared = frame.copy()
        prepared["sampled_at"] = timestamps
        prepared["elapsed_seconds"] = (
            timestamps - self.reference_time_
        ).dt.total_seconds()

        return (
            prepared.sort_values("sampled_at")
            .reset_index(drop=True)
        )

    def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_frame).transform(training_frame)