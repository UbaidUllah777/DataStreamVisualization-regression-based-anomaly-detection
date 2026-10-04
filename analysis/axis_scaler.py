import numpy as np
import pandas as pd


class AxisScaler:
    """Scale axis measurements using parameters learned from training data."""

    def __init__(self, axes: list[str]):
        self.axes = list(axes)
        self.parameters_ = None

    def _values(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[self.axes].astype(float)

        if values.empty:
            raise ValueError("The dataset is empty.")

        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Axis values must be finite and non-missing.")

        return values

    def fit(self, training_frame: pd.DataFrame):
        values = self._values(training_frame)

        self.parameters_ = pd.DataFrame({
            "minimum": values.min(),
            "maximum": values.max(),
            "mean": values.mean(),
            "std": values.std(ddof=0),
        })
        self.parameters_["range"] = (
            self.parameters_["maximum"] - self.parameters_["minimum"]
        )

        return self

    def _offset_and_scale(self, method: str):
        if self.parameters_ is None:
            raise RuntimeError("Call fit() on training data first.")

        if method == "minmax":
            offset = self.parameters_["minimum"]
            scale = self.parameters_["range"]
        elif method == "zscore":
            offset = self.parameters_["mean"]
            scale = self.parameters_["std"]
        else:
            raise ValueError("method must be 'minmax' or 'zscore'.")

        # A constant training axis uses a denominator of 1.
        return offset, scale.replace(0, 1.0)

    def transform(
        self,
        frame: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        values = self._values(frame)
        offset, scale = self._offset_and_scale(method)

        result = frame.copy()
        result[self.axes] = (values - offset) / scale
        return result

    def inverse_transform(
        self,
        frame: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        values = self._values(frame)
        offset, scale = self._offset_and_scale(method)

        result = frame.copy()
        result[self.axes] = values * scale + offset
        return result