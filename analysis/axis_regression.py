import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class AxisRegression:
    """Fit one elapsed-time regression for each axis."""

    def __init__(self, axes: list[str]):
        self.axes = list(axes)
        self.models_ = {}
        self.summary_ = None

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame[["elapsed_seconds"]].astype(float)

        if features.empty:
            raise ValueError("The dataset is empty.")

        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("Elapsed time must contain finite values.")

        return features

    def fit(self, training_frame: pd.DataFrame):
        features = self._features(training_frame)

        if features["elapsed_seconds"].nunique() < 2:
            raise ValueError("Training requires at least two distinct times.")

        targets = training_frame[self.axes].astype(float)
        if not np.isfinite(targets.to_numpy()).all():
            raise ValueError("Training axis values must be finite.")

        models = {}
        rows = []

        for axis in self.axes:
            observed = targets[axis]
            model = LinearRegression()
            model.fit(features, observed)
            predicted = model.predict(features)

            models[axis] = model
            rows.append({
                "axis": axis,
                "slope_per_second": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "training_r2": r2_score(observed, predicted),
                "training_mae": mean_absolute_error(observed, predicted),
                "training_rmse": np.sqrt(
                    mean_squared_error(observed, predicted)
                ),
            })

        self.models_ = models
        self.summary_ = pd.DataFrame(rows).set_index("axis")
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("Call fit() before predict().")

        features = self._features(frame)

        return pd.DataFrame(
            {
                axis: self.models_[axis].predict(features)
                for axis in self.axes
            },
            index=frame.index,
        )

    def plot_fits(self, frame: pd.DataFrame):
        ordered = frame.sort_values("elapsed_seconds")
        predictions = self.predict(ordered)

        # Display hours for readability; models still use seconds.
        hours = ordered["elapsed_seconds"] / 3600

        figure, panels = plt.subplots(
            4, 2, figsize=(14, 14), sharex=True
        )

        for axis, panel in zip(self.axes, panels.flat):
            panel.scatter(
                hours,
                ordered[axis],
                s=3,
                alpha=0.2,
                color="steelblue",
                label="Observed",
                rasterized=True,
            )
            panel.plot(
                hours,
                predictions[axis],
                color="darkorange",
                linewidth=2,
                label="Linear regression",
            )
            panel.set_title(f"Axis #{axis.removeprefix('j')}")
            panel.set_ylabel("Measurement (original units)")
            panel.grid(alpha=0.2)

        for panel in panels[-1]:
            panel.set_xlabel("Elapsed hours from training reference")

        handles, labels = panels.flat[0].get_legend_handles_labels()

        figure.suptitle(
            "Historical measurements and time-based regression",
            y=0.995,
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        return figure