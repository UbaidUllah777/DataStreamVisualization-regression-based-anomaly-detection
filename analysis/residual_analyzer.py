import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ResidualAnalyzer:
    """Analyze aligned observations and predictions for each axis."""

    def __init__(
        self,
        observations: pd.DataFrame,
        predictions: pd.DataFrame,
        axes: list[str],
    ):
        self.axes = list(axes)

        if not observations.index.equals(predictions.index):
            raise ValueError("Observation and prediction indexes must match.")

        observed = observations[self.axes].astype(float)
        predicted = predictions[self.axes].astype(float)

        if observed.empty:
            raise ValueError("Observations are empty.")

        if not np.isfinite(observed.to_numpy()).all():
            raise ValueError("Observations must contain finite values.")

        if not np.isfinite(predicted.to_numpy()).all():
            raise ValueError("Predictions must contain finite values.")

        self.observations = observations.copy()
        self.residuals = observed - predicted

    def summary(self) -> pd.DataFrame:
        residuals = self.residuals

        q1 = residuals.quantile(0.25)
        q3 = residuals.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr

        return pd.DataFrame({
            "mean": residuals.mean(),
            "std": residuals.std(ddof=0),
            "minimum": residuals.min(),
            "median": residuals.median(),
            "q95": residuals.quantile(0.95),
            "q99": residuals.quantile(0.99),
            "maximum": residuals.max(),
            "upper_iqr_fence": upper_fence,
            "above_upper_fence": residuals.gt(upper_fence).sum(),
            "above_upper_fence_pct": (
                residuals.gt(upper_fence).mean() * 100
            ),
        }).rename_axis("axis")

    def state_summary(self) -> pd.DataFrame:
        """Compare residuals during recorded idle and running states."""
        rows = []

        for state, group in self.observations.groupby("state"):
            state_residuals = self.residuals.loc[group.index]

            for axis in self.axes:
                values = state_residuals[axis]
                rows.append({
                    "state": state,
                    "axis": axis,
                    "samples": len(values),
                    "mean_residual": values.mean(),
                    "median_residual": values.median(),
                    "q95_residual": values.quantile(0.95),
                })

        return pd.DataFrame(rows).set_index(["state", "axis"])

    def plot_distributions(self):
        figure, panels = plt.subplots(4, 2, figsize=(14, 13))

        for axis, panel in zip(self.axes, panels.flat):
            panel.hist(
                self.residuals[axis],
                bins=60,
                color="steelblue",
                edgecolor="white",
                linewidth=0.3,
            )
            panel.axvline(0, color="black", linestyle="--", linewidth=1)
            panel.set_yscale("log")
            panel.set_title(f"Axis #{axis.removeprefix('j')}")
            panel.set_xlabel("Residual (original units)")
            panel.set_ylabel("Sample count (log scale)")
            panel.grid(axis="y", alpha=0.2)

        figure.suptitle(
            "Training residual distributions — dashed line marks zero"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        return figure

    def plot_over_time(self):
        order = self.observations.sort_values("elapsed_seconds").index
        hours = self.observations.loc[order, "elapsed_seconds"] / 3600

        figure, panels = plt.subplots(
            4, 2, figsize=(14, 13), sharex=True
        )

        for axis, panel in zip(self.axes, panels.flat):
            panel.scatter(
                hours,
                self.residuals.loc[order, axis],
                s=3,
                alpha=0.2,
                color="steelblue",
                rasterized=True,
            )
            panel.axhline(0, color="black", linestyle="--", linewidth=1)
            panel.set_title(f"Axis #{axis.removeprefix('j')}")
            panel.set_ylabel("Residual (original units)")
            panel.grid(alpha=0.2)

        for panel in panels[-1]:
            panel.set_xlabel("Elapsed hours from training reference")

        figure.suptitle(
            "Training residuals over time — dashed line marks zero"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        return figure