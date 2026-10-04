import numpy as np
import pandas as pd


class ThresholdExplorer:
    """Compare positive residual thresholds and observed exceedance runs."""

    def __init__(
        self,
        observations: pd.DataFrame,
        residuals: pd.DataFrame,
        axes: list[str],
        max_gap_seconds: float,
    ):
        if not observations.index.equals(residuals.index):
            raise ValueError("Observation and residual indexes must match.")

        if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive.")

        self.axes = list(axes)
        order = observations.sort_values("sampled_at").index

        self.timestamps = pd.to_datetime(
            observations.loc[order, "sampled_at"],
            utc=True,
        ).reset_index(drop=True)

        self.residuals = (
            residuals.loc[order, self.axes]
            .astype(float)
            .reset_index(drop=True)
        )

        if self.timestamps.empty:
            raise ValueError("The dataset is empty.")

        if self.timestamps.isna().any() or self.timestamps.duplicated().any():
            raise ValueError("Timestamps must be present and unique.")

        if not np.isfinite(self.residuals.to_numpy()).all():
            raise ValueError("Residuals must be finite.")

        self.max_gap_seconds = float(max_gap_seconds)

    def exceedance_runs(
        self,
        axis: str,
        threshold: float,
    ) -> pd.DataFrame:
        """Return runs with residual >= threshold and acceptable gaps."""
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Threshold must be finite and positive.")

        values = self.residuals[axis]
        above = values.ge(threshold)

        gaps = self.timestamps.diff().dt.total_seconds()
        starts = above & (
            ~above.shift(fill_value=False)
            | gaps.gt(self.max_gap_seconds)
        )
        run_ids = starts.cumsum()

        selected = pd.DataFrame({
            "run_id": run_ids[above],
            "timestamp": self.timestamps[above],
            "residual": values[above],
        })

        runs = (
            selected.groupby("run_id")
            .agg(
                start=("timestamp", "min"),
                end=("timestamp", "max"),
                samples=("timestamp", "size"),
                peak_residual=("residual", "max"),
            )
            .reset_index(drop=True)
        )

        runs["duration_seconds"] = (
            runs["end"] - runs["start"]
        ).dt.total_seconds()

        return runs

    def compare(
        self,
        quantiles=(0.95, 0.975, 0.99),
        durations=(4, 8, 12, 20),
    ) -> pd.DataFrame:
        rows = []

        for axis in self.axes:
            for quantile in quantiles:
                threshold = float(
                    self.residuals[axis].quantile(quantile)
                )

                if threshold <= 0:
                    raise ValueError(
                        f"{axis}: quantile {quantile} is not positive."
                    )

                runs = self.exceedance_runs(axis, threshold)
                observed_durations = runs["duration_seconds"]

                row = {
                    "axis": axis,
                    "quantile": quantile,
                    "threshold": threshold,
                    "exceeding_samples": int(
                        self.residuals[axis].ge(threshold).sum()
                    ),
                    "run_count": len(runs),
                    "longest_run_seconds": (
                        float(observed_durations.max())
                        if not runs.empty else 0.0
                    ),
                }

                for duration in durations:
                    row[f"runs_ge_{duration}s"] = int(
                        observed_durations.ge(duration).sum()
                    )

                rows.append(row)

        return pd.DataFrame(rows)