import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """Generate synthetic baseline measurements from historical blocks."""

    def __init__(
        self,
        axes: list[str],
        block_size: int = 128,
        relative_noise: float = 0.02,
    ):
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if relative_noise < 0:
            raise ValueError("relative_noise cannot be negative.")

        self.axes = list(axes)
        self.block_size = block_size
        self.relative_noise = relative_noise
        self.training_values_ = None
        self.training_end_ = None

    def fit(self, training_frame: pd.DataFrame):
        ordered = training_frame.sort_values("sampled_at")
        values = ordered[self.axes].to_numpy(dtype=float)

        if len(values) < self.block_size:
            raise ValueError("Training data is shorter than one block.")
        if not np.isfinite(values).all():
            raise ValueError("Training measurements must be finite.")
        if (values < 0).any():
            raise ValueError("This generator expects nonnegative measurements.")

        self.training_values_ = values.copy()
        self.training_end_ = pd.to_datetime(
            ordered["sampled_at"], utc=True
        ).max()
        return self

    def generate(
        self,
        n_samples: int,
        seed: int,
        interval_seconds: float = 2.0,
    ) -> pd.DataFrame:
        if self.training_values_ is None:
            raise RuntimeError("Call fit() before generate().")
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if not np.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive.")

        rng = np.random.default_rng(seed)
        number_of_blocks = (
            n_samples + self.block_size - 1
        ) // self.block_size

        starts = rng.integers(
            0,
            len(self.training_values_) - self.block_size + 1,
            size=number_of_blocks,
        )

        values = np.concatenate([
            self.training_values_[start:start + self.block_size]
            for start in starts
        ])[:n_samples].copy()

        # Mean-one, positive multiplicative noise preserves zero readings.
        sigma = self.relative_noise
        factors = rng.lognormal(
            mean=-0.5 * sigma**2,
            sigma=sigma,
            size=values.shape,
        )
        values *= factors

        timestamps = self.training_end_ + pd.to_timedelta(
            np.arange(1, n_samples + 1) * interval_seconds,
            unit="s",
        )

        result = pd.DataFrame(values, columns=self.axes)
        result.insert(0, "sampled_at", timestamps)
        result["trait"] = "current"
        result["state"] = np.where(
            result[self.axes].ne(0).any(axis=1),
            "RUNNING",
            "IDLE",
        )
        return result[["sampled_at", "trait", "state", *self.axes]]

    def compare(self, synthetic_frame: pd.DataFrame) -> pd.DataFrame:
        if self.training_values_ is None:
            raise RuntimeError("Call fit() before compare().")

        historical = pd.DataFrame(
            self.training_values_, columns=self.axes
        )
        synthetic = synthetic_frame[self.axes]

        report = pd.DataFrame({
            "training_mean": historical.mean(),
            "synthetic_mean": synthetic.mean(),
            "training_std": historical.std(ddof=0),
            "synthetic_std": synthetic.std(ddof=0),
            "training_zero_pct": historical.eq(0).mean() * 100,
            "synthetic_zero_pct": synthetic.eq(0).mean() * 100,
        })

        report["mean_change_pct"] = (
            (report["synthetic_mean"] - report["training_mean"])
            / report["training_mean"].replace(0, np.nan)
            * 100
        )
        report["std_change_pct"] = (
            (report["synthetic_std"] - report["training_std"])
            / report["training_std"].replace(0, np.nan)
            * 100
        )
        return report