from pathlib import Path

import pandas as pd


class SyntheticCSVExporter:
    """Export synthetic measurements in the historical CSV layout."""

    def export(self, frame: pd.DataFrame, path: str | Path) -> Path:
        path = Path(path)
        axes = [f"j{number}" for number in range(1, 9)]

        required = {"sampled_at", "trait", *axes}
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        output = pd.DataFrame(index=frame.index)
        output["Trait"] = frame["trait"]

        for number in range(1, 15):
            output[f"Axis #{number}"] = (
                frame[f"j{number}"] if number <= 8 else float("nan")
            )

        timestamps = pd.to_datetime(
            frame["sampled_at"], utc=True, errors="raise"
        )
        if timestamps.isna().any():
            raise ValueError("Missing timestamps.")

        output["Time"] = timestamps.map(
            lambda timestamp: timestamp.isoformat().replace("+00:00", "Z")
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        return path