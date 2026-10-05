import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from acquisition.controller_feed import JOINTS
from persistence.event_logger import EventCSVLogger


class DashboardService:
    """Load and verify a completed streaming run for the dashboard."""

    def __init__(self, store, results_root: str | Path):
        self.store = store
        self.results_root = Path(results_root)

    @staticmethod
    def _records(frame):
        # Convert timestamps and numeric types into JSON-compatible values.
        return json.loads(
            frame.to_json(orient="records", date_format="iso")
        )

    def load_run(self, run_id: str) -> dict:
        # Canonical UUID also prevents arbitrary directory paths.
        run_id = str(uuid.UUID(run_id))
        run_directory = self.results_root / run_id

        predictions = pd.read_csv(
            run_directory / "predictions.csv"
        )
        predictions["sampled_at"] = pd.to_datetime(
            predictions["sampled_at"], utc=True
        )

        events = EventCSVLogger(
            run_directory / "events.csv"
        ).load()

        configuration = pd.read_csv(
            run_directory / "configuration.csv"
        )

        for frame in (predictions, events):
            if not frame["run_id"].eq(run_id).all():
                raise ValueError("Saved records belong to a different run.")

        if predictions.duplicated(["sample_id", "axis"]).any():
            raise ValueError("Duplicate prediction-log keys.")

        if (
            configuration["axis"].duplicated().any()
            or set(configuration["axis"]) != set(JOINTS)
        ):
            raise ValueError("Expected one configuration row per axis.")

        database = self.store.read_run(run_id)
        summary = self.store.run_summary(run_id)

        if database.empty or summary.empty:
            raise ValueError("No persisted measurements found for this run.")

        database_long = database.melt(
            id_vars=["sample_id", "sampled_at"],
            value_vars=JOINTS,
            var_name="axis",
            value_name="database_observed",
        )

        matched = predictions.merge(
            database_long,
            on=["sample_id", "axis"],
            how="outer",
            suffixes=("_log", "_database"),
            indicator=True,
            validate="one_to_one",
        )

        if not matched["_merge"].eq("both").all():
            raise ValueError("Database and prediction-log keys differ.")

        if not matched["sampled_at_log"].eq(
            matched["sampled_at_database"]
        ).all():
            raise ValueError("Database and logged timestamps differ.")

        np.testing.assert_allclose(
            matched["observed"],
            matched["database_observed"],
            rtol=1e-10,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            predictions["observed"] - predictions["predicted"],
            predictions["residual"],
            rtol=1e-10,
            atol=1e-10,
        )

        counts = {
            "samples": len(database),
            "axis_measurements": len(database_long),
            "Alert": int(events["severity"].eq("Alert").sum()),
            "Error": int(events["severity"].eq("Error").sum()),
        }

        return {
            "run_id": run_id,
            "mode": "Completed-run review",
            "database_verified_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "summary": self._records(summary)[0],
            "counts": counts,
            "configuration": self._records(configuration),
            "predictions": self._records(predictions),
            "events": self._records(events),
        }