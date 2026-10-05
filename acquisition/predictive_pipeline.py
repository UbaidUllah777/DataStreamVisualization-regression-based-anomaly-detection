import time
from pathlib import Path

import numpy as np
import pandas as pd

from acquisition.controller_feed import JOINTS, StreamingSimulator
from analysis.streaming_detector import StreamingAnomalyDetector
from persistence.event_logger import EventCSVLogger


class PredictiveStreamingPipeline:
    """Persist each reading, query it back, then predict and detect."""

    def __init__(
        self,
        store,
        regression,
        time_preprocessor,
        axis_scaler,
        thresholds,
        duration_seconds,
        max_gap_seconds,
    ):
        self.store = store
        self.regression = regression
        self.time_preprocessor = time_preprocessor
        self.axis_scaler = axis_scaler
        self.thresholds = thresholds.copy()
        self.duration_seconds = duration_seconds
        self.max_gap_seconds = max_gap_seconds

    def run(
        self,
        csv_path,
        run_id,
        output_directory,
        interval_seconds=2.0,
    ):
        summary = self.store.run_summary(run_id)

        if summary.empty:
            raise ValueError("Run ID does not exist.")
        if int(summary.iloc[0]["sample_count"]) != 0:
            raise ValueError(
                "This run already contains samples. "
                "Create a new run before replaying."
            )

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        logger = EventCSVLogger(output_directory / "events.csv")
        detector = StreamingAnomalyDetector(
            thresholds=self.thresholds,
            duration_seconds=self.duration_seconds,
            max_gap_seconds=self.max_gap_seconds,
        )
        simulator = StreamingSimulator(
            csv_path,
            interval=interval_seconds,
        )

        last_sample_id = 0
        processed = 0
        prediction_rows = []
        started = time.perf_counter()

        while True:
            record = simulator.nextDataPoint()
            if record is None:
                break

            inserted_id = self.store.write_record(run_id, record)

            # Predictions use committed data queried from Neon.
            stored = self.store.read_run(
                run_id,
                after_id=last_sample_id,
            )

            if len(stored) != 1:
                raise RuntimeError(
                    "Expected exactly one newly committed sample."
                )

            if int(stored.iloc[0]["sample_id"]) != inserted_id:
                raise RuntimeError("Retrieved sample ID does not match.")

            prepared = self.time_preprocessor.transform(stored)

            # Apply historical scaling parameters to each streamed sample.
            normalized = self.axis_scaler.transform(
                prepared, method="minmax"
            )
            standardized = self.axis_scaler.transform(
                prepared, method="zscore"
            )

            # Regression targets and detection thresholds use original units.
            predictions = self.regression.predict(prepared)
            residuals = prepared[JOINTS] - predictions

            timestamp = prepared.iloc[0]["sampled_at"]
            status = detector.update(
                timestamp,
                residuals.iloc[0].to_dict(),
            )

            for axis in JOINTS:
                prediction_rows.append({
                    "run_id": run_id,
                    "sample_id": inserted_id,
                    "sampled_at": timestamp,
                    "axis": axis,
                    "observed": float(prepared.iloc[0][axis]),
                    "normalized": float(normalized.iloc[0][axis]),
                    "standardized": float(standardized.iloc[0][axis]),
                    "predicted": float(predictions.iloc[0][axis]),
                    "residual": float(residuals.iloc[0][axis]),
                    "detector_status": status[axis],
                })

            # Save the latest event snapshot, including active durations.
            logger.save(detector.event_frame(), run_id)

            last_sample_id = inserted_id
            processed += 1

            if processed % 50 == 0:
                print(
                    f"Processed {processed} readings; "
                    f"recorded {len(detector.events)} threshold events",
                    flush=True,
                )

        detector.finish()
        events = detector.event_frame()
        logger.save(events, run_id)

        predictions_log = pd.DataFrame(prediction_rows)
        predictions_log.to_csv(
            output_directory / "predictions.csv",
            index=False,
        )

        elapsed = time.perf_counter() - started

        print(f"Completed: {processed} readings")
        print(f"Elapsed wall-clock time: {elapsed:.1f} seconds")
        print(f"Events saved: {len(events)}")

        return predictions_log, events