import uuid

import pandas as pd
from sqlalchemy import text

from persistence.telemetry_store import SCHEMA, TelemetryStore


class StreamRunStore(TelemetryStore):
    """Extend telemetry storage with identifiable synthetic stream runs."""

    def initialise_runs(self):
        # Existing tables and historical records are preserved.
        self.initialise()

        with self.connection() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.stream_run (
                    run_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))

            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.stream_run_sample (
                    sample_id BIGINT PRIMARY KEY
                        REFERENCES {SCHEMA}.sample(sample_id)
                        ON DELETE CASCADE,
                    run_id TEXT NOT NULL
                        REFERENCES {SCHEMA}.stream_run(run_id)
                )
            """))

            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS stream_run_sample_run_idx
                ON {SCHEMA}.stream_run_sample(run_id)
            """))

    def create_run(self, source_file: str) -> str:
        run_id = str(uuid.uuid4())

        with self.connection() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.stream_run (run_id, source_file)
                    VALUES (:run_id, :source_file)
                """),
                {"run_id": run_id, "source_file": source_file},
            )

        return run_id

    def write_record(self, run_id: str, record: pd.DataFrame) -> int:
        if len(record) != 1:
            raise ValueError("Expected exactly one streaming observation.")

        # The measurement and its run association commit together.
        with self.connection() as conn:
            sample_ids = self.write(conn, record, origin="stream")
            sample_id = sample_ids[0]

            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.stream_run_sample
                        (sample_id, run_id)
                    VALUES (:sample_id, :run_id)
                """),
                {"sample_id": sample_id, "run_id": run_id},
            )

        return sample_id

    def read_run(self, run_id: str, after_id: int = 0) -> pd.DataFrame:
        frame = self.ask(
            f"""
                SELECT w.*, r.run_id
                FROM {SCHEMA}.sample_wide AS w
                JOIN {SCHEMA}.stream_run_sample AS r
                    ON r.sample_id = w.sample_id
                WHERE r.run_id = :run_id
                  AND w.sample_id > :after_id
                ORDER BY w.sample_id
            """,
            run_id=run_id,
            after_id=after_id,
        )

        frame["sampled_at"] = pd.to_datetime(
            frame["sampled_at"], utc=True
        )
        return frame

    def run_summary(self, run_id: str) -> pd.DataFrame:
        return self.ask(
            f"""
                SELECT
                    r.run_id,
                    r.source_file,
                    r.created_at,
                    COUNT(DISTINCT s.sample_id) AS sample_count,
                    COUNT(c.joint_id) AS axis_value_count,
                    MIN(s.sampled_at) AS first_reading,
                    MAX(s.sampled_at) AS last_reading
                FROM {SCHEMA}.stream_run AS r
                LEFT JOIN {SCHEMA}.stream_run_sample AS link
                    ON link.run_id = r.run_id
                LEFT JOIN {SCHEMA}.sample AS s
                    ON s.sample_id = link.sample_id
                LEFT JOIN {SCHEMA}.joint_current AS c
                    ON c.sample_id = s.sample_id
                WHERE r.run_id = :run_id
                GROUP BY r.run_id, r.source_file, r.created_at
            """,
            run_id=run_id,
        )