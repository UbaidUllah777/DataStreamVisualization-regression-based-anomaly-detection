import argparse

from flask import Flask, render_template

from monitor_web.dashboard_service import DashboardService
from persistence.stream_run_store import StreamRunStore
from persistence.telemetry_store import PROJECT_ROOT


class RegressionDashboard:
    """Serve a database-verified completed-run dashboard."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.service = DashboardService(
            store=StreamRunStore(),
            results_root=PROJECT_ROOT / "results" / "stream_runs",
        )
        self.app = Flask(__name__)
        self.app.add_url_rule("/", view_func=self.index)

    def index(self):
        payload = self.service.load_run(self.run_id)
        return render_template(
            "regression_dashboard.html",
            payload=payload,
        )

    def run(self):
        self.app.run(
            host="127.0.0.1",
            port=8061,
            debug=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    RegressionDashboard(args.run_id).run()