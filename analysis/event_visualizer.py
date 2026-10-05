import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class EventVisualizer:
    """Plot streamed measurements, predictions, and detected events."""

    def __init__(self, axes):
        self.axes = list(axes)

    def plot(self, prediction_log, events):
        readings = prediction_log.copy()
        event_data = events.copy()

        readings["sampled_at"] = pd.to_datetime(
            readings["sampled_at"], utc=True
        )
        for column in ("start", "triggered_at", "end"):
            event_data[column] = pd.to_datetime(
                event_data[column], utc=True
            )

        reference = readings["sampled_at"].min()

        figure, panels = plt.subplots(
            4, 2, figsize=(15, 15), sharex=True
        )

        styles = {
            "Alert": {"color": "#b36b00", "marker": "o"},
            "Error": {"color": "#c62828", "marker": "^"},
        }

        for axis, panel in zip(self.axes, panels.flat):
            axis_readings = (
                readings.loc[readings["axis"].eq(axis)]
                .sort_values("sampled_at")
                .copy()
            )
            minutes = (
                axis_readings["sampled_at"] - reference
            ).dt.total_seconds() / 60

            # Scatter avoids drawing a measured line across recording gaps.
            panel.scatter(
                minutes,
                axis_readings["observed"],
                s=10,
                alpha=0.5,
                color="steelblue",
            )
            panel.plot(
                minutes,
                axis_readings["predicted"],
                color="black",
                linewidth=1.5,
            )

            axis_events = event_data.loc[
                event_data["axis"].eq(axis)
            ]

            for event in axis_events.itertuples(index=False):
                style = styles[event.severity]

                trigger_reading = axis_readings.loc[
                    axis_readings["sampled_at"].eq(event.triggered_at)
                ]
                if len(trigger_reading) != 1:
                    raise ValueError(
                        "Each event trigger must match one observation."
                    )

                trigger_minutes = (
                    event.triggered_at - reference
                ).total_seconds() / 60

                start_minutes = (
                    event.start - reference
                ).total_seconds() / 60

                end_minutes = (
                    event.end - reference
                ).total_seconds() / 60

                observed = float(trigger_reading.iloc[0]["observed"])

                panel.axvspan(
                    start_minutes,
                    end_minutes,
                    color=style["color"],
                    alpha=0.08,
                )

                # A large hollow Alert marker remains visible behind
                # the smaller Error marker when triggers coincide.
                panel.scatter(
                    [trigger_minutes],
                    [observed],
                    s=110 if event.severity == "Alert" else 55,
                    marker=style["marker"],
                    facecolors=(
                        "none" if event.severity == "Alert"
                        else style["color"]
                    ),
                    edgecolors=style["color"],
                    linewidths=1.5,
                    zorder=5 if event.severity == "Alert" else 6,
                )

                offset = (
                    (-12, 24) if event.severity == "Alert"
                    else (12, -30)
                )

                panel.annotate(
                    f"{event.severity}: {event.duration_seconds:.0f}s",
                    xy=(trigger_minutes, observed),
                    xytext=offset,
                    textcoords="offset points",
                    ha="right" if event.severity == "Alert" else "left",
                    fontsize=8,
                    color=style["color"],
                    arrowprops={
                        "arrowstyle": "-",
                        "color": style["color"],
                    },
                )

            panel.set_title(f"Axis #{axis.removeprefix('j')}")
            panel.set_ylabel("Measurement (original units)")
            panel.grid(alpha=0.2)
            panel.margins(y=0.3)

        for panel in panels[-1]:
            panel.set_xlabel("Minutes since first streamed measurement")

        legend = [
            Line2D(
                [], [], color="steelblue", marker="o",
                linestyle="None", label="Observed"
            ),
            Line2D([], [], color="black", label="Regression prediction"),
            Line2D(
                [], [], color="#b36b00", marker="o",
                markerfacecolor="none", linestyle="None",
                label="Alert trigger"
            ),
            Line2D(
                [], [], color="#c62828", marker="^",
                linestyle="None", label="Error trigger"
            ),
        ]

        figure.suptitle(
            "Verified synthetic stream: predictions and detected events",
            y=0.995,
        )
        figure.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=4,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        return figure