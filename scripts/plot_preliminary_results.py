#!/usr/bin/env python3
"""Plot presentation figures for the current preliminary GovSim results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCENARIO_LABELS = {
    "fishing": "Fishery",
    "sheep": "Pasture",
}

EXPERIMENT_LABELS = {
    "fish_no_communication": "Fishery\nNo comm.",
    "fish_free_communication": "Fishery\nFree comm.",
    "fish_costly_punishment": "Fishery\nPunishment",
    "sheep_no_communication": "Pasture\nNo comm.",
    "sheep_free_communication": "Pasture\nFree comm.",
    "sheep_costly_punishment": "Pasture\nPunishment",
}

INSTITUTION_LABELS = {
    "no_communication": "No comm.",
    "free_communication": "Free comm.",
    "costly_punishment": "Punishment",
}

STATUS_COLORS = {
    "completed": "#2f855a",
    "running": "#b7791f",
    "timed_out": "#c53030",
    "not_run": "#718096",
}

STATUS_LABELS = {
    "completed": "Done",
    "running": "Running",
    "timed_out": "Timeout",
    "not_run": "Not run",
}

BAR_COLOR = "#3b6ea8"
ACCENT_COLOR = "#d95f02"
GRID_COLOR = "#d9dee7"
TEXT_COLOR = "#20242a"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sort_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {
        "fish_no_communication": 0,
        "fish_free_communication": 1,
        "fish_costly_punishment": 2,
        "sheep_no_communication": 3,
        "sheep_free_communication": 4,
        "sheep_costly_punishment": 5,
    }
    return sorted(rows, key=lambda row: order.get(row["experiment"], 99))


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4b2")
    ax.spines["bottom"].set_color("#9aa4b2")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def add_value_labels(ax, bars, fmt="{:.0f}", y_offset=0.02) -> None:
    ylim_top = ax.get_ylim()[1]
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ylim_top * y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_COLOR,
        )


def plot_outcomes(rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = sort_result_rows(rows)
    labels = [EXPERIMENT_LABELS[row["experiment"]] for row in rows]
    rounds = [float(row["rounds_survived"]) for row in rows]
    final_resource = [float(row["final_resource"]) for row in rows]
    overuse = [float(row["overuse_actions"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), constrained_layout=True)
    fig.suptitle("Preliminary GovSim Outcomes", fontsize=13, color=TEXT_COLOR)

    bars = axes[0].bar(labels, rounds, color=BAR_COLOR)
    axes[0].set_title("Rounds Survived", fontsize=10)
    axes[0].set_ylim(0, max(3, max(rounds) + 1))
    style_axis(axes[0])
    add_value_labels(axes[0], bars)

    bars = axes[1].bar(labels, final_resource, color=BAR_COLOR)
    axes[1].set_title("Final Resource", fontsize=10)
    axes[1].set_ylim(0, 105)
    style_axis(axes[1])
    add_value_labels(axes[1], bars)

    bars = axes[2].bar(labels, overuse, color=ACCENT_COLOR)
    axes[2].set_title("Overuse Actions", fontsize=10)
    axes[2].set_ylim(0, max(overuse) + 1.5)
    style_axis(axes[2])
    add_value_labels(axes[2], bars)

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=0)

    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.06, hspace=0.05, wspace=0.05)
    fig.savefig(out_dir / "preliminary_outcomes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_welfare_inequality(rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = sort_result_rows(rows)
    labels = [EXPERIMENT_LABELS[row["experiment"]] for row in rows]
    welfare = [float(row["net_social_welfare"]) for row in rows]
    gini = [float(row["net_gini"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)
    fig.suptitle("Preliminary Welfare and Inequality", fontsize=13, color=TEXT_COLOR)

    bars = axes[0].bar(labels, welfare, color=BAR_COLOR)
    axes[0].set_title("Net Social Welfare", fontsize=10)
    axes[0].set_ylim(0, max(110, max(welfare) + 10))
    style_axis(axes[0])
    add_value_labels(axes[0], bars)

    bars = axes[1].bar(labels, gini, color=ACCENT_COLOR)
    axes[1].set_title("Net Gini", fontsize=10)
    axes[1].set_ylim(0, max(gini) + 0.08)
    style_axis(axes[1])
    add_value_labels(axes[1], bars, fmt="{:.3f}", y_offset=0.03)

    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.06, hspace=0.05, wspace=0.05)
    fig.savefig(out_dir / "preliminary_welfare_inequality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_status(rows: list[dict[str, str]], out_dir: Path) -> None:
    scenarios = ["fishing", "sheep"]
    institutions = ["no_communication", "free_communication", "costly_punishment"]
    row_by_key = {(row["scenario"], row["institution"]): row for row in rows}

    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    ax.set_title("Preliminary Matrix Coverage", fontsize=13, color=TEXT_COLOR)

    for y, scenario in enumerate(scenarios):
        for x, institution in enumerate(institutions):
            row = row_by_key[(scenario, institution)]
            status = row["status"]
            ax.scatter(
                x,
                y,
                s=1600,
                marker="s",
                color=STATUS_COLORS[status],
                edgecolor="white",
                linewidth=1.5,
            )
            ax.text(
                x,
                y,
                STATUS_LABELS[status],
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
            )

    ax.set_xticks(range(len(institutions)), [INSTITUTION_LABELS[i] for i in institutions])
    ax.set_yticks(range(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_xlim(-0.6, len(institutions) - 0.4)
    ax.set_ylim(-0.6, len(scenarios) - 0.4)
    ax.invert_yaxis()
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    handles = [
        Patch(facecolor=STATUS_COLORS[key], edgecolor="white", label=key.replace("_", " ").title())
        for key in ["completed", "running", "timed_out", "not_run"]
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=8)

    fig.savefig(out_dir / "preliminary_matrix_status.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("analysis_outputs/preliminary_results.csv"))
    parser.add_argument("--status", type=Path, default=Path("analysis_outputs/preliminary_status.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_rows = load_csv(args.results)
    status_rows = load_csv(args.status)

    plot_outcomes(result_rows, args.out_dir)
    plot_welfare_inequality(result_rows, args.out_dir)
    plot_status(status_rows, args.out_dir)

    for path in sorted(args.out_dir.glob("preliminary_*.png")):
        print(path)


if __name__ == "__main__":
    main()
