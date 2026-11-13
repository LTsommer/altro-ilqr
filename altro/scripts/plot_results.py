#!/usr/bin/env python3
"""Plot the CSV files written by the example programs.

    ./build/altro/alilqr [output_dir]   ->  alilqr_trajectory.csv, alilqr_road.csv
    ./build/altro/mp     [output_dir]   ->  mp_trajectory.csv, mp_road.csv, mp_path_validation.json

Usage:
    python3 altro/scripts/plot_results.py alilqr            # show windows
    python3 altro/scripts/plot_results.py mp --dir out      # read the CSV files from out/
    python3 altro/scripts/plot_results.py mp --save figs    # write PNG files instead

Only needs matplotlib.
"""
import argparse
import csv
import json
import math
import os
import sys


def read_csv(path):
    """Returns {column name: list of floats}. Empty cells (shorter columns) are skipped."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        names = next(reader)
        columns = {name: [] for name in names}
        for row in reader:
            for name, cell in zip(names, row):
                if cell != "":
                    columns[name].append(float(cell))
    return columns


def plot_path(ax, traj, road, ref_names, ref_label):
    ax.plot(road[ref_names[0]], road[ref_names[1]], "--", color="gray", label=ref_label)
    ax.plot(road["left_x"], road["left_y"], color="black", label="left boundary")
    ax.plot(road["right_x"], road["right_y"], color="black", linestyle="-.", label="right boundary")
    ax.plot(traj["x"], traj["y"], color="tab:red", label="optimized")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("path")
    ax.legend()


def plot_series(ax, x, xlabel, title, series):
    """series: list of (label, values). Each one is drawn against the leading part of x."""
    for label, values in series:
        n = min(len(x), len(values))
        ax.plot(x[:n], values[:n], label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()


def alilqr_figures(plt, directory):
    traj = read_csv(os.path.join(directory, "alilqr_trajectory.csv"))
    road = read_csv(os.path.join(directory, "alilqr_road.csv"))
    figures = {}

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_path(ax, traj, road, ("centerline_x", "centerline_y"), "center line")
    figures["alilqr_path"] = fig

    s = traj["s"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plot_series(axes[0][0], s, "s [m]", "heading [rad]",
                [("reference", traj["ref_yaw"]), ("optimized", traj["theta"])])
    plot_series(axes[0][1], s, "s [m]", "curvature [1/m]", [("kappa", traj["kappa"])])
    plot_series(axes[1][0], s, "s [m]", "d(kappa)/ds", [("dkappa", traj["dkappa"])])
    plot_series(axes[1][1], s, "s [m]", "control: d2(kappa)/ds2", [("ddkappa", traj["ddkappa"])])
    fig.tight_layout()
    figures["alilqr_states"] = fig
    return figures


def mp_clearance_figure(plt, traj, report):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon

    radius = report["vehicle_radius"]
    path = report["path"]
    failure = report["failure"]
    clearances = report["clearances"]
    closest = min(clearances, key=lambda item: item["center_distance"], default=None)
    details = [item for item in clearances if item["object"].startswith("obstacle ")]
    if closest is not None and closest not in details:
        details.append(closest)

    # Each path segment swept by a circle is a capsule. Draw the union using
    # the same opaque light green fill on every capsule.
    capsules = []
    for a, b in zip(path, path[1:]):
        if None in a or None in b:
            continue
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        capsule = []
        for (x, y), angle in ((b, heading - math.pi / 2), (a, heading + math.pi / 2)):
            for i in range(49):
                theta = angle + math.pi * i / 48
                capsule.append((x + radius * math.cos(theta), y + radius * math.sin(theta)))
        capsules.append(capsule)

    rows = max(1, len(details))
    fig = plt.figure(figsize=(11, max(6, 3.65 * rows)), constrained_layout=True)
    grid = fig.add_gridspec(rows, 2, width_ratios=(1.35, 1))
    overview = fig.add_subplot(grid[:, 0] if details else grid[:, :])
    left, right = report["left_boundary"], report["right_boundary"]
    boundaries = [left, right, [left[0], right[0]], [left[-1], right[-1]]]
    x, y = zip(*(p if None not in p else (math.nan, math.nan) for p in path)) if path else ([], [])

    def draw_scene(ax):
        ax.add_collection(PolyCollection(
            capsules, facecolor="#bde7cd", edgecolor="none",
            label=f"Swept area · r = {radius:g} m"))
        for i, edge in enumerate(boundaries):
            ax.plot(*zip(*edge), color="#637386", linestyle="--", linewidth=1,
                    label="Road boundaries" if i == 0 else None)
        for i, polygon in enumerate(report["obstacles"]):
            ax.add_patch(Polygon(polygon, facecolor="#ef9967a6", edgecolor="#b75028",
                                 label="Static obstacles" if i == 0 else None))
        ax.plot(x, y, color="#184673", linewidth=1.6, label="Optimized path")
        ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
        ax.set_axisbelow(True)
        ax.grid(alpha=0.22)

    def mark_gap(ax, item):
        p = item["path_point"]
        q = item["object_point"]
        distance = item["center_distance"]
        if distance > radius:
            w = [a + (b - a) * radius / distance for a, b in zip(p, q)]
            ax.plot(*zip(p, w), ":", color="#184673", linewidth=1.4)
            ax.plot(*zip(w, q), "o-", color="#93264f", linewidth=2.5, markersize=4)
        else:
            ax.plot(*zip(p, q), "o-", color="#93264f", linewidth=2.5, markersize=4)
        ax.scatter(*p, color="#184673", s=20, zorder=5)

    draw_scene(overview)
    overview.set_title("Path overview", loc="left")
    finite = [p for p in path if None not in p]
    if finite:
        for axis, values in zip((overview.set_xlim, overview.set_ylim), zip(*finite)):
            axis(min(values) - 5 * radius, max(values) + 5 * radius)
    if failure and failure["segment"] is not None:
        i = failure["segment"]
        if i + 1 < len(path):
            overview.plot(x[i:i+2], y[i:i+2], color="#b61735", linewidth=3,
                          label="First failed segment", zorder=6)
    overview.legend(loc="lower left", fontsize=8)

    for i, item in enumerate(details):
        p = item["path_point"]
        overview.annotate(str(i+1), p, xytext=(6, 5), textcoords="offset points",
                          color="#93264f", weight="bold")
        overview.scatter(*p, s=30, color="#93264f", edgecolor="white", zorder=6)
        ax = fig.add_subplot(grid[i, 1])
        draw_scene(ax)
        mark_gap(ax, item)
        q = item["object_point"]
        margin = 1.45 * radius
        ax.set_xlim(min(p[0], q[0]) - margin, max(p[0], q[0]) + margin)
        ax.set_ylim(min(p[1], q[1]) - margin, max(p[1], q[1]) + margin)
        gap = item["center_distance"] - radius
        label = f"{100*gap:.2f} cm" if gap > 1e-9 else "Contact or overlap"
        ax.set_title(f"{i+1}. {item['object'].capitalize()} · {label}",
                     loc="left", color="#93264f" if item is closest else "#184673")
        segment = item["segment"]
        caption = f"segment [{segment}, {segment+1}]"
        if segment + 1 < len(traj["t"]):
            a, b = path[segment:segment+2]
            delta = [v - u for u, v in zip(a, b)]
            length_squared = sum(d*d for d in delta)
            fraction = (sum((v-u)*d for v, u, d in zip(p, a, delta)) /
                        length_squared) if length_squared else 0.0
            fraction = min(1.0, max(0.0, fraction))
            t0, t1 = traj["t"][segment:segment+2]
            caption = f"t ≈ {t0 + fraction*(t1-t0):.2f} s · {caption}"
        ax.text(0.03, 0.04, caption, transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#dbe5ef"})

    status = f"Step 4 failed · {failure['reason']}" if failure else "Step 4 passed"
    if not failure and closest and closest["center_distance"] > radius + 1e-9:
        gap = closest["center_distance"] - radius
        status += f" · Minimum clearance {100*gap:.2f} cm ({closest['object']})"
    fig.suptitle(f"mp path clearance\n{status}", fontsize=14, weight="bold")
    return fig


def add_mp_clearance(plt, directory, traj, figures):
    filename = os.path.join(directory, "mp_path_validation.json")
    try:
        with open(filename, encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as error:
        print(f"Cannot read {filename}: {error}", file=sys.stderr)
        return

    path = [[v if math.isfinite(v) else None for v in point]
            for point in zip(traj["x"], traj["y"])]
    if not isinstance(report, dict) or report.get("path") != path:
        print(f"Skipping clearance figure: {filename} does not match the trajectory CSV",
              file=sys.stderr)
        return
    figures["mp_clearance"] = mp_clearance_figure(plt, traj, report)


def mp_figures(plt, directory):
    traj = read_csv(os.path.join(directory, "mp_trajectory.csv"))
    road = read_csv(os.path.join(directory, "mp_road.csv"))
    figures = {}

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_path(ax, traj, road, ("ref_x", "ref_y"), "reference line")
    figures["mp_path"] = fig

    t = traj["t"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    plot_series(axes[0][0], t, "t [s]", "heading [rad]",
                [("reference", traj["ref_yaw"]), ("optimized", traj["theta"])])
    plot_series(axes[0][1], t, "t [s]", "speed [m/s]",
                [("reference", traj["ref_v"]), ("optimized", traj["v"])])
    plot_series(axes[1][0], t, "t [s]", "acceleration [m/s^2]", [("a", traj["a"])])
    plot_series(axes[1][1], t, "t [s]", "curvature [1/m]", [("kappa", traj["kappa"])])
    plot_series(axes[2][0], t, "t [s]", "control: jerk [m/s^3]", [("jerk", traj["jerk"])])
    plot_series(axes[2][1], t, "t [s]", "control: d(kappa)/dt", [("dkappa", traj["dkappa"])])
    fig.tight_layout()
    figures["mp_states"] = fig
    add_mp_clearance(plt, directory, traj, figures)
    return figures


def main():
    parser = argparse.ArgumentParser(description="Plot the CSV results of the alilqr / mp examples.")
    parser.add_argument("example", choices=["alilqr", "mp"], help="which example's results to plot")
    parser.add_argument("--dir", default=".", help="directory containing the CSV files (default: .)")
    parser.add_argument("--save", metavar="DIR", help="write PNG files to DIR instead of showing windows")
    args = parser.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    figures = (alilqr_figures if args.example == "alilqr" else mp_figures)(plt, args.dir)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figures.items():
            path = os.path.join(args.save, name + ".png")
            fig.savefig(path, dpi=150)
            print("saved", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
