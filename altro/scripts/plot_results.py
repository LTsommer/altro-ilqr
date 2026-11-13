#!/usr/bin/env python3
"""Plot the CSV files written by the example programs.

    ./build/altro/alilqr [output_dir]   ->  alilqr_trajectory.csv, alilqr_road.csv
    ./build/altro/mp     [output_dir]   ->  mp_trajectory.csv,     mp_road.csv

Usage:
    python3 altro/scripts/plot_results.py alilqr            # show windows
    python3 altro/scripts/plot_results.py mp --dir out      # read the CSV files from out/
    python3 altro/scripts/plot_results.py mp --save figs    # write PNG files instead

Only needs matplotlib.
"""
import argparse
import csv
import os


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
