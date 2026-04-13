from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_hamming_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "image_a",
        "image_b",
        "bit_hamming_ratio",
        "bit_hamming_distance",
        "byte_hamming_ratio",
        "byte_hamming_distance",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def build_heatmap_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    names = sorted(set(df["image_a"]).union(set(df["image_b"])))
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for name in names:
        matrix.loc[name, name] = 0.0

    for _, row in df.iterrows():
        a = row["image_a"]
        b = row["image_b"]
        v = float(row[value_col])
        matrix.loc[a, b] = v
        matrix.loc[b, a] = v

    return matrix


def parse_name(name: str) -> tuple[str, str]:
    """
    Parse names like:
      base_white
      base_white__one_pixel_flip
      base_simple_doodle__rotate_90_cw

    Returns:
      (base_family, perturbation)
    """
    if "__" in name:
        base, perturb = name.split("__", 1)
        return base, perturb
    return name, "base"


def build_perturbation_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Keep only pairs of the form:
      base_family  vs  base_family__perturbation

    This is what you usually want for avalanche boxplots.
    """
    rows = []

    for _, row in df.iterrows():
        a = str(row["image_a"])
        b = str(row["image_b"])
        val = float(row[metric])

        base_a, pert_a = parse_name(a)
        base_b, pert_b = parse_name(b)

        # Keep comparisons within the same family where one side is the base image
        if base_a == base_b:
            if pert_a == "base" and pert_b != "base":
                rows.append({
                    "base_family": base_a,
                    "perturbation": pert_b,
                    "metric": val,
                    "base_image": a,
                    "variant_image": b,
                })
            elif pert_b == "base" and pert_a != "base":
                rows.append({
                    "base_family": base_a,
                    "perturbation": pert_a,
                    "metric": val,
                    "base_image": b,
                    "variant_image": a,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No base-vs-perturbation rows found. "
            "Expected names like 'base_white' and 'base_white__one_pixel_flip'."
        )
    return out


def prettify_label(label: str) -> str:
    mapping = {
        "one_pixel_flip": "1-pixel flip",
        "patch_15x15_flip": "15x15 patch flip",
        "rotate_90_cw": "Rotate 90 deg",
        "shift_1px_down_right": "Shift 1 px",
        "brightness_plus_16": "Brightness +16",
        "gaussian_blur_r1": "Blur r=1",
        "base": "Base",
    }
    return mapping.get(label, label.replace("_", " "))


def plot_heatmap(matrix: pd.DataFrame, out_path: Path, title: str) -> None:
    n = len(matrix)
    # Scale figure size with matrix dimensions
    fig_w = max(8, min(24, n * 0.45))
    fig_h = max(7, min(22, n * 0.40))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bit Hamming Ratio")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(
    pert_df: pd.DataFrame,
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    order = [
        "one_pixel_flip",
        "patch_15x15_flip",
        "brightness_plus_16",
        "gaussian_blur_r1",
        "shift_1px_down_right",
        "rotate_90_cw",
    ]

    available = [p for p in order if p in set(pert_df["perturbation"])]
    if not available:
        available = sorted(set(pert_df["perturbation"]))

    data = [pert_df.loc[pert_df["perturbation"] == p, "metric"].tolist() for p in available]
    labels = [prettify_label(p) for p in available]

    fig_w = max(8, 1.3 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(pert_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append("Perturbation summary (base image vs perturbed variant)\n")
    grouped = pert_df.groupby("perturbation")["metric"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()

    for _, row in summary.iterrows():
        lines.append(
            f"{prettify_label(str(row['perturbation']))}: "
            f"n={int(row['count'])}, "
            f"mean={row['mean']:.6f}, "
            f"std={0.0 if pd.isna(row['std']) else row['std']:.6f}, "
            f"min={row['min']:.6f}, "
            f"max={row['max']:.6f}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a heatmap and perturbation boxplot from hamming_results.csv"
    )
    parser.add_argument("input_csv", type=Path, help="Path to hamming_results.csv")
    parser.add_argument(
        "--metric",
        default="bit_hamming_ratio",
        choices=[
            "bit_hamming_ratio",
            "bit_hamming_distance",
            "byte_hamming_ratio",
            "byte_hamming_distance",
        ],
        help="Metric column to plot",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to write output figures",
    )
    parser.add_argument(
        "--heatmap-name",
        default="hamming_heatmap.png",
        help="Filename for the heatmap figure",
    )
    parser.add_argument(
        "--boxplot-name",
        default="perturbation_boxplot.png",
        help="Filename for the perturbation boxplot figure",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_hamming_csv(args.input_csv)
    matrix = build_heatmap_matrix(df, args.metric)
    pert_df = build_perturbation_df(df, args.metric)

    heatmap_path = args.out_dir / args.heatmap_name
    boxplot_path = args.out_dir / args.boxplot_name
    summary_path = args.out_dir / "perturbation_summary.txt"

    metric_title = args.metric.replace("_", " ").title()

    plot_heatmap(
        matrix,
        heatmap_path,
        title=f"Hamming Heatmap ({metric_title})",
    )
    plot_boxplot(
        pert_df,
        boxplot_path,
        title=f"Base vs Perturbation ({metric_title})",
        y_label=metric_title,
    )
    write_summary_text(pert_df, summary_path)

    print(f"Wrote heatmap: {heatmap_path.resolve()}")
    print(f"Wrote boxplot: {boxplot_path.resolve()}")
    print(f"Wrote summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
