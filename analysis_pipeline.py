#!/usr/bin/env python3
"""Reproducible LDA and bar-plot analysis for the L-PGDS fatty-acid sensor array."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

CHANNELS = ["WT", "LA", "LB", "LC", "LD", "LE"]
CLASS_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
]
CHANNEL_COLORS = {
    "WT": "#4DBBD5", "LA": "#E64B35", "LB": "#00A087",
    "LC": "#3C5488", "LD": "#F39B7F", "LE": "#8491B4",
}
CHI2_95_SCALE_2D = 2.44774683068

DEFAULT_ANALYSES = [
    {
        "lda_sheet": "不饱和度LDA", "bar_sheet": "不饱和度柱状图",
        "key": "unsaturation", "lda_title": "Unsaturation discrimination",
        "bar_title": "Unsaturation response profile", "bar_xlabel": "Fatty acid",
    },
    {
        "lda_sheet": "不同链长LDA", "bar_sheet": "不同链长柱状图",
        "key": "chain_length", "lda_title": "Chain-length discrimination",
        "bar_title": "Chain-length response profile", "bar_xlabel": "Fatty acid",
    },
    {
        "lda_sheet": "不同比例LDA", "bar_sheet": "不同比例柱状图",
        "key": "mixture_ratio", "lda_title": "Mixture-ratio discrimination",
        "bar_title": "Mixture-ratio response profile",
        "bar_xlabel": "C18:C18-1 ratio",
    },
    {
        "lda_sheet": "C18浓度LDA", "bar_sheet": "C18浓度柱状图",
        "key": "C18_concentration", "lda_title": "C18 concentration discrimination",
        "bar_title": "C18 concentration response profile",
        "bar_xlabel": "Concentration (μM)",
    },
    {
        "lda_sheet": "C18-2浓度LDA", "bar_sheet": "C18-2浓度柱状图",
        "key": "C18_2_concentration", "lda_title": "C18:2 concentration discrimination",
        "bar_title": "C18:2 concentration response profile",
        "bar_xlabel": "Concentration (μM)",
    },
]


@dataclass
class SummaryRecord:
    analysis: str
    seed: int
    n_classes: int
    training_per_class: int
    prediction_per_class: int
    n_prediction: int
    n_correct: int
    prediction_accuracy: float
    ld1_explained_percent: float
    ld2_explained_percent: float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--train-per-class", type=int, default=6)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def load_analyses(path):
    if path is None:
        return DEFAULT_ANALYSES
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.1,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, length=5, pad=6)
    ax.grid(False)


def read_lda_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None).dropna(how="all").iloc[:, :7].copy()
    if df.shape[1] != 7:
        raise ValueError(f"{sheet}: expected label + six sensor columns.")
    df.columns = ["label"] + CHANNELS
    df["label"] = df["label"].astype(str).str.strip()
    for c in CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df.reset_index(drop=True)


def read_bar_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None).dropna(how="all").iloc[:, :13].copy()
    if df.shape[1] != 13:
        raise ValueError(f"{sheet}: expected condition + six mean/SD pairs.")
    df = df.iloc[1:].copy()
    cols = ["condition"]
    for c in CHANNELS:
        cols += [f"{c}_mean", f"{c}_sd"]
    df.columns = cols
    df["condition"] = df["condition"].astype(str).str.strip()
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df.reset_index(drop=True)


def stratified_split(df, train_per_class, seed):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label, group in df.groupby("label", sort=False):
        idx = group.index.to_numpy()
        if len(idx) <= train_per_class:
            raise ValueError(f"{label}: not enough replicates.")
        chosen = np.sort(rng.choice(idx, train_per_class, replace=False))
        chosen_set = set(chosen.tolist())
        train += chosen.tolist()
        test += [int(i) for i in idx if i not in chosen_set]
    return np.array(sorted(train)), np.array(sorted(test))


def ellipse_params(points):
    cov = np.cov(np.asarray(points, float), rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2 * CHI2_95_SCALE_2D * np.sqrt(vals[0])
    height = 2 * CHI2_95_SCALE_2D * np.sqrt(vals[1])
    return np.asarray(points).mean(axis=0), width, height, angle


def ellipse_bbox(center, width, height, angle_deg, n=720):
    t = np.linspace(0, 2 * np.pi, n)
    a, b = width / 2, height / 2
    ang = np.deg2rad(angle_deg)
    x = center[0] + a*np.cos(t)*np.cos(ang) - b*np.sin(t)*np.sin(ang)
    y = center[1] + a*np.cos(t)*np.sin(ang) + b*np.sin(t)*np.cos(ang)
    return x.min(), x.max(), y.min(), y.max()


def display_label(value, key, bar=False):
    s = str(value)
    if key == "mixture_ratio":
        return s.split("=")[-1].strip()
    if key in {"C18_concentration", "C18_2_concentration"}:
        out = s.replace("c = ", "").strip()
        return out if bar else out + " μM"
    return s


def run_lda(df, analysis, seed, n_train, fig_dir, dpi, pdf):
    X = df[CHANNELS].to_numpy(float)
    y = df["label"].to_numpy(object)
    classes = list(dict.fromkeys(y.tolist()))
    train, test = stratified_split(df, n_train, seed)

    scaler = StandardScaler().fit(X[train])
    X_scaled = scaler.transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2, solver="svd")
    lda.fit(X_scaled[train], y[train])
    scores = lda.transform(X_scaled)
    pred = lda.predict(X_scaled)
    evr = lda.explained_variance_ratio_

    n_correct = int(np.sum(pred[test] == y[test]))
    summary = SummaryRecord(
        analysis=analysis["key"], seed=seed, n_classes=len(classes),
        training_per_class=n_train,
        prediction_per_class=int(len(test) / len(classes)),
        n_prediction=len(test), n_correct=n_correct,
        prediction_accuracy=n_correct / len(test),
        ld1_explained_percent=float(evr[0] * 100),
        ld2_explained_percent=float(evr[1] * 100),
    )

    split_rows = []
    train_set = set(train.tolist())
    for cls in classes:
        idxs = np.where(y == cls)[0]
        for rep, idx in enumerate(idxs, 1):
            split_rows.append({
                "analysis": analysis["key"],
                "seed": seed,
                "class_label": str(cls),
                "replicate": rep,
                "role": "Training" if idx in train_set else "Prediction",
                "true_label": str(y[idx]),
                "predicted_label": str(pred[idx]),
                "correct": bool(pred[idx] == y[idx]),
                "ld1": float(scores[idx, 0]),
                "ld2": float(scores[idx, 1]),
            })

    fig = plt.figure(figsize=(9.2, 6.0), facecolor="white")
    ax = fig.add_axes([0.10, 0.16, 0.60, 0.72])
    style_axes(ax)
    colors = {c: CLASS_COLORS[i % len(CLASS_COLORS)] for i, c in enumerate(classes)}

    ellipse_info, bounds = {}, []
    for cls in classes:
        idx = np.array([i for i in train if y[i] == cls])
        info = ellipse_params(scores[idx, :2])
        ellipse_info[cls] = info
        bounds.append(ellipse_bbox(*info))

    for cls in classes:
        center, width, height, angle = ellipse_info[cls]
        ax.add_patch(Ellipse(
            center, width, height, angle=angle,
            facecolor=colors[cls], edgecolor=colors[cls],
            alpha=0.10, lw=1.5, clip_on=False, zorder=1
        ))

    for cls in classes:
        idx = np.array([i for i in train if y[i] == cls])
        ax.scatter(scores[idx, 0], scores[idx, 1], s=95,
                   facecolors=colors[cls], edgecolors="#222222",
                   linewidths=0.9, zorder=3)
        idx = np.array([i for i in test if y[i] == cls])
        ax.scatter(scores[idx, 0], scores[idx, 1], s=125,
                   facecolors="white", edgecolors=colors[cls],
                   linewidths=2.3, zorder=4)

    x0 = min(scores[:, 0].min(), min(v[0] for v in bounds))
    x1 = max(scores[:, 0].max(), max(v[1] for v in bounds))
    y0 = min(scores[:, 1].min(), min(v[2] for v in bounds))
    y1 = max(scores[:, 1].max(), max(v[3] for v in bounds))
    xp = max((x1 - x0) * 0.06, 0.35)
    yp = max((y1 - y0) * 0.08, 0.35)
    ax.set_xlim(x0 - xp, x1 + xp)
    ax.set_ylim(y0 - yp, y1 + yp)

    ax.set_xlabel(f"LD1 ({evr[0]*100:.1f}%)", labelpad=8)
    ax.set_ylabel(f"LD2 ({evr[1]*100:.1f}%)", labelpad=8)
    ax.set_title(analysis["lda_title"], pad=18)

    class_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=colors[c], markeredgecolor="#222222",
               markersize=10, label=display_label(c, analysis["key"]))
        for c in classes
    ]
    fig.legend(handles=class_handles, frameon=False, loc="upper left",
               bbox_to_anchor=(0.73, 0.89), handletextpad=0.6)

    split_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="#666666", markeredgecolor="#222222",
               markersize=10, label=f"Training (n={n_train})"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="white", markeredgecolor="#555555",
               markeredgewidth=2.0, markersize=10,
               label=f"Prediction (n={summary.prediction_per_class})"),
    ]
    fig.legend(handles=split_handles, frameon=False, loc="upper left",
               bbox_to_anchor=(0.73, 0.40), handletextpad=0.6)

    fig.savefig(fig_dir / f"{analysis['key']}_LDA.png", dpi=dpi, facecolor="white")
    fig.savefig(fig_dir / f"{analysis['key']}_LDA.svg", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return summary, split_rows


def plot_bar(df, analysis, fig_dir, dpi, pdf):
    fig = plt.figure(figsize=(9.2, 6.0), facecolor="white")
    ax = fig.add_axes([0.10, 0.16, 0.84, 0.68])
    style_axes(ax)

    labels = [display_label(v, analysis["key"], bar=True) for v in df["condition"]]
    means = np.column_stack([df[f"{c}_mean"].to_numpy(float) for c in CHANNELS])
    sds = np.column_stack([df[f"{c}_sd"].to_numpy(float) for c in CHANNELS])
    x = np.arange(len(labels), dtype=float)
    width = 0.12 if len(labels) >= 5 else 0.13
    offsets = (np.arange(6) - 2.5) * width

    for j, c in enumerate(CHANNELS):
        ax.bar(x + offsets[j], means[:, j], width=width, yerr=sds[:, j],
               capsize=2.5, color=CHANNEL_COLORS[c],
               edgecolor="#222222", linewidth=0.65,
               error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(analysis["bar_xlabel"], labelpad=8)
    ax.set_ylabel("Normalized fluorescence response", labelpad=8)
    ax.set_title(analysis["bar_title"], pad=18)

    low = float(np.min(means - sds))
    high = float(np.max(means + sds))
    span = max(high - low, 0.1)
    ax.set_ylim(low - 0.12 * span, high + 0.20 * span)

    handles = [
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=CHANNEL_COLORS[c], markeredgecolor="#222222",
               markersize=10, label=c)
        for c in CHANNELS
    ]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="upper center",
               bbox_to_anchor=(0.52, 0.93), columnspacing=1.2)

    fig.savefig(fig_dir / f"{analysis['key']}_bar.png", dpi=dpi, facecolor="white")
    fig.savefig(fig_dir / f"{analysis['key']}_bar.svg", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def consistency_check(raw, bar, key):
    rows = []
    for _, r in bar.iterrows():
        cond = str(r["condition"]).strip()
        group = raw.loc[raw["label"].astype(str).str.strip() == cond]
        for c in CHANNELS:
            raw_mean = float(group[c].mean()) if len(group) else np.nan
            raw_sd = float(group[c].std(ddof=1)) if len(group) else np.nan
            rows.append({
                "analysis": key, "condition": cond, "channel": c,
                "raw_n": int(len(group)), "raw_mean": raw_mean,
                "summary_mean": float(r[f"{c}_mean"]),
                "mean_difference": float(r[f"{c}_mean"]) - raw_mean,
                "raw_sd": raw_sd, "summary_sd": float(r[f"{c}_sd"]),
                "sd_difference": float(r[f"{c}_sd"]) - raw_sd,
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    analyses = load_analyses(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    setup_style()

    summaries, split_rows, qc_frames = [], [], []
    with PdfPages(args.output_dir / "all_figures.pdf") as pdf:
        for i, analysis in enumerate(analyses):
            raw = read_lda_sheet(args.input, analysis["lda_sheet"])
            bar = read_bar_sheet(args.input, analysis["bar_sheet"])
            seed = args.seed + i * 101

            summary, rows = run_lda(
                raw, analysis, seed, args.train_per_class,
                fig_dir, args.dpi, pdf
            )
            summaries.append(asdict(summary))
            split_rows.extend(rows)
            plot_bar(bar, analysis, fig_dir, args.dpi, pdf)
            qc_frames.append(consistency_check(raw, bar, analysis["key"]))

    pd.DataFrame(summaries).to_csv(args.output_dir / "lda_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(
        args.output_dir / "lda_predictions_and_split.csv", index=False
    )
    pd.concat(qc_frames, ignore_index=True).to_csv(
        args.output_dir / "bar_summary_consistency_check.csv", index=False
    )

    metadata = {
        "input_file": str(args.input),
        "base_seed": args.seed,
        "train_per_class": args.train_per_class,
        "sensor_channels": CHANNELS,
        "analyses": analyses,
        "notes": {
            "scaling": "StandardScaler fitted on training data only.",
            "lda": "LDA fitted on training data only.",
            "prediction": "Held-out samples are used only for evaluation.",
            "ellipse": "95% covariance ellipses use training LDA scores only.",
            "bar_error": "Bar plots use mean ± SD from dedicated summary sheets."
        }
    }
    with (args.output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Analysis complete.")
    for s in summaries:
        print(f"{s['analysis']}: {s['n_correct']}/{s['n_prediction']} "
              f"({100*s['prediction_accuracy']:.1f}%)")


if __name__ == "__main__":
    main()
