from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.criteo_fwfm.encoders.integer_sl import SLIntegerEncoder, SLIntegerEncoderConfig
from experiments.criteo_fwfm.schema import ALL_COLUMNS


def _load_terminal_plot_module(path: Path):
    spec = importlib.util.spec_from_file_location("terminal_plot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import terminal plot module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_int_columns(
    *,
    data_path: Path,
    start_row: int,
    n_rows: int,
    columns: list[str],
) -> pd.DataFrame:
    dtype = {col: "Int64" for col in columns}
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=ALL_COLUMNS,
        usecols=columns,
        skiprows=int(start_row),
        nrows=int(n_rows),
        na_values=[""],
        keep_default_na=True,
        dtype=dtype,
    )
    if n_rows > 0 and len(df) != n_rows:
        raise RuntimeError(f"Expected {n_rows} rows, got {len(df)}")
    return df


def _uniform_u_indices(max_index: int, max_points: int) -> np.ndarray:
    if max_index <= 0:
        return np.array([0], dtype=np.int64)
    n = min(max_points, max_index + 1)
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    idx = np.rint(np.expm1(u * np.log1p(float(max_index)))).astype(np.int64)
    idx = np.unique(np.clip(idx, 0, max_index))
    return idx


def _even_indices(max_index: int, max_points: int) -> np.ndarray:
    if max_index <= 0:
        return np.array([0], dtype=np.int64)
    n = min(max_points, max_index + 1)
    idx = np.rint(np.linspace(0.0, float(max_index), n, dtype=np.float64)).astype(np.int64)
    idx = np.unique(np.clip(idx, 0, max_index))
    return idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit per-column SL integer encoders on a Criteo split and render eigenfunctions "
            "as Unicode terminal plots (using the terminal-plotter skill script)."
        )
    )
    p.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to a journaled Optuna results.json (used for data_path/train_range and SL u_exp_valley params).",
    )
    p.add_argument(
        "--columns",
        type=str,
        default="I3,I4,I6,I7,I8,I9,I11",
        help="Comma-separated integer columns to plot.",
    )
    p.add_argument(
        "--plot-k",
        type=int,
        default=4,
        help="Number of leading eigenfunctions to plot per column (k=0..plot_k-1).",
    )
    p.add_argument(
        "--potential-family",
        type=str,
        default="none",
        choices=["none", "inverse_square"],
        help="Optional diagonal potential term q(x).",
    )
    p.add_argument(
        "--potential-kappa",
        type=float,
        default=0.0,
        help="Potential strength (used for inverse_square).",
    )
    p.add_argument(
        "--potential-x0",
        type=float,
        default=0.0,
        help="Shift for x in q(x) = kappa / (x + x0)^2, where x is the positive integer value (1..cap).",
    )
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--charset", type=str, default="braille", choices=["ascii", "block", "braille"])
    p.add_argument(
        "--max-points",
        type=int,
        default=256,
        help="Downsample points per eigenfunction before plotting.",
    )
    p.add_argument(
        "--x-axis",
        type=str,
        default="u",
        choices=["u", "x"],
        help="Plot eigenfunctions vs u=log1p(i)/log1p(max_i) or vs x=i+1.",
    )
    p.add_argument(
        "--terminal-plot-script",
        type=Path,
        default=Path("/home/alex/.codex/skills/terminal-plotter/scripts/terminal_plot.py"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    payload = json.loads(args.results_json.read_text())
    data_path = Path(payload["data_path"])
    train_start, train_end = payload["train_range"]
    train_start = int(train_start)
    train_end = int(train_end)
    n_rows = train_end - train_start

    conductance_family = "u_exp_valley"
    u0 = float(payload.get("sl_u0", 0.0))
    left_slope = float(payload.get("sl_left_slope", 1.0))
    right_slope = float(payload.get("sl_right_slope", 1.0))
    num_basis = int(payload.get("sl_num_basis", 10))

    sl_config = SLIntegerEncoderConfig(
        cap_max=10_000_000,
        num_basis=num_basis,
        prior_count=0.5,
        cutoff_quantile=0.99,
        cutoff_factor=1.1,
        curvature_alpha=1.0,
        curvature_beta=0.0,
        curvature_center=0.0,
        conductance_eps=1e-8,
        positive_overflow="clip_to_cap",
        conductance_family=conductance_family,
        uvalley_u0=u0,
        uvalley_left_slope=left_slope,
        uvalley_right_slope=right_slope,
        potential_family=str(args.potential_family),
        potential_kappa=float(args.potential_kappa),
        potential_x0=float(args.potential_x0),
    )

    columns = [c.strip() for c in str(args.columns).split(",") if c.strip()]
    df = _read_int_columns(
        data_path=data_path,
        start_row=train_start,
        n_rows=n_rows,
        columns=columns,
    )

    terminal_plot = _load_terminal_plot_module(args.terminal_plot_script)

    potential_note = ""
    if args.potential_family != "none" and float(args.potential_kappa) != 0.0:
        potential_note = (
            f" potential='{args.potential_family}' kappa={float(args.potential_kappa):.6g} x0={float(args.potential_x0):.6g}"
        )
    else:
        potential_note = " (no q potential)"

    print(
        f"SL eigenfunctions{potential_note} on train rows [{train_start}, {train_end}) "
        f"using conductance='{conductance_family}' u0={u0:.6g} left={left_slope:.6g} right={right_slope:.6g} "
        f"num_basis={num_basis}"
    )
    print(f"x-axis: {args.x_axis}  charset={args.charset}  width={args.width} height={args.height}")
    print()

    for col in columns:
        encoder = SLIntegerEncoder(sl_config).fit(df[col])
        support_size = int(encoder.basis_matrix.shape[0])
        cap_value = int(encoder.cap_value)
        max_index = max(support_size - 1, 0)

        print(f"== {col}  cap_value={cap_value}  support_size={support_size} ==")
        plot_k = max(1, min(int(args.plot_k), encoder.num_basis, support_size))
        for k in range(plot_k):
            phi = encoder.basis_matrix[:, k].astype(np.float64, copy=False)
            if args.x_axis == "u":
                denom = np.log1p(float(max_index))
                xs = np.zeros(support_size, dtype=np.float64) if denom <= 0 else np.log1p(
                    np.arange(support_size, dtype=np.float64)
                ) / denom
                idx = _uniform_u_indices(max_index=max_index, max_points=int(args.max_points))
            else:
                xs = 1.0 + np.arange(support_size, dtype=np.float64)
                idx = _even_indices(max_index=max_index, max_points=int(args.max_points))

            x_s = xs[idx].tolist()
            y_s = phi[idx].tolist()

            title = f"{col}  phi_{k}({args.x_axis})"
            rendered = terminal_plot.render_xy_plot(
                x_s,
                y_s,
                mode="line",
                width=int(args.width),
                height=int(args.height),
                title=title,
                ascii_only=False,
                charset=str(args.charset),
            )
            print(rendered)
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
