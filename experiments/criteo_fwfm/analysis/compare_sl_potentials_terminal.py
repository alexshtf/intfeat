from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
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


def _side_by_side(left: str, right: str, sep: str = "  ") -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    width_left = max((len(line) for line in left_lines), default=0)
    height = max(len(left_lines), len(right_lines))
    left_lines = left_lines + [""] * (height - len(left_lines))
    right_lines = right_lines + [""] * (height - len(right_lines))
    out: list[str] = []
    for l, r in zip(left_lines, right_lines, strict=True):
        out.append(l.ljust(width_left) + sep + r)
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare two diagonal potential families for SL integer eigenfunctions and render "
            "side-by-side Unicode terminal plots. Left: right-edge inverse-square in u. "
            "Right: monotone u^p confining potential."
        )
    )
    p.add_argument("--results-json", type=Path, required=True)
    p.add_argument(
        "--columns",
        type=str,
        default="I3,I4,I6,I7,I8,I9,I11",
        help="Comma-separated integer columns to plot.",
    )
    p.add_argument("--plot-k", type=int, default=3)
    p.add_argument("--kappa", type=float, default=3.0)
    p.add_argument(
        "--right-eps",
        type=float,
        default=0.01,
        help="Epsilon for the right-edge barrier: V(u)=kappa/(1-u+eps)^2.",
    )
    p.add_argument(
        "--u-power",
        type=float,
        default=2.0,
        help="Power p for monotone potential: V(u)=kappa*u^p.",
    )
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--charset", type=str, default="braille", choices=["ascii", "block", "braille"])
    p.add_argument("--max-points", type=int, default=256)
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

    u0 = float(payload.get("sl_u0", 0.0))
    left_slope = float(payload.get("sl_left_slope", 1.0))
    right_slope = float(payload.get("sl_right_slope", 1.0))
    num_basis = int(payload.get("sl_num_basis", 10))

    base = SLIntegerEncoderConfig(
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
        conductance_family="u_exp_valley",
        uvalley_u0=u0,
        uvalley_left_slope=left_slope,
        uvalley_right_slope=right_slope,
    )

    columns = [c.strip() for c in str(args.columns).split(",") if c.strip()]
    df = _read_int_columns(
        data_path=data_path,
        start_row=train_start,
        n_rows=n_rows,
        columns=columns,
    )
    terminal_plot = _load_terminal_plot_module(args.terminal_plot_script)

    kappa = float(args.kappa)
    right_eps = float(args.right_eps)
    u_power = float(args.u_power)

    print(
        "SL eigenfunction comparison on train "
        f"[{train_start}, {train_end}) conductance=u_exp_valley(u0={u0:.6g}, left={left_slope:.6g}, right={right_slope:.6g}) "
        f"num_basis={num_basis}"
    )
    print(f"Left: V(u)=kappa/(1-u+eps)^2  (kappa={kappa:.6g}, eps={right_eps:.6g})")
    print(f"Right: V(u)=kappa*u^p         (kappa={kappa:.6g}, p={u_power:.6g})")
    print(f"Plots: x=u, charset={args.charset}, width={args.width}, height={args.height}")
    print()

    cfg_left = replace(
        base,
        potential_family="u_right_inverse_square",
        potential_kappa=kappa,
        potential_eps=right_eps,
    )
    cfg_right = replace(
        base,
        potential_family="u_power",
        potential_kappa=kappa,
        potential_power=u_power,
    )

    for col in columns:
        enc_left = SLIntegerEncoder(cfg_left).fit(df[col])
        enc_right = SLIntegerEncoder(cfg_right).fit(df[col])

        support_size = int(enc_left.basis_matrix.shape[0])
        if support_size != int(enc_right.basis_matrix.shape[0]):
            raise RuntimeError("Support size mismatch between potential variants")

        max_index = max(support_size - 1, 0)
        denom = np.log1p(float(max_index))
        xs = np.zeros(support_size, dtype=np.float64) if denom <= 0 else np.log1p(
            np.arange(support_size, dtype=np.float64)
        ) / denom
        idx = _uniform_u_indices(max_index=max_index, max_points=int(args.max_points))
        x_s = xs[idx].tolist()

        print(f"== {col}  cap_value={enc_left.cap_value}  support_size={support_size} ==")
        plot_k = max(1, min(int(args.plot_k), num_basis, support_size))
        for k in range(plot_k):
            y_left = enc_left.basis_matrix[:, k].astype(np.float64, copy=False)[idx].tolist()
            y_right = enc_right.basis_matrix[:, k].astype(np.float64, copy=False)[idx].tolist()

            left_title = f"{col} phi_{k}  opt1: right barrier"
            right_title = f"{col} phi_{k}  opt2: u^p confine"
            rendered_left = terminal_plot.render_xy_plot(
                x_s,
                y_left,
                mode="line",
                width=int(args.width),
                height=int(args.height),
                title=left_title,
                ascii_only=False,
                charset=str(args.charset),
            )
            rendered_right = terminal_plot.render_xy_plot(
                x_s,
                y_right,
                mode="line",
                width=int(args.width),
                height=int(args.height),
                title=right_title,
                ascii_only=False,
                charset=str(args.charset),
            )
            print(_side_by_side(rendered_left, rendered_right))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

