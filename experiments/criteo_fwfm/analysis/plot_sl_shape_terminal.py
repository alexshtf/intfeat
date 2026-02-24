from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.criteo_fwfm.config import get_config_value, load_yaml
from experiments.criteo_fwfm.encoders.integer_sl import SLIntegerEncoder, SLIntegerEncoderConfig
from experiments.criteo_fwfm.schema import ALL_COLUMNS


def _load_terminal_plot_module(path: Path):
    spec = importlib.util.spec_from_file_location("terminal_plot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import terminal plot module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_int_column(
    *,
    data_path: Path,
    start_row: int,
    n_rows: int,
    column: str,
) -> pd.Series:
    dtype = {column: "Int64"}
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=ALL_COLUMNS,
        usecols=[column],
        skiprows=int(start_row),
        nrows=int(n_rows),
        na_values=[""],
        keep_default_na=True,
        dtype=dtype,
    )
    if n_rows > 0 and len(df) != n_rows:
        raise RuntimeError(f"Expected {n_rows} rows, got {len(df)}")
    return df[column]


def _infer_hybrid_config(results_json: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit

    cmd_path = results_json.parent / "cmd.txt"
    if not cmd_path.exists():
        return None

    raw = cmd_path.read_text().strip()
    if not raw:
        return None

    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    for idx, tok in enumerate(tokens):
        if tok == "--hybrid-config" and idx + 1 < len(tokens):
            return Path(tokens[idx + 1])
    return None


def _apply_sl_config_from_yaml(
    *,
    yaml_path: Path,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    config = load_yaml(yaml_path)
    out = dict(defaults)

    out["cap_max"] = int(get_config_value(config, "model.integer.sl.cap_max", default=out["cap_max"]))
    out["cap_mode"] = str(get_config_value(config, "model.integer.sl.cap_mode", default=out["cap_mode"]))
    out["positive_overflow"] = str(
        get_config_value(config, "model.integer.sl.positive_overflow", default=out["positive_overflow"])
    )
    out["right_boundary"] = str(
        get_config_value(config, "model.integer.sl.right_boundary", default=out.get("right_boundary", "neumann_midpoint"))
    )
    out["conductance_eps"] = float(
        get_config_value(config, "model.integer.sl.conductance_eps", default=out["conductance_eps"])
    )

    out["prior_count"] = float(
        get_config_value(config, "model.integer.sl.hist.prior_count", default=out["prior_count"])
    )
    out["cutoff_quantile"] = float(
        get_config_value(config, "model.integer.sl.hist.cutoff_quantile", default=out["cutoff_quantile"])
    )
    out["cutoff_factor"] = float(
        get_config_value(config, "model.integer.sl.hist.cutoff_factor", default=out["cutoff_factor"])
    )

    out["curvature_alpha"] = float(
        get_config_value(config, "model.integer.sl.curvature.alpha", default=out["curvature_alpha"])
    )
    out["curvature_beta"] = float(
        get_config_value(config, "model.integer.sl.curvature.beta", default=out["curvature_beta"])
    )
    out["curvature_center"] = float(
        get_config_value(config, "model.integer.sl.curvature.center", default=out["curvature_center"])
    )

    out["conductance_family"] = str(
        get_config_value(config, "model.integer.sl.conductance.family", default=out["conductance_family"])
    )
    if out["conductance_family"] == "u_exp_valley":
        out["uvalley_u0"] = float(
            get_config_value(
                config,
                "model.integer.sl.conductance.u_exp_valley.u0",
                default=out["uvalley_u0"],
            )
        )
        out["uvalley_left_slope"] = float(
            get_config_value(
                config,
                "model.integer.sl.conductance.u_exp_valley.left_slope",
                default=out["uvalley_left_slope"],
            )
        )
        out["uvalley_right_slope"] = float(
            get_config_value(
                config,
                "model.integer.sl.conductance.u_exp_valley.right_slope",
                default=out["uvalley_right_slope"],
            )
        )

    out["potential_family"] = str(
        get_config_value(config, "model.integer.sl.potential.family", default=out["potential_family"])
    )
    out["potential_kappa"] = float(
        get_config_value(config, "model.integer.sl.potential.kappa", default=out["potential_kappa"])
    )
    out["potential_x0"] = float(
        get_config_value(config, "model.integer.sl.potential.x0", default=out["potential_x0"])
    )
    out["potential_eps"] = float(
        get_config_value(config, "model.integer.sl.potential.eps", default=out["potential_eps"])
    )
    out["potential_power"] = float(
        get_config_value(config, "model.integer.sl.potential.power", default=out["potential_power"])
    )
    return out


def _uniform_u_indices(max_index: int, max_points: int) -> np.ndarray:
    if max_index <= 0:
        return np.array([0], dtype=np.int64)
    n = min(max_points, max_index + 1)
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    idx = np.rint(np.expm1(u * np.log1p(float(max_index)))).astype(np.int64)
    idx = np.unique(np.clip(idx, 0, max_index))
    return idx


def _effective_potential_v(
    *,
    family: str,
    support_size: int,
    kappa: float,
    x0: float,
    eps: float,
    power: float,
) -> np.ndarray:
    if family in {"none", "null", "off"} or kappa == 0.0:
        return np.zeros(support_size, dtype=np.float64)

    max_index = max(support_size - 1, 0)
    denom = np.log1p(float(max_index))
    u = (
        np.zeros(support_size, dtype=np.float64)
        if denom <= 0
        else np.log1p(np.arange(support_size, dtype=np.float64)) / denom
    )

    if family == "inverse_square":
        x = (1.0 + np.arange(support_size, dtype=np.float64)) + float(x0)
        x = np.maximum(x, 1e-12)
        return float(kappa) / (x * x)
    if family == "u_right_inverse_square":
        eps = max(float(eps), 1e-12)
        dist = 1.0 - u
        return float(kappa) / ((dist + eps) * (dist + eps))
    if family == "u_power":
        power = max(float(power), 0.0)
        return float(kappa) * np.power(u, power)

    raise ValueError(f"Unknown potential family: {family!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot SL conductance c, effective potential V (before multiplying by w), and "
            "leading eigenfunctions for a single integer column as Unicode terminal plots."
        )
    )
    p.add_argument("--results-json", type=Path, required=True)
    p.add_argument(
        "--hybrid-config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config path to load SL conductance/potential parameters. "
            "If omitted, tries to infer it from a sibling cmd.txt (via --hybrid-config)."
        ),
    )
    p.add_argument("--column", type=str, default="I5")
    p.add_argument(
        "--plot-k",
        type=int,
        default=5,
        help="Number of leading eigenfunctions to plot (k=0..plot_k-1).",
    )
    p.add_argument("--width", type=int, default=72)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--charset", type=str, default="braille", choices=["ascii", "block", "braille"])
    p.add_argument(
        "--max-points",
        type=int,
        default=256,
        help="Downsample points per curve before plotting.",
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

    results = payload.get("results") or []
    best_entry = results[0] if results else {}
    best = dict(best_entry.get("best_params") or {})

    # Defaults: match the Criteo hybrid setup (quantile cap + per-column LARGE token),
    # but allow config-based or Optuna-tuned overrides.
    sl_params: dict[str, Any] = {
        "cap_max": int(payload.get("sl_cap_max", 10_000_000)),
        "cap_mode": str(payload.get("sl_cap_mode", "quantile")),
        "positive_overflow": str(payload.get("sl_positive_overflow", "large_token")),
        "right_boundary": str(payload.get("sl_right_boundary", "neumann_midpoint")),
        "prior_count": float(payload.get("sl_prior_count", 0.5)),
        "cutoff_quantile": float(payload.get("sl_cutoff_quantile", 0.99)),
        "cutoff_factor": float(payload.get("sl_cutoff_factor", 1.1)),
        "curvature_alpha": float(payload.get("sl_curvature_alpha", 1.0)),
        "curvature_beta": float(payload.get("sl_curvature_beta", 0.0)),
        "curvature_center": float(payload.get("sl_curvature_center", 0.0)),
        "conductance_eps": float(payload.get("sl_conductance_eps", 1e-8)),
        "conductance_family": str(payload.get("sl_conductance_family", "u_exp_valley")),
        "uvalley_u0": float(payload.get("sl_u0", 0.0)),
        "uvalley_left_slope": float(payload.get("sl_left_slope", 1.0)),
        "uvalley_right_slope": float(payload.get("sl_right_slope", 1.0)),
        "potential_family": str(payload.get("sl_potential_family", "none")),
        "potential_kappa": float(payload.get("sl_potential_kappa", 0.0)),
        "potential_x0": float(payload.get("sl_potential_x0", 0.0)),
        "potential_eps": float(payload.get("sl_potential_eps", 0.01)),
        "potential_power": float(payload.get("sl_potential_power", 2.0)),
    }

    hybrid_cfg_path = _infer_hybrid_config(args.results_json, args.hybrid_config)
    if hybrid_cfg_path is not None:
        sl_params = _apply_sl_config_from_yaml(yaml_path=hybrid_cfg_path, defaults=sl_params)

    # Optuna-tuned overrides (if present).
    sl_params["cap_max"] = int(best.get("sl_cap_max", sl_params["cap_max"]))
    sl_params["cap_mode"] = str(best.get("sl_cap_mode", sl_params["cap_mode"]))
    sl_params["positive_overflow"] = str(
        best.get("sl_positive_overflow", sl_params["positive_overflow"])
    )
    sl_params["uvalley_u0"] = float(best.get("sl_u0", sl_params["uvalley_u0"]))
    sl_params["uvalley_left_slope"] = float(
        best.get("sl_left_slope", sl_params["uvalley_left_slope"])
    )
    sl_params["uvalley_right_slope"] = float(
        best.get("sl_right_slope", sl_params["uvalley_right_slope"])
    )
    sl_params["potential_family"] = str(
        best.get("sl_potential_family", sl_params["potential_family"])
    )
    sl_params["potential_kappa"] = float(
        best.get("sl_potential_kappa", sl_params["potential_kappa"])
    )
    sl_params["potential_x0"] = float(
        best.get("sl_potential_x0", sl_params["potential_x0"])
    )
    sl_params["potential_eps"] = float(
        best.get("sl_potential_eps", sl_params["potential_eps"])
    )
    sl_params["potential_power"] = float(
        best.get("sl_potential_power", sl_params["potential_power"])
    )

    num_basis = int(payload.get("sl_num_basis", 10))
    cfg = SLIntegerEncoderConfig(
        cap_max=int(sl_params["cap_max"]),
        num_basis=num_basis,
        prior_count=float(sl_params["prior_count"]),
        cutoff_quantile=float(sl_params["cutoff_quantile"]),
        cutoff_factor=float(sl_params["cutoff_factor"]),
        curvature_alpha=float(sl_params["curvature_alpha"]),
        curvature_beta=float(sl_params["curvature_beta"]),
        curvature_center=float(sl_params["curvature_center"]),
        conductance_eps=float(sl_params["conductance_eps"]),
        positive_overflow=str(sl_params["positive_overflow"]),
        cap_mode=str(sl_params["cap_mode"]),
        right_boundary=str(sl_params["right_boundary"]),
        conductance_family=str(sl_params["conductance_family"]),
        uvalley_u0=float(sl_params["uvalley_u0"]),
        uvalley_left_slope=float(sl_params["uvalley_left_slope"]),
        uvalley_right_slope=float(sl_params["uvalley_right_slope"]),
        potential_family=str(sl_params["potential_family"]),
        potential_kappa=float(sl_params["potential_kappa"]),
        potential_x0=float(sl_params["potential_x0"]),
        potential_eps=float(sl_params["potential_eps"]),
        potential_power=float(sl_params["potential_power"]),
    )

    potential_family = cfg.potential_family
    pot_kappa = cfg.potential_kappa
    pot_x0 = cfg.potential_x0
    pot_eps = cfg.potential_eps
    pot_power = cfg.potential_power

    series = _read_int_column(
        data_path=data_path,
        start_row=train_start,
        n_rows=n_rows,
        column=str(args.column),
    )

    terminal_plot = _load_terminal_plot_module(args.terminal_plot_script)

    encoder = SLIntegerEncoder(cfg).fit(series)
    support_size = int(encoder.basis_matrix.shape[0])
    cap_value = int(encoder.cap_value)
    plot_k = max(1, min(int(args.plot_k), encoder.num_basis, support_size))

    # Curves: plot against x in [1, cap_value] but sample indices uniformly in u=log1p(i)/log1p(max_i).
    max_index = max(support_size - 1, 0)
    idx_nodes = _uniform_u_indices(max_index=max_index, max_points=int(args.max_points))
    x_nodes = (1.0 + idx_nodes.astype(np.float64)).tolist()

    cs = encoder._compute_conductances(support_size)
    max_edge = max(support_size - 2, 0)
    idx_edges = _uniform_u_indices(max_index=max_edge, max_points=int(args.max_points))
    x_edges = (1.0 + idx_edges.astype(np.float64)).tolist()

    v = _effective_potential_v(
        family=potential_family,
        support_size=support_size,
        kappa=pot_kappa,
        x0=pot_x0,
        eps=pot_eps,
        power=pot_power,
    )

    print(
        f"SL shape for {args.column} on train rows [{train_start}, {train_end}) "
        f"(cap_value={cap_value}, support_size={support_size})"
    )
    if cfg.conductance_family == "u_exp_valley":
        print(
            "conductance=u_exp_valley("
            f"u0={cfg.uvalley_u0:.6g}, left={cfg.uvalley_left_slope:.6g}, right={cfg.uvalley_right_slope:.6g}"
            ")"
        )
    else:
        print(f"conductance={cfg.conductance_family}")
    print(
        f"potential={potential_family}(kappa={pot_kappa:.6g}, x0={pot_x0:.6g}, eps={pot_eps:.6g}, power={pot_power:.6g})"
    )
    print(f"plots: x in [1, {cap_value}] (sampled uniformly in u), charset={args.charset}")
    print()

    y_c = cs[idx_edges].astype(np.float64, copy=False).tolist()
    rendered_c = terminal_plot.render_xy_plot(
        x_edges,
        y_c,
        mode="line",
        width=int(args.width),
        height=int(args.height),
        title=f"{args.column} conductance c(x)",
        ascii_only=False,
        charset=str(args.charset),
    )
    print(rendered_c)
    print()

    y_v = v[idx_nodes].astype(np.float64, copy=False).tolist()
    rendered_v = terminal_plot.render_xy_plot(
        x_nodes,
        y_v,
        mode="line",
        width=int(args.width),
        height=int(args.height),
        title=f"{args.column} effective potential V(x) (pre-w)",
        ascii_only=False,
        charset=str(args.charset),
    )
    print(rendered_v)
    print()

    for k in range(plot_k):
        phi = encoder.basis_matrix[:, k].astype(np.float64, copy=False)
        y_phi = phi[idx_nodes].tolist()
        rendered_phi = terminal_plot.render_xy_plot(
            x_nodes,
            y_phi,
            mode="line",
            width=int(args.width),
            height=int(args.height),
            title=f"{args.column} phi_{k}(x)",
            ascii_only=False,
            charset=str(args.charset),
        )
        print(rendered_phi)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
