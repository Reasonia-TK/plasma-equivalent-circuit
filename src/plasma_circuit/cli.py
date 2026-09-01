"""Command-line entry point for the reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasma_circuit.config import load_config
from plasma_circuit.coupling import (
    coupled_result_to_dict,
    matched_result_to_dict,
    solve_self_consistent_density,
    solve_self_consistent_matching,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/schmidt2018.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/schmidt2018/paper_caps"))
    parser.add_argument(
        "--optimize-matching",
        action="store_true",
        help="alternate global-model convergence with the paper's L-network matching update",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.optimize_matching:
        result = solve_self_consistent_matching(config, args.output)
        summary = matched_result_to_dict(result)
        converged = result.converged and result.coupled_result.converged
    else:
        result = solve_self_consistent_density(config, args.output)
        summary = coupled_result_to_dict(result)
        converged = result.converged
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
