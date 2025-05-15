#!/usr/bin/env python3
"""
Hyperparameter sweep script for eval.py
Runs multiple configurations and analyzes the results
"""

import itertools
import multiprocessing
import typing as t
from dataclasses import dataclass

import draccus
from eval import DeployConfig, deploy


def run_single_experiment(params: dict[str, t.Any]) -> None:
    """Run a single experiment with the given parameters"""
    config = DeployConfig(**params)
    deploy(config)


def run_parallel_sweep(
    config: dict[str, t.Any],
    param_grid: dict[str, list[t.Any]],
    num_workers: int = 2,
) -> list[dict[str, t.Any]]:
    """Run a parallel sweep over the parameter grid"""
    # Create all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    values = [v if isinstance(v, list) else [v] for v in values]
    combinations = list(itertools.product(*values))
    print(f"combinations: {combinations}")

    # Create parameter dictionaries for each combination
    all_params = []
    for combo in combinations:
        params = {keys[i]: combo[i] for i in range(len(keys))}
        all_params.append({**config, **params})

    # Run experiments in parallel
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(run_single_experiment, all_params)

    return all_params


@dataclass
class SweepConfig:
    n_iters: int = 1
    host: str = "localhost"
    port: int = 8000
    sequential: bool = False
    model_name: str = "gpt-4o-mini"


@draccus.wrap()
def main(cfg: SweepConfig) -> None:
    history_lengths = [0, 1, 2, 4, 8, 16, 32]
    history_choices = ["all", "last", "first", "alternate", "third"]

    # Base configuration shared across sweeps
    base_config = {
        "host": cfg.host,
        "port": cfg.port,
        "sequential": cfg.sequential,
        "model_name": cfg.model_name,
    }
    param_grid = {
        "external_history_length": history_lengths,
        "external_history_choice": history_choices,
    }

    # History parameter sweep
    for _ in range(cfg.n_iters):
        run_parallel_sweep(config=base_config, param_grid=param_grid, num_workers=1)


if __name__ == "__main__":
    main()
