#!/usr/bin/env python3
"""
Hyperparameter sweep script for test_policy_client_mse.py
Runs multiple configurations and analyzes the results
"""

import itertools
import multiprocessing
import typing as t

from test_policy_client_mse import DeployConfig, deploy


def run_single_experiment(params: dict[str, t.Any]) -> None:
    """Run a single experiment with the given parameters"""
    config = DeployConfig(**params)
    deploy(config)


def run_parallel_sweep(
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
        all_params.append(params)

    # Run experiments in parallel
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(run_single_experiment, all_params)

    return all_params


def history_sweep(
    base_config: t.Optional[dict[str, t.Any]] = None, num_workers: int = 2
) -> None:
    """Run a sweep over history length and history choice parameters"""
    if base_config is None:
        base_config = {}

    param_grid = {
        **base_config,
        "external_history_length": [10],
        "external_history_choice": ["all", "last", "first", "alternate", "third"],
    }

    run_parallel_sweep(param_grid=param_grid, num_workers=num_workers)


if __name__ == "__main__":
    # Example usage - uncomment the sweep you want to run

    # Base configuration shared across sweeps
    base_config = {
        "host": "localhost",
        "port": 8000,
        "sequential": False,
        "model_name": "gemini-2-5-pro",
    }

    # History parameter sweep
    history_sweep(base_config, num_workers=2)
