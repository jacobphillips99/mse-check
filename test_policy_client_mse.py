"""
Test the sanity of the policy from a policy client,
by computing various metrics against ground-truth actions from 10 trajs sampled from Bridge V2

Pulled from https://github.com/zhouzypaul/mse-check/blob/main/test_policy_client_mse.py
"""

import asyncio
import datetime
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import json_numpy
import numpy as np
from tqdm import tqdm
from utils.data import load_data
from utils.eval import analyze_saved_results, evaluate_actions
from utils.server import PolicyClient, get_url

json_numpy.patch()


RESULTS_DIR_PATH = str(Path(__file__).parent / "results")


def assemble_history_dict(
    step_idx: int,
    obs_list: list[np.ndarray],
    gt_actions: list[np.ndarray],
    external_history_length: int,
    external_history_choice: str,
) -> dict:
    """
    Assembles a history dict for a given step index from the full set of observations and actions.
    - external_history_length is the number of steps to look back
    - external_history_choice is the choice of steps to include in the history dict
        - "all": include all steps
        - "last": include only the last step
        - "first": include only the first step
        - "alternate": include every other step
        - "third": include every third step
    """
    historical_obs = obs_list[max(0, step_idx - external_history_length) : step_idx]
    historical_actions = gt_actions[
        max(0, step_idx - external_history_length) : step_idx
    ]
    assert external_history_choice in ["all", "last", "first", "alternate", "third"]
    if external_history_choice == "all":
        inds = np.arange(len(historical_obs))
    elif external_history_choice == "last":
        inds = [len(historical_obs) - 1]
    elif external_history_choice == "first":
        inds = [0]
    elif external_history_choice == "alternate":
        inds = np.arange(0, len(historical_obs), 2)
    elif external_history_choice == "third":
        inds = np.arange(0, len(historical_obs), 3)
    else:
        raise ValueError(f"Invalid external_history_choice: {external_history_choice}")
    print(
        f"selecting inds {inds} for history given external_history_choice"
        f"{external_history_choice} over {len(historical_actions)} actions"
    )
    history_dict = {
        # FIXME FIXME --> need to convert action numbers to strings w ECOT
        "steps": [
            {"description": str(historical_actions[i]), "images": [historical_obs[i]]}
            for i in inds
        ]
    }
    return history_dict


async def collect_actions(
    policy_client: PolicyClient,
    trajs: list[dict],
    subsample_rate: int = 20,
    sequential: bool = False,
    external_history_length: Optional[int] = None,
    external_history_choice: Optional[str] = None,
) -> list[dict]:
    """
    Collect actions from the policy client.
    We support collecting actions in parallel or sequentially, but limited by each episode.
    """
    all_actions = []
    await policy_client.async_init()

    for traj_idx, traj in enumerate(
        tqdm(trajs, desc=f"Collecting actions with subsample rate {subsample_rate}")
    ):
        await policy_client.server_reset()
        obs_list = [t["images0"] for t in traj["observations"]]  # list of obs
        gt_actions = traj["actions"]  # list of ground truth actions
        language_instruction = traj["language"][0] if "language" in traj else None

        # Subsample the observations and ground truth actions
        subsampled_inds = np.arange(0, len(obs_list), subsample_rate)
        subsampled_obs_list = obs_list[::subsample_rate]
        subsampled_gt_actions = gt_actions[::subsample_rate]

        payloads = []
        for i, obs in enumerate(subsampled_obs_list):
            obs_dict = {"image_primary": obs}
            if (
                external_history_length is not None
                and external_history_choice is not None
            ):
                step_idx = subsampled_inds[i]
                history_dict = assemble_history_dict(
                    step_idx,
                    obs_list,
                    gt_actions,
                    external_history_length,
                    external_history_choice,
                )
            else:
                history_dict = None
            payloads.append(
                dict(
                    obs_dict=obs_dict,
                    language_instruction=language_instruction,
                    history_dict=history_dict,
                )
            )

        print(
            f"setup {len(payloads)} payloads for {'sequential' if sequential else 'parallel'} execution"
        )

        if sequential:
            results = []
            for i, payload in enumerate(payloads):
                if i < 2:
                    continue
                res = policy_client(**payload)
                results.append(res)
        else:
            results = await asyncio.gather(
                *(policy_client.async_call(**payload) for payload in payloads),
                return_exceptions=False,
            )

        pred_actions = [res[0] for res in results]
        vlm_responses = [res[1] for res in results]
        print(f"Collected {len(pred_actions)} actions for trajectory {traj_idx+1}")

        # Filter out any None values that might have been added during sequential processing
        valid_indices = [i for i, a in enumerate(pred_actions) if a is not None]
        collected_pred_actions = [pred_actions[i] for i in valid_indices]
        collected_gt_actions = [subsampled_gt_actions[i] for i in valid_indices]
        breakpoint()
        # Make sure we have actions to process
        if not collected_pred_actions:
            print(f"No valid actions collected for trajectory {traj_idx+1}, skipping")
            continue

        traj_actions = {
            "traj_idx": traj_idx,
            "instruction": language_instruction,
            "pred_actions": np.array(collected_pred_actions),
            "gt_actions": np.array(collected_gt_actions),
            "num_actions": len(collected_gt_actions),
            "vlm_responses": vlm_responses,
        }
        all_actions.append(traj_actions)

    return all_actions


async def async_evaluate_policy(
    policy_client: PolicyClient,
    trajs: list[dict],
    subsample_rate: int = 10,
    save_dir: str = RESULTS_DIR_PATH,
    sequential: bool = False,
    model_name: str = "",
    external_history_length: Optional[int] = None,
    external_history_choice: Optional[str] = None,
) -> str:
    """First collect actions, save them, then compute metrics"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a timestamp string
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create file prefix with model name if provided
    file_prefix = f"{model_name}_{timestamp}" if model_name else timestamp

    # Step 1: Collect all actions
    all_actions = await collect_actions(
        policy_client,
        trajs,
        subsample_rate,
        sequential,
        external_history_length,
        external_history_choice,
    )

    # Step 2: Save raw actions to disk before computing any metrics
    actions_file = f"{save_dir}/actions_{file_prefix}.pkl"
    print(f"Saving {len(all_actions)} trajectories of actions to {actions_file}")
    with open(actions_file, "wb") as f:
        pickle.dump(all_actions, f)

    # Step 3: Evaluate the actions
    results = evaluate_actions(all_actions)

    # Save the computed metrics
    metrics_file = f"{save_dir}/metrics_{file_prefix}.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(results, f)

    # Also save the files with timestamp and model name in the results object
    results["timestamp"] = timestamp
    results["model_name"] = model_name
    results["files"] = {"actions": actions_file, "metrics": metrics_file}
    return metrics_file


def evaluate_policy(
    policy_client: PolicyClient,
    trajs: list[dict],
    subsample_rate: int = 10,
    save_dir: str = "results",
    sequential: bool = False,
    model_name: str = "",
    external_history_length: Optional[int] = None,
    external_history_choice: Optional[str] = None,
) -> str:
    """
    Evaluate the policy using the policy client.
    """
    return asyncio.run(
        async_evaluate_policy(
            policy_client,
            trajs,
            subsample_rate,
            save_dir,
            sequential,
            model_name,
            external_history_length,
            external_history_choice,
        )
    )


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "localhost"  # Policy server IP address
    port: int = 8000  # Policy server port
    subsample: int = 10  # Subsample rate for trajectory steps (every N steps)
    save_dir: str = (
        RESULTS_DIR_PATH  # Directory to save results relative to script location
    )
    analyze_only: bool = False  # Only analyze previously saved results
    sequential: bool = False  # Process observations one at a time instead of in batch
    model_name: str = ""  # Name of the model being evaluated
    external_history_length: Optional[int] = (
        None  # Optional history length (how many steps to remember)
    )
    external_history_choice: Optional[str] = None


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    # Create policy client
    policy_client = PolicyClient(host=cfg.host, port=cfg.port)
    print(f"Connecting to policy server at {get_url(cfg.host, cfg.port, '')}")

    # Load data
    bridge_trajs = load_data()
    print(f"Loaded {len(bridge_trajs)} trajectories")

    try:
        if cfg.analyze_only:
            analyze_saved_results(cfg.save_dir, model_name=cfg.model_name)
        else:
            metrics_file = evaluate_policy(
                policy_client,
                bridge_trajs,
                subsample_rate=cfg.subsample,
                save_dir=cfg.save_dir,
                sequential=cfg.sequential,
                model_name=cfg.model_name,
                external_history_length=cfg.external_history_length,
                external_history_choice=cfg.external_history_choice,
            )
        print(f"Metrics file saved to {metrics_file}")
        analyze_saved_results(cfg.save_dir, model_name=cfg.model_name)
    finally:
        policy_client.close()


if __name__ == "__main__":
    deploy()
