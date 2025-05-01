"""
Test the sanity of the policy from a policy client
by computing various metrics against ground-truth actions from 10 trajs sampled from Bridge V2

Pulled from https://github.com/zhouzypaul/mse-check/blob/main/test_policy_client_mse.py

"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

import json_numpy
json_numpy.patch()
import requests
import asyncio
import aiohttp
import os
import pickle
import typing as t
from tqdm import tqdm
import argparse
import datetime
import time

from data_utils import load_data
from server_utils import PolicyClient, get_url
from eval_utils import evaluate_actions, analyze_saved_results

"""
Collect actions first, then compute metrics
"""
async def collect_actions(policy_client: PolicyClient, trajs: list[dict], subsample_rate: int = 20,  sequential: bool = False) -> list[dict]:
    all_actions = []
    await policy_client.async_init()
    
    for traj_idx, traj in enumerate(tqdm(trajs, desc=f"Collecting actions with subsample rate {subsample_rate}")):
        await policy_client.server_reset()
        obs_list = [t["images0"] for t in traj["observations"]]  # list of obs
        gt_actions = traj["actions"]  # list of ground truth actions
        language_instruction = traj["language"][0] if "language" in traj else None
        
        # Subsample the observations and ground truth actions
        subsampled_obs_list = obs_list[::subsample_rate]
        subsampled_gt_actions = gt_actions[::subsample_rate]
        
        pred_actions = []
        if sequential: 
            for i, obs in enumerate(subsampled_obs_list):
                obs_dict = {"image_primary": obs}
                # Use the synchronous call method instead of async_call
                action = policy_client(obs_dict, language_instruction)
                pred_actions.append(action)
            print(f"Collected {len(pred_actions)} actions for trajectory {traj_idx+1}")
        else:
            tasks = []
            for obs in subsampled_obs_list:
                obs_dict = {"image_primary": obs}
                tasks.append(policy_client.async_call(obs_dict, language_instruction))
            pred_actions = await asyncio.gather(*tasks)
            print(f"Collected {len(pred_actions)} actions for trajectory {traj_idx+1}")
            
        # Filter out any None values that might have been added during sequential processing
        valid_indices = [i for i, a in enumerate(pred_actions) if a is not None]
        pred_actions = [pred_actions[i] for i in valid_indices]
        
        # Make sure we have actions to process
        if not pred_actions:
            print(f"No valid actions collected for trajectory {traj_idx+1}, skipping")
            continue
            
        # Convert to numpy arrays for easier computation
        pred_actions = np.array(pred_actions)
        gt_actions = np.array(subsampled_gt_actions[:len(pred_actions)])  # Trim ground truth to match predictions
    
        # Save actions for this trajectory
        traj_actions = {
            'traj_idx': traj_idx,
            'instruction': language_instruction,
            'pred_actions': pred_actions,
            'gt_actions': gt_actions,
            'num_actions': len(gt_actions),
        }
        all_actions.append(traj_actions)
    
    return all_actions

async def async_evaluate_policy(policy_client: PolicyClient, trajs: list[dict],  subsample_rate: int = 10,  save_dir: str = "results",   sequential: bool = False,model_name: str = "") -> dict[str, t.Any]:
    """First collect actions, save them, then compute metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a timestamp string
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create file prefix with model name if provided
    file_prefix = f"{model_name}_{timestamp}" if model_name else timestamp
    
    # Step 1: Collect all actions
    all_actions = await collect_actions(policy_client, trajs, subsample_rate, sequential)
    
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
    results['timestamp'] = timestamp
    results['model_name'] = model_name
    results['files'] = {
        'actions': actions_file,
        'metrics': metrics_file
    }
    
    return results

def evaluate_policy(policy_client: PolicyClient,
                   trajs: list[dict],
                   subsample_rate: int = 10,
                   save_dir: str = "results",
                   sequential: bool = False,
                   model_name: str = "") -> dict[str, t.Any]:
    return asyncio.run(async_evaluate_policy(policy_client, trajs, subsample_rate, save_dir, sequential, model_name))

  
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test policy client by computing metrics')
    parser.add_argument('--ip', type=str, default='localhost', help='Policy server IP address')
    parser.add_argument('--port', type=int, default=8000, help='Policy server port')
    parser.add_argument('--subsample', type=int, default=10, help='Subsample rate for trajectory steps (every N steps)')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze previously saved results')
    parser.add_argument('--sequential', action='store_true', help='Process observations one at a time instead of in batch')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model being evaluated')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create policy client
    policy_client = PolicyClient(host=args.ip, port=args.port)
    print(f"Connecting to policy server at {get_url(args.ip, args.port, '')}")
    
    # Load data
    bridge_trajs = load_data()
    print(f"Loaded {len(bridge_trajs)} trajectories")
    
    # Evaluate policy
    try:
        results = evaluate_policy(
            policy_client, 
            bridge_trajs, 
            subsample_rate=args.subsample,
            save_dir=args.save_dir,
            sequential=args.sequential,
            model_name=args.model_name
        )
        
        # Additional analysis
        analyze_saved_results(args.save_dir, model_name=args.model_name)
        
    finally:
        # Clean up
        policy_client.close()