import os
import pickle
import typing as t

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(
    pred_actions: np.ndarray, gt_actions: np.ndarray
) -> dict[str, float]:
    """Compute various metrics between predicted and ground truth actions"""
    results = {}

    # MSE - Mean Squared Error
    breakpoint()
    results["mse"] = np.mean(np.square(pred_actions - gt_actions))

    # MAE - Mean Absolute Error
    results["mae"] = np.mean(np.abs(pred_actions - gt_actions))

    # Normalized MSE - divide by variance of ground truth
    gt_var = np.var(gt_actions)
    if gt_var > 0:
        results["nmse"] = results["mse"] / gt_var
    else:
        results["nmse"] = float("inf")

    # Cosine similarity between flattened vectors
    pred_flat = pred_actions.reshape(-1)
    gt_flat = gt_actions.reshape(-1)

    pred_norm = np.linalg.norm(pred_flat)
    gt_norm = np.linalg.norm(gt_flat)

    if pred_norm > 0 and gt_norm > 0:
        results["cosine_sim"] = np.dot(pred_flat, gt_flat) / (pred_norm * gt_norm)
    else:
        results["cosine_sim"] = 0

    # R-squared - coefficient of determination
    ss_tot = np.sum(np.square(gt_flat - np.mean(gt_flat)))
    ss_res = np.sum(np.square(gt_flat - pred_flat))

    if ss_tot > 0:
        results["r2"] = 1 - (ss_res / ss_tot)
    else:
        results["r2"] = 0

    # Calculate correlations
    try:
        results["pearson_r"], results["pearson_p"] = pearsonr(pred_flat, gt_flat)
    except Exception as e:
        print(f"Error computing Pearson correlation: {e}")
        results["pearson_r"], results["pearson_p"] = 0, 1

    try:
        results["spearman_r"], results["spearman_p"] = spearmanr(pred_flat, gt_flat)
    except Exception as e:
        print(f"Error computing Spearman correlation: {e}")
        results["spearman_r"], results["spearman_p"] = 0, 1

    # Action magnitude comparison
    results["pred_mag"] = np.mean(np.linalg.norm(pred_actions, axis=1))
    results["gt_mag"] = np.mean(np.linalg.norm(gt_actions, axis=1))
    results["mag_ratio"] = (
        results["pred_mag"] / results["gt_mag"]
        if results["gt_mag"] > 0
        else float("inf")
    )

    return results


def evaluate_actions(all_actions: list[dict]) -> dict[str, t.Any]:
    """Compute metrics on previously collected actions"""
    all_results = []

    total_sq_diff = 0
    total_actions = 0
    total_inference_time = 0
    first_call_times = []
    other_call_times = []

    for traj_data in all_actions:
        traj_idx = traj_data["traj_idx"]
        pred_actions = traj_data["pred_actions"]
        gt_actions = traj_data["gt_actions"]

        # Extract first call time

        # Compute metrics for this trajectory
        results = compute_metrics(pred_actions, gt_actions)
        results["traj_idx"] = traj_idx
        results["num_actions"] = traj_data["num_actions"]

        all_results.append(results)

        # Log results
        print(f"Trajectory {traj_idx+1} metrics:")
        print(f"  MSE: {results['mse']:.6f} (n={traj_data['num_actions']})")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  Normalized MSE: {results['nmse']:.6f}")
        print(f"  Cosine similarity: {results['cosine_sim']:.6f}")
        print(f"  R²: {results['r2']:.6f}")
        print(f"  Pearson r: {results['pearson_r']:.6f} (p={results['pearson_p']:.6f})")
        print(
            f"  Action magnitude - Pred: {results['pred_mag']:.6f}, GT: {results['gt_mag']:.6f}, Ratio: {results['mag_ratio']:.6f}"
        )

        # Update totals for weighted average calculation
        total_sq_diff += np.sum(np.square(pred_actions - gt_actions))
        total_actions += traj_data["num_actions"]

    # Compute overall MSE (weighted by number of actions in each trajectory)
    if total_actions == 0:
        print("Warning: No actions were processed successfully")
        return {"results": all_results}

    overall_mse = total_sq_diff / total_actions
    print(f"Overall MSE (weighted): {overall_mse:.6f}")

    # Compute first call and other call averages
    avg_first_call = np.mean(first_call_times) if first_call_times else 0
    avg_other_calls = np.mean(other_call_times) if other_call_times else 0

    print(f"\nAverage first call time: {avg_first_call*1000:.2f}ms")
    print(f"Average time (excluding first calls): {avg_other_calls*1000:.2f}ms")

    # Compute average of other metrics
    avg_metrics = {
        k: np.mean([r[k] for r in all_results])
        for k in all_results[0]
        if k not in ["traj_idx", "num_actions"]
    }
    print("\nAverage metrics across all trajectories:")
    for k, v in avg_metrics.items():
        if "time" in k:
            if "total" in k:
                print(f"  {k}: {v:.2f}s")
            else:
                print(f"  {k}: {v*1000:.2f}ms")
        else:
            print(f"  {k}: {v:.6f}")

    # Add overall timing metrics
    print(
        f"\nTotal inference time across all trajectories: {total_inference_time:.2f}s"
    )
    print(f"Throughput: {total_actions/total_inference_time:.2f} actions/sec")

    return {
        "results": all_results,
        "avg_metrics": avg_metrics,
        "first_call_avg": avg_first_call,
        "other_calls_avg": avg_other_calls,
        "total_inference_time": total_inference_time,
        "throughput": total_actions / total_inference_time,
    }


def analyze_saved_results(
    save_dir: str = "results",
    timestamp: t.Optional[str] = None,
    host: t.Optional[str] = None,
    port: t.Optional[int] = None,
    model_name: t.Optional[str] = None,
) -> None:
    """Analyze saved results to identify potential issues"""
    try:
        # If all specific identifiers are provided, use them to load specific files
        if timestamp and host and port:
            file_prefix = f"{host}_{port}_{timestamp}"
            if model_name:
                file_prefix = f"{model_name}_{file_prefix}"
            actions_file = f"{save_dir}/actions_{file_prefix}.pkl"
            metrics_file = f"{save_dir}/metrics_{file_prefix}.pkl"
        elif timestamp and model_name:
            # Find files with matching timestamp and model name
            files = os.listdir(save_dir)
            matching_files = [
                f
                for f in files
                if timestamp in f
                and model_name in f
                and f.startswith("actions_")
                and f.endswith(".pkl")
            ]
            if not matching_files:
                print(
                    f"No files found with timestamp {timestamp} and model {model_name}"
                )
                return
            actions_file = f"{save_dir}/{matching_files[0]}"
            prefix = matching_files[0].replace("actions_", "").replace(".pkl", "")
            metrics_file = f"{save_dir}/metrics_{prefix}.pkl"
        elif timestamp:
            # Find files with matching timestamp
            files = os.listdir(save_dir)
            matching_files = [
                f
                for f in files
                if timestamp in f and f.startswith("actions_") and f.endswith(".pkl")
            ]
            if not matching_files:
                print(f"No files found with timestamp {timestamp}")
                return
            actions_file = f"{save_dir}/{matching_files[0]}"
            prefix = matching_files[0].replace("actions_", "").replace(".pkl", "")
            metrics_file = f"{save_dir}/metrics_{prefix}.pkl"
        elif model_name:
            # Find the most recent files for the specific model
            files = os.listdir(save_dir)
            action_files = [
                f
                for f in files
                if model_name in f and f.startswith("actions_") and f.endswith(".pkl")
            ]

            if not action_files:
                print(f"No files found for model {model_name}")
                return
            # Sort files by timestamp (most recent first)
            action_files.sort(reverse=True)
            prefix = action_files[0].replace("actions_", "").replace(".pkl", "")
            actions_file = f"{save_dir}/{action_files[0]}"
            metrics_file = f"{save_dir}/metrics_{prefix}.pkl"
        else:
            # Find the most recent files if no identifiers are provided
            files = os.listdir(save_dir)
            action_files = [
                f for f in files if f.startswith("actions_") and f.endswith(".pkl")
            ]

            if not action_files:
                # Fall back to default filename if no timestamped files exist
                actions_file = f"{save_dir}/actions.pkl"
                metrics_file = f"{save_dir}/metrics.pkl"
            else:
                # Sort files by timestamp (most recent first)
                action_files.sort(reverse=True)
                prefix = action_files[0].replace("actions_", "").replace(".pkl", "")
                actions_file = f"{save_dir}/{action_files[0]}"
                metrics_file = f"{save_dir}/metrics_{prefix}.pkl"

        print(f"Analyzing results from: {actions_file}")
        with open(actions_file, "rb") as f:
            all_actions = pickle.load(f)

        # If metrics haven't been calculated yet, do it now
        try:
            with open(metrics_file, "rb") as fr:
                metrics_data = pickle.load(fr)
                print("Loaded pre-computed metrics")
        except (FileNotFoundError, EOFError):
            print("Computing metrics from saved actions")
            metrics_data = evaluate_actions(all_actions)
            with open(metrics_file, "wb") as fw:
                pickle.dump(metrics_data, fw)

        # Check for zero policy
        zero_policy = True
        for traj in all_actions:
            if np.any(traj["pred_actions"] != 0):
                zero_policy = False
                break

        if zero_policy:
            print("WARNING: Your policy appears to be outputting all zeros!")

        # Check action magnitudes
        pred_mags = [
            np.mean(np.linalg.norm(traj["pred_actions"], axis=1))
            for traj in all_actions
        ]
        gt_mags = [
            np.mean(np.linalg.norm(traj["gt_actions"], axis=1)) for traj in all_actions
        ]

        avg_pred_mag = np.mean(pred_mags)
        avg_gt_mag = np.mean(gt_mags)

        print("\nAction Magnitude Analysis:")
        print(
            f"Average action magnitude - Predicted: {avg_pred_mag:.6f}, Ground Truth: {avg_gt_mag:.6f}"
        )
        print(f"Ratio (Pred/GT): {avg_pred_mag/avg_gt_mag:.6f}")

        if avg_pred_mag < 0.1 * avg_gt_mag:
            print(
                "WARNING: Predicted actions are much smaller than ground truth actions"
            )
        elif avg_pred_mag > 2.0 * avg_gt_mag:
            print(
                "WARNING: Predicted actions are much larger than ground truth actions"
            )

        # Calculate dimension-wise correlations
        print("\nDimension-wise Analysis:")
        action_dim = all_actions[0]["pred_actions"].shape[1]
        for dim in range(action_dim):
            dim_corrs = []
            dim_errors = []
            for traj in all_actions:
                pred_dim = traj["pred_actions"][:, dim]
                gt_dim = traj["gt_actions"][:, dim]
                dim_errors.append(np.mean(np.square(pred_dim - gt_dim)))
                try:
                    r, _ = pearsonr(pred_dim, gt_dim)
                    dim_corrs.append(r)
                except Exception as e:
                    print(
                        f"Error computing Pearson correlation for dimension {dim}: {e}"
                    )
                    pass

            if dim_corrs:
                print(f"Action dimension {dim}:")
                print(f"  Correlation: {np.mean(dim_corrs):.6f}")
                print(f"  MSE: {np.mean(dim_errors):.6f}")

            print(f"Analyzed results from: {actions_file}")

    except Exception as e:
        print(f"Error analyzing saved results: {e}")
