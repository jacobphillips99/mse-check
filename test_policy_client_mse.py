"""
Test the sanity of the policy from a policy client
by computing the action MSE against ground-truth actions from 10 trajs sampled from Bridge V2
"""
import numpy as np
import requests
from typing import Dict, Any, Optional
from tqdm import tqdm
import argparse

"""
load the data
"""
def load_data():
    return np.load("bridge_v2_10_trajs.npy", allow_pickle=True)

"""
prepare to query the policy client for actions
"""
class PolicyClient:
    """
    A simple client to query actions from the policy server
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._session = requests.Session()
        
        # Patch json to handle numpy arrays
        try:
            import json_numpy
            json_numpy.patch()
        except ImportError:
            print("Warning: json_numpy not found. Numpy arrays may not serialize correctly.")

    def __call__(self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None):
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict["image_primary"].shape

        # Use the session for connection reuse
        action = self._session.post(
            f"http://{self.host}:{self.port}/act",
            json={
                "image": obs_dict["image_primary"],
                "instruction": language_instruction,
            },
        ).json()

        # the original action is not modifiable, cannot clip boundaries after the fact for example
        if type(action) not in (np.ndarray, list):
            raise RuntimeError(
                "Policy server returned invalid action. It must return a numpy array or a list. Received: "
                + str(action)
            )
        return action.copy()
    
    def close(self):
        """Explicitly close connections"""
        if hasattr(self, "_session"):
            self._session.close()

    def __del__(self):
        self.close()

"""
compute the action MSE
"""
def compute_action_mse(policy_client, trajs):
    total_mse = 0
    total_actions = 0
    
    for traj_idx, traj in enumerate(tqdm(trajs, desc="Processing trajectories")):
        obs_list = [t["images0"] for t in traj["observations"]]  # list of obs
        gt_actions = traj["actions"]  # list of ground truth actions
        language_instruction = traj["language"][0] if "language" in traj else None
        
        pred_actions = []
        for obs in tqdm(obs_list, desc=f"Traj {traj_idx+1}", leave=False):
            # Create observation dictionary
            obs_dict = {"image_primary": obs}
            
            # Query the policy client for actions
            pred_action = policy_client(obs_dict, language_instruction)
            pred_actions.append(pred_action)
        
        # Convert to numpy arrays for easier computation
        pred_actions = np.array(pred_actions)
        gt_actions = np.array(gt_actions)
        
        # Compute MSE for this trajectory
        traj_mse = np.mean(np.square(pred_actions - gt_actions))
        print(f"Trajectory {traj_idx+1} MSE: {traj_mse:.6f}")
        
        total_mse += np.sum(np.square(pred_actions - gt_actions))
        total_actions += len(gt_actions)
    
    # Compute overall MSE
    overall_mse = total_mse / total_actions
    print(f"Overall MSE: {overall_mse:.6f}")
    return overall_mse
  
  
def parse_args():
    parser = argparse.ArgumentParser(description='Test policy client by computing action MSE')
    parser.add_argument('--ip', type=str, default='localhost', help='Policy server IP address')
    parser.add_argument('--port', type=int, default=8000, help='Policy server port')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create policy client
    policy_client = PolicyClient(host=args.ip, port=args.port)
    print(f"Connecting to policy server at {args.ip}:{args.port}")
    
    # Load data
    bridge_trajs = load_data()
    print(f"Loaded {len(bridge_trajs)} trajectories")
    
    # Compute action MSE
    try:
        overall_mse = compute_action_mse(policy_client, bridge_trajs)
        print(f"Overall MSE: {overall_mse:.6f}")
    finally:
        # Clean up
        policy_client.close()