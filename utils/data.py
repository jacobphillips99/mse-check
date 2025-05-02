from pathlib import Path

import numpy as np

"""
load the data -- some weird tenses and mispellings are tripping up smarter models
"""
TASK_EDITS = {
    0: "open the drawer",
    2: "remove the green thing from the drawer and place it on the left side of the table",
    4: "take the blue stuffed animal and leave it inside the drawer",
    9: "remove the blue object from the drawer and put it on the lower left side of the table",
}

BRIDGE_FILE_PATH = Path(__file__).parent.parent / "bridge_v2_10_trajs.npy"


def load_data() -> np.ndarray:
    bridge_trajs = np.load(BRIDGE_FILE_PATH, allow_pickle=True)
    for k, v in TASK_EDITS.items():
        bridge_trajs[k]["language"][0] = v
    return bridge_trajs
