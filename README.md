# MSE-Check

A tool for evaluating policy server performance by computing Mean Squared Error (MSE) against ground-truth actions from Bridge V2 trajectories. Forked from [zhouzypaul/mse-check](https://github.com/zhouzypaul/mse-check).

## Overview

This tool provides a framework for:
- Evaluating policy server actions against ground truth trajectories
- Computing performance metrics including MSE
- Supporting both sequential and parallel action collection
- Handling historical context with configurable history lengths and sampling strategies

## Installation

### Dependencies
- numpy
- draccus
- json_numpy
- tqdm
- asyncio
- pickle

Install dependencies using:
```bash
uv pip install numpy draccus json-numpy tqdm
```

## Usage

1. First, ensure your policy server is running at your desired `<server_ip>:<server_port>`

2. Run the evaluation script:
```bash
python test_policy_client_mse.py --ip <server_ip> --port <server_port>
```

### Advanced Configuration

The tool supports several configuration options through the `DeployConfig` class:

```python
config = DeployConfig(
    host="localhost",          # Policy server IP address
    port=8000,                # Policy server port
    subsample=10,             # Subsample rate for trajectory steps
    save_dir="results",       # Directory to save results
    analyze_only=False,       # Only analyze previously saved results
    sequential=False,         # Process observations sequentially vs in parallel
    model_name="",           # Name of the model being evaluated
    external_history_length=None,  # Number of historical steps to include
    external_history_choice=None   # How to sample historical steps
)
```

### History Options

When using historical context, the following sampling strategies are available:
- `"all"`: Include all historical steps
- `"last"`: Include only the last step
- `"first"`: Include only the first step
- `"alternate"`: Include every other step
- `"third"`: Include every third step

## Output

The tool generates two main output files in the specified `save_dir`:
- `actions_{timestamp}.pkl`: Raw collected actions
- `metrics_{timestamp}.pkl`: Computed performance metrics

Each file includes:
- Trajectory index
- Language instructions (if available)
- Predicted actions
- Ground truth actions
- Number of actions
- VLM responses
- Timestamp
- Model name
