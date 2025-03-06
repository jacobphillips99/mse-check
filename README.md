# mse-check
Check the policy server's action MSE against 10 BridgeV2 trajs. The provides a sanity check to the performance of the policy.

## Usage
First, host your policy server at `<server_ip>:<server_port>`
```bash
python test_policy_client_mse.py --ip <server_ip> --port <server_port>
```

## Dependencies
- numpy