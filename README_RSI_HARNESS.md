# Drone RSI Harness

This harness makes the RSI loop explicit and runnable before the real Unreal /
Colosseum adapter is ready.

## What Improves

1. MongoDB route memory: exact successful trajectories are cached and replayed.
2. Rollout-derived training data: high-reward state/action pairs are exported as
   `prompt/completion` JSONL for SFT or NTK-Mirror.
3. NTK-Mirror behavior controllers: tiny controller artifacts can be trained
   from successful rollouts and registered in MongoDB.

Base Gemma weights, Unreal, and drone physics do not improve.

## Runtime Flow

```text
mission -> retrieve Mongo memory/controllers -> build prompt with K-step context
        -> Gemma outputs rung-3 action JSON -> harness executes in sim
        -> verifier scores -> Mongo logs rollout -> best data exported
        -> optional NTK-Mirror fit -> evaluate/promote controller metadata
```

## Commands

Run one local episode with the built-in simulated environment:

```powershell
python drone_rsi_harness.py episode --request "Fly to the Ferry Building and return safely" --goal "Ferry Building, San Francisco"
```

Run a small autonomous RSI cycle:

```powershell
python drone_rsi_harness.py cycle --rollouts-per-mission 4 --max-steps 40 --train-controller
```

Export high-reward training examples:

```powershell
python drone_rsi_harness.py export-ntk --out runs/ntk/drone_train.jsonl --min-reward 0.2
```

Fit an NTK-Mirror controller if `ntkmirror` is installed and the selected model
is supported by its text-only `AutoModelForCausalLM` path:

```powershell
python drone_rsi_harness.py train-ntk --train runs/ntk/drone_train.jsonl --out runs/ntk/drone_controller.pt
```

## MongoDB Collections

- `mission_runs`: attempt summaries for dashboard and evaluation.
- `trajectories`: exact successful action sequences for route replay.
- `lessons`: compact failure lessons injected into future prompts.
- `rsi_rollouts`: full structured trajectories with step-level state/action/reward.
- `rsi_training_examples`: `prompt/completion` rows generated from high-reward steps.
- `rsi_controllers`: NTK-Mirror controller registry and metrics.
- `rsi_policy_versions`: active policy/controller records.

## Notes

The default environment is a deterministic local simulator so the harness can
be tested immediately. Replace `SimulatedDroneEnvironment` with
`ColosseumAirSimEnvironment` once the Unreal process is reachable.

NTK-Mirror here is used as RL-derived SFT: grouped rollouts provide reward
ranking, then the harness distills the best behavior into tiny supervised
controllers. It is not PPO. PPO should be skipped for the hackathon unless the
basic loop is already working.
