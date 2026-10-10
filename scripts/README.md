# Experiment guide

Run these commands from the repository root after the installation in [README](../README.md). Training evaluates held-out contexts at true starts. Commands use the existing `pcrc` package and condition names.

## Training entry points

| Script | Suggested config | Context-bank input |
|---|---|---|
| `pointmaze_sac_her.py` | `config/pointmaze_curriculum.yaml` | Built by the driver, or `--bank-path` |
| `minigrid_ppo.py` | `config/minigrid_doorkey.yaml` | Built by the driver, or `--bank-path` |
| `fetch_sac_her.py` | `config/fetch_pick_and_place.yaml` | `--bank-path results/banks/pnp_conflict_bank.json` |
| `metaworld_sac_her.py` | `config/metaworld_pickplace.yaml` | Built by the driver |

All four accept `--config`, `--exp-name`, `--conditions`, `--seed`, `--n-iterations`, and `--results-root`. Use `--help` for task-specific options. Configurations without a `scaffold` block (for example `pointmaze_uncertainty.yaml` and `metaworld_push.yaml`) are sampler-baseline experiments; use their configured sampler names rather than scaffold-controller conditions.

Method IDs and labels come from [`src/pcrc/registry.py`](../src/pcrc/registry.py). Use `unassisted` for target-start training in every task, and `trajectory_per_context_success` for trajectory-derived scaffolds.

## Fetch banks and demos

The README conflict command uses a committed bank and needs no calibration checkpoint. For the all-in-air bank expected by the default `fetch_pick_and_place.yaml` config:

```bash
uv run --no-sync python scripts/build_inair_pnp_bank.py
uv run --no-sync python scripts/fetch_sac_her.py \
    --config config/fetch_pick_and_place.yaml \
    --exp-name fetch_inair/s42 --conditions per_context_success,unassisted --seed 42
```

`build_inair_pnp_bank.py` writes `results/priority_1a_pnp_sparse_calibration/calibration/bank.json`. It samples simulator poses and does not need a trained model. `build_pnp_bank.py` builds mixed on-table/in-air banks; `build_pnp_withingroup_bank.py` builds in-air sub-band conflicts. Use their `--help` for construction parameters. `build_ontable_pnp_bank.py` expects its documented calibrated pool and is an optional historical workflow.

The `priority_1a` through `priority_4` scripts retain the older calibration, PPO, sparse-HER, and ACDC-lite experiments. Their `fetch_priority_*.yaml` configs require calibration outputs, and some require a starter checkpoint produced by `train_fetch_ppo_warmup.py`; they are separate from the README's from-scratch runs. Each script's `--help` lists required inputs.

`collect_pnp_demos.py` collects scripted Fetch demonstrations. The optional Meta-World demonstration configs require:

```bash
uv run --no-sync python scripts/collect_metaworld_demos.py \
    --env-id push-v3 --out results/mw_demos/push_demos.npz
```

Demonstration files are generated locally and ignored by Git.

## Automatic trajectory scaffolds

A ladder maps each training context to restorable states along a reference trajectory. Level zero remains the normal unassisted reset. Build and validate a DoorKey ladder:

```bash
uv run --no-sync python scripts/build_doorkey_traj_scaffold.py \
    --out results/banks/scaffold_ladders/doorkey_8x8_s0.json \
    --bank-out results/banks/doorkey_8x8_bank_s0.json
uv run --no-sync python scripts/validate_traj_scaffold.py \
    --task doorkey --bank results/banks/doorkey_8x8_bank_s0.json \
    --ladder results/banks/scaffold_ladders/doorkey_8x8_s0.json
uv run --no-sync python scripts/minigrid_ppo.py \
    --config config/minigrid_doorkey.yaml --exp-name doorkey_auto/s42 \
    --bank-path results/banks/doorkey_8x8_bank_s0.json \
    --ladder-path results/banks/scaffold_ladders/doorkey_8x8_s0.json \
    --conditions trajectory_per_context_success --seed 42
```

Other builders are `build_pointmaze_traj_scaffold.py` and `build_fetch_traj_scaffold.py`; their `--help` specifies bank and environment options. `build_self_traj_scaffold.py` combines successful policy rollouts harvested by the training drivers with a supplied fallback ladder. `build_coldstart_ladder.py` creates a true-start-only ladder for the cold-start control.

## Outputs and portability

Training outputs default to this checkout's `results/`. Override with `--results-root` or `PCRC_SCRATCH_ROOT` (which places outputs under `<scratch>/results`). Bank inputs and builder output arguments are paths relative to the working directory unless absolute. `scripts/spark_env.sh`, `lambda_env.sh`, and `farmshare_env.sh` configure their respective hosts. Farmshare training must use SLURM with `--qos=gpu` and large outputs in scratch.

The fixed task-bank split and paired seeds support comparisons, but new runs need not reproduce results bit for bit across simulator versions or hardware. The lockfile retains the development stack. Historical plan references in implementation comments describe development provenance; those private planning documents are not runtime inputs.
