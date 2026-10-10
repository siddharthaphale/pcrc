# Context banks and scaffold fixtures

The JSON files in this directory are simulator task inputs, not training-result logs. They specify context seeds/goals, train/held-out splits, and (where applicable) difficulty groups. Existing filenames are retained so configurations and tests can resolve them.

- `pnp_conflict_bank.json` supplies the README's mixed on-table/in-air Fetch conflict.
- `pnp_withingroup_bank.json` and `pnp_inair2band_bank.json` contain within-group Fetch variants.
- `pointmaze_*.json` supply saved navigation banks; the main PointMaze driver can also build its own bank.
- `doorkey*.json` include DoorKey layouts, candidate-pool features, and easy/hard group assignments.
- `metaworld_pickplace_bank_seed0.json` and `antmaze_large_feas.json` are additional experiment inputs.
- `scaffold_ladders/doorkey_8x8_coldstart.json` is the retained true-start-only control.
- `scaffold_ladders/doorkey_conflict_{easy,hard}_auto_s0.json` are deterministic BFS ladders regenerated from the corresponding bundled sub-banks for the integration tests. They cover training contexts only.

The easy/hard ladder fixtures are regenerated using the existing builder API:

```bash
uv run --no-sync python - <<'PYCODE'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from build_doorkey_traj_scaffold import build_ladder
from pcrc.envs.minigrid_env import MiniGridTaskBank
base = Path("results/banks")
for group in ("easy", "hard"):
    bank = MiniGridTaskBank.load(base / f"doorkey_conflict_{group}_s0.json")
    ladder, _ = build_ladder(bank, 160)
    ladder.save(base / "scaffold_ladders" / f"doorkey_conflict_{group}_auto_s0.json")
PYCODE
```

Other trajectory ladders and demo files are generated on demand with the scripts in the [experiment guide](../../scripts/README.md). Pair each ladder with the exact bank used to build it: sub-bank context IDs can differ from full-bank IDs.
