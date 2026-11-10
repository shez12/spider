# Local SimToolReal snapshot

This directory contains actual copied files, not links to the source checkout.
The copied environment and vendored SAPG files remain byte-identical to the
source snapshot. `COPY_MANIFEST.json` records the source HEAD, working-tree
status and SHA256 of every copied file. `README_LOCAL.md` itself is local
documentation added after the snapshot and is not included in that manifest.

Copied components:

- `isaacsimenvs/tasks/simtoolreal`: original action/reset/observation/reward,
  target sampling, termination and tolerance curriculum.
- `isaacsimenvs/cfg`, `isaacsimenvs/utils`, `isaacsimenvs/train.py`.
- `rl_games/rl_games`: the exact vendored SAPG implementation.
- `assets`: original robot, table and tool assets with local meshes.
- Necessary `isaacgymenvs` adjacency-map/path helpers and `simtoolreal_shared`.
- `reference/2026-09-14_14-35-39`: saved historical configuration and overrides.

The source was the current working tree, including its uncommitted changes,
not a reconstructed September source release. No training outputs or historical
checkpoints were copied here.

Experiment-specific right-hand/mustard adaptation lives outside this directory:
`/home/szc/work/spider/experiments/mustard_simtoolreal/`.
Isaac Sim / Isaac Lab / PyTorch still come from the installed conda environment;
those binary runtimes are not duplicated.
