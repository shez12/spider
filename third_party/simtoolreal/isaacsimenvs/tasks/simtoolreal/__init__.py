"""SimToolReal task registration.

Registers ``Isaacsimenvs-SimToolReal-Direct-v0`` (KUKA + SharPa),
``Isaacsimenvs-SimToolRealYimu-Direct-v0`` (KUKA + Yimu-left), and
``Isaacsimenvs-SimToolRealYimuTraj-Direct-v0`` (Yimu + continuous
multi-subtarget goal trajectory, used by isaacsimenvs/eval_trajectory.py).

Entry points:
- ``env_cfg_entry_point``           → SimToolRealEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task YAML overlay
- ``rl_games_cfg_entry_point``      → cfg/train/SimToolRealPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/SimToolRealSAPG.yaml (default)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .simtoolreal_env import SimToolRealEnv
from .simtoolreal_env_cfg import SimToolRealEnvCfg

__all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

_TASK_KWARGS = {
    "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg:SimToolRealEnvCfg",
    "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
    "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
}

gym.register(
    id="Isaacsimenvs-SimToolReal-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        **_TASK_KWARGS,
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolReal.yaml"),
    },
)

gym.register(
    id="Isaacsimenvs-SimToolRealYimu-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        **_TASK_KWARGS,
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolRealYimu.yaml"),
    },
)

gym.register(
    id="Isaacsimenvs-SimToolRealYimuTraj-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        **_TASK_KWARGS,
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolRealYimuTraj.yaml"),
    },
)
