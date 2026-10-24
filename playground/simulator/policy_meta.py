"""Compatibility metadata stored next to every trained policy.

Byte-identical copies live in platform/training (written by trainers, checked
by the test track) and playground/simulator (checked before a robot executes
a policy); platform/training/test_parity.py fails CI if they drift.

A policy is only meaningful for the observation layout it was trained on.
Executing it on a different layout produces confident nonsense, so robots,
inference and evaluation refuse policies whose obs_version doesn't match.
"""

import json
from pathlib import Path
from typing import Optional

from observation import OBS_VERSION

META_FILENAME = "policy_meta.json"


def write_policy_meta(output_dir) -> Path:
    """Write policy_meta.json into a trainer's output dir (uploaded with the policy)."""
    path = Path(output_dir) / META_FILENAME
    path.write_text(json.dumps({"obs_version": OBS_VERSION}))
    return path


def read_policy_meta(path) -> Optional[dict]:
    """Read policy_meta.json; None if it doesn't exist."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def incompatibility(meta: Optional[dict]) -> Optional[str]:
    """Why a policy can't be executed here, or None if it's compatible."""
    if meta is None:
        return (f"no {META_FILENAME}: policy predates observation version {OBS_VERSION} "
                "(world-frame observations); retrain it")
    version = meta.get("obs_version")
    if version != OBS_VERSION:
        return f"policy uses observation version {version}, this build uses {OBS_VERSION}; retrain it"
    return None
