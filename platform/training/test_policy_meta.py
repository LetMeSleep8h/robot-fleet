"""Policies carry the observation version they were trained on; anything
trained on a different version must be refused, never executed."""

import ast
import json
from pathlib import Path

import policy_meta as pm
from observation import OBS_VERSION

TRAINING = Path(__file__).resolve().parent


def test_round_trip(tmp_path):
    path = pm.write_policy_meta(tmp_path)
    assert path.name == pm.META_FILENAME
    meta = pm.read_policy_meta(path)
    assert meta == {"obs_version": OBS_VERSION}
    assert pm.incompatibility(meta) is None


def test_missing_meta_is_incompatible(tmp_path):
    meta = pm.read_policy_meta(tmp_path / pm.META_FILENAME)
    assert meta is None
    assert "policy_meta.json" in pm.incompatibility(meta)


def test_other_version_is_incompatible(tmp_path):
    (tmp_path / pm.META_FILENAME).write_text(json.dumps({"obs_version": 2}))
    reason = pm.incompatibility(pm.read_policy_meta(tmp_path / pm.META_FILENAME))
    assert reason is not None and "2" in reason and str(OBS_VERSION) in reason


def test_every_trainer_writes_policy_meta():
    for script in ["train_locomotion.py", "train_offline.py", "train_reach.py"]:
        tree = ast.parse((TRAINING / script).read_text())
        calls = {getattr(n.func, "id", getattr(n.func, "attr", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "write_policy_meta" in calls, f"{script} uploads policies without policy_meta.json"
