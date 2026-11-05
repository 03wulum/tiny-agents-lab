"""Load experiment 01's code (simulator, features, models, training and evaluation) for experiment 02.

02 changes only switches that 01's code already has (all off by default, so 01 stays reproducible):
the fixed clause parser + relative clause ids (features.CueState(relative=True)) and float16
checkpoint storage (model_torch.save_npz(dtype="float16")).
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP01 = HERE.parent / "01-pocket-actions"
if str(EXP01) not in sys.path:
    sys.path.insert(1, str(EXP01))   # after this folder, so 02's criteria.py wins


def load01(module):
    """Import 01's train.py / run.py under another name (02 has files with the same names)."""
    spec = importlib.util.spec_from_file_location(f"{module}01", EXP01 / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
